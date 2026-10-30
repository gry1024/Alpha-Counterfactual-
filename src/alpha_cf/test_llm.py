"""Streaming response handling, without network or market data."""
from contextlib import redirect_stdout
from io import StringIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import httpx
from openai import APIError

from .trainer import AlphaCFTrainer


class StreamingTests(unittest.TestCase):
    def trainer(self, chunks):
        trainer = AlphaCFTrainer.__new__(AlphaCFTrainer)
        trainer.args = SimpleNamespace(llm_attempts=1, llm_timeout=30,
                                       llm_max_tokens=16384, temperature=0.5)
        trainer.model = "test-model"
        trainer.record = Mock()
        trainer.client = Mock()
        trainer.client.chat.completions.create.return_value.__enter__ = Mock(return_value=iter(chunks))
        trainer.client.chat.completions.create.return_value.__exit__ = Mock(return_value=False)
        return trainer

    def chunk(self, content=None, reasoning=None, finish=None):
        return SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content, reasoning_content=reasoning), finish_reason=finish)])

    def ask(self, trainer):
        return trainer.ask("Generate {parent}", {"parent": {"expression": "$close", "mechanisms": []}})

    def test_reasoning_is_separate_and_output_budget_is_explicit(self):
        result = {"updated_understanding": "A hypothesis", "offspring": []}
        trainer = self.trainer([self.chunk(reasoning="private reasoning"),
                                self.chunk(content=json.dumps(result), finish="stop")])
        self.assertEqual(self.ask(trainer), result)
        self.assertEqual(trainer.client.chat.completions.create.call_args.kwargs["max_tokens"], 16384)
        response = next(c.kwargs for c in trainer.record.call_args_list if c.args == ("llm_response",))
        self.assertEqual(response["reasoning_chars"], len("private reasoning"))
        self.assertGreater(response["content_chars"], 0)

    def test_truncation_reports_budget_instead_of_empty_json(self):
        trainer = self.trainer([self.chunk(reasoning="unfinished", finish="length")])
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "output token limit.*16384"):
            self.ask(trainer)

    def test_provider_stream_error_retries_then_accepts_final_json(self):
        result = {"updated_understanding": "A hypothesis", "offspring": []}
        trainer = self.trainer([])
        trainer.args.llm_attempts = 2
        trainer.args.llm_retry_wait = 1
        failure = APIError("An error occurred during streaming",
                           httpx.Request("POST", "http://example.invalid"),
                           body={"message": "temporary upstream error"})
        def broken_stream():
            yield self.chunk(reasoning="partial thinking")
            raise failure
        failed = Mock()
        failed.__enter__ = Mock(return_value=broken_stream())
        failed.__exit__ = Mock(return_value=False)
        success = Mock()
        success.__enter__ = Mock(return_value=iter([self.chunk(content=json.dumps(result), finish="stop")]))
        success.__exit__ = Mock(return_value=False)
        trainer.client.chat.completions.create.side_effect = [failed, success]
        with redirect_stdout(StringIO()), patch("alpha_cf.trainer.time.sleep"):
            self.assertEqual(self.ask(trainer), result)
        self.assertEqual(trainer.client.chat.completions.create.call_count, 2)
        error = next(c.kwargs for c in trainer.record.call_args_list if c.args == ("llm_error",))
        self.assertTrue(error["retryable"])
        self.assertIn("temporary upstream error", error["error_detail"])

    def test_empty_final_content_still_fails(self):
        trainer = self.trainer([self.chunk(reasoning="not a final answer", finish="stop")])
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "empty"):
            self.ask(trainer)


if __name__ == "__main__":
    unittest.main()
