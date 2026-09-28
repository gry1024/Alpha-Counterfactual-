import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import spearmanr
import torch
from alphagen_qlib.stock_data import StockData
from alpha_cf.alpha_pool import AlphaCFPool, icir, rank, spearman
from alpha_cf.expression import ablate, parameter_variants, parse


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.args = SimpleNamespace(max_nodes=60, max_depth=10, max_backtrack=252,
            chunk_size=7, windows=[5, 10, 20], pool_capacity=3, no_pool_selection=False,
            alpha=1.0, beta=1.0, gamma=0.2, cost_weight=0.1)
        self.data = StockData.__new__(StockData)
        self.data.max_backtrack_days, self.data.max_future_days = 252, 0
        self.data.data = torch.rand(332, 6, 12) + 1
        self.target = torch.randn(80, 12)
        self.pool = AlphaCFPool(self.data, self.target, self.args)

    def test_spearman_matches_scipy_with_ties_and_missing_values(self):
        x = torch.tensor([[1., 1., 4., 2., float("nan")], [1., 2., 3., 4., 5.]])
        y = torch.tensor([[4., 2., 1., 3., 5.], [5., 2., 2., float("inf"), 1.]])
        actual = spearman(x, y).numpy()
        expected = []
        for a, b in zip(x.numpy(), y.numpy()):
            valid = np.isfinite(a) & np.isfinite(b)
            expected.append(spearmanr(a[valid], b[valid]).statistic)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertTrue(torch.isnan(spearman(torch.ones_like(x), y)).all())
        self.assertAlmostEqual(icir(torch.tensor([0.1, -0.2, -0.3])).item(),
                               -icir(torch.tensor([-0.1, 0.2, 0.3])).item(), places=6)

    def test_local_ablation_changes_one_occurrence(self):
        expr = parse("Mul(TsMean($close,5),TsMean($close,5))", self.args)
        changed = ablate(expr, dict(path=[0], mode="remove", replacement="$close"), self.args)
        self.assertEqual(str(changed), "Mul($close,TsMean($close,5))")
        self.assertEqual(str(expr), "Mul(TsMean($close,5),TsMean($close,5))")
        with self.assertRaises(ValueError):
            ablate(expr, dict(path=[0], mode="neutralize", replacement="$volume"), self.args)

    def test_future_lookback_and_malformed_expressions_rejected(self):
        for text in ["Ref($close,-1)", "TsMean($close,0)", "TsMean(TsMean($close,200),100)",
                     "Add($close,$open,$volume)"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text, self.args)

    def test_seed_library_and_full_window_grid(self):
        seeds = json.loads((Path(__file__).resolve().parents[1] / "start_pool.json").read_text())["exprs"]
        self.assertEqual(len(set(seeds)), 200)
        for text in seeds:
            parse(text, self.args)
        expr = parse("Div(TsMean($close,5),TsMean($close,10))", self.args)
        variants = {str(e) for e in parameter_variants(expr, self.args)}
        self.assertEqual(len(variants), 8)
        self.assertIn("Div(TsMean($close,20),TsMean($close,20))", variants)

    def test_chunked_execution_and_pool_credit(self):
        expressions = [parse(text, self.args) for text in
                       ["TsMean($close,5)", "Div($volume,TsMean($volume,10))", "Sub($close,$open)"]]
        for expr in expressions:
            actual = self.pool.evaluate(expr)["signal"]
            raw = expr.evaluate(self.data)
            expected = rank(raw) / (raw.isfinite().sum(1, keepdim=True) - 1) - 0.5
            torch.testing.assert_close(actual, expected)
        self.pool.exprs = expressions
        child = parse("TsMean($close,10)", self.args)
        total = sum(self.pool.signal(e) for e in expressions)
        replacement = (total - self.pool.signal(expressions[0]) + self.pool.signal(child)) / 3
        incremental = self.pool.score_signals(replacement.unsqueeze(0))[0].item()
        direct = self.pool.utility([child, *expressions[1:]])
        self.assertAlmostEqual(incremental, direct, places=5)
        self.assertEqual(len(self.pool.select([*expressions, child], 3)), 3)

    def test_complete_round_with_model_stub(self):
        from alpha_cf.trainer import AlphaCFTrainer
        from alpha_cf.expression import walk
        self.args.rounds, self.args.parents, self.args.mechanisms = 0, 2, 2
        self.args.offspring, self.args.refine_top_k = 3, 1
        self.args.no_cf_evidence = self.args.no_pool_credit = False
        self.args.no_refinement = self.args.no_memory = self.args.random_crossover = False
        with TemporaryDirectory() as directory:
            trainer = AlphaCFTrainer(self.pool, self.args, Path(directory))
            self.args.rounds = 1

            def ask(prompt, context):
                if "mechanism_count" in context:
                    parent = parse(context["parent"], self.args)
                    descendant = next(n for path, n in walk(parent) if path and n.is_featured)
                    return {"mechanisms": [dict(path=[], description="test transform", mode="remove",
                                 replacement=str(descendant), reason="Retain the input")]}
                return dict(expressions=["TsMean($open,10)", "TsMean($high,10)", "Div($close,TsMean($close,5))"],
                            operations=["mutation"] * 3, explanations=["test hypothesis"] * 3)

            trainer.ask = ask
            result = trainer.train(["TsMean($close,5)", "Div($volume,TsMean($volume,10))",
                                    "Sub($close,$open)", "TsMean($low,5)"])
            self.assertEqual(len(result), 3)
            self.assertEqual(len(trainer.memory), 2)
            self.assertTrue((Path(directory) / "pool_1.json").exists())
            records = [json.loads(line) for line in (Path(directory) / "round_1.jsonl").read_text().splitlines()]
            self.assertTrue(any(row["event"] == "refinement" for row in records))
            self.assertEqual(records[-1]["event"], "update")

    def test_load_data_purges_boundary_labels(self):
        import pandas as pd
        import train_cf
        calendar = pd.bdate_range("2008-01-01", "2026-05-01")
        self.args.instrument, self.args.device, self.args.horizon = "fixture", "cpu", 20
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "calendars"
            folder.mkdir()
            (folder / "day.txt").write_text("\n".join(str(day.date()) for day in calendar))
            self.args.qlib_path = directory

            def stock_data(**kwargs):
                left = calendar.searchsorted(pd.Timestamp(kwargs["start_time"]))
                right = calendar.searchsorted(pd.Timestamp(kwargs["end_time"]), side="right")
                data = StockData.__new__(StockData)
                data.max_backtrack_days, data.max_future_days = kwargs["max_backtrack_days"], 0
                data._dates = calendar[left - data.max_backtrack_days:right]
                data.data = torch.arange(1, len(data._dates) + 1).float()[:, None, None].expand(-1, 6, 3)
                return data

            with patch.object(train_cf, "StockData", side_effect=stock_data):
                data, target = train_cf.load_data(self.args, "train")
            self.assertEqual(len(target), data.n_days - 20)
            self.assertLessEqual(data._dates[-1], pd.Timestamp("2021-12-31"))
            close = data.data[data.max_backtrack_days:, 1]
            torch.testing.assert_close(target[-1], close[-1] / close[-21] - 1)

    def test_shared_llm_call_and_nested_json(self):
        from alpha_cf.trainer import AlphaCFTrainer
        from alpha_cf.prompt import PROMPT_DIAGNOSIS, PROMPT_EVOLUTION
        from utils.llm import OpenAIModel
        self.args.rounds, self.args.temperature = 0, 0.5
        with TemporaryDirectory() as directory:
            trainer = AlphaCFTrainer(self.pool, self.args, Path(directory))
            trainer.client = Mock()
            trainer.model = OpenAIModel("fixture", "", None)
            expected = {"mechanisms": [{"path": [1], "description": "volume scaling", "mode": None}]}
            for text in (json.dumps(expected), "```json\n" + json.dumps(expected) + "\n```",
                         '<think>```json\n{"draft":true}, incomplete\n```</think>' + json.dumps(expected)):
                trainer.client.chat.completions.create.return_value = SimpleNamespace(choices=[
                    SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])
                result = trainer.ask(PROMPT_DIAGNOSIS, dict(parent="TsMean($close,5)", reward=0.1,
                    nodes=[dict(path=[], expression="TsMean($close,5)")], mechanism_count=3, cf_enabled=True))
                self.assertEqual(result, expected)
            call = trainer.client.chat.completions.create.call_args.kwargs
            self.assertEqual(call["model"], "fixture")
            self.assertNotIn("max_tokens", call)
            self.assertIn("Given parent:", call["messages"][1]["content"])
            self.assertIn("NOT Boolean comparisons", call["messages"][1]["content"])
            trainer.ask(PROMPT_EVOLUTION, dict(parent={}, donor={}, historical_memory=[],
                existing_expressions=[], offspring_count=3, random_crossover=False, limits={}))
            self.assertIn("all three arrays have 3 entries", trainer.client.chat.completions.create.call_args.kwargs["messages"][1]["content"])

    def test_final_pool_calls_existing_script(self):
        import train_cf
        self.args.instrument, self.args.horizon, self.args.cuda = "csi300", 20, 0
        self.args.seed, self.args.n_factors = 7, 10
        with TemporaryDirectory() as directory:
            source = Path(directory) / "final.json"
            source.write_text(json.dumps(dict(split="valid", exprs=["$close"], weights=[1.0])))
            with patch.object(train_cf.subprocess, "run") as run:
                train_cf.test(self.args, source)
            command = run.call_args.args[0]
            self.assertEqual(Path(command[1]).name, "run_adaptive_combination.py")
            self.assertEqual(command[command.index("--expressions_file") + 1], str(source.resolve()))
            self.assertEqual(command[command.index("--label_days") + 1], "20")
            self.assertTrue(run.call_args.kwargs["check"])
            self.assertIn("src", run.call_args.kwargs["env"]["PYTHONPATH"])


if __name__ == "__main__":
    unittest.main()
