"""Selection ablations: no market data or LLM requests."""
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .alpha_pool import AlphaCFPool
from .expression import parse
from .trainer import AlphaCFTrainer


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(no_pool_selection=True, no_memory=False,
            max_nodes=60, max_depth=10, max_backtrack=100, rounds=1,
            reasoning_workers=1, correlation_threshold=0.8,
            alpha=5, beta=800, gamma=2, cost_weight=2)
        self.parent, self.peer, self.a, self.b = [parse(s, self.args)
            for s in ("$close", "$volume", "$open", "$high")]
        self.pool = AlphaCFPool.__new__(AlphaCFPool)
        self.pool.args, self.pool.exprs = self.args, [self.parent, self.peer]
        self.pool.evaluate = Mock(return_value=dict(reward=0.0, cost=1.0))
        for method in ("utility", "correlation", "equivalent", "signal_distance"):
            setattr(self.pool, method, Mock(side_effect=AssertionError("Selection metric called")))

    def test_draw_precedes_measurement_and_ignores_quality(self):
        def draw(children):
            self.pool.evaluate.assert_not_called()
            self.assertEqual(children, [self.a, self.b])
            return children[1]
        with patch("alpha_cf.alpha_pool.random.choice", side_effect=draw):
            result = self.pool.replace_parent(self.parent,
                [self.parent, self.a, self.a, self.peer, self.b])
        self.pool.evaluate.assert_called_once_with(self.b)
        self.assertIs(self.pool.exprs[0], self.b)
        self.assertEqual(result["selection_policy"], "uniform_random_v1")
        self.assertEqual([c["selection_probability"] for c in result["candidates"]], [0.5, 0.5])
        self.assertEqual(result["reason"], "random_offspring")

    def test_unusable_draw_does_not_resample(self):
        self.pool.evaluate.side_effect = ValueError("constant signal")
        with patch("alpha_cf.alpha_pool.random.choice", return_value=self.a) as draw:
            result = self.pool.replace_parent(self.parent, [self.a, self.b])
        draw.assert_called_once()
        self.pool.evaluate.assert_called_once_with(self.a)
        self.assertIs(self.pool.exprs[0], self.parent)
        self.assertEqual(result["reason"], "invalid_random_offspring")
        self.assertEqual(result["drawn_child"], str(self.a))

    def test_no_new_child_retains_without_measurement(self):
        with patch("alpha_cf.alpha_pool.random.choice") as draw:
            result = self.pool.replace_parent(self.parent, [self.parent, self.peer])
        draw.assert_not_called()
        self.pool.evaluate.assert_not_called()
        self.assertEqual(result["reason"], "no_new_offspring")

    def test_normal_mode_keeps_positive_credit_gate(self):
        self.args.no_pool_selection = False
        self.pool.utility = Mock(side_effect=lambda xs: 0.11 if self.b in xs
                                 else 0.09 if self.a in xs else 0.1)
        self.pool.correlation = Mock(return_value=0.0)
        self.pool.signal_distance = Mock(return_value=0.5)
        self.pool.equivalent = lambda a, b: str(a) == str(b)
        result = self.pool.replace_parent(self.parent, [self.a, self.b])
        self.assertEqual(result["child"], str(self.b))
        self.assertFalse(result["candidates"][0]["credit_ok"])
        self.assertTrue(result["candidates"][1]["credit_ok"])

    def make_trainer(self):
        trainer = AlphaCFTrainer.__new__(AlphaCFTrainer)
        trainer.args, trainer.pool = self.args, self.pool
        trainer.memory = {"f0": []}
        trainer.factor_ids = {str(self.parent): "f0"}
        trainer.chain_ids = {str(self.parent): "f0"}
        trainer.next_factor_id, trainer.lineage, trainer.visits = 1, [], Counter()
        trainer.record, trainer.save = Mock(), Mock()
        return trainer

    def test_generation_parses_without_measuring_candidates(self):
        trainer = self.make_trainer()
        trainer.ask = Mock(return_value=dict(updated_understanding="A hypothesis",
            offspring=[dict(expression=s, description="An edit", evidence_refs=[])
                       for s in ("$open", "$high", "$open", "bad_formula")]))
        with redirect_stdout(StringIO()):
            children, details = trainer.evolve(dict(expression=str(self.parent), mechanisms=[]), [])
        self.pool.evaluate.assert_not_called()
        self.assertEqual([str(e) for e in children], [str(self.a), str(self.b)])
        policy = trainer.ask.call_args.args[1]["selection_policy"]
        self.assertIn("uniformly at random", policy)
        self.assertNotIn("score_weights", trainer.ask.call_args.args[1])
        self.assertEqual(set(details), {str(self.a), str(self.b)})

    def test_random_update_preserves_lineage_and_memory(self):
        trainer = self.make_trainer()
        trainer.initialize = Mock()
        trainer.select_parents = Mock(return_value=[self.parent])
        trainer.reasoning_context = Mock(return_value={})
        trainer.ask = Mock(return_value={})
        trainer.reason_about = Mock(return_value={})
        trainer.evolve = Mock(return_value=([self.a],
            {str(self.a): dict(description="An edit", evidence_refs=[])}))
        self.pool.utility = Mock(return_value=0.1)
        self.pool.keep = Mock()
        with redirect_stdout(StringIO()):
            trainer.train([])
        self.assertEqual(trainer.factor_ids[str(self.a)], "f1")
        self.assertEqual(trainer.chain_ids[str(self.a)], "f0")
        self.assertEqual(len(trainer.memory["f0"]), 1)
        self.assertEqual(trainer.lineage[0]["selection_policy"], "uniform_random_v1")
        self.assertEqual(trainer.lineage[0]["child_description"], "An edit")


    def test_llm_only_parents_and_context_use_no_numerical_feedback(self):
        self.args.llm_only = True
        self.args.parents = 1
        self.args.no_memory = self.args.no_cf_evidence = self.args.no_pool_credit = True
        trainer = self.make_trainer()
        self.pool.evaluate.side_effect = AssertionError("Numerical feedback requested")
        with patch("alpha_cf.trainer.random.sample", return_value=[0]) as sample, redirect_stdout(StringIO()):
            selected = trainer.select_parents()
            context = trainer.reasoning_context(self.parent)
            parent = trainer.reason_about(self.parent, None, dict(understanding="A hypothesis",
                mechanisms=[dict(path=[], replacement="$open", description="Change input", reason="A prediction")]))
            trainer.ask = Mock(return_value=dict(updated_understanding="Still untested", offspring=[]))
            trainer.evolve(parent, [])
        sample.assert_called_once_with(range(2), 1)
        self.assertEqual(selected, [self.parent])
        self.assertIsNone(context["reward"])
        self.assertEqual(context["previous_understanding"], {})
        self.assertFalse(context["cf_enabled"])
        self.assertIsNone(parent["reward"])
        for key in ("delta_cf", "pool_credit", "signal_distance", "coverage_change", "common_rank_correlation"):
            self.assertIsNone(parent["mechanisms"][0][key])
        self.assertEqual(trainer.ask.call_args.args[1]["historical_memory"], [])
        self.pool.utility.assert_not_called()
        self.pool.evaluate.assert_not_called()

    def test_llm_only_cli_enables_all_removal_flags(self):
        import train_cf
        with patch("sys.argv", ["train_cf.py", "--llm-only", "--rounds", "20"]), \
                patch.object(train_cf, "load_dotenv"), patch.object(train_cf, "train") as train:
            train_cf.main()
        args = train.call_args.args[0]
        self.assertTrue(args.llm_only)
        self.assertEqual(args.rounds, 20)
        self.assertTrue(all(getattr(args, key) for key in
            ("no_cf_evidence", "no_pool_credit", "no_memory", "no_pool_selection")))


if __name__ == "__main__":
    unittest.main()
