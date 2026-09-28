"""Diagnosis -> mechanism memory -> evolution -> refinement -> pool update."""
from collections import Counter
import json
import os
import random
import re
import time

from openai import OpenAI
from utils.llm import OpenAIModel
from alphagen.data.expression import Sub
from .expression import ablate, parse, parameter_variants, walk
from .prompt import PROMPT_HEAD, PROMPT_FEATURES_AND_OPERATORS, PROMPT_DIAGNOSIS, PROMPT_EVOLUTION


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class AlphaCFTrainer:
    def __init__(self, pool, args, log_dir):
        self.pool, self.args, self.log_dir = pool, args, log_dir
        self.memory, self.visits = [], Counter()
        self.client = self.model = None
        if args.rounds:
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                                 timeout=args.llm_timeout, max_retries=0)
            self.model = OpenAIModel(os.environ["OPENAI_MODEL_NAME"], "", None)
        self.step = 0

    def record(self, event, **values):
        with (self.log_dir / f"round_{self.step}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(event=event, **values), ensure_ascii=False, allow_nan=False) + "\n")

    def ask(self, prompt, context):
        user_prompt = PROMPT_FEATURES_AND_OPERATORS + prompt.format(
            **{key: json.dumps(value, ensure_ascii=False) for key, value in context.items()})
        self.record("llm_request", system_prompt=PROMPT_HEAD, user_prompt=user_prompt)
        text, finish_reason = self.model.chat_generate(
            self.client, system_prompt=PROMPT_HEAD, user_prompt=user_prompt, temperature=self.args.temperature)
        self.record("llm_response", response=text, finish_reason=finish_reason)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
            if block is None:
                raise ValueError("Model response is not a JSON object")
            result = json.loads(block[1])
        if not isinstance(result, dict):
            raise ValueError("Model response must be a JSON object")
        return result

    def evaluate(self, text):
        try:
            expr = parse(text, self.args)
            result = self.pool.evaluate(expr)
            self.record("factor", expression=str(expr), **{k: v for k, v in result.items() if k != "signal"})
            return expr
        except (ValueError, IndexError) as exc:
            self.record("invalid", expression=text, error=str(exc))
            print(f"  Skip {text}: {exc}", flush=True)
            return None

    def initialize(self, seeds):
        candidates = []
        for seed in seeds:
            expr = self.evaluate(seed)
            if expr is None:
                continue
            # Seed direction is learned once on Train, never on Validation/Test.
            if self.pool.evaluate(expr)["reward"] < 0:
                expr = self.evaluate(str(Sub(0.0, expr)))
            if expr is not None:
                candidates.append(expr)
        self.pool.update(candidates)
        self.save()

    def select_parents(self):
        ranked = sorted(self.pool.exprs, key=lambda e: self.pool.evaluate(e)["reward"], reverse=True)
        count = min(self.args.parents, len(ranked))
        # Half strong parents, half least visited; every pool member gets opportunities.
        selected = sorted(ranked, key=lambda e: self.visits[str(e)])[:count // 2]
        selected += [e for e in ranked if str(e) not in {str(x) for x in selected}][:count - len(selected)]
        for expr in selected:
            self.visits[str(expr)] += 1
        return selected

    def diagnose(self, parent, utility, total):
        reward = self.pool.evaluate(parent)["reward"]
        proposals = self.ask(PROMPT_DIAGNOSIS, dict(parent=str(parent), reward=reward,
            nodes=[dict(path=p, expression=str(n)) for p, n in walk(parent)],
            mechanism_count=self.args.mechanisms, cf_enabled=not self.args.no_cf_evidence))["mechanisms"]
        evidence = []
        for mechanism in proposals[:self.args.mechanisms]:
            try:
                node = dict(walk(parent))[tuple(mechanism["path"])]
                row = dict(factor=str(parent), expression=str(node), **mechanism,
                           delta_cf=None, pool_credit=None, action="Explore")
                if not self.args.no_cf_evidence and mechanism.get("mode"):
                    changed = ablate(parent, mechanism, self.args)
                    result = self.pool.evaluate(changed)
                    row["ablation"] = str(changed)
                    row["delta_cf"] = result["reward"] - reward
                    if not self.args.no_pool_credit:
                        signal = (total - self.pool.signal(parent) + self.pool.signal(changed)) / len(self.pool.exprs)
                        row["pool_credit"] = utility - self.pool.score_signals(signal.unsqueeze(0))[0].item()
                    credit = row["pool_credit"] or 0.0
                    if row["delta_cf"] < -1e-5 and credit >= -1e-5:
                        row["action"] = "Preserve"
                    elif row["delta_cf"] > 1e-5 and credit <= 1e-5:
                        row["action"] = "Replace"
                evidence.append(row)
                self.record("mechanism", **row)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                self.record("invalid_mechanism", factor=str(parent), proposal=mechanism, error=str(exc))
                print(f"  Skip mechanism: {exc}", flush=True)
        measured = [row for row in evidence if row["delta_cf"] is not None]
        self.memory.extend(measured)
        print(f"  Diagnosed {len(evidence)} mechanisms, measured {len(measured)}", flush=True)
        return dict(expression=str(parent), reward=reward, mechanisms=evidence)

    def evolve(self, parent, donor, previous_children):
        existing = [str(e) for e in self.pool.exprs + previous_children]
        result = self.ask(PROMPT_EVOLUTION, dict(parent=parent, donor=donor,
            historical_memory=[] if self.args.no_memory else self.memory,
            existing_expressions=existing, offspring_count=self.args.offspring,
            random_crossover=self.args.random_crossover,
            limits=dict(nodes=self.args.max_nodes, depth=self.args.max_depth, lookback=self.args.max_backtrack)))
        expressions, operations, explanations = (result[key] for key in ("expressions", "operations", "explanations"))
        if not all(isinstance(items, list) and len(items) == self.args.offspring
                   for items in (expressions, operations, explanations)):
            raise ValueError("Expected matching expressions, operations and explanations for each offspring")
        children, seen = [], set(existing)
        for text, operation, explanation in zip(expressions, operations, explanations):
            self.record("proposal", parent=parent["expression"], donor=donor["expression"],
                        expression=text, operation=operation, reason=explanation)
            child = self.evaluate(text)
            if child is not None and str(child) not in seen:
                children.append(child)
                seen.add(str(child))
        print(f"  Evaluated {len(children)} new offspring", flush=True)
        return children

    def refine(self, children):
        if self.args.no_refinement:
            return children
        best = sorted(children, key=lambda e: self.pool.evaluate(e)["reward"], reverse=True)[:self.args.refine_top_k]
        refined = []
        for child in best:
            winner = child
            for variant in parameter_variants(child, self.args):
                result = self.evaluate(str(variant))
                if result is not None and self.pool.evaluate(result)["reward"] > self.pool.evaluate(winner)["reward"]:
                    winner = result
                # Keep only the best grid point's tensor, not every evaluated variant.
                self.pool.keep(self.pool.exprs + children + refined + [winner])
            refined.append(winner)
            self.record("refinement", original=str(child), refined=str(winner))
        return children + refined

    def save(self):
        save_json(self.log_dir / f"pool_{self.step}.json", self.pool.to_dict())
        save_json(self.log_dir / "memory.json", self.memory)
        print(f"Pool {self.step}: size={len(self.pool.exprs)}, RankICIR={self.pool.utility():.4f}", flush=True)

    def train(self, seeds):
        self.initialize(seeds)
        for step in range(1, self.args.rounds + 1):
            self.step = step
            started = time.monotonic()
            parents = self.select_parents()
            total = sum(self.pool.signal(e) for e in self.pool.exprs)
            utility = self.pool.utility()
            print(f"Round {step}/{self.args.rounds}: {len(parents)} parents", flush=True)
            evidence = [self.diagnose(parent, utility, total) for parent in parents]
            children = []
            for parent in evidence:
                donors = [row for row in evidence if row is not parent] or [parent]
                if self.args.random_crossover:
                    donor = random.choice(donors)
                else:
                    donor = max(donors, key=lambda row: max(
                        [(m["pool_credit"] or 0.0) - (m["delta_cf"] or 0.0) for m in row["mechanisms"]], default=0.0))
                children.extend(self.evolve(parent, donor, children))
                self.pool.keep(self.pool.exprs + children)
            children = list({str(expr): expr for expr in children}.values())
            if not children:
                raise ValueError(f"Round {step} produced no usable offspring; see round_{step}.jsonl")
            children = self.refine(children)
            old = {str(e) for e in self.pool.exprs}
            self.pool.update(children)
            added = [str(e) for e in self.pool.exprs if str(e) not in old]
            self.record("update", offspring=len(children), added=added, utility=self.pool.utility())
            self.save()
            print(f"  {len(children)} offspring, {len(added)} entered pool, {time.monotonic() - started:.1f}s", flush=True)
        return self.pool.exprs
