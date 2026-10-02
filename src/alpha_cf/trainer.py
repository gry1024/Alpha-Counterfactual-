"""Diagnosis -> mechanism memory -> evolution -> pool update."""
from collections import Counter
import json
import math
import os
import re
import time

from openai import OpenAI
from utils.llm import OpenAIModel
from .expression import ablate, at, complexity, parse, walk
from .prompt import PROMPT_HEAD, PROMPT_FEATURES_AND_OPERATORS, PROMPT_DIAGNOSIS, PROMPT_EVOLUTION


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class AlphaCFTrainer:
    def __init__(self, pool, args, log_dir):
        self.pool, self.args, self.log_dir = pool, args, log_dir
        self.memory, self.visits = [], Counter()
        self.lineage, self.factor_ids = [], {}
        self.next_factor_id = 0
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
        for attempt in range(1, 4):
            self.record("llm_request", attempt=attempt, system_prompt=PROMPT_HEAD, user_prompt=user_prompt)
            text, finish_reason = self.model.chat_generate(
                self.client, system_prompt=PROMPT_HEAD, user_prompt=user_prompt, temperature=self.args.temperature)
            self.record("llm_response", attempt=attempt, response=text, finish_reason=finish_reason)
            try:
                text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
                if not text:
                    raise ValueError("Model response is empty after removing thinking content")
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
                    if block is None:
                        raise ValueError("Model response is not a JSON object")
                    result = json.loads(block[1])
                if not isinstance(result, dict):
                    raise ValueError("Model response must be a JSON object")
                keys = ("mechanisms",) if prompt == PROMPT_DIAGNOSIS else ("offspring",)
                if not all(isinstance(result.get(key), list) for key in keys):
                    raise ValueError(f"Expected required list fields: {', '.join(keys)}")
                if prompt == PROMPT_EVOLUTION:
                    if len(result["offspring"]) > 5:
                        raise ValueError("At most 5 offspring are allowed")
                    if not all(isinstance(child, dict) and all(
                            isinstance(child.get(key), str) and child[key].strip()
                            for key in ("expression", "description")) for child in result["offspring"]):
                        raise ValueError("Each offspring needs a nonempty expression and description")
                if prompt == PROMPT_DIAGNOSIS and len(result["mechanisms"]) > 5:
                    raise ValueError("At most 5 meaningful mechanisms are allowed")
                return result
            except ValueError as exc:
                self.record("invalid_llm_response", attempt=attempt, error=str(exc))
                if attempt == 3:
                    raise

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
            if expr is not None:
                candidates.append(expr)
        self.pool.initialize(candidates)
        for expr in self.pool.exprs:
            factor_id = f"f{self.next_factor_id}"
            self.next_factor_id += 1
            self.factor_ids[str(expr)] = factor_id
            self.lineage.append(dict(round=0, outcome="seed", child=str(expr), child_id=factor_id,
                                     parent=None, parent_id=None))
        self.save()

    def select_parents(self):
        ranked = sorted(self.pool.exprs, key=lambda e: self.pool.evaluate(e)["reward"], reverse=True)
        count = min(self.args.parents, len(ranked))
        # Default: top 5, then 5 least visited among the remaining pool members.
        selected = ranked[:count // 2]
        remaining = ranked[count // 2:]
        selected += sorted(remaining, key=lambda e: self.visits[str(e)])[:count - len(selected)]
        for expr in selected:
            self.visits[str(expr)] += 1
        return selected

    def diagnose(self, parent, utility, total):
        reward = self.pool.evaluate(parent)["reward"]
        proposals = self.ask(PROMPT_DIAGNOSIS, dict(parent=str(parent), reward=reward,
            nodes=[dict(path=p, expression=str(n)) for p, n in walk(parent)],
            cf_enabled=not self.args.no_cf_evidence,
            limits=dict(nodes=self.args.max_nodes, depth=self.args.max_depth, lookback=self.args.max_backtrack)))["mechanisms"]
        evidence = []
        for mechanism in proposals[:5]:
            try:
                node = at(parent, mechanism["path"])
                row = dict(factor=str(parent), expression=str(node),
                           **{key: mechanism[key] for key in ("path", "description", "replacement", "reason")},
                           delta_cf=None, pool_credit=None, signal_distance=None)
                if not self.args.no_cf_evidence:
                    changed = ablate(parent, mechanism, self.args)
                    result = self.pool.evaluate(changed)
                    correlation = self.pool.correlation(parent, changed)
                    if not math.isfinite(correlation):
                        raise ValueError("Counterfactual has no shared varying observations")
                    row["counterfactual"] = str(changed)
                    row["delta_cf"] = abs(result["reward"]) - abs(reward)
                    row["signal_distance"] = 1 - correlation
                    if not self.args.no_pool_credit:
                        # C_pool = U(P_{-f} ∪ {f'}) - U(P); larger means the edit improves the pool.
                        signal = (total - self.pool.signal(parent) + self.pool.signal(changed)) / len(self.pool.exprs)
                        credit = self.pool.score_signals(signal.unsqueeze(0))[0].item() - utility
                        if not math.isfinite(credit):
                            raise ValueError("Counterfactual pool has no usable variation")
                        row["pool_credit"] = credit
                evidence.append(row)
                self.record("mechanism", **row)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                self.record("invalid_mechanism", factor=str(parent), proposal=mechanism, error=str(exc))
                print(f"  Skip mechanism: {exc}", flush=True)
        measured = [row for row in evidence if row["delta_cf"] is not None]
        self.memory.extend(measured)
        print(f"  Diagnosed {len(evidence)} mechanisms, measured {len(measured)}", flush=True)
        return dict(expression=str(parent), reward=reward, complexity=complexity(parent), mechanisms=evidence)

    def evolve(self, parent, previous_children):
        existing = [str(e) for e in self.pool.exprs] + previous_children
        factor_memory = ([] if self.args.no_memory
                         else [m for m in self.memory if m.get("factor") == parent["expression"]])
        result = self.ask(PROMPT_EVOLUTION, dict(parent=parent,
            historical_memory=factor_memory,
            existing_expressions=existing,
            limits=dict(nodes=self.args.max_nodes, depth=self.args.max_depth, lookback=self.args.max_backtrack)))
        children, descriptions, seen = [], {}, set(existing)
        for proposal in result["offspring"]:
            text, description = proposal["expression"], proposal["description"].strip()
            self.record("proposal", parent=parent["expression"],
                        parent_id=self.factor_ids[parent["expression"]],
                        expression=text, description=description)
            child = self.evaluate(text)
            if child is not None and str(child) not in seen:
                children.append(child)
                descriptions[str(child)] = description
                seen.add(str(child))
        print(f"  Evaluated {len(children)} new offspring", flush=True)
        return children, descriptions

    def save(self):
        payload = self.pool.to_dict()
        payload["split"] = "train"
        payload["factor_ids"] = [self.factor_ids[str(e)] for e in self.pool.exprs]
        payload["visits"] = dict(self.visits)
        save_json(self.log_dir / f"pool_{self.step}.json", payload)
        save_json(self.log_dir / "memory.json", self.memory)
        save_json(self.log_dir / "lineage.json", self.lineage)
        print(f"Pool {self.step}: size={len(self.pool.exprs)}, RankIC={self.pool.utility():.4f}", flush=True)

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
            previous_children, added = [], []
            for parent_expr, parent in zip(parents, evidence):
                children, descriptions = self.evolve(parent, previous_children)
                previous_children.extend(str(child) for child in children)
                decision = self.pool.replace_parent(parent_expr, children)
                for candidate in decision["candidates"]:
                    candidate["description"] = descriptions[candidate["expression"]]
                decision["child_description"] = descriptions.get(decision["child"])
                decision.update(round=step, parent_id=self.factor_ids[str(parent_expr)], child_id=None)
                if decision["child"] is not None:
                    child_id = f"f{self.next_factor_id}"
                    self.next_factor_id += 1
                    self.factor_ids.pop(str(parent_expr))
                    self.factor_ids[decision["child"]] = child_id
                    decision["child_id"] = child_id
                    added.append(decision["child"])
                self.lineage.append(decision)
                self.record("replacement", **decision)
                self.pool.keep(self.pool.exprs)
            self.record("update", offspring=len(previous_children), added=added, utility=self.pool.utility())
            self.save()
            print(f"  {len(previous_children)} offspring, {len(added)} parents replaced, "
                  f"{time.monotonic() - started:.1f}s", flush=True)
        return self.pool.exprs
