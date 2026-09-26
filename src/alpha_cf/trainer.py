"""Counterfactual diagnosis, evidence retrieval and LLM-driven mechanism evolution."""
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import re
import time

from openai import OpenAI, OpenAIError
from .expression import ablate, at, parse, replace, structure, walk, parameter_specs, parameter_variants
from .prompt import SYSTEM, DIAGNOSE, EVOLVE


def _invalid_json_number(value):
    raise ValueError(f"Invalid JSON number: {value}")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def make_logger(log_dir):
    started = time.monotonic()
    def log(kind, **fields):
        record = dict(kind=kind, elapsed=round(time.monotonic() - started, 3), **fields)
        with (Path(log_dir) / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        summary = {k: v for k, v in fields.items() if k in
                   ("split", "step", "phase", "size", "eval_cnt", "reward", "utility", "seconds", "error", "reason")}
        print(f"[{kind}] {summary}", flush=True)
    return log


def mechanism_key(m):
    return m["factor"], tuple(m["path"])


class AlphaCFTrainer:
    def __init__(self, pool, args, log_dir):
        self.pool, self.args, self.log_dir, self.log = pool, args, Path(log_dir), pool.log
        self.memory, self.seen, self.step, self.calls = [], set(), 0, 0
        self.last_diagnosed, self.refinement = {}, {}
        self.model = os.environ["OPENAI_MODEL_NAME"]
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                             base_url=os.environ["OPENAI_BASE_URL"], timeout=90, max_retries=1)

    @contextmanager
    def phase(self, name):
        started = time.monotonic()
        before = self.pool.eval_cnt, self.pool.pool_scores, self.calls
        try:
            yield
        finally:
            self.log("phase_done", step=self.step, phase=name, seconds=time.monotonic() - started,
                     factor_evals=self.pool.eval_cnt - before[0],
                     pool_scores=self.pool.pool_scores - before[1], llm_calls=self.calls - before[2])

    def ask(self, prompt):
        self.pool.check_time()
        if self.calls >= self.args.max_llm_calls:
            raise TimeoutError("LLM call limit reached")
        remaining = self.pool.deadline - time.monotonic()
        if remaining < 5:
            raise TimeoutError("Insufficient time for another LLM call")
        self.calls += 1
        started = time.monotonic()
        self.log("llm_request", step=self.step, call=self.calls, model=self.model, system=SYSTEM, prompt=prompt)
        try:
            response = self.client.with_options(timeout=min(90, (remaining - 2) / 2)).chat.completions.create(
                model=self.model, messages=[{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": prompt}],
                temperature=self.args.temperature, max_tokens=self.args.max_output_tokens)
            text = response.choices[0].message.content or ""
            self.log("llm_response", step=self.step, call=self.calls, response=text,
                     finish_reason=response.choices[0].finish_reason,
                     usage=response.usage.model_dump() if response.usage else None,
                     seconds=time.monotonic() - started)
            if text.strip().startswith(chr(96) * 3):
                text = "\n".join(text.strip().splitlines()[1:-1])
            result = json.loads(text, parse_constant=_invalid_json_number)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except (OpenAIError, ValueError, IndexError) as exc:
            self.log("llm_failed", step=self.step, call=self.calls, error=type(exc).__name__,
                     seconds=time.monotonic() - started)
            return {}

    def select_parents(self):
        members = list(self.pool.exprs)
        random.shuffle(members)
        quality = sorted(members, key=lambda e: self.pool.cache[str(e)]["reward"], reverse=True)
        marginal = {r["expression"]: r.get("marginal", 0.0) for r in self.pool.selection_log}
        synergy = sorted(members, key=lambda e: marginal.get(str(e), 0.0), reverse=True)
        exploration = sorted(members, key=lambda e: self.last_diagnosed.get(str(e), -1))
        parents, used = [], set()
        queues = [quality, synergy, exploration]
        while len(parents) < min(self.args.parents, len(members)):
            for queue in queues:
                while queue and str(queue[0]) in used:
                    queue.pop(0)
                if queue and len(parents) < self.args.parents:
                    parent = queue.pop(0)
                    parents.append(parent)
                    used.add(str(parent))
        return parents

    def diagnose(self, parents):
        evidence = []
        use_cf = not self.args.no_cf_evidence
        use_pool = use_cf and not self.args.no_pool_credit
        base = self.pool.utility(self.pool.exprs) if use_pool else None
        members = {str(e) for e in self.pool.exprs}
        counts = Counter(returned=0, proposed=0, identified=0, structural=0, measured=0, pool_measured=0)
        for parent in parents:
            self.last_diagnosed[str(parent)] = self.step
            context = dict(parent=str(parent), metrics=self.pool.metrics(self.pool.cache[str(parent)]),
                           nodes=[dict(path=p, subtree=str(n)) for p, n in walk(parent)],
                           mechanism_limit=self.args.mechanisms, cf_enabled=use_cf,
                           max_nodes=self.args.max_nodes, max_depth=self.args.max_depth,
                           max_backtrack=self.args.max_backtrack)
            mechanisms = self.ask(DIAGNOSE + json.dumps(context, ensure_ascii=False)).get("mechanisms", [])
            if not isinstance(mechanisms, list):
                self.log("invalid_diagnosis", step=self.step, parent=str(parent), reason="Expected mechanisms list")
                continue
            counts["returned"] += len(mechanisms)
            paths = set()
            for proposal in mechanisms[:self.args.mechanisms]:
                counts["proposed"] += 1
                cf, row = None, None
                try:
                    if not isinstance(proposal, dict):
                        raise ValueError("Mechanism must be an object")
                    path = tuple(proposal["path"])
                    subtree = str(at(parent, path))
                    if subtree != str(parse(proposal["subtree"], self.args)) or path in paths:
                        raise ValueError("Invalid or duplicate source mechanism")
                    if not isinstance(proposal["description"], str) or not proposal["description"].strip():
                        raise ValueError("Missing mechanism description")
                    paths.add(path)
                    counts["identified"] += 1
                    row = dict(step=self.step, pool_snapshot=f"pool_{self.step - 1}.json",
                               factor=str(parent), path=path, subtree=subtree,
                               description=proposal["description"], status="unmeasured",
                               delta_cf=None, pool_credit=None, action="Unmeasured")
                    evidence.append(row)
                    if not use_cf:
                        row["reason"] = "Counterfactual evidence disabled"
                        continue
                    row.update(mode=proposal.get("mode"), reason=proposal.get("reason"),
                               replacement=proposal.get("replacement"))
                    if proposal.get("replacement") is None:
                        row["status"] = "unsupported"
                        continue
                    cf, baseline_kind = ablate(parent, proposal, self.args)
                    counts["structural"] += 1
                    row.update(ablated_expr=str(cf), baseline_kind=baseline_kind)
                    before, after = self.pool.evaluate(parent), self.pool.evaluate(cf)
                    if after is None:
                        row.update(status="invalid", error="Intervened factor failed evaluation")
                        continue
                    bm, am = before["valid_mask"], after["valid_mask"]
                    common = (bm & am).sum().item()
                    row.update(status="measured", action="Explore", delta_cf=after["reward"] - before["reward"],
                               parent_reward=before["reward"], ablated_reward=after["reward"],
                               parent_coverage=before["coverage"], ablated_coverage=after["coverage"],
                               parent_valid_days=before["valid_days"], ablated_valid_days=after["valid_days"],
                               common_valid_days=common, valid_day_jaccard=common / (bm | am).sum().item())
                    counts["measured"] += 1
                    # Keep measured factor evidence even if pool credit is undefined or time runs out.
                    self.memory.append(row)
                    if use_pool:
                        try:
                            swapped = self.pool.utility([e for e in self.pool.exprs if str(e) != str(parent)] + [cf])
                            row.update(pool_before=base, pool_after=swapped, pool_credit=base - swapped)
                            counts["pool_measured"] += 1
                        except ValueError as exc:
                            row["pool_error"] = str(exc)
                    delta, credit = row["delta_cf"], row["pool_credit"]
                    positive = delta < 0 and (not use_pool or credit is not None and credit > 0)
                    negative = delta > 0 and (not use_pool or credit is not None and credit < 0)
                    row["action"] = "Preserve" if positive else "Replace" if negative else "Explore"
                except (ValueError, TypeError, KeyError, IndexError, AssertionError) as exc:
                    if row is not None:
                        row.update(status="invalid", error=str(exc))
                    self.log("invalid_intervention", step=self.step, parent=str(parent),
                             proposal=proposal, error=str(exc))
                finally:
                    if row is not None:
                        self.log("mechanism", **row)
                    if cf is not None and str(cf) not in members:
                        self.pool.cache.pop(str(cf), None)
        self.log("diagnosis_coverage", step=self.step, cf_enabled=use_cf, **counts,
                 measured_fraction=counts["measured"] / counts["proposed"] if use_cf and counts["proposed"] else None)
        return evidence

    def visible_evidence(self, row):
        hidden = {"pool_credit", "pool_before", "pool_after", "pool_error"} if self.args.no_pool_credit else set()
        return {k: v for k, v in row.items() if k not in hidden}

    def retrieve_memory(self, parents):
        if self.args.no_memory or not self.args.memory_limit:
            return []
        history = [m for m in self.memory if m["step"] < self.step]
        # No embedding service: shared features/operators identify related structures.
        tokens = lambda text: set(re.findall(r"\$[a-z]+|[A-Za-z_]\w*(?=\()", text))
        query = set().union(*(tokens(str(p)) for p in parents))
        def similarity(m):
            signature = tokens(m["subtree"])
            return len(query & signature) / max(1, len(query | signature))
        def strength(m):
            return max(abs(m["delta_cf"]), 0.0 if self.args.no_pool_credit else abs(m.get("pool_credit") or 0.0))
        positive, negative, mixed = [], [], []
        for m in history:
            delta, credit = m["delta_cf"], m.get("pool_credit")
            if delta < 0 and (self.args.no_pool_credit or credit is not None and credit > 0):
                positive.append(m)
            elif delta > 0 and (self.args.no_pool_credit or credit is not None and credit < 0):
                negative.append(m)
            else:
                mixed.append(m)
        queues = [sorted(history, key=lambda m: (similarity(m), m["step"]), reverse=True),
                  sorted(positive, key=strength, reverse=True),
                  sorted(negative, key=strength, reverse=True),
                  sorted(mixed, key=strength, reverse=True), list(reversed(history))]
        selected, seen = [], set()
        while any(queues) and len(selected) < self.args.memory_limit:
            for queue in queues:
                while queue:
                    m = queue.pop(0)
                    key = (m["step"], *mechanism_key(m), m["ablated_expr"])
                    if key not in seen:
                        selected.append(self.visible_evidence(m))
                        seen.add(key)
                        break
                if len(selected) >= self.args.memory_limit:
                    break
        return selected

    def evolve(self, parents, evidence):
        parent_map = {str(p): p for p in parents}
        mechanisms = {mechanism_key(m): m for m in evidence}
        if not mechanisms:
            self.log("no_evolution", step=self.step, reason="No identified mechanisms")
            return []
        reference = lambda m: dict(parent=m["factor"], path=m["path"], subtree=m["subtree"])
        random_pairs = None
        if self.args.random_crossover:
            groups = {}
            for m in evidence:
                groups.setdefault(m["factor"], []).append(m)
            random_pairs = []
            if len(groups) >= 2:
                for _ in range(self.args.offspring):
                    names = random.sample(list(groups), 2)
                    random_pairs.append([reference(random.choice(groups[name])) for name in names])
        history = self.retrieve_memory(parents)
        self.log("memory_retrieved", step=self.step, size=len(history),
                 records=[dict(step=m["step"], factor=m["factor"], path=m["path"]) for m in history])
        context = dict(parents=[dict(expression=name, **self.pool.metrics(self.pool.cache[name])) for name in parent_map],
                       current_evidence=[self.visible_evidence(m) for m in evidence],
                       historical_memory=history, random_pairs=random_pairs, offspring_limit=self.args.offspring,
                       max_nodes=self.args.max_nodes, max_depth=self.args.max_depth,
                       max_backtrack=self.args.max_backtrack, window_grid=self.args.windows,
                       constant_grid=self.args.constants, max_refine_trials=self.args.max_refine_trials)
        proposals = self.ask(EVOLVE + json.dumps(context, ensure_ascii=False)).get("children", [])
        if not isinstance(proposals, list):
            return []
        offspring, accepted, crossover_index = [], Counter(), 0
        self.refinement = {}
        for proposal in proposals[:self.args.offspring]:
            try:
                if not isinstance(proposal, dict):
                    raise ValueError("Child proposal must be an object")
                operation, names = proposal["operation"], proposal["parents"]
                forced_pair = None
                if operation == "crossover" and random_pairs is not None:
                    forced_pair = random_pairs[crossover_index]
                    crossover_index += 1
                if operation not in ("mutation", "replacement", "crossover"):
                    raise ValueError("Unknown evolution operation")
                expected = 2 if operation == "crossover" else 1
                if not isinstance(names, list) or len(names) != expected or len(set(names)) != expected or any(n not in parent_map for n in names):
                    raise ValueError("Invalid current parents")
                if not isinstance(proposal.get("reason"), str) or not proposal["reason"].strip():
                    raise ValueError("Missing structural hypothesis")
                child = parse(proposal["expression"], self.args)
                if str(child) in self.seen:
                    raise ValueError("Previously proposed expression")
                if any(structure(child) == structure(parent_map[n]) for n in names):
                    raise ValueError("Parameter-only change belongs in refinement")
                kept = []
                for item in proposal.get("keep", []):
                    m = mechanisms[(item["parent"], tuple(item["path"]))]
                    if item["parent"] not in names or str(parse(item["subtree"], self.args)) != m["subtree"]:
                        raise ValueError("Incorrect mechanism source")
                    kept.append(m)
                available = Counter(str(node) for _, node in walk(child))
                if any(available[s] < count for s, count in Counter(m["subtree"] for m in kept).items()):
                    raise ValueError("Declared mechanism is not preserved")
                if operation == "mutation" and not kept:
                    raise ValueError("Mutation must retain a declared mechanism")
                if operation == "crossover":
                    if {m["factor"] for m in kept} != set(names):
                        raise ValueError("Crossover must retain mechanisms from both parents")
                    if forced_pair is not None:
                        expected_keys = {(m["parent"], tuple(m["path"])) for m in forced_pair}
                        if len(kept) != 2 or {mechanism_key(m) for m in kept} != expected_keys:
                            raise ValueError("Crossover must use the next assigned random pair")
                if operation == "replacement":
                    path = tuple(proposal["path"])
                    if (names[0], path) not in mechanisms:
                        raise ValueError("Replacement must address an identified mechanism")
                    if str(replace(parent_map[names[0]], path, at(child, path))) != str(child):
                        raise ValueError("Replacement changed surrounding structure")
                self.seen.add(str(child))
                self.log("offspring", step=self.step, expression=str(child), proposal=proposal)
                if self.pool.evaluate(child) is not None:
                    offspring.append(child)
                    accepted[operation] += 1
                    if not self.args.no_refinement:
                        try:
                            specs = parameter_specs(child, proposal.get("refine"), self.args)
                        except (ValueError, TypeError, KeyError, IndexError) as exc:
                            self.log("invalid_refinement_spec", expression=str(child), error=str(exc))
                            specs = parameter_specs(child, None, self.args)
                        self.refinement[str(child)] = specs
            except (ValueError, TypeError, KeyError, IndexError, AssertionError) as exc:
                self.log("invalid_offspring", step=self.step, proposal=proposal, error=str(exc))
        self.log("evolution_done", step=self.step, proposed=min(len(proposals), self.args.offspring),
                 size=len(offspring), accepted_by_operation=dict(accepted))
        return offspring

    def refine(self, offspring):
        if self.args.no_refinement:
            return offspring
        eligible = sorted(offspring, key=lambda e: (-self.pool.cache[str(e)]["reward"], str(e)))[:self.args.refine_top_k]
        result = list(offspring)
        for original in eligible:
            best = original
            specs = self.refinement.get(str(original), parameter_specs(original, None, self.args))
            for variant, changes in parameter_variants(original, specs, self.args):
                if str(variant) in self.seen:
                    continue
                self.seen.add(str(variant))
                entry = self.pool.evaluate(variant)
                self.log("refinement", step=self.step, parent=str(original), expression=str(variant),
                         parameters=changes, reward=entry["reward"] if entry else None)
                if entry and entry["reward"] > self.pool.cache[str(best)]["reward"]:
                    if best is not original:
                        self.pool.cache.pop(str(best), None)
                    best = variant
                else:
                    self.pool.cache.pop(str(variant), None)
            result[result.index(original)] = best
        return result

    def save(self):
        save_json(self.log_dir / f"pool_{self.step}.json", self.pool.to_dict())
        save_json(self.log_dir / "memory.json", self.memory)

    def train(self, initial_exprs):
        try:
            with self.phase("initialize"):
                initial = []
                for text in initial_exprs:
                    try:
                        expr = parse(text, self.args)
                        if str(expr) not in self.seen:
                            self.seen.add(str(expr))
                            if self.pool.evaluate(expr) is not None:
                                initial.append(expr)
                    except (ValueError, IndexError, AssertionError) as exc:
                        self.log("invalid_seed", expression=text, error=str(exc))
                self.pool.exprs = self.pool.select(initial, self.args.pool_capacity)
                self.pool.keep(self.pool.exprs)
                self.save()
            for self.step in range(1, self.args.rounds + 1):
                self.pool.check_time()
                if self.pool.eval_cnt >= self.args.max_evals:
                    raise TimeoutError("Factor evaluation limit reached")
                parents = self.select_parents()
                self.log("round_start", step=self.step, parents=[str(e) for e in parents], eval_cnt=self.pool.eval_cnt)
                with self.phase("diagnosis"):
                    evidence = self.diagnose(parents)
                with self.phase("evolution"):
                    offspring = self.evolve(parents, evidence)
                with self.phase("refinement"):
                    offspring = self.refine(offspring)
                with self.phase("selection"):
                    self.pool.update(offspring)
                    utility = self.pool.utility(self.pool.exprs)
                self.log("round_done", step=self.step, size=len(self.pool.exprs), utility=utility,
                         eval_cnt=self.pool.eval_cnt)
                self.save()
        except TimeoutError as exc:
            self.log("search_stopped", step=self.step, reason=str(exc), eval_cnt=self.pool.eval_cnt)
            if not self.pool.exprs:
                raise
            self.pool.keep(self.pool.exprs)
            self.save()
        finally:
            self.client.close()
        return self.pool.exprs
