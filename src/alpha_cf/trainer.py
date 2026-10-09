"""Diagnosis -> mechanism memory -> evolution -> pool update."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
import math
import os
import random
import re
import time

from openai import APIConnectionError, APIStatusError, OpenAI
from .expression import ablate, at, complexity, parse, walk
from .prompt import PROMPT_HEAD, PROMPT_FEATURES_AND_OPERATORS, PROMPT_DIAGNOSIS, PROMPT_EVOLUTION


class LLMUnavailable(RuntimeError):
    """Transient API failures exhausted the configured attempts."""


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metric(value):
    return "unmeasured" if value is None else f"{value:+.6f}"


class AlphaCFTrainer:
    def __init__(self, pool, args, log_dir):
        self.pool, self.args, self.log_dir = pool, args, log_dir
        self.memory, self.visits = {}, Counter()
        self.lineage, self.factor_ids, self.chain_ids = [], {}, {}
        self.next_factor_id = 0
        self.client = self.model = None
        if args.rounds:
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                                 timeout=args.llm_timeout, max_retries=0)
            self.model = os.environ["OPENAI_MODEL_NAME"]
        self.step = 0
        self.log_lock = Lock()

    def record(self, event, **values):
        with self.log_lock, (self.log_dir / f"round_{self.step}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(event=event, **values), ensure_ascii=False, allow_nan=False) + "\n")

    def ask(self, prompt, context):
        base_prompt = PROMPT_FEATURES_AND_OPERATORS + prompt.format(
            **{key: json.dumps(value, ensure_ascii=False) for key, value in context.items()})
        factor = context["parent"]
        log_context = dict(task="diagnosis" if prompt == PROMPT_DIAGNOSIS else "evolution",
                           factor=factor if isinstance(factor, str) else factor["expression"])
        error = ""
        attempts = self.args.llm_attempts
        for attempt in range(1, attempts + 1):
            user_prompt = base_prompt + (f"\nPrevious response failed validation: {error}\n"
                "Correct that error. Return only a valid JSON object matching the required schema, "
                "without control characters or thinking text." if error else "")
            timeout = self.args.llm_timeout * min(attempt, 2)
            self.record("llm_request", attempt=attempt, **log_context, started_at=time.time(),
                        timeout_seconds=timeout, prompt_chars=len(PROMPT_HEAD) + len(user_prompt),
                        system_prompt=PROMPT_HEAD, user_prompt=user_prompt)
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(model=self.model,
                    messages=[dict(role="system", content=PROMPT_HEAD), dict(role="user", content=user_prompt)],
                    temperature=self.args.temperature, top_p=1, timeout=timeout)
            except (APIConnectionError, APIStatusError) as exc:
                retryable = isinstance(exc, APIConnectionError) or exc.status_code in (408, 409, 429) or exc.status_code >= 500
                label = type(exc).__name__ + (f" (HTTP {exc.status_code})" if isinstance(exc, APIStatusError) else "")
                self.record("llm_error", attempt=attempt, **log_context, error=label,
                            retryable=retryable, elapsed_seconds=time.monotonic() - started)
                print(f"  LLM request failed (attempt {attempt}/{attempts}, timeout={timeout}s): {label}", flush=True)
                if not retryable:
                    raise
                if attempt == attempts:
                    raise LLMUnavailable(f"{label} after {attempts} attempts") from exc
                delay = min(self.args.llm_retry_wait * 2 ** (attempt - 1), 60)
                # Respect provider cooldowns, bounded to avoid an unbounded stall.
                if isinstance(exc, APIStatusError):
                    try:
                        delay = max(delay, min(float(exc.response.headers.get("retry-after", 0)), 300))
                    except ValueError:
                        pass
                delay += random.SystemRandom().uniform(0, min(delay * 0.2, 5))
                self.record("llm_retry", attempt=attempt, **log_context, wait_seconds=delay)
                print(f"  Retrying in {delay:.1f}s", flush=True)
                time.sleep(delay)
                continue
            text, finish_reason = response.choices[0].message.content, response.choices[0].finish_reason
            self.record("llm_response", attempt=attempt, **log_context,
                        elapsed_seconds=time.monotonic() - started, response=text, finish_reason=finish_reason)
            try:
                text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
                if not text:
                    raise ValueError("Model response is empty after removing thinking content")
                try:
                    result = json.loads(text)
                except json.JSONDecodeError as exc:
                    block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
                    if block is None:
                        raise ValueError(f"Invalid JSON: {exc}") from exc
                    result = json.loads(block[1])
                if not isinstance(result, dict):
                    raise ValueError("Model response must be a JSON object")
                keys = ("mechanisms",) if prompt == PROMPT_DIAGNOSIS else ("offspring",)
                if not all(isinstance(result.get(key), list) for key in keys):
                    raise ValueError(f"Expected required list fields: {', '.join(keys)}")
                diagnosis = prompt == PROMPT_DIAGNOSIS
                summary = "understanding" if diagnosis else "updated_understanding"
                if not isinstance(result.get(summary), str) or not result[summary].strip():
                    raise ValueError(f"Expected nonempty {summary}")
                items = result["mechanisms" if diagnosis else "offspring"]
                fields = ("description", "replacement", "reason") if diagnosis else ("expression", "description")
                if len(items) > 5 or not all(isinstance(item, dict) and all(
                        isinstance(item.get(key), str) and item[key].strip() for key in fields) for item in items):
                    raise ValueError(f"Expected at most 5 items with nonempty {fields}")
                if not diagnosis:
                    ids = {row["id"] for row in context["parent"]["mechanisms"]}
                    if not all(isinstance(child.get("evidence_refs"), list) and all(
                            isinstance(ref, str) and ref in ids for ref in child["evidence_refs"]) for child in items):
                        raise ValueError("evidence_refs must list current mechanism IDs, or [] for exploration")
                return result
            except ValueError as exc:
                error = str(exc)
                self.record("invalid_llm_response", attempt=attempt, **log_context, error=error)
                print(f"  Invalid LLM response (attempt {attempt}/{attempts}): {exc}", flush=True)
                if attempt == attempts:
                    raise

    def evaluate(self, text):
        try:
            expr = parse(text, self.args)
            result = self.pool.evaluate(expr)
            self.record("factor", expression=str(expr), **{k: v for k, v in result.items() if k != "signal"})
            print(f"> Evaluated: RankIC={result['reward']:+.6f}, |R|={abs(result['reward']):.6f}, "
                  f"cost={result['cost']:.6f}, nodes={complexity(expr)}, expr={expr}", flush=True)
            return expr
        except (ValueError, IndexError) as exc:
            self.record("invalid", expression=text, error=str(exc))
            print(f"  Skip {text}: {exc}", flush=True)
            return None

    def initialize(self, seeds):
        if len(seeds) != self.args.pool_capacity:
            raise ValueError(f"Need exactly {self.args.pool_capacity} seeds, got {len(seeds)}")
        print(f"=== Initialization: loading {len(seeds)} curated factors ===", flush=True)
        candidates = []
        for i, seed in enumerate(seeds, 1):
            print(f"> Seed #{i}/{len(seeds)}", flush=True)
            expr = self.evaluate(seed)
            if expr is None:
                raise ValueError(f"Invalid initial factor #{i}: {seed}")
            candidates.append(expr)
        self.pool.initialize(candidates)
        for expr in self.pool.exprs:
            factor_id = f"f{self.next_factor_id}"
            self.next_factor_id += 1
            self.factor_ids[str(expr)] = factor_id
            self.chain_ids[str(expr)] = factor_id
            self.memory[factor_id] = []
            self.lineage.append(dict(round=0, outcome="seed", child=str(expr), child_id=factor_id,
                                     parent=None, parent_id=None, chain_id=factor_id))
        self.save()

    def select_parents(self):
        exprs = self.pool.exprs
        count = min(self.args.parents, len(exprs))
        utility = self.pool.utility()
        contributions = [utility - self.pool.utility(exprs[:i] + exprs[i + 1:])
                         for i in range(len(exprs))]
        low_pool = sorted(range(len(exprs)), key=lambda i: contributions[i])[:(len(exprs) + 1) // 2]
        low = random.sample(low_pool, count // 2)
        selected_indices = low + random.sample([i for i in range(len(exprs)) if i not in low], count - len(low))
        random.shuffle(selected_indices)
        selected = [exprs[i] for i in selected_indices]
        self.record("parent_selection", utility=utility, parents=[str(e) for e in selected],
                    members=[dict(expression=str(e), deletion_credit=contributions[i],
                                  source="low_contribution" if i in low else "random" if i in selected_indices else None)
                             for i, e in enumerate(exprs)])
        print(f"> Selected parents: {len(low)} low-contribution, {count - len(low)} random", flush=True)
        for expr in selected:
            self.visits[str(expr)] += 1
        return selected

    def diagnosis_context(self, parent):
        history = [] if self.args.no_memory else self.memory.get(self.chain_ids.get(str(parent)), [])
        previous = history[-1]["parent"] if history else {}
        return dict(parent=str(parent), reward=self.pool.evaluate(parent)["reward"],
            previous_understanding={k: previous[k] for k in ("expression", "updated_understanding") if k in previous},
            nodes=[dict(path=p, expression=str(n)) for p, n in walk(parent)],
            cf_enabled=not self.args.no_cf_evidence,
            limits=dict(nodes=self.args.max_nodes, depth=self.args.max_depth, lookback=self.args.max_backtrack))

    def diagnose(self, parent, utility, diagnosis=None):
        reward = self.pool.evaluate(parent)["reward"]
        print(f"> Parent {self.factor_ids[str(parent)]}: RankIC={reward:+.6f}, "
              f"nodes={complexity(parent)}, expr={parent}", flush=True)
        if diagnosis is None:
            diagnosis = self.ask(PROMPT_DIAGNOSIS, self.diagnosis_context(parent))
        understanding = diagnosis["understanding"].strip()
        self.record("understanding", factor=str(parent), understanding=understanding)
        print(f"  Understanding: {understanding}", flush=True)
        evidence = []
        index = next(i for i, e in enumerate(self.pool.exprs) if str(e) == str(parent))
        peers = self.pool.exprs[:index] + self.pool.exprs[index + 1:]
        for i, mechanism in enumerate(diagnosis["mechanisms"], 1):
            try:
                node = at(parent, mechanism["path"])
                row = dict(id=f"m{i}", factor=str(parent), expression=str(node),
                           **{key: mechanism[key] for key in ("path", "description", "replacement", "reason")},
                           delta_cf=None, pool_credit=None, signal_distance=None,
                           coverage_change=None, common_rank_correlation=None)
                if not self.args.no_cf_evidence:
                    changed = ablate(parent, mechanism, self.args)
                    result = self.pool.evaluate(changed)
                    row["counterfactual"] = str(changed)
                    row["delta_cf"] = abs(result["reward"]) - abs(reward)
                    row["signal_distance"] = self.pool.signal_distance(parent, changed)
                    row.update(self.pool.signal_comparison(parent, changed))
                    if not self.args.no_pool_credit:
                        # Use the same full-Train OLS comparison as parent replacement.
                        credit = self.pool.utility(peers + [changed]) - utility
                        if not math.isfinite(credit):
                            raise ValueError("Counterfactual pool has no usable variation")
                        row["pool_credit"] = credit
                evidence.append(row)
                self.record("mechanism", **row)
                print(f"  > Mechanism #{i}: path={row['path']}, {row['description']}\n"
                      f"    Edit: {row['expression']} -> {row['replacement']}\n"
                      f"    Hypothesis / prediction: {row['reason']}\n"
                      f"    CF expr: {row.get('counterfactual', 'unmeasured')}\n"
                      f"    delta_cf={metric(row['delta_cf'])}, pool_credit={metric(row['pool_credit'])}, "
                      f"signal_distance={metric(row['signal_distance'])}, "
                      f"coverage_change={metric(row['coverage_change'])}, "
                      f"common_rank_correlation={metric(row['common_rank_correlation'])}", flush=True)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                self.record("invalid_mechanism", factor=str(parent), proposal=mechanism, error=str(exc))
                print(f"  Skip mechanism: {exc}", flush=True)
        measured = [row for row in evidence if row["delta_cf"] is not None]
        print(f"  Diagnosed {len(evidence)} mechanisms, measured {len(measured)}", flush=True)
        return dict(expression=str(parent), reward=reward, complexity=complexity(parent),
                    understanding=understanding, mechanisms=evidence)

    def evolve(self, parent, previous_children):
        existing = [str(e) for e in self.pool.exprs] + previous_children
        factor_memory = ([] if self.args.no_memory
                         else self.memory[self.chain_ids[parent["expression"]]])
        print(f"> Evolving {self.factor_ids[parent['expression']]}: "
              f"{len(parent['mechanisms'])} mechanisms, {len(factor_memory)} past evolutions, "
              f"expr={parent['expression']}", flush=True)
        result = self.ask(PROMPT_EVOLUTION, dict(parent=parent,
            historical_memory=factor_memory,
            existing_expressions=existing,
            correlation_threshold=self.args.correlation_threshold,
            limits=dict(nodes=self.args.max_nodes, depth=self.args.max_depth, lookback=self.args.max_backtrack)))
        parent["updated_understanding"] = result["updated_understanding"].strip()
        parent["offspring"] = result["offspring"]
        self.record("understanding_update", factor=parent["expression"],
                    updated_understanding=parent["updated_understanding"])
        print(f"  Updated understanding: {parent['updated_understanding']}", flush=True)
        children, details, seen = [], {}, set(existing)
        for i, proposal in enumerate(result["offspring"], 1):
            text, description = proposal["expression"], proposal["description"].strip()
            print(f"  > Offspring #{i}/{len(result['offspring'])}: {text}\n"
                  f"    Evidence: {proposal['evidence_refs']}\n    Description: {description}", flush=True)
            self.record("proposal", parent=parent["expression"],
                        parent_id=self.factor_ids[parent["expression"]],
                        expression=text, description=description, evidence_refs=proposal["evidence_refs"])
            child = self.evaluate(text)
            if child is not None and str(child) not in seen:
                children.append(child)
                details[str(child)] = dict(description=description, evidence_refs=proposal["evidence_refs"])
                seen.add(str(child))
            elif child is not None:
                print(f"  Skip duplicate offspring: {child}", flush=True)
        print(f"  Evaluated {len(children)} new offspring", flush=True)
        return children, details

    def save(self):
        payload = self.pool.to_dict()
        payload["split"] = "train"
        payload["factor_ids"] = [self.factor_ids[str(e)] for e in self.pool.exprs]
        payload["chain_ids"] = [self.chain_ids[str(e)] for e in self.pool.exprs]
        payload["visits"] = dict(self.visits)
        save_json(self.log_dir / f"pool_{self.step}.json", payload)
        save_json(self.log_dir / "memory.json", self.memory)
        save_json(self.log_dir / "lineage.json", self.lineage)
        print(f"=== Logging at iteration {self.step} ===\n" + "-" * 45, flush=True)
        for i, (expr, factor_id, values, weight) in enumerate(zip(
                payload['exprs'], payload['factor_ids'], payload['metrics'], payload['weights']), 1):
            print(f"> Alpha #{i} ({factor_id}): RankIC={values['reward']:+.6f}, "
                  f"|R|={abs(values['reward']):.6f}, cost={values['cost']:.6f}, "
                  f"nodes={complexity(self.pool.exprs[i - 1])}, weight={weight:+.6f}, expr={expr}", flush=True)
        rewards = [abs(m['reward']) for m in payload['metrics']]
        print(f">> Best single |RankIC|: {max(rewards):.6f}, mean |RankIC|: {sum(rewards) / len(rewards):.6f}\n"
              f">> Pool: size={len(rewards)}, OLS_U={payload['utility']:+.6f}, "
              f"memory={sum(map(len, self.memory.values()))}\n"
              f">> Saved: {self.log_dir / f'pool_{self.step}.json'}\n" + "-" * 45, flush=True)

    def train(self, seeds, on_round=None):
        self.initialize(seeds)
        if on_round is not None:
            on_round(0)
        for step in range(1, self.args.rounds + 1):
            self.step = step
            started = time.monotonic()
            parents = self.select_parents()
            utility = self.pool.utility()
            print(f"=== Iteration {step}/{self.args.rounds}: {len(parents)} parents, OLS_U={utility:+.6f} ===", flush=True)
            evidence = []
            contexts = [self.diagnosis_context(parent) for parent in parents]
            with ThreadPoolExecutor(max_workers=self.args.diagnosis_workers) as executor:
                requests = [executor.submit(self.ask, PROMPT_DIAGNOSIS, context) for context in contexts]
                for i, (parent, request) in enumerate(zip(parents, requests), 1):
                    print(f"[Diagnosis {i}/{len(parents)}]", flush=True)
                    try:
                        diagnosis = request.result()
                    except LLMUnavailable as exc:
                        self.record("llm_parent_skipped", task="diagnosis", factor=str(parent), error=str(exc))
                        print(f"  Skip parent diagnosis; retained {parent}: {exc}", flush=True)
                        continue
                    evidence.append((parent, self.diagnose(parent, utility, diagnosis)))
            previous_children, added = [], []
            for i, (parent_expr, parent) in enumerate(evidence, 1):
                print(f"[Evolution {i}/{len(evidence)}]", flush=True)
                try:
                    children, details = self.evolve(parent, previous_children)
                except LLMUnavailable as exc:
                    self.record("llm_parent_skipped", task="evolution", factor=str(parent_expr), error=str(exc))
                    print(f"  Skip parent evolution; retained {parent_expr}: {exc}", flush=True)
                    continue
                previous_children.extend(str(child) for child in children)
                decision = self.pool.replace_parent(parent_expr, children)
                print(f"> Parent score: S={decision['parent_score']:.6f}, "
                      f"group_S={decision['parent_group_score']:.6f}, "
                      f"RankIC={decision['parent_reward']:+.6f}, nodes={decision['parent_complexity']}", flush=True)
                for candidate in decision["candidates"]:
                    candidate.update(details[candidate["expression"]])
                    status = "selected" if candidate['expression'] == decision['child'] else "not selected" if candidate['eligible'] else "not eligible"
                    print(f"  > Candidate ({status}): RankIC={candidate['reward']:+.6f}, nodes={candidate['complexity']}, "
                          f"max_corr={candidate['max_correlation']:.6f}, correlation_ok={candidate['correlation_ok']}, credit_ok={candidate['credit_ok']}, distance={metric(candidate['signal_distance'])}, "
                          f"C_pool={candidate['pool_credit']:+.6f}, cost={candidate['cost']:.6f}, "
                          f"S={candidate['score']:.6f}, expr={candidate['expression']}", flush=True)
                if decision['child'] is not None:
                    print(f"[Pool Pop] {decision['parent']}\n[Pool Add] {decision['child']}\n"
                          f"  Replacement reason: {decision['reason']}", flush=True)
                else:
                    print(f"[Pool Reject] offspring: {decision['reason']}; retained {decision['parent']}", flush=True)
                decision["child_description"] = details.get(decision["child"], {}).get("description")
                decision["child_evidence_refs"] = details.get(decision["child"], {}).get("evidence_refs", [])
                chain_id = self.chain_ids[str(parent_expr)]
                decision.update(round=step, parent_id=self.factor_ids[str(parent_expr)], child_id=None,
                                chain_id=chain_id)
                if decision["child"] is not None:
                    child_id = f"f{self.next_factor_id}"
                    self.next_factor_id += 1
                    self.factor_ids.pop(str(parent_expr))
                    self.factor_ids[decision["child"]] = child_id
                    self.chain_ids.pop(str(parent_expr))
                    self.chain_ids[decision["child"]] = chain_id
                    decision["child_id"] = child_id
                    added.append(decision["child"])
                self.lineage.append(decision)
                self.memory[chain_id].append(dict(parent=parent, decision=decision))
                self.record("replacement", **decision)
                self.pool.keep(self.pool.exprs)
            self.record("update", offspring=len(previous_children), added=added, utility=self.pool.utility())
            self.save()
            if on_round is not None:
                on_round(step)
            print(f"  {len(previous_children)} offspring, {len(added)} parents replaced, "
                  f"{time.monotonic() - started:.1f}s", flush=True)
        return self.pool.exprs
