"""LLM adapters for Phase A: propose minimal edits, then attribute to a slimmer factor.

Public surface
--------------
- propose(expr, k, recent)  -> List[EditAction]         (3 rounds per seed)
- attribute(f, records)     -> dict (necessary/redundant/final_factor/reason)
- record_illegal / set_illegal_log: shared by phaseA + llm itself

JSON parsing
------------
LLM outputs may be wrapped in markdown ```json ... ```, contain <think>...</think>
prefixes, or be slightly malformed. _json_obj() tries (in order):
  1. direct json.loads on the whole text
  2. extract ```json ... ``` block, json.loads
  3. raw_decode every top-level {...} starting at each '{'
  4. salvage by re-extracting the "edits" array if present
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
import httpx

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagen.data.expression import (
    BinaryOperator,
    Expression,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)

from alpha_cf.config import (
    CONSTANTS,
    DELTA_TIMES,
    FEATURE_NAMES,
    K,
    OPERATORS,
)
from alpha_cf.edits import EDIT_KINDS, apply_edit
from alpha_cf.pool import parse_expr
from alpha_cf.types import CFRecord, EditAction, Site

_REPO = Path(__file__).resolve().parents[2]
_ILLEGAL_LOG: Optional[str] = None  # train_cf sets this to <out_dir>/illegal.jsonl


# path or None -> where skipped illegal edits are appended as JSONL.
def set_illegal_log(path: Optional[str]) -> None:
    global _ILLEGAL_LOG
    _ILLEGAL_LOG = path


# print + optional jsonl: one illegal T(f,a)=None; does not raise.
def record_illegal(source: str, expr, action: EditAction, **extra) -> None:
    print(
        f"[{source}] ILLEGAL skip  kind={action.kind}  site_id={action.site_id}  "
        f"new_value={action.new_value!r}  T(f,a)=None  f={expr}",
        flush=True,
    )
    if action.hypothesis:
        print(f"[{source}]          hypothesis={action.hypothesis}", flush=True)
    if _ILLEGAL_LOG:
        rec = {
            "source": source,
            "f": str(expr),
            "kind": action.kind,
            "site_id": action.site_id,
            "new_value": action.new_value,
            "hypothesis": action.hypothesis,
            "T": None,
        }
        rec.update(extra)
        with open(_ILLEGAL_LOG, "a") as w:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")


class LLMError(RuntimeError):
    pass


PROMPT_HEAD = """
You are an expert on quantitative finance and alpha factor mining, acting as a structural counterfactual intervention proposer.
You propose WHAT to test. You do NOT compute IC / RankIC / Delta, do NOT backtest, and do NOT pick the final factor. Real market evaluation handles those.
Strictly follow the user instructions. Output ONLY a JSON object — no code fences, no <think> tags, no prose.
"""


PROMPT_FEATURES_AND_OPERATORS = """
1. Features: $open, $high, $low, $close, $vwap, $volume.
2. Rolling windows MUST be integers from WINDOWS = {windows}. Arithmetic constants MUST be floats from CONSTANTS = {constants}. No scientific notation.
3. Operators (use exact names, same arity):
   Abs, SLog1p, Inv, Sign, Log, Rank,
   Add, Sub, Mul, Div, Pow, Greater, Less,
   Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin,
   TsMed, TsMad, TsRank, TsDelta, TsDiv, TsPctChange, TsWMA, TsEMA,
   TsCov, TsCorr
4. Syntax: Op(arg, ...) or $feature or a constant. Rolling / PairRolling take the window as the LAST integer arg. Greater/Less require BOTH children to contain a market feature.
Examples: TsMean($close,20)  Div(Sub($close,$open),Add(Sub($high,$low),0.01))  TsCorr($close,$volume,10)
"""


PROMPT_TASK = """
Task: propose K={k} MINIMAL one-site counterfactual edits of the current factor f.
The PURPOSE of Phase A is to UNDERSTAND WHICH SUB-STRUCTURES OF f ARE REDUNDANT OR LOAD-BEARING, not to invent a stronger alpha. Every edit here is a probe, not an improvement.
1. Each edit changes exactly one site_id, keeps the rest unchanged. The edit is applied to the FROZEN ORIGINAL f, never to a previous f'.
2. For each non-leaf site in f, you SHOULD propose at least one PROBE that targets it — feature_replace / operator_replace / window_replace at that site, OR subtree_delete of that subtree. Probes tell us what the site contributes; without them we cannot attribute.
3. subtree_delete is the most informative probe: it tests "is this subtree redundant?". Prioritise subtree_delete whenever the candidate deletion is legal (see rule 8).
4. For richer factors (>= 4 sites), include BOTH probing edits AND a few candidate-improvement edits (feature/operator/window replace). The improvement edits give the attribution step something to weigh the probes against; they are NOT the goal.
5. For very short factors (1–2 sites, e.g. $volume or $close), the catalog of useful edits is small. Propose what you can; the attribution step may decide to keep f unchanged.
6. Use only features/operators from the catalog. Rolling windows MUST be integers in WINDOWS; arithmetic constants MUST be floats in CONSTANTS.
7. Operator replacement MUST stay in the same category (UNARY/BINARY/ROLLING/PAIR_ROLLING); use the per-site ∈ set.
8. Four kinds (Phase A only allows 减法 / 等量替换型编辑; wrap has been removed):
   feature_replace  swap a $feature  →  tests whether alpha is price- vs volume-based, etc.
   operator_replace swap same-arity operator → tests mean vs extreme, Add vs Sub, etc.
   window_replace   change time scale → tests horizon; new_value as a string e.g. "10".
   subtree_delete   remove that subtree and keep a valid EXPR. new_value MUST be null. Rules: root (site_id 0) cannot be deleted; under Binary/PairRolling the sibling branch replaces the parent; under Unary/Rolling the operand replaces the op (unwrap); deleting a Feature/Constant that is the only child of Unary/Rolling is illegal — delete the parent op instead.
9. {k} edits must be semantically different (not synonymous window tweaks).
10. hypothesis MUST be a falsifiable causal sentence. For probes, the hypothesis should state WHAT EVIDENCE the edit would produce if the targeted subtree is / is not redundant.
11. If Observed traces are not empty, they are real one-site interventions vs the frozen original f. You do NOT see numeric IC / Delta. Later batches MUST prioritize (kind, site_id) pairs listed under Untested.
12. Do NOT output anything except the JSON object. Length MUST be exactly {k}.

Output JSON:
{{"edits":[
  {{"hypothesis":"...","kind":"feature_replace|operator_replace|window_replace|subtree_delete",
    "site_id":0,"new_value":"... or null when kind=subtree_delete"}}
]}}

Current factor:
f = {expr}

Editable sites (site_id is preorder; new_value must come from that site's ∈ set):
{sites}

Observed Delta traces (real backtests from earlier rounds):
{recent}

Untested (kind, site_id) pairs (prioritize these):
{untested}
"""


PROMPT_ATTRIBUTE = """
You are an expert on alpha factor mining. Given a seed factor f and 15 (or fewer) real counterfactual backtest records on f, decide which sub-structures of f are REDUNDANT (safe to drop) and which are NECESSARY (load-bearing). Your output is a TRIMMED version of f, used to UNDERSTAND its mechanism — NOT to invent a stronger alpha.

You may rewrite f freely — you are NOT constrained to apply a specific edit. Use the catalog of features/operators/windows below.

CRITICAL: if the records show that every probe targeting a subtree leaves inc_residual ~0 AND delta_ic ~0, that subtree is redundant — drop it. Conversely, if a probe causes large |delta_ic| or |inc_residual|, that subtree is load-bearing — keep it.

KEEP f UNCHANGED when it is already minimal:
- f is a single feature (e.g. $volume, $close) — there is nothing to drop.
- f has 1–2 sites and ALL probes show inc_residual ~0 — the seed is already a clean irreducible unit.
In these cases, set final_factor EXACTLY to the original f string below. Returning f unchanged is the CORRECT answer, not a fallback.

How to use the records:
- Each record has: kind, site_id, new_value, hypothesis, inc_residual, ic_f, ic_fp, delta_ic, f_prime.
- inc_residual = Pearson IC of residual(f' | f) vs forward return. Near 0 means f' carries NO info beyond f; far from 0 (positive OR negative) means f' has INDEPENDENT signal vs f. This is the KEY signal for redundancy: if dropping/changing a subtree barely changes inc_residual, the subtree is likely redundant.
- delta_ic = ic(f') − ic(f). Positive means the edit IMPROVED IC; negative means it HURT IC.
- ic_f / ic_fp are the raw Pearson ICs of f and f' on train (you can sanity-check delta_ic from these).
- Compare across records: if all (kind, site_id) edits targeting a subtree leave inc_residual ~0 and delta_ic ~0, that subtree is redundant. If changing it causes a large |delta_ic| or |inc_residual|, it is load-bearing.
- You see multiple edit kinds at multiple sites. Use the whole picture to identify the essential mechanism vs accidental complexity.

Output ONLY this JSON object (no code fences, no <think> tags, no prose):
{{
  "necessary": "what load-bearing mechanism you kept, and why",
  "redundant":  "what sub-structure(s) you dropped, and the evidence (cite records by site_id / kind)",
  "final_factor": "a legal alpha expression using the catalog; return the ORIGINAL f string verbatim when f is already minimal",
  "reason": "1-2 sentences tying necessary+redundant to the final_factor"
}}

Catalog (exact names, same arity as defined):
$open $high $low $close $vwap $volume (features)
WINDOWS = {windows}
CONSTANTS = {constants}
Operators: Abs, SLog1p, Inv, Sign, Log, Rank, Add, Sub, Mul, Div, Pow, Greater, Less, Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin, TsMed, TsMad, TsRank, TsDelta, TsDiv, TsPctChange, TsWMA, TsEMA, TsCov, TsCorr
(ROLLING/PairRolling take the window as the LAST int arg; Greater/Less need TWO featured children; constants need a decimal point.)

Original factor (return this verbatim if f is already minimal):
f = {expr}

Observed counterfactual records (real backtests):
{records}
"""


# expr, IC(f), k -> legal EditActions from the LLM (illegal items printed and dropped).
#
# Pipeline:
#   1. Load .env (api key / base url / model).
#   2. Build messages with PROMPT_HEAD (system) + PROMPT_TASK + per-site list.
#   3. _chat() retries on transient errors with exponential backoff up to
#      LLM_RETRY_BUDGET seconds.
#   4. Strip <think>...</think> prefix, parse JSON, drop edits whose T(f, a)
#      produces f' == f (no-op) or fails apply_edit.
def propose(expr, k: int = K, recent: Optional[Sequence[CFRecord]] = None) -> List[EditAction]:
    cfg = _dotenv()
    sites = _sites_with_synthesis_safe_includes(expr)
    print(f"[llm] propose k={k}", flush=True)
    data = _chat(cfg, _messages_propose(expr, k, sites, recent))
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    text = _answer_content(msg)
    print(
        f"[llm] finish_reason={choice.get('finish_reason')}  "
        f"reasoning_chars={_reasoning_chars(msg)}",
        flush=True,
    )
    if not text:
        print(f"[llm] empty propose content; returning empty", flush=True)
        return []
    print("[llm] raw output >>>", flush=True)
    print(text, flush=True)
    print("[llm] raw output <<<", flush=True)
    actions = _parse_edits(text, expr)
    if len(actions) != k:
        print(f"[llm] expected {k} edits, got {len(actions)}; keeping parsed items", flush=True)
    legal: List[EditAction] = []
    # Verify each parsed edit actually produces a featured f' (LLM may propose
    # an edit whose new_value isn't in the site's ∈ set — apply_edit returns None).
    for a in actions:
        fp = apply_edit(expr, a)
        if fp is None:
            record_illegal("llm", expr, a)
            continue
        print(f"[llm] ok  {a.kind}@{a.site_id} {a.new_value!r}  ->  {fp}", flush=True)
        legal.append(a)
    if not legal:
        print("[llm] all edits illegal this round; returning empty", flush=True)
    elif len(legal) != len(actions):
        print(f"[llm] kept {len(legal)} legal / {len(actions)} parsed", flush=True)
    return legal


# After probe records: returns {"necessary", "redundant", "final_factor", "reason"}.
# If LLM fails, returns a fallback dict with final_factor = str(f).
def attribute(f: Expression, records: Sequence[dict]) -> dict:
    fallback = {
        "necessary": "",
        "redundant": "",
        "final_factor": str(f),
        "reason": "fallback: LLM call failed; keeping original f",
    }
    cfg = _dotenv()
    print(f"[llm] attribute  records={len(records)}", flush=True)
    try:
        data = _chat(cfg, _messages_attribute(f, records))
    except LLMError as e:
        print(f"[llm] attribute failed: {e}; returning fallback", flush=True)
        return fallback
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    text = _answer_content(msg)
    print(
        f"[llm] finish_reason={choice.get('finish_reason')}  "
        f"reasoning_chars={_reasoning_chars(msg)}",
        flush=True,
    )
    if not text:
        print("[llm] empty attribute content; returning fallback", flush=True)
        return fallback
    print("[llm] raw output >>>", flush=True)
    print(text, flush=True)
    print("[llm] raw output <<<", flush=True)
    parsed = _parse_attribute(text, fallback)
    # verify final_factor parses to a featured expression
    ff = parsed.get("final_factor") or ""
    try:
        expr = parse_expr(ff)
        if not getattr(expr, "is_featured", True):
            print(f"[llm] attribute final_factor unfeatured: {expr}", flush=True)
            return fallback
        # Normalise to canonical str so downstream string-compare matches.
        parsed["final_factor"] = str(expr)
        return parsed
    except Exception as e:
        print(f"[llm] attribute final_factor parse failed: {e}; raw={ff!r}", flush=True)
        return fallback


# repo .env -> LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (+ optional knobs).
def _dotenv() -> Dict[str, str]:
    path = _REPO / ".env"
    if not path.is_file():
        raise LLMError("missing .env")
    cfg = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        cfg[key.strip()] = val.strip().strip("'").strip('"')
    for req in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if not cfg.get(req):
            raise LLMError(f"{req} empty in .env")
    return cfg


def _vocab() -> str:
    return "\n".join([
        "FEATURES: " + ", ".join(FEATURE_NAMES),
        "WINDOWS: " + ", ".join(map(str, DELTA_TIMES)),
        "CONSTANTS: " + ", ".join(map(str, CONSTANTS)),
        "UNARY_OPS: " + ", ".join(op.__name__ for op in OPERATORS
                                   if op.category_type() is UnaryOperator),
        "BINARY_OPS: " + ", ".join(op.__name__ for op in OPERATORS
                                    if op.category_type() is BinaryOperator),
        "ROLLING_OPS: " + ", ".join(op.__name__ for op in OPERATORS
                                    if op.category_type() is RollingOperator),
        "PAIR_ROLLING_OPS: " + ", ".join(op.__name__ for op in OPERATORS
                                         if op.category_type() is PairRollingOperator),
    ])


# Local import here to avoid a top-level cycle: edits imports from alpha_cf.types,
# and we want to call enumerate_sites without importing alpha_cf.edits at module
# load (which would re-import this file).
def _sites_with_synthesis_safe_includes(expr: Expression) -> List[Site]:
    from alpha_cf.edits import enumerate_sites
    return enumerate_sites(expr)


def _format_sites(sites: List[Site]) -> str:
    blocks = []
    for s in sites:
        lines = [f"[site_id={s.site_id}] type={s.kind}  subtree={s.expr_str}  current={s.current}"]
        if s.kind == "feature":
            lines.append("  allowed: feature_replace, new_value in {" + ", ".join(s.allowed) + "}")
        elif s.kind == "operator":
            lines.append("  allowed: operator_replace, new_value in {" + ", ".join(s.allowed) + "}")
            if s.allowed_windows:
                lines.append("  allowed: window_replace, new_value in {" + ", ".join(map(str, s.allowed_windows)) + "}")
        if s.can_delete:
            lines.append("  allowed: subtree_delete, new_value must be null")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _last_n_rounds(recent: Sequence[CFRecord], n: int = 1) -> Sequence[CFRecord]:
    """Keep only records from the most recent `n` rounds (by extra["round"]).

    phaseA passes back round_idx-1 in extra["round"] so subsequent rounds see
    only their immediate predecessor, encouraging the LLM to explore new
    (kind, site) pairs instead of re-trying the previous batch.
    """
    seen: List[int] = []
    for rec in recent:
        rnd = (rec.extra or {}).get("round")
        if isinstance(rnd, int) and rnd not in seen:
            seen.append(rnd)
    if len(seen) <= n:
        return recent
    keep = set(seen[-n:])
    return [rec for rec in recent if (rec.extra or {}).get("round") in keep]


def _format_recent(recent: Sequence[CFRecord]) -> str:
    lines = []
    for rec in recent:
        a = rec.action
        parts = [
            f"f={rec.f}",
            f"f'={rec.f_prime}",
            f"kind={a.kind}",
            f"site_id={a.site_id}",
            f"new_value={a.new_value}",
        ]
        if a.hypothesis:
            parts.append(f"hypothesis={a.hypothesis}")
        extra = rec.extra or {}
        for key in ("round",):
            if key in extra:
                parts.append(f"{key}={extra[key]}")
        lines.append("- " + "  ".join(parts))
    return "\n".join(lines)


def _format_records_for_attr(records: Sequence[dict]) -> str:
    """Pretty-print records for the attribution prompt.

    Each record is the dict shape produced by phaseA: keys include kind, site_id,
    new_value, hypothesis, inc_residual, ic_f, ic_fp, delta_ic, f_prime.
    Missing keys are skipped.
    """
    lines = []
    for i, rec in enumerate(records, 1):
        parts = [f"#{i}  kind={rec.get('kind')}  site_id={rec.get('site_id')}  new_value={rec.get('new_value')}"]
        if rec.get("hypothesis"):
            parts.append(f"hypothesis={rec['hypothesis']}")
        for k in ("inc_residual", "ic_f", "ic_fp", "delta_ic"):
            if k in rec:
                v = rec[k]
                if isinstance(v, float):
                    parts.append(f"{k}={v:+.4f}")
                else:
                    parts.append(f"{k}={v}")
        if rec.get("f_prime"):
            parts.append(f"f'={rec['f_prime']}")
        lines.append("- " + "  ".join(parts))
    if not lines:
        return "(empty: no legal edits were produced for this seed)"
    return "\n".join(lines)


def _untested_text(expr: Expression, recent: Optional[Sequence[CFRecord]]) -> str:
    # (kind, site_id) pairs the LLM has NOT tried yet → tell it to prioritise
    # these in the next batch so coverage expands across rounds.
    from alpha_cf.edits import enumerate_sites, kinds_for_site
    tried = {(rec.action.kind, rec.action.site_id) for rec in (recent or [])}
    lines = []
    for s in enumerate_sites(expr):
        missing = [k for k in kinds_for_site(s) if (k, s.site_id) not in tried]
        if missing:
            lines.append(f"site {s.site_id} current={s.current}: untested {', '.join(missing)}")
    return "\n".join(lines) if lines else "(all listed kinds on all sites were tried at least once)"


def _messages_propose(expr: Expression, k: int, sites: List[Site],
                      recent: Optional[Sequence[CFRecord]]):
    recent_txt = (
        _format_recent(_last_n_rounds(recent))
        if recent
        else "(empty. This is the first intervention batch for this factor.)"
    )
    untested = (
        _untested_text(expr, recent)
        if recent
        else "(first batch: no coverage constraint)"
    )
    user = (
        PROMPT_FEATURES_AND_OPERATORS
        + PROMPT_TASK.format(
            k=k,
            windows=", ".join(map(str, DELTA_TIMES)),
            constants=", ".join(map(str, CONSTANTS)),
            expr=expr,
            sites=_format_sites(sites),
            recent=recent_txt,
            untested=untested,
        )
    )
    return [
        {"role": "system", "content": PROMPT_HEAD.strip()},
        {"role": "user", "content": user.strip()},
    ]


def _messages_attribute(f: Expression, records: Sequence[dict]):
    user = PROMPT_ATTRIBUTE.format(
        windows=", ".join(map(str, DELTA_TIMES)),
        constants=", ".join(map(str, CONSTANTS)),
        expr=f,
        records=_format_records_for_attr(records),
    )
    return [
        {"role": "system", "content": PROMPT_HEAD.strip()},
        {"role": "user", "content": user.strip()},
    ]


# OpenAI-compatible chat; retries transient errors until LLM_RETRY_BUDGET (default 300s).
def _chat(cfg: Dict[str, str], messages) -> dict:
    timeout = float(cfg["LLM_TIMEOUT"]) if cfg.get("LLM_TIMEOUT") else 600.0
    budget = float(cfg["LLM_RETRY_BUDGET"]) if cfg.get("LLM_RETRY_BUDGET") else 300.0
    thinking = cfg.get("LLM_THINKING") or "adaptive"
    # Some upstream models (e.g. Claude on certain proxies) split reasoning into
    # a separate billing bucket; reasoning_split=false merges them back.
    extra: Dict = {
        "thinking": {"type": thinking},
        "reasoning_split": thinking != "disabled",
    }
    kwargs = {
        "model": cfg["LLM_MODEL"],
        "messages": messages,
        "max_tokens": int(cfg["LLM_MAX_TOKENS"]) if cfg.get("LLM_MAX_TOKENS") else 16384,
        "extra_body": extra,
    }
    if cfg.get("LLM_TEMPERATURE"):
        kwargs["temperature"] = float(cfg["LLM_TEMPERATURE"])

    attempt = 0
    t0 = time.monotonic()
    last_err: Optional[Exception] = None
    while True:
        try:
            client = OpenAI(
                api_key=cfg["LLM_API_KEY"],
                base_url=cfg["LLM_BASE_URL"].rstrip("/"),
                timeout=httpx.Timeout(timeout, connect=30.0),
            )
            resp = client.chat.completions.create(**kwargs)
            return resp.model_dump()
        except Exception as e:
            last_err = e
            elapsed = time.monotonic() - t0
            # Non-transient errors (auth, bad request) → fail immediately;
            # transient errors (5xx, rate-limit, timeout) → retry with exp backoff
            # capped at LLM_RETRY_BUDGET wall-clock seconds total.
            if not _transient(e) or elapsed >= budget:
                break
            attempt += 1
            sleep = min(60.0, 5.0 * (2 ** (attempt - 1)))
            remain = budget - elapsed
            if remain <= 0:
                break
            sleep = min(sleep, remain)
            print(
                f"[llm] transient {type(e).__name__}: {e}  "
                f"retry={attempt}  sleep={sleep:.0f}s  elapsed={elapsed:.0f}/{budget:.0f}s",
                flush=True,
            )
            time.sleep(sleep)
    raise LLMError(f"LLM request failed: {type(last_err).__name__}: {last_err}") from last_err


def _transient(exc: Exception) -> bool:
    # Network / rate-limit / 5xx → retry. Auth / 4xx (other than 408/409/425/429) → don't.
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
        return True
    return False


def _answer_content(msg: dict) -> str:
    """Strip <think>...</think> and concat all content parts into one string."""
    raw = msg.get("content")
    if isinstance(raw, list):
        # Some providers return content as [{type: text, text: ...}, ...]
        parts = []
        for p in raw:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or "")
        raw = "".join(parts)
    text = (raw or "").strip()
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _reasoning_chars(msg: dict) -> int:
    """Length of the model's hidden reasoning chain (for logging / debugging)."""
    n = len(msg.get("reasoning_content") or "")
    details = msg.get("reasoning_details") or []
    if isinstance(details, list):
        for d in details:
            if isinstance(d, dict):
                n += len(d.get("text") or "")
            elif isinstance(d, str):
                n += len(d)
    return n


def _has_edits(obj) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("edits"), list)


def _has_attribution(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("final_factor"), str):
        return False
    # need at least one of necessary / redundant / reason
    return any(isinstance(obj.get(k), str) for k in ("necessary", "redundant", "reason"))


def _is_payload(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    return _has_edits(obj) or _has_attribution(obj)


def _extract_array(text: str, key: str) -> Optional[List[dict]]:
    # Last-resort salvage: if the wrapper JSON is unparseable but a "edits": [...]
    # array is somewhere in the text, pull it out so we lose a few items but
    # keep the round productive instead of returning empty.
    needle = f'"{key}"'
    at = text.find(needle)
    if at < 0:
        return None
    i = text.find("[", at)
    if i < 0:
        return None
    i += 1
    dec = json.JSONDecoder()
    items: List[dict] = []
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            items.append(obj)
        i = end
    return items or None


# model text -> {"edits": [...]} or {"final_factor": ..., ...} payload.
def _json_obj(text: str) -> dict:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    try:
        obj = json.loads(text)
        if _is_payload(obj):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if _is_payload(obj):
                return obj
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if _is_payload(obj):
            return obj
    salvaged = _extract_array(text, "edits")
    if salvaged is not None:
        print(f"[llm] wrapper JSON invalid; salvaged {len(salvaged)} edit objects", flush=True)
        return {"edits": salvaged}
    raise LLMError("response is not JSON")


# JSON text -> EditAction list; bad items printed and dropped, not raised.
def _parse_edits(text: str, expr=None) -> List[EditAction]:
    try:
        obj = _json_obj(text)
    except LLMError as e:
        print(f"[llm] parse failed: {e}; returning empty", flush=True)
        return []
    if not _has_edits(obj):
        if not _has_attribution(obj):
            print("[llm] JSON missing edits list; returning empty", flush=True)
        return []
    out = []
    for i, item in enumerate(obj["edits"]):
        if not isinstance(item, dict):
            print(f"[llm] ILLEGAL skip  edits[{i}] not an object: {item!r}", flush=True)
            continue
        kind = item.get("kind")
        if kind not in EDIT_KINDS:
            dummy = EditAction(str(kind), int(item["site_id"]) if "site_id" in item else -1,
                               item.get("new_value"), item.get("hypothesis"))
            record_illegal("llm", expr, dummy, reason="bad kind")
            continue
        try:
            site_id = int(item["site_id"])
        except (KeyError, TypeError, ValueError):
            print(f"[llm] ILLEGAL skip  edits[{i}] missing/bad site_id: {item!r}", flush=True)
            continue
        new_value = None if kind == "subtree_delete" else (
            None if item.get("new_value") is None else str(item["new_value"])
        )
        if kind != "subtree_delete" and new_value is None:
            dummy = EditAction(kind, site_id, None, item.get("hypothesis"))
            record_illegal("llm", expr, dummy, reason="null new_value")
            continue
        out.append(EditAction(kind, site_id, new_value, item.get("hypothesis")))
    return out


# JSON text -> {"necessary", "redundant", "final_factor", "reason"}; missing keys -> "".
def _parse_attribute(text: str, fallback: dict) -> dict:
    try:
        obj = _json_obj(text)
    except LLMError as e:
        print(f"[llm] attribute parse failed: {e}; returning fallback", flush=True)
        return fallback
    if not _has_attribution(obj):
        print(f"[llm] attribute JSON invalid; returning fallback", flush=True)
        return fallback
    raw_factor = obj.get("final_factor")
    if not isinstance(raw_factor, str):
        return fallback
    return {
        "necessary": str(obj.get("necessary") or ""),
        "redundant": str(obj.get("redundant") or ""),
        "final_factor": raw_factor.strip(),
        "reason": str(obj.get("reason") or ""),
    }


if __name__ == "__main__":
    from alpha_cf.pool import parse_expr

    f = parse_expr(sys.argv[1] if len(sys.argv) > 1 else "TsMean($close,20)")
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    actions = propose(f, k)
    print(f"[llm] standalone got {len(actions)} legal edits")
    for a in actions:
        print(f"T(f,a)  {a.kind}@{a.site_id} {a.new_value!r}  ->  {apply_edit(f, a)}")