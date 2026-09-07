import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from openai import OpenAI

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagen.data.expression import (
    BinaryOperator,
    Expression,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)

from alpha_cf.config import CONSTANTS, DELTA_TIMES, FEATURE_NAMES, K, OPERATORS
from alpha_cf.edits import EDIT_KINDS, apply_edit, enumerate_sites
from alpha_cf.types import CFRecord, EditAction, Site

_REPO = Path(__file__).resolve().parents[2]
_ILLEGAL_LOG = None  # train_cf sets this to <out_dir>/illegal.jsonl


# path or None -> where skipped illegal edits are appended as JSONL
def set_illegal_log(path: Optional[str]) -> None:
    global _ILLEGAL_LOG
    _ILLEGAL_LOG = path


# print + optional jsonl: one illegal T(f,a)=None; does not raise
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


# raised when .env, HTTP, or JSON is invalid (illegal edits are skipped, not raised)
class LLMError(RuntimeError):
    pass


PROMPT_HEAD = """
You are an expert on quantitative finance and alpha factor mining, acting as a structural counterfactual intervention proposer.
You propose WHAT to test. You do NOT compute IC / RankIC / Delta, do NOT backtest, and do NOT pick the final factor. Real market evaluation and any later Q model handle those.
Strictly follow the instructions given by the user below. Make sure the output ONLY CONTAINS A JSON FORMAT. Do not output anything else other than the JSON object, including code block markers like ``` or ```json, <think> tags, or any prose before or after the JSON.
"""

PROMPT_FEATURES_AND_OPERATORS = """
The available features, constants and operators are listed below.
1. You can use the following features:
   $open: Opening price.
   $high: Daily highest price.
   $low: Daily lowest price.
   $close: Closing price.
   $vwap: Daily average price, weighted by the volume of trades at each price.
   $volume: Trading number of shares.
2. You can use int constants from WINDOWS during rolling (time-series) calculations, and float constants from CONSTANTS during arithmetic calculations. Other constants are not allowed. Do NOT use scientific notation.
3. The following operators are available:
(BEGIN OF FEATURES AND OPERATORS DEFINITIONS)
    Abs(x): Absolute value of x
    SLog1p(x): Signed log transform: sign(x) * log(1 + |x|)
    Inv(x): Reciprocal 1/x
    Sign(x): Sign of x: 1 if x > 0, -1 if x < 0, 0 if x = 0
    Log(x): Natural logarithm of x
    Rank(x): Cross-sectional rank of x (across stocks on the same day)
    Add(x, y): x + y
    Sub(x, y): x - y
    Mul(x, y): x * y
    Div(x, y): x / y
    Pow(x, y): x raised to the power of y; y must be a constant
    Greater(x, y): elementwise max of x and y; BOTH sides must contain a market feature (a constant child is illegal)
    Less(x, y): elementwise min of x and y; same featured constraint as Greater
    Ref(x, d): Value of x d days ago
    TsMean(x, d): Rolling mean of x over the past d days
    TsSum(x, d): Rolling sum of x over the past d days
    TsStd(x, d): Rolling standard deviation of x over the past d days
    TsIr(x, d): Rolling information ratio (mean/std) over the past d days
    TsMinMaxDiff(x, d): TsMax(x, d) - TsMin(x, d)
    TsMaxDiff(x, d): today's x minus TsMax(x, d)
    TsMinDiff(x, d): today's x minus TsMin(x, d)
    TsVar(x, d): Rolling variance of x over the past d days
    TsSkew(x, d): Rolling skewness of x over the past d days
    TsKurt(x, d): Rolling kurtosis of x over the past d days
    TsMax(x, d): Rolling maximum of x over the past d days
    TsMin(x, d): Rolling minimum of x over the past d days
    TsMed(x, d): Rolling median of x over the past d days
    TsMad(x, d): Rolling mean absolute deviation over the past d days
    TsRank(x, d): Time-series rank of x over the past d days
    TsDelta(x, d): Today's value of x minus the value of x d days ago
    TsDiv(x, d): Today's x divided by the rolling mean of x over d days
    TsPctChange(x, d): Percentage change in x over the past d days
    TsWMA(x, d): Weighted moving average over the past d days with linearly decaying weights
    TsEMA(x, d): Exponential moving average of x with span d
    TsCov(x, y, d): Time-series covariance of x and y for the past d days
    TsCorr(x, y, d): Time-series correlation of x and y for the past d days
(END OF FEATURES AND OPERATORS DEFINITIONS)
4. Expression syntax: Op(arg, ...) or $feature or a constant. Rolling / PairRolling take the window as the last integer argument. Unary ops take one argument. Binary ops take two. PairRolling (TsCov, TsCorr) take two expressions plus a window.
Examples of valid alpha expressions:
TsMean($close,20)
Div(Sub($close,$open),Add(Sub($high,$low),0.01))
TsCorr($close,$volume,10)
Rank($volume)
"""

PROMPT_TASK = """
Your task is to propose K={k} MINIMAL structural counterfactual edits of the current factor f, such that:
1. Each edit is a one-site intervention T(f, a): change exactly one site_id, keep the rest of the tree unchanged. Do NOT change a feature and a window in the same edit.
2. First understand the semantic meaning of f in quantitative finance (momentum, reversal, volatility, volume confirmation, residual noise, etc.). Then map each economic hypothesis to one concrete EditAction. This is Semantic Proposal + Intervention Proposal.
3. You can only use the features, constants and operators given above in the (FEATURES AND OPERATORS DEFINITIONS). DO NOT MODIFY the name of any features or operators. All operators must be used with the same arity as defined.
4. As for constants: rolling windows MUST be integers from WINDOWS = {windows}; arithmetic constants MUST be floats from CONSTANTS = {constants}. Do NOT use other constants. Do NOT use scientific notation.
5. Operator replacement MUST stay in the same category: UNARY only to UNARY, BINARY only to BINARY, ROLLING only to ROLLING, PAIR_ROLLING only to PAIR_ROLLING. The per-site allowed sets already enforce this — pick new_value from those sets.
6. Five kinds and how to use them:
   6.1 feature_replace: swap the input at a feature site. Tests whether alpha comes from price vs volume, close vs high, etc. new_value ∈ that site's allowed FEATURE set.
   6.2 operator_replace: swap an operator for a same-arity cousin. Tests mean vs extreme, mean vs volatility, Add vs Sub (sign flip), etc. new_value ∈ that site's allowed operator set. That set is T-filtered: only operators for which apply_edit keeps a featured tree. Greater/Less need BOTH children featured, so they are omitted when a child is a constant (e.g. Pow($low,0.5) cannot become Greater).
   6.3 window_replace: change only the time scale; the mechanism stays. new_value is a window integer written as a string, e.g. "10". new_value ∈ that site's allowed WINDOWS.
   6.4 wrap: put ONE operator layer around the subtree at site_id (only when that site lists wrap). Inverse of unwrap-delete. new_value is a template with exactly one $_ (stands for this site's subtree). Write Abs($_) / Log($_) / Rank($_), not Abs(the subtree) or Log($vwap). Other examples: TsMean($_,20) ; Div($_,1.0) ; Div(1.0,$_) ; Add($_,$volume) ; TsCorr($_,$volume,10). For Binary / PairRolling the other operand MUST be a FEATURE or CONSTANT leaf. Constants need a decimal point (1.0 not 1). Greater($_,1.0) is illegal. Root wrap is legal when listed: TsMean($close,20) + wrap site 0 with Div($_,1.0) → Div(TsMean($close,20),1.0).
   6.5 subtree_delete: remove the subtree at site_id and keep a valid EXPR. new_value MUST be null. Rules: (a) site_id 0 (root) is illegal — you cannot delete the whole factor; (b) if the site is a child of Binary / PairRolling, replace the parent operator with the sibling branch (prune this child); (c) if the site is a Unary / Rolling operator that is NOT the root, unwrap — replace that op with its operand (inverse of wrap); (d) deleting a Feature / Constant that is the only child of Unary / Rolling is illegal (would empty the parent) — delete the parent op site instead, do not auto-unwrap a different site.
7. Give {k} edits with DIFFERENT modification strategies, sound and explainable. They should look semantically different, or use totally different ways to probe the same factor (for example: cross-sectional Rank vs time-series Ts op; different statistical measures; different but related semantic meanings such as dropping volume confirmation vs flipping momentum to reversal vs changing the horizon). Do NOT emit {k} synonymous window tweaks (10 then 20 then 30).
8. Each hypothesis MUST be a falsifiable causal sentence, e.g. "If the signal is short-horizon reversal rather than a 20-day trend, shortening the window should raise |IC|", not "try another operator".
9. If Observed Delta traces are not empty, they are real backtest outcomes from earliest to latest, NOT numbers you invented. Each line is one complete intervention: parent f, the edit (kind / site_id / new_value / hypothesis), child f', Δ (= r − r_seed), plus observed ic / rank_ic / r. site_id is relative to that line's f, which may be an ancestor of the current factor. Learn from them: do not repeat a kind/site/value on the same parent that already produced Δ≤0; a direction with Δ>0 may be continued with a different legal value under the same hypothesis.
10. After you generate each edit, check whether it is valid. Some INVALID operations are:
   10.1 Modify the name of operators or features. Eg: using "*" instead of "Mul", using "closing_price" instead of "$close", using "TSMAD" instead of "TsMad".
   10.2 The number of operands is not correct. Eg: Add(x), Div(x,y,z), missing rolling window in TsSkew / TsMean, etc.
   10.3 Using a window not in WINDOWS, or a constant not in CONSTANTS, or scientific notation (1e-4). Transfer into a listed float, e.g. 0.01.
   10.4 Using operators or features not listed in FEATURES AND OPERATORS DEFINITIONS.
   10.5 Changing two things at once, or picking a site_id that is not in the editable-site list.
   10.6 operator_replace across categories (TsMean → Add, Rank → TsMax).
   10.7 subtree_delete with new_value other than null; wrap / feature_replace / operator_replace / window_replace with null.
   10.8 wrap template missing $_, using two holes, wrapping more than one layer, other operand not a leaf, constant written as 1 instead of 1.0, an unknown operator, or a wrap that leaves the tree unfeatured (Greater/Less around a constant).
   10.9 subtree_delete at site_id 0, or at a Feature / Constant under Unary / Rolling.
   10.10 new_value not in that site's allowed set for typed replaces (the ∈ set printed for the site). Those sets are already filtered by T: only values for which apply_edit succeeds.
   10.11 operator_replace that leaves the tree with no market feature (Greater/Less require two featured children; Pow($low,0.5)→Greater is illegal).
   All the mentioned above in 10 are INVALID. Try your best to AVOID them.
11. You do NOT compute IC, RankIC, or Δ. R(f) below is given only as context for how strong the current factor is.

Given the current factor, you should ONLY and STRICTLY output a JSON object which contains the following contents:
{{
  "edits": [
    {{
      "hypothesis": "falsifiable causal sentence, string",
      "kind": "feature_replace|operator_replace|window_replace|subtree_delete|wrap",
      "site_id": 0,
      "new_value": "string, or null when kind is subtree_delete"
    }}
  ]
}}
The length of edits MUST be {k}. Each element MUST contain hypothesis, kind, site_id, new_value. site_id MUST be an integer that appears in the editable-site list. new_value MUST fall in that site's allowed set.

The example below is ONLY schema illustration for f = TsMean($close,20) where site_id=0 is TsMean and site_id=1 is $close. This tree has NO legal subtree_delete (root cannot be deleted; $close is the only child of a rolling op). For Add($close,$open), subtree_delete at site_id=2 with null yields $close. The example has 5 edits; you must still output exactly {k} edits for the CURRENT f, re-numbering site_id from the site list above, not by copying this example unless the tree is identical:
{{"edits":[
  {{"hypothesis":"If the effective signal is a shorter-horizon price trend rather than a 20-day mean, shortening the window should raise |IC|","kind":"window_replace","site_id":0,"new_value":"10"}},
  {{"hypothesis":"Close may only proxy liquidity; swapping in volume tests whether the alpha is price-based or flow-based","kind":"feature_replace","site_id":1,"new_value":"$volume"}},
  {{"hypothesis":"Replacing the mean with an extreme tests whether alpha comes from tails rather than average drift","kind":"operator_replace","site_id":0,"new_value":"TsMax"}},
  {{"hypothesis":"Normalizing the rolling mean by 1.0 tests a no-op scale wrap that later edits can turn into a real denominator","kind":"wrap","site_id":0,"new_value":"Div($_,1.0)"}},
  {{"hypothesis":"Cross-sectional ranking inside the time-series mean tests whether relative strength, not raw level, is the source","kind":"wrap","site_id":1,"new_value":"Rank($_)"}}
]}}

Current factor:
f = {expr}
R(f) = |daily-mean Pearson IC| = {r:.6f}

Legal vocabulary (typed replaces use the per-site ∈ sets; wrap uses this catalog plus the $_ template):
{vocab}
Wrap templates (exactly one $_):
Abs($_)  Log($_)  Rank($_)  TsMean($_,20)  Div($_,1.0)  Div(1.0,$_)  Add($_,$volume)  TsCorr($_,$volume,10)

Editable sites (site_id is preorder; new_value must be taken from that site's ∈ set):
{sites}

Observed Delta traces (real backtests, not computed by you):
{recent}

Do not output anything else other than the JSON object, including code block markers like ``` or ```json.
Now output a JSON object whose edits list has length exactly {k}.
"""


# expr, R(f), k -> legal EditActions from the LLM (illegal items printed and dropped)
def propose(expr, r, k=K, recent=None) -> List[EditAction]:
    cfg = _dotenv()
    sites = enumerate_sites(expr)
    print(f"[llm] propose k={k}", flush=True)
    data = _chat(cfg, _messages(expr, r, k, sites, recent))
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    thinking = cfg.get("LLM_THINKING") or "adaptive"
    text = _answer_content(msg)
    print(
        f"[llm] finish_reason={choice.get('finish_reason')}  "
        f"thinking={thinking}  reasoning_chars={_reasoning_chars(msg)}",
        flush=True,
    )
    if not text:
        print(
            "[llm] empty content after thinking split "
            f"(finish_reason={choice.get('finish_reason')}); returning empty",
            flush=True,
        )
        return []
    print("[llm] raw output >>>", flush=True)
    print(text, flush=True)
    print("[llm] raw output <<<", flush=True)
    actions = _parse(text, expr)
    if len(actions) != k:
        print(f"[llm] expected {k} edits, got {len(actions)}; keeping parsed items", flush=True)
    legal: List[EditAction] = []
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


# repo .env -> LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (+ optional knobs)
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


# operator class -> comma-separated names in that category
def _ops(cls) -> str:
    return ", ".join(op.__name__ for op in OPERATORS if op.category_type() is cls)


# FEATURES / WINDOWS / CONSTANTS / ops by category, for the prompt
def _vocab() -> str:
    return "\n".join([
        "FEATURES: " + ", ".join(FEATURE_NAMES),
        "WINDOWS: " + ", ".join(map(str, DELTA_TIMES)),
        "CONSTANTS: " + ", ".join(map(str, CONSTANTS)),
        "UNARY_OPS: " + _ops(UnaryOperator),
        "BINARY_OPS: " + _ops(BinaryOperator),
        "ROLLING_OPS: " + _ops(RollingOperator),
        "PAIR_ROLLING_OPS: " + _ops(PairRollingOperator),
    ])


# Site list -> per-site allowed-set text for the prompt
def _format_sites(sites: List[Site]) -> str:
    blocks = []
    for s in sites:
        lines = [f"[site_id={s.site_id}] type={s.kind}  subtree={s.expr_str}  current={s.current}"]
        if s.kind == "feature":
            lines.append("  allowed: feature_replace, new_value ∈ {" + ", ".join(s.allowed) + "}")
        elif s.kind == "operator":
            lines.append("  allowed: operator_replace, new_value ∈ {" + ", ".join(s.allowed) + "}")
            if s.allowed_windows:
                lines.append("  allowed: window_replace, new_value ∈ {" + ", ".join(map(str, s.allowed_windows)) + "}")
        if s.can_delete:
            lines.append("  allowed: subtree_delete, new_value must be null")
        if s.can_wrap:
            lines.append("  allowed: wrap, new_value e.g. Abs($_) / Log($_) / TsMean($_,20) / Div($_,1.0)")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


# keep traces from the last n search rounds; no-op if history has <= n rounds
def _last_n_rounds(recent: Sequence[CFRecord], n: int = 3) -> Sequence[CFRecord]:
    seen: List[int] = []
    for rec in recent:
        rnd = (rec.extra or {}).get("round")
        if isinstance(rnd, int) and rnd not in seen:
            seen.append(rnd)
    if len(seen) <= n:
        return recent
    keep = set(seen[-n:])
    return [rec for rec in recent if (rec.extra or {}).get("round") in keep]


# CFRecord list -> Observed Delta traces block
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
        parts.extend([
            f"Δ={rec.delta:+.4f}",
            f"ic={rec.ic:.4f}",
            f"rank_ic={rec.rank_ic:.4f}",
            f"r={rec.r:.6f}",
            f"r_seed={rec.r_seed:.6f}",
        ])
        extra = rec.extra or {}
        for key in ("round", "seed"):
            if key in extra:
                parts.append(f"{key}={extra[key]}")
        lines.append("- " + "  ".join(parts))
    return "\n".join(lines)


# expr, r, k, sites, traces -> OpenAI-style system+user messages
def _messages(expr: Expression, r: float, k: int, sites: List[Site], recent: Optional[Sequence[CFRecord]]):
    recent_txt = (
        _format_recent(_last_n_rounds(recent))
        if recent
        else "(empty. This is the first intervention batch for this factor.)"
    )

    user = PROMPT_FEATURES_AND_OPERATORS + PROMPT_TASK.format(
        k=k,
        windows=", ".join(map(str, DELTA_TIMES)),
        constants=", ".join(map(str, CONSTANTS)),
        expr=expr,
        r=r,
        vocab=_vocab(),
        sites=_format_sites(sites),
        recent=recent_txt,
    )
    return [
        {"role": "system", "content": PROMPT_HEAD.strip()},
        {"role": "user", "content": user.strip()},
    ]


# OpenAI-compatible chat; MiniMax extras go in extra_body
def _chat(cfg: Dict[str, str], messages) -> dict:
    client = OpenAI(
        api_key=cfg["LLM_API_KEY"],
        base_url=cfg["LLM_BASE_URL"].rstrip("/"),
        timeout=float(cfg["LLM_TIMEOUT"]) if cfg.get("LLM_TIMEOUT") else 600.0,
    )
    thinking = cfg.get("LLM_THINKING") or "adaptive"
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
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        raise LLMError(f"LLM request failed: {type(e).__name__}: {e}") from e
    return resp.model_dump()


# message dict -> final answer only (CoT lives in reasoning_* after split)
def _answer_content(msg: dict) -> str:
    raw = msg.get("content")
    if isinstance(raw, list):
        parts = []
        for p in raw:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or "")
        raw = "".join(parts)
    text = (raw or "").strip()
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


# message dict -> length of split-off reasoning fields
def _reasoning_chars(msg: dict) -> int:
    n = len(msg.get("reasoning_content") or "")
    details = msg.get("reasoning_details") or []
    if isinstance(details, list):
        for d in details:
            if isinstance(d, dict):
                n += len(d.get("text") or "")
            elif isinstance(d, str):
                n += len(d)
    return n


# True iff obj is the {"edits": [...]} payload
def _has_edits(obj) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("edits"), list)


# broken wrapper -> edit dicts from the "edits" array, skipping extra/missing braces
def _extract_edits_array(text: str) -> Optional[List[dict]]:
    key = text.find('"edits"')
    if key < 0:
        return None
    i = text.find("[", key)
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


# model text -> {"edits": [...]} (raw, fenced, or salvaged array items)
def _json_obj(text: str) -> dict:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    try:
        obj = json.loads(text)
        if _has_edits(obj):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if _has_edits(obj):
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
        if _has_edits(obj):
            return obj
    salvaged = _extract_edits_array(text)
    if salvaged is not None:
        print(f"[llm] wrapper JSON invalid; salvaged {len(salvaged)} edit objects", flush=True)
        return {"edits": salvaged}
    raise LLMError("response is not JSON")


# JSON text -> EditAction list; bad items printed and dropped, not raised
def _parse(text: str, expr=None) -> List[EditAction]:
    try:
        obj = _json_obj(text)
    except LLMError as e:
        print(f"[llm] parse failed: {e}; returning empty", flush=True)
        return []
    if not _has_edits(obj):
        print("[llm] JSON missing edits list; returning empty", flush=True)
        return []
    out = []
    for i, item in enumerate(obj["edits"]):
        if not isinstance(item, dict):
            print(f"[llm] ILLEGAL skip  edits[{i}] not an object: {item!r}", flush=True)
            continue
        kind = item.get("kind")
        if kind not in EDIT_KINDS:
            dummy = EditAction(str(kind), int(item["site_id"]) if "site_id" in item else -1, item.get("new_value"), item.get("hypothesis"))
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


if __name__ == "__main__":
    from alpha_cf.pool import parse_expr

    f = parse_expr(sys.argv[1] if len(sys.argv) > 1 else "TsMean($close,20)")
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    actions = propose(f, 0.06, k)
    print(f"[llm] standalone got {len(actions)} legal edits")
    for a in actions:
        print(f"T(f,a)  {a.kind}@{a.site_id} {a.new_value!r}  ->  {apply_edit(f, a)}")
