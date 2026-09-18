from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Editable tree node: kind, current token, legal replacements
#
# Produced by edits.enumerate_sites / edits._site_of and fed into the LLM prompt
# so it knows which site_ids exist and what each site's ∈ set looks like.
# `allowed_windows` only populated for rolling/pair-rolling operator sites;
# `can_delete` is False for sites where subtree_delete would yield an
# unfeatured (non-evaluable) expression.
@dataclass(frozen=True)
class Site:
    site_id: int
    kind: str  # feature | operator | constant
    current: str
    allowed: List[str]
    expr_str: str
    window: Optional[int] = None          # set only for rolling op sites
    allowed_windows: Tuple[int, ...] = ()  # other DELTA_TIMES choices
    can_delete: bool = False


# One-site edit T(f, a); new_value is None for subtree_delete
#
# kind ∈ {"feature_replace", "operator_replace", "window_replace", "subtree_delete"}.
# For "subtree_delete", new_value MUST be None and edits.apply_edit ignores it.
# For the other three, new_value must come from the site's ∈ set — apply_edit
# will return None otherwise (logged as illegal by llm.record_illegal).
# `hypothesis` is the LLM's falsifiable causal sentence explaining WHY this edit
# is worth testing; recorded in phaseA_records.jsonl for downstream debugging.
@dataclass(frozen=True)
class EditAction:
    kind: str  # feature_replace | operator_replace | window_replace | subtree_delete
    site_id: int
    new_value: Optional[str] = None
    hypothesis: Optional[str] = None


# One backtested (f, a, f', Δ, IC, inc) row.
#
# Built per edit by phaseA._process_one_seed and rehydrated into CFRecord
# objects when feeding back into llm.propose() as `recent`. The ic / r /
# rank_ic triple is redundant on purpose: r == ic (raw Pearson), and rank_ic
# is currently filled from ic_fp too (ric dropped from records to keep
# records.jsonl lean — see phaseA record dict).
@dataclass
class CFRecord:
    f: str                       # seed factor string
    action: EditAction           # the edit that produced f'
    f_prime: str                 # edited factor string
    delta: float                 # ic(f') - ic(f)
    ic: float                    # Pearson IC of f' on train
    rank_ic: float               # currently mirrors ic (ric dropped)
    r: float                     # same as ic (raw Pearson IC, not |IC|)
    r_seed: float                # seed IC (ic(f))
    inc: float = 0.0             # Pearson IC of residual(f'|f) on train — KEY signal for redundancy
    extra: Dict[str, Any] = field(default_factory=dict)  # round index, etc.

    # dataclass -> JSON-serializable dict (used by llm._format_recent etc.)
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
