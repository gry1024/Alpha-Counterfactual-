from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Editable tree node: kind, current token, legal replacements
@dataclass(frozen=True)
class Site:
    site_id: int
    kind: str  # feature | operator | constant
    current: str
    allowed: List[str]
    expr_str: str
    window: Optional[int] = None
    allowed_windows: Tuple[int, ...] = ()
    can_delete: bool = False
    can_wrap: bool = True


# One-site edit T(f, a); new_value is None for subtree_delete
@dataclass(frozen=True)
class EditAction:
    kind: str  # feature_replace | operator_replace | window_replace | subtree_delete | wrap
    site_id: int
    new_value: Optional[str] = None
    hypothesis: Optional[str] = None


# One backtested (f, a, f', Δ, IC) row
@dataclass
class CFRecord:
    f: str
    action: EditAction
    f_prime: str
    delta: float
    ic: float
    rank_ic: float
    r: float
    r_seed: float
    extra: Dict[str, Any] = field(default_factory=dict)

    # dataclass -> JSON-serializable dict
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
