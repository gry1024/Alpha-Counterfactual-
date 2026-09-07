import json
from typing import List, Optional

from alphagen.data.expression import Expression
from alphagen.data.tree import ExpressionParser

_PARSER = ExpressionParser()


# AlphaSAGE string -> Expression tree
def parse_expr(expr_str: str) -> Expression:
    return _PARSER.parse(expr_str)


# pool JSON path -> unique Expression list, optionally truncated
def load_pool(path: str, max_n: Optional[int] = None) -> List[Expression]:
    with open(path, "r") as f:
        raw = json.load(f)
    strings: List[str] = raw["exprs"] if isinstance(raw, dict) else raw  # {"exprs": [...]} or a bare list

    out: List[Expression] = []
    seen = set()
    for s in strings:
        expr = parse_expr(s)
        key = str(expr)
        if key in seen:
            continue  # skip duplicate canonical strings
        seen.add(key)
        out.append(expr)
        if max_n is not None and len(out) >= max_n:
            break
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "data/gfn_logs/pool_50/"
        "gfn_gnn_sp500_50_0-0.01-1.0-1.0-1.0-0.3-linear-0.0/pool_9999.json"
    )
    exprs = load_pool(path, max_n=None)
    print(f"loaded {len(exprs)} unique seeds from {path}")
    for e in exprs:
        print(" ", e)
