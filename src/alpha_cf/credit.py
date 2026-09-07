import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagen.data.expression import Feature

from alpha_cf.edits import enumerate_sites, parse_wrap_template
from alpha_cf.pool import parse_expr
from alpha_cf.types import CFRecord, EditAction, Site


# jsonl of CFRecord.to_dict rows -> CFRecord list
def records_from_jsonl(path: str) -> List[CFRecord]:
    out: List[CFRecord] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(_record_from_dict(json.loads(line)))
    return out


# CFRecords -> RQ1 report (skip empty groupings)
def attribute(records: Sequence[CFRecord]) -> Dict[str, Any]:
    by_kind: Dict[str, List[float]] = defaultdict(list)
    by_new_feature: Dict[str, List[float]] = defaultdict(list)
    by_new_operator: Dict[str, List[float]] = defaultdict(list)
    by_new_window: Dict[str, List[float]] = defaultdict(list)
    by_old: Dict[str, List[float]] = defaultdict(list)
    by_wrap_side: Dict[str, List[float]] = defaultdict(list)
    site_cache: Dict[str, Optional[List[Site]]] = {}

    for rec in records:
        a = rec.action
        d = rec.delta
        by_kind[a.kind].append(d)

        if a.kind == "feature_replace" and a.new_value:
            by_new_feature[a.new_value].append(d)
        elif a.kind == "operator_replace" and a.new_value:
            by_new_operator[a.new_value].append(d)
        elif a.kind == "window_replace" and a.new_value is not None:
            wk = _window_key(a.new_value)
            if wk is not None:
                by_new_window[wk].append(d)
        elif a.kind == "wrap":
            spec = parse_wrap_template(a.new_value or "")
            if spec is not None:
                by_new_operator[spec.op_name].append(d)
                by_wrap_side[spec.side].append(d)
                if spec.window is not None:
                    by_new_window[str(int(spec.window))].append(d)
                if spec.other is not None and isinstance(spec.other, Feature):
                    by_new_feature[str(spec.other)].append(d)

        site = _site_of(rec.f, a.site_id, site_cache)
        if site is not None:
            by_old[_old_token(site, a)].append(d)

    report: Dict[str, Any] = {"n_records": len(records)}
    for name, buckets in (
        ("by_kind", by_kind),
        ("by_new_feature", by_new_feature),
        ("by_new_operator", by_new_operator),
        ("by_new_window", by_new_window),
        ("by_old", by_old),
        ("by_wrap_side", by_wrap_side),
    ):
        grouped = _finalize(buckets)
        if grouped:
            report[name] = grouped
    return report


# CFRecords, dest path -> write credit.json and return the report
def write_credit(records: Sequence[CFRecord], path: str) -> Dict[str, Any]:
    report = attribute(records)
    with open(path, "w") as w:
        json.dump(report, w, indent=2, ensure_ascii=False)
        w.write("\n")
    return report


# json object -> CFRecord (nested action dict from asdict)
def _record_from_dict(d: Mapping[str, Any]) -> CFRecord:
    a = d["action"]
    return CFRecord(
        f=d["f"],
        action=EditAction(
            kind=a["kind"],
            site_id=a["site_id"],
            new_value=a.get("new_value"),
            hypothesis=a.get("hypothesis"),
        ),
        f_prime=d["f_prime"],
        delta=float(d["delta"]),
        ic=float(d["ic"]),
        rank_ic=float(d["rank_ic"]),
        r=float(d["r"]),
        r_seed=float(d["r_seed"]),
        extra=dict(d["extra"]) if d.get("extra") else {},
    )


# deltas -> n / mean_delta / median_delta / frac_positive
def _stats(deltas: Sequence[float]) -> Dict[str, Any]:
    n = len(deltas)
    return {
        "n": n,
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "frac_positive": sum(1 for x in deltas if x > 0) / n,
    }


# key -> deltas -> skip-empty stats, sorted by mean Δ desc
def _finalize(buckets: Mapping[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    items = [(k, _stats(v)) for k, v in buckets.items() if v]
    items.sort(key=lambda kv: (-kv[1]["mean_delta"], kv[0]))
    return {k: v for k, v in items}


# f string, site_id -> Site at that preorder id, or None
def _site_of(
    f_str: str, site_id: int, cache: Dict[str, Optional[List[Site]]]
) -> Optional[Site]:
    if f_str not in cache:
        try:
            cache[f_str] = enumerate_sites(parse_expr(f_str))
        except Exception:
            cache[f_str] = None
    sites = cache[f_str]
    if sites is None or site_id < 0 or site_id >= len(sites):
        return None
    return sites[site_id]


# Site, action -> old token (feature / op class / window if rolling)
def _old_token(site: Site, action: EditAction) -> str:
    if action.kind == "window_replace" and site.window is not None:
        return str(int(site.window))
    return site.current


# window token -> canonical key, or None
def _window_key(value: Any) -> Optional[str]:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    recs = [
        CFRecord(
            "TsMean($close,20)",
            EditAction("wrap", 0, "Div($_,1.0)"),
            "Div(TsMean($close,20),1.0)",
            0.02, 0.12, 0.11, 0.12, 0.10,
        ),
        CFRecord(
            "TsMean($close,20)",
            EditAction("window_replace", 0, "10"),
            "TsMean($close,10)",
            -0.01, 0.09, 0.09, 0.09, 0.10,
        ),
        CFRecord(
            "Add($close,$open)",
            EditAction("subtree_delete", 1, None),
            "$open",
            0.005, 0.08, 0.08, 0.085, 0.08,
        ),
        CFRecord(
            "TsMean($close,20)",
            EditAction("feature_replace", 1, "$volume"),
            "TsMean($volume,20)",
            0.03, 0.13, 0.12, 0.13, 0.10,
        ),
        CFRecord(
            "TsMean($close,20)",
            EditAction("wrap", 0, "Add($volume,$_)"),
            "Add($volume,TsMean($close,20))",
            0.01, 0.11, 0.10, 0.11, 0.10,
        ),
    ]
    report = attribute(recs)
    assert report["n_records"] == 5
    for key in ("by_kind", "by_new_feature", "by_new_operator",
                "by_new_window", "by_old", "by_wrap_side"):
        assert key in report and report[key], key
    assert report["by_kind"]["wrap"]["n"] == 2
    assert report["by_kind"]["window_replace"]["n"] == 1
    assert report["by_kind"]["subtree_delete"]["n"] == 1
    assert report["by_new_operator"]["Div"]["n"] == 1
    assert report["by_new_window"]["10"]["mean_delta"] == -0.01
    assert report["by_old"]["TsMean"]["n"] == 2  # wrap Div + wrap Add
    assert report["by_old"]["20"]["n"] == 1
    assert report["by_old"]["$close"]["n"] == 2  # delete + feature_replace
    assert report["by_wrap_side"]["left"]["n"] == 1
    assert report["by_wrap_side"]["right"]["n"] == 1
    assert report["by_new_feature"]["$volume"]["n"] == 2
    assert report["by_kind"]["wrap"]["frac_positive"] == 1.0

    import os
    import tempfile

    td = tempfile.mkdtemp(prefix="cf_credit_")
    jsonl_path = os.path.join(td, "cf_records.jsonl")
    with open(jsonl_path, "w") as w:
        for rec in recs:
            w.write(json.dumps(rec.to_dict()) + "\n")
    loaded = records_from_jsonl(jsonl_path)
    assert len(loaded) == 5 and loaded[0].action.kind == "wrap"
    credit_path = os.path.join(td, "credit.json")
    write_credit(loaded, credit_path)
    with open(credit_path) as f:
        dumped = json.load(f)
    assert dumped["n_records"] == 5
    assert "by_kind" in dumped
    print(json.dumps(report, indent=2))
    print("ok", credit_path)
