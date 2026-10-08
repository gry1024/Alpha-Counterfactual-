"""Run the fixed Qlib Alpha158 feature set through the shared historical OLS backtest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from qlib.contrib.data.loader import Alpha158DL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", default="csi300")
    parser.add_argument("--output-dir")
    for name, default in dict(train_end_year=2021, label_days=20, n_factors=20,
                              chunk_size=64, cuda=0, seed=0).items():
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir) if args.output_dir else root / "data/alpha158_logs" / args.instruments
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields, names = Alpha158DL.get_feature_config()
    manifest = output / "alpha158.json"
    manifest.write_text(json.dumps(dict(feature_set="Alpha158", names=names, fields=fields,
        args=vars(args), combination="historical OLS; fixed feature set; no factor search"), indent=2), encoding="utf-8")
    command = [sys.executable, str(root / "run_adaptive_combination.py"),
               "--expressions_file", str(manifest), "--metrics_file", str(output / "metrics.json"),
               "--instruments", args.instruments]
    for name in ("train_end_year", "label_days", "n_factors", "chunk_size", "cuda", "seed"):
        command.extend(["--" + name, str(getattr(args, name))])
    # Omit --use_weights: argparse's existing type=bool would treat "False" as True.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")]))
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
