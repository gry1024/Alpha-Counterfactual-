"""Convert QCM validation-selected CSV tables to expression pools."""
import argparse
import csv
import json
import math
from pathlib import Path


def convert(source):
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    factors = [row for row in rows if row['exprs'].strip() != 'Ensemble']
    if not factors:
        raise ValueError(f'No factors in {source}')
    weights = [float(row['weight']) for row in factors]
    ics = [float(row['ic']) for row in factors]
    if not all(math.isfinite(v) for v in weights + ics):
        raise ValueError(f'Non-finite weights or ICs in {source}')
    pool = {
        'exprs': [row['exprs'].strip() for row in factors],
        'weights': weights,
        'train_ics': ics,
        'source': source.name,
        'selection': 'best_validation_ic',
    }
    ensemble = [row for row in rows if row['exprs'].strip() == 'Ensemble']
    if ensemble:
        pool['validation_ic'] = float(ensemble[-1]['ic'])
        pool['test_ic'] = float(ensemble[-1]['weight'])
    output = source.with_name('pool.json')
    output.write_text(json.dumps(pool, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Exported {len(factors)} factors to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='CSV file or directory of validation tables')
    args = parser.parse_args()
    sources = sorted(args.source.rglob('valid_best_table.csv')) if args.source.is_dir() else [args.source]
    if not sources:
        raise FileNotFoundError(f'No valid_best_table.csv under {args.source}')
    for source in sources:
        convert(source)


if __name__ == '__main__':
    main()