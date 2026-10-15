"""Export GP cache candidates for run_adaptive_combination.py."""
import argparse
import json
import math
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--top', type=int, default=50)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be positive')
    cache = json.loads(args.source.read_text(encoding='utf-8'))['cache']
    ranked = sorted(
        ((formula, float(score)) for formula, score in cache.items()
         if math.isfinite(float(score)) and float(score) > -1),
        key=lambda item: item[1], reverse=True,
    )
    exprs, scores, seen = [], [], set()
    for formula, score in ranked:
        formula = re.sub(r'\b(open_|close|high|low|volume|vwap)\b',
                         lambda m: '$' + m[0].rstrip('_'), formula)
        if formula in seen:
            continue
        seen.add(formula)
        exprs.append(formula)
        scores.append(score)
        if len(exprs) == args.top:
            break
    if not exprs:
        raise ValueError('No finite candidates with train IC > -1')
    output = args.output or args.source.with_name(f'pool_{args.top}.json')
    output.write_text(json.dumps({
        'exprs': exprs, 'train_ics': scores,
        'source': args.source.name, 'selection': 'descending_signed_train_ic',
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(exprs)} factors to {output}')


if __name__ == '__main__':
    main()