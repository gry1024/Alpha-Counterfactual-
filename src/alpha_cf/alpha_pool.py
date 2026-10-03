"""Signed RankIC, initial pool selection and per-parent replacement."""
from copy import copy
import numpy as np
import torch
from alphagen.utils.correlation import batch_pearsonr
from .expression import complexity


def rank(x):
    # Average tied ranks without the stock-by-stock matrix in batch_spearmanr.
    finite = x.isfinite()
    values, order = x.masked_fill(~finite, torch.inf).sort(dim=1)
    positions = torch.arange(x.shape[1], device=x.device).expand_as(order)
    boundary = values[:, 1:] != values[:, :-1]
    edge = torch.ones_like(values[:, :1], dtype=torch.bool)
    first, last = torch.cat((edge, boundary), 1), torch.cat((boundary, edge), 1)
    starts = torch.where(first, positions, 0).cummax(1).values
    ends = torch.where(last, positions, x.shape[1] - 1).flip(1).cummin(1).values.flip(1)
    return torch.empty_like(x).scatter_(1, order, (starts + ends).to(x.dtype) / 2).masked_fill(~finite, torch.nan)


def spearman(x, y):
    mask = x.isfinite() & y.isfinite()
    x, y = rank(x.masked_fill(~mask, torch.nan)), rank(y.masked_fill(~mask, torch.nan))
    valid = (mask.sum(1) >= 2) & (x.nan_to_num().sum(1) > 0) & (y.nan_to_num().sum(1) > 0)
    for value in (x, y):
        valid &= value.nan_to_num(nan=-torch.inf).amax(1) > value.nan_to_num(nan=torch.inf).amin(1)
    return batch_pearsonr(x, y).clamp(-1, 1).masked_fill(~valid, torch.nan)


def _ols_weighted_signal(signals, target, nan_to_num=0.0):
    """Compute the in-sample OLS-weighted signal matching `run_adaptive_combination.py`.

    Args:
        signals: [N, T, S] per-factor cross-sectional signals.
        target:  [T, S] forward returns (same shape as one signal).
        nan_to_num: fill NaN with this value before regression (matches test side).

    Returns:
        Tensor [T, S] the OLS-weighted combined signal. Falls back to equal-weight
        mean if regression fails or the factor set is degenerate.
    """
    n = signals.shape[0]
    if n == 0:
        return None
    t, s = target.shape
    if n == 1:
        return signals[0]
    # Reshape to [T*S, N] samples x factors, matching test-side convention.
    x = signals.permute(1, 2, 0).reshape(-1, n)
    y = target.reshape(-1)
    valid_mask = torch.isfinite(y) & torch.isfinite(x).all(dim=1)
    if valid_mask.sum() < n + 1:
        return signals.mean(0)
    x_v = x[valid_mask]
    y_v = y[valid_mask].unsqueeze(1)
    try:
        coef = torch.linalg.lstsq(x_v, y_v, rcond=1e-15).solution  # [N, 1]
        # Drop intercept-like behaviour: clip non-negative weights (factors are
        # already normalised; OLS can produce arbitrary signs). Keep signs.
        coef = coef.squeeze(1)
        weighted = (signals * coef.view(n, 1, 1)).sum(0)
        return torch.nan_to_num(weighted, nan=nan_to_num)
    except Exception:
        return signals.mean(0)


class AlphaCFPool:
    def __init__(self, data, target, args):
        self.data, self.target, self.args = data, target, args
        self.exprs, self.cache = [], {}

    @torch.no_grad()
    def evaluate(self, expr):
        key = str(expr)
        if key not in self.cache:
            values = []
            for start in range(0, len(self.target), self.args.chunk_size):
                block = copy(self.data)
                block.data = self.data.data[start:self.data.max_backtrack_days + min(start + self.args.chunk_size, len(self.target))]
                value = expr.evaluate(block)
                values.append(value.masked_fill(~value.isfinite(), torch.nan))
            value = torch.cat(values)
            count = value.isfinite().sum(1, keepdim=True)
            signal = (rank(value) / (count - 1).clamp_min(1) - 0.5).masked_fill(count < 2, torch.nan)
            daily = spearman(signal, self.target)
            reward = daily.nanmean().item()
            cost = (1 - spearman(signal[1:], signal[:-1]).nanmean().item()) / 2
            if not np.isfinite([reward, cost]).all():
                raise ValueError("Factor has no usable cross-sectional variation")
            self.cache[key] = dict(signal=signal, reward=reward, ic=daily.nanmean().item(), cost=cost)
        return self.cache[key]

    def signal(self, expr):
        # Missing exposures receive neutral rank, retaining fixed equal weights.
        return self.evaluate(expr)["signal"].nan_to_num().double()

    def score_signals(self, signals):
        shape = signals.shape
        target = self.target.expand_as(signals).flatten(0, 1)
        daily = spearman(signals.flatten(0, 1), target).reshape(shape[:2])
        return daily.nanmean(1)

    def utility(self, exprs=None):
        exprs = self.exprs if exprs is None else exprs
        if not exprs:
            return 0.0
        signals = torch.stack([self.signal(expr) for expr in exprs])  # [N, T, S]
        combined = _ols_weighted_signal(signals, self.target)
        if combined is None:
            return 0.0
        return self.score_signals(combined.unsqueeze(0))[0].item()

    @torch.no_grad()
    def select(self, candidates, size):
        remaining = list({str(e): e for e in candidates}.values())
        for expr in remaining:
            self.evaluate(expr)
        if len(remaining) < size:
            raise ValueError(f"Need {size} usable unique factors, got {len(remaining)}")
        if self.args.no_pool_selection:
            return sorted(remaining, key=lambda e: abs(self.evaluate(e)["reward"]), reverse=True)[:size]
        selected, correlations = [], {str(e): 0.0 for e in remaining}
        weights = np.array([self.args.alpha, self.args.beta, self.args.gamma, self.args.cost_weight])
        while len(selected) < size:
            utilities = []
            for start in range(0, len(remaining), 4):
                batch = remaining[start:start + 4]
                candidate_signals = torch.stack([self.signal(e) for e in batch])
                # OLS-weighted utility on (selected + candidate). Falls back to mean
                # if lstsq fails; matches run_adaptive_combination semantics.
                if selected:
                    pool_signals = torch.stack([self.signal(e) for e in selected])
                    stacked = torch.cat([pool_signals, candidate_signals], dim=0)
                else:
                    stacked = candidate_signals
                combined = _ols_weighted_signal(stacked, self.target)
                utilities.extend(self.score_signals(combined.unsqueeze(0)).tolist())
            pool_utility = self.utility(selected) if selected else 0.0
            # Columns: |R|, C_pool = U(P∪{e})-U(P), D, 1-C_cost (all larger-is-better).
            rows = [[abs(self.evaluate(e)["reward"]), u - pool_utility,
                     1 - correlations[str(e)], 1 - self.evaluate(e)["cost"]]
                    for e, u in zip(remaining, utilities)]
            scores = np.array(rows) @ weights
            scores[~np.isfinite(utilities)] = -np.inf
            if not np.isfinite(scores).any():
                raise ValueError("No valid pool extension")
            index = int(scores.argmax())
            winner = remaining.pop(index)
            selected.append(winner)
            for start in range(0, len(remaining), 4):
                batch = remaining[start:start + 4]
                signals = torch.stack([self.evaluate(e)["signal"] for e in batch])
                rhs = self.evaluate(winner)["signal"].expand_as(signals)
                corr = spearman(signals.flatten(0, 1), rhs.flatten(0, 1)).reshape(signals.shape[:2]).abs().nanmean(1)
                for expr, value in zip(batch, corr.tolist()):
                    correlations[str(expr)] = max(correlations[str(expr)], value if np.isfinite(value) else 1.0)
        return selected

    def initialize(self, candidates):
        self.exprs = self.select(candidates, self.args.pool_capacity)
        self.keep(self.exprs)

    def correlation(self, left, right, absolute=False):
        daily = spearman(self.evaluate(left)["signal"], self.evaluate(right)["signal"])
        return (daily.abs() if absolute else daily).nanmean().item()

    def replace_parent(self, parent, offspring):
        index = next(i for i, e in enumerate(self.exprs) if str(e) == str(parent))
        peers = self.exprs[:index] + self.exprs[index + 1:]
        existing = {str(e) for e in self.exprs}
        parent_nodes = complexity(parent)
        pool_utility = self.utility(self.exprs)

        children = [c for c in {str(c): c for c in offspring}.values() if str(c) not in existing]
        for child in children:
            self.evaluate(child)

        # Diversity and C_pool are always measured against the peers (the pool excluding
        # this parent), so S(f|P) never contains f itself.
        def diversity(expr):
            if not peers:
                return 1.0
            corr = max((c if np.isfinite(c) else 1.0
                        for c in (self.correlation(expr, peer, absolute=True) for peer in peers)),
                       default=0.0)
            return 1.0 - corr

        def c_pool(expr):
            # C_pool = U(P_{-f} ∪ {e}) - U(P); larger means the swap improves the pool.
            value = self.utility(peers + [expr]) - pool_utility
            return value if np.isfinite(value) else 0.0

        distance = {}
        for child in children:
            corr = self.correlation(parent, child)
            distance[str(child)] = 1.0 - corr if np.isfinite(corr) else None

        # Case 1: an equivalent child (signal_distance ~ 0) with fewer nodes wins directly.
        equivalent = [child for child in children
                      if distance[str(child)] is not None and distance[str(child)] <= 1e-6
                      and complexity(child) < parent_nodes]
        equivalent.sort(key=complexity)

        weights = np.array([self.args.alpha, self.args.beta, self.args.gamma, self.args.cost_weight])
        members = [parent] + children
        # Natural-scale weighted score (no minmax); weights are chosen so a
        # "typical" factor contributes ~1.0 from each axis.
        rows = [[abs(self.evaluate(e)["reward"]), c_pool(e), diversity(e),
                 1 - self.evaluate(e)["cost"]]
                for e in members]
        scores = np.array(rows) @ weights
        parent_score = scores[0]

        candidates, winner = [], None
        if equivalent:
            winner = equivalent[0]
        elif children:
            # Case 2: highest-scoring child strictly above the parent replaces it.
            best = int(np.argmax(scores[1:])) + 1
            if scores[best] > parent_score:
                winner = members[best]

        for child, score in zip(children, scores[1:]):
            candidates.append(dict(expression=str(child), reward=self.evaluate(child)["reward"],
                                   complexity=complexity(child), max_correlation=1.0 - diversity(child),
                                   signal_distance=distance[str(child)], score=float(score),
                                   eligible=child in equivalent or float(score) > float(parent_score)))
        if winner is not None:
            self.exprs[index] = winner
        return dict(parent=str(parent), parent_reward=self.evaluate(parent)["reward"],
                    parent_complexity=parent_nodes,
                    child=str(winner) if winner is not None else None,
                    outcome="replaced" if winner is not None else "retained", candidates=candidates)

    def keep(self, exprs):
        keys = {str(e) for e in exprs}
        self.cache = {k: v for k, v in self.cache.items() if k in keys}

    def to_dict(self):
        # Store actual OLS weights from the same regression the pool utility uses,
        # so the saved JSON reflects the in-sample combination that produced U(P).
        n = len(self.exprs)
        if n == 0:
            weights = []
        else:
            signals = torch.stack([self.signal(e) for e in self.exprs])  # [N, T, S]
            x = signals.permute(1, 2, 0).reshape(-1, n)
            y = self.target.reshape(-1)
            valid_mask = torch.isfinite(y) & torch.isfinite(x).all(dim=1)
            try:
                coef = torch.linalg.lstsq(x[valid_mask], y[valid_mask].unsqueeze(1),
                                          rcond=1e-15).solution.squeeze(1).tolist()
            except Exception:
                coef = [1.0 / n] * n
            weights = [float(c) for c in coef]
        return dict(exprs=[str(e) for e in self.exprs], weights=weights,
                    metrics=[{k: v for k, v in self.evaluate(e).items() if k != "signal"} for e in self.exprs],
                    utility=self.utility())
