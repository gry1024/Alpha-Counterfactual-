"""Signed RankIC, direct seed loading and per-parent replacement."""
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


def _ols_weights(signals, target):
    """Signed least squares; failures are explicit, never replaced by equal weights."""
    n = signals.shape[0]
    x = signals.permute(1, 2, 0).reshape(-1, n)
    y = target.to(signals).reshape(-1, 1)
    valid = y[:, 0].isfinite() & x.isfinite().all(1)
    if valid.sum() < n + 1:
        raise ValueError("Not enough valid samples for OLS")
    weights = torch.linalg.lstsq(x[valid], y[valid], rcond=1e-15).solution[:, 0]
    if not weights.isfinite().all():
        raise ValueError("OLS produced nonfinite coefficients")
    return weights


def _ols_weighted_signal(signals, target, nan_to_num=0.0):
    if signals.shape[0] == 0:
        return None
    weights = _ols_weights(signals, target)
    return torch.nan_to_num((signals * weights[:, None, None]).sum(0), nan=nan_to_num)


class AlphaCFPool:
    def __init__(self, data, target, args):
        self.data, self.target, self.args = data, target, args
        self.exprs, self.cache = [], {}
        self.label_days = spearman(target, target).isfinite()
        if not self.label_days.any():
            raise ValueError("Train labels have no usable cross-sectional variation")

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
            # Undefined factor IC contributes zero on every usable label day.
            if not daily.isfinite().any():
                raise ValueError("Factor has no usable cross-sectional variation")
            reward = daily[self.label_days].nan_to_num().mean().item()
            cost = (1 - spearman(signal[1:], signal[:-1]).nanmean().item()) / 2
            if not np.isfinite([reward, cost]).all():
                raise ValueError("Factor has no usable cross-sectional variation")
            self.cache[key] = dict(signal=signal, reward=reward, ic=reward, cost=cost)
        return self.cache[key]

    def signal(self, expr):
        # Missing exposures receive neutral rank before OLS combination.
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

    def initialize(self, candidates):
        if len(candidates) != self.args.pool_capacity or len({str(e) for e in candidates}) != len(candidates):
            raise ValueError(f"Initial pool must contain exactly {self.args.pool_capacity} unique usable factors")
        for expr in candidates:
            self.evaluate(expr)
        self.exprs = list(candidates)
        self.keep(self.exprs)

    def correlation(self, left, right, absolute=False):
        daily = spearman(self.evaluate(left)["signal"], self.evaluate(right)["signal"])
        return (daily.abs() if absolute else daily).nanmean().item()

    def equivalent(self, left, right):
        x, y = (self.evaluate(expr)["signal"] for expr in (left, right))
        return torch.equal(x.isfinite(), y.isfinite()) and (
            torch.allclose(x, y, rtol=0, atol=1e-6, equal_nan=True)
            or torch.allclose(x, -y, rtol=0, atol=1e-6, equal_nan=True))

    def signal_comparison(self, left, right):
        x, y = (self.evaluate(expr)["signal"] for expr in (left, right))
        masks = x.isfinite(), y.isfinite()
        union = (masks[0] | masks[1]).sum().item()
        correlation = self.correlation(left, right)
        return dict(coverage_change=(masks[0] ^ masks[1]).sum().item() / union if union else None,
                    common_rank_correlation=correlation if np.isfinite(correlation) else None)

    def signal_distance(self, left, right):
        x, y = (self.evaluate(expr)["signal"] for expr in (left, right))
        masks = x.isfinite(), y.isfinite()
        union = (masks[0] | masks[1]).sum().item()
        if not union:
            return None
        if self.equivalent(left, right):
            return 0.0
        coverage = (masks[0] & masks[1]).sum().item() / union
        a, b = x.nan_to_num().double().flatten(), y.nan_to_num().double().flatten()
        norms = a.norm(), b.norm()
        denom = norms[0] * norms[1]
        correlation = (a.dot(b) / denom).clamp(-1, 1).abs().item() if denom > 0 else float(norms[0] == norms[1])
        # A tiny numerical error must not label unequal full rankings exactly equivalent.
        return max(float(np.clip(1.0 - coverage * correlation, 0, 1)), np.finfo(float).eps)

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
            # Both pools fit and evaluate on the complete Train dataset.
            return 0.0 if str(expr) == str(parent) else self.utility(peers + [expr]) - pool_utility

        distance = {str(child): self.signal_distance(parent, child) for child in children}

        weights = np.array([self.args.alpha, self.args.beta, self.args.gamma, self.args.cost_weight])
        members = [parent] + children
        rows = [[abs(self.evaluate(e)["reward"]), c_pool(e), diversity(e),
                 1 - self.evaluate(e)["cost"]]
                for e in members]
        scores = np.array(rows) @ weights
        parent_score = scores[0]

        # Apply both update gates before canonicalizing equivalent children.
        correlation_ok = [True] + [1.0 - row[2] <= self.args.correlation_threshold for row in rows[1:]]
        credit_ok = [True] + [row[1] > 1e-6 for row in rows[1:]]
        representatives = []
        allowed = [i for i in range(len(members)) if correlation_ok[i] and credit_ok[i]]
        for i in sorted(allowed, key=lambda i: (complexity(members[i]), i)):
            if not any(self.equivalent(members[i], members[j]) for j in representatives):
                representatives.append(i)
        parent_rep = next(i for i in representatives if self.equivalent(parent, members[i]))
        best = max(representatives, key=lambda i: (scores[i], -complexity(members[i]), -i))
        selected = best if scores[best] > scores[parent_rep] else parent_rep
        winner = members[selected] if selected != 0 else None
        candidates = []

        for i, (child, row, score) in enumerate(zip(children, rows[1:], scores[1:]), 1):
            candidates.append(dict(expression=str(child), reward=self.evaluate(child)["reward"],
                                   complexity=complexity(child), max_correlation=1.0 - row[2],
                                   pool_credit=row[1], cost=self.evaluate(child)["cost"],
                                   signal_distance=distance[str(child)], score=float(score),
                                   correlation_ok=bool(correlation_ok[i]), credit_ok=bool(credit_ok[i]),
                                   representative=i in representatives,
                                   eligible=bool(i in representatives and (i == parent_rep or score > scores[parent_rep]))))
        if winner is not None:
            self.exprs[index] = winner
        return dict(parent=str(parent), parent_reward=self.evaluate(parent)["reward"],
                    parent_complexity=parent_nodes, parent_score=float(parent_score),
                    parent_group_score=float(scores[parent_rep]),
                    reason=("equivalent_simplification" if winner is not None and selected == parent_rep else "score_improvement"
                            if winner is not None else "no_new_offspring" if not children else "correlation_threshold"
                            if not any(correlation_ok[1:]) else "pool_credit_threshold"
                            if len(allowed) == 1 else "no_score_improvement"),
                    child=str(winner) if winner is not None else None,
                    outcome="replaced" if winner is not None else "retained", candidates=candidates)

    def keep(self, exprs):
        keys = {str(e) for e in exprs}
        self.cache = {k: v for k, v in self.cache.items() if k in keys}

    def to_dict(self):
        # Store actual OLS weights from the same regression the pool utility uses,
        # so the saved JSON reflects the in-sample combination that produced U(P).
        signals = torch.stack([self.signal(e) for e in self.exprs]) if self.exprs else None
        weights = _ols_weights(signals, self.target) if self.exprs else None
        combined = (signals * weights[:, None, None]).sum(0) if self.exprs else None
        return dict(exprs=[str(e) for e in self.exprs],
                    weights=weights.tolist() if self.exprs else [],
                    metrics=[{k: v for k, v in self.evaluate(e).items() if k != "signal"} for e in self.exprs],
                    utility=self.score_signals(combined.unsqueeze(0))[0].item() if self.exprs else 0.0)
