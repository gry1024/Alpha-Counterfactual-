"""Signed RankICIR and greedy pool selection from idea.md."""
from copy import copy
import numpy as np
import torch
from alphagen.utils.correlation import batch_pearsonr


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


def icir(ic):
    count = ic.isfinite().sum(-1)
    mean = ic.nan_to_num().sum(-1) / count.clamp_min(1)
    var = (ic - mean.unsqueeze(-1)).square().nan_to_num().sum(-1) / count.clamp_min(1)
    return (mean / (var.sqrt() + 1e-8)).masked_fill(count < 2, torch.nan)


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
            reward = icir(daily).item()
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
        return icir(daily)

    def utility(self, exprs=None):
        exprs = self.exprs if exprs is None else exprs
        if not exprs:
            return 0.0
        signal = sum(self.signal(expr) for expr in exprs) / len(exprs)
        return self.score_signals(signal.unsqueeze(0))[0].item()

    @torch.no_grad()
    def select(self, candidates, size):
        remaining = list({str(e): e for e in candidates}.values())
        for expr in remaining:
            self.evaluate(expr)
        if len(remaining) < size:
            raise ValueError(f"Need {size} usable unique factors, got {len(remaining)}")
        if self.args.no_pool_selection:
            return sorted(remaining, key=lambda e: self.evaluate(e)["reward"], reverse=True)[:size]
        selected, total, utility = [], torch.zeros_like(self.target, dtype=torch.float64), 0.0
        correlations = {str(e): 0.0 for e in remaining}
        weights = np.array([self.args.alpha, self.args.beta, self.args.gamma, -self.args.cost_weight])
        while len(selected) < size:
            utilities = []
            for start in range(0, len(remaining), 4):
                batch = remaining[start:start + 4]
                signals = torch.stack([self.signal(e) for e in batch])
                utilities.extend(self.score_signals((total + signals) / (len(selected) + 1)).tolist())
            rows = np.array([[self.evaluate(e)["reward"], u - utility,
                              1 - correlations[str(e)], self.evaluate(e)["cost"]]
                             for e, u in zip(remaining, utilities)])
            scores = ((rows - rows.min(0)) / np.maximum(np.ptp(rows, axis=0), 1e-12)) @ weights
            scores[~np.isfinite(utilities)] = -np.inf
            if not np.isfinite(scores).any():
                raise ValueError("No valid pool extension")
            index = int(scores.argmax())
            winner, utility = remaining.pop(index), utilities[index]
            selected.append(winner)
            total += self.signal(winner)
            for start in range(0, len(remaining), 4):
                batch = remaining[start:start + 4]
                signals = torch.stack([self.evaluate(e)["signal"] for e in batch])
                rhs = self.evaluate(winner)["signal"].expand_as(signals)
                corr = spearman(signals.flatten(0, 1), rhs.flatten(0, 1)).reshape(signals.shape[:2]).abs().nanmean(1)
                for expr, value in zip(batch, corr.tolist()):
                    correlations[str(expr)] = max(correlations[str(expr)], value if np.isfinite(value) else 1.0)
        return selected

    def update(self, offspring):
        self.exprs = self.select(self.exprs + offspring, self.args.pool_capacity)
        self.keep(self.exprs)

    def keep(self, exprs):
        keys = {str(e) for e in exprs}
        self.cache = {k: v for k, v in self.cache.items() if k in keys}

    def to_dict(self):
        return dict(exprs=[str(e) for e in self.exprs], weights=[1 / len(self.exprs)] * len(self.exprs),
                    metrics=[{k: v for k, v in self.evaluate(e).items() if k != "signal"} for e in self.exprs],
                    utility=self.utility())
