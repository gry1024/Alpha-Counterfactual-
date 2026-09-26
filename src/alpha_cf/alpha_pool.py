"""Signed RankICIR, equal-weight rank signals and exact pool-aware selection."""
from copy import copy
import time

import torch
from alphagen.utils.correlation import batch_pearsonr
from alphagen_qlib.stock_data import FeatureType


def rank(x):
    """Average tied ranks, with missing positions preserved; no stocks² tensor."""
    finite = torch.isfinite(x)
    values, order = x.masked_fill(~finite, torch.inf).sort(dim=1)
    positions = torch.arange(x.shape[1], device=x.device).expand_as(order)
    boundary = values[:, 1:] != values[:, :-1]
    edge = torch.ones_like(values[:, :1], dtype=torch.bool)
    first, last = torch.cat((edge, boundary), 1), torch.cat((boundary, edge), 1)
    starts = torch.where(first, positions, 0).cummax(1).values
    ends = torch.where(last, positions, x.shape[1] - 1).flip(1).cummin(1).values.flip(1)
    result = torch.empty_like(x).scatter_(1, order, (starts + ends).to(x.dtype) / 2)
    return result.masked_fill(~finite, torch.nan)


def _correlate(rx, ry, min_stocks=2):
    mask = torch.isfinite(rx) & torch.isfinite(ry)
    rx, ry = rx.masked_fill(~mask, torch.nan), ry.masked_fill(~mask, torch.nan)
    def varying(r):
        return r.nan_to_num(nan=-torch.inf).amax(1) > r.nan_to_num(nan=torch.inf).amin(1)
    valid = (mask.sum(1) >= min_stocks) & varying(rx) & varying(ry)
    corr = batch_pearsonr(rx, ry).clamp(-1, 1)
    return corr.masked_fill(~valid | ~torch.isfinite(corr), torch.nan)


def spearman(x, y, chunk_size=64, min_stocks=2):
    result = []
    for a, b in zip(x.split(chunk_size), y.split(chunk_size)):
        mask = torch.isfinite(a) & torch.isfinite(b)
        result.append(_correlate(rank(a.masked_fill(~mask, torch.nan)),
                                 rank(b.masked_fill(~mask, torch.nan)), min_stocks))
    return torch.cat(result)


def valid_values(values, min_days=2):
    values = values[torch.isfinite(values)]
    if len(values) < min_days:
        raise ValueError(f"Insufficient valid observations: {len(values)} < {min_days}")
    return values


def icir(ic, min_days=2):
    ic = valid_values(ic, min_days)
    return (ic.mean() / (ic.std(correction=0) + 1e-8)).item()


class AlphaCFPool:
    def __init__(self, data, target, args, log, deadline=float("inf"), max_evals=None):
        self.data, self.target, self.args, self.log = data, target, args, log
        self.deadline, self.max_evals = deadline, max_evals
        self.exprs, self.cache, self.selection_log, self.pair_cache = [], {}, [], {}
        self.eval_cnt = self.pool_scores = self.pair_hits = self.pair_scores = 0
        start = data.max_backtrack_days
        close = data.data[start:start + len(target), int(FeatureType.CLOSE), :]
        self.market = torch.isfinite(close) & (close > 0)
        self.target = target.masked_fill(~self.market, torch.nan)
        self.target_rank = rank(self.target)
        self.eligible = torch.isfinite(self.target)
        self.eligible_days = torch.isfinite(_correlate(
            self.target_rank, self.target_rank, args.min_stocks))
        if self.eligible_days.sum().item() < args.min_valid_days:
            raise ValueError("Insufficient eligible market days")

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Run time limit reached")

    def summarize_ic(self, daily_ic):
        valid = torch.isfinite(daily_ic)
        coverage = (valid.sum() / self.eligible_days.sum()).item()
        if coverage < self.args.min_day_coverage:
            raise ValueError(f"IC day coverage {coverage:.3f} below minimum")
        values = valid_values(daily_ic, self.args.min_valid_days)
        return dict(reward=icir(values), mean_ic=values.mean().item(),
                    std_ic=values.std(correction=0).item(),
                    valid_days=len(values), day_coverage=coverage)

    @staticmethod
    def metrics(entry):
        return {k: v for k, v in entry.items() if k not in ("signal", "valid_mask")}

    @torch.no_grad()
    def evaluate(self, expr):
        key = str(expr)
        if key in self.cache:
            return self.cache[key]
        self.check_time()
        if self.max_evals is not None and self.eval_cnt >= self.max_evals:
            raise TimeoutError("Factor evaluation limit reached")
        self.eval_cnt += 1
        started, signals, ics = time.monotonic(), [], []
        try:
            for start in range(0, len(self.target), self.args.chunk_size):
                self.check_time()
                end = min(start + self.args.chunk_size, len(self.target))
                block = copy(self.data)
                block.data = self.data.data[start:self.data.max_backtrack_days + end]
                value = expr.evaluate(block)
                # Signal normalization uses current market availability, not future labels.
                value = value.masked_fill(~torch.isfinite(value) | ~self.market[start:end], torch.nan)
                ics.append(spearman(value, self.target[start:end], self.args.chunk_size, self.args.min_stocks))
                count = torch.isfinite(value).sum(1, keepdim=True)
                signal = rank(value) / (count - 1).clamp_min(1) - 0.5
                signals.append(signal.masked_fill(count < 2, torch.nan))
            signal, daily_ic = torch.cat(signals), torch.cat(ics)
            coverage = ((torch.isfinite(signal) & self.eligible).sum() / self.eligible.sum()).item()
            if coverage < self.args.min_coverage:
                raise ValueError(f"Value coverage {coverage:.3f} below minimum")
            turnover = spearman(signal[1:], signal[:-1], self.args.chunk_size, self.args.min_stocks)
            stats = dict(**self.summarize_ic(daily_ic), coverage=coverage,
                         cost=(1 - valid_values(turnover, self.args.min_valid_days).mean().item()) / 2)
            self.cache[key] = dict(signal=signal, valid_mask=torch.isfinite(daily_ic), **stats)
            self.log("evaluate", expression=key, eval_cnt=self.eval_cnt,
                     seconds=time.monotonic() - started, **stats)
        except torch.cuda.OutOfMemoryError:
            raise
        except (ValueError, IndexError, RuntimeError) as exc:
            self.cache[key] = None
            self.log("invalid_factor", expression=key, error=str(exc), eval_cnt=self.eval_cnt)
        return self.cache[key]

    def keep(self, exprs):
        keys = {str(expr) for expr in exprs}
        self.cache = {key: value for key, value in self.cache.items() if key in keys}
        self.pair_cache = {pair: value for pair, value in self.pair_cache.items() if set(pair) <= keys}

    def signal(self, expr):
        result = self.evaluate(expr)
        if result is None:
            raise ValueError(f"Invalid factor: {expr}")
        return result["signal"]

    @torch.no_grad()
    def score_signal(self, signal):
        self.check_time()
        self.pool_scores += 1
        ics = []
        for start in range(0, len(signal), self.args.chunk_size):
            self.check_time()
            end = start + self.args.chunk_size
            x = signal[start:end].masked_fill(~self.eligible[start:end], torch.nan)
            ics.append(_correlate(rank(x), self.target_rank[start:end], self.args.min_stocks))
        return self.summarize_ic(torch.cat(ics))["reward"]

    def utility(self, exprs):
        exprs = list({str(expr): expr for expr in exprs}.values())
        if not exprs:
            return 0.0
        signal = sum(torch.nan_to_num(self.signal(expr)) for expr in exprs) / len(exprs)
        return self.score_signal(signal)

    def correlation(self, lhs, rhs):
        self.check_time()
        key = tuple(sorted((str(lhs), str(rhs))))
        if key in self.pair_cache:
            self.pair_hits += 1
            return self.pair_cache[key]
        daily = spearman(self.signal(lhs), self.signal(rhs), self.args.chunk_size, self.args.min_stocks)
        value = valid_values(daily, self.args.min_valid_days).abs().mean().item()
        self.pair_cache[key] = value
        self.pair_scores += 1
        return value

    @torch.no_grad()
    def select(self, candidates, k):
        started = time.monotonic()
        remaining = [expr for _, expr in sorted({str(e): e for e in candidates}.items())
                     if self.evaluate(expr) is not None]
        if len(remaining) < k:
            raise ValueError(f"Need {k} valid unique factors, got {len(remaining)}")
        records = []
        if self.args.no_pool_selection:
            selected = sorted(remaining, key=lambda e: (-self.cache[str(e)]["reward"], str(e)))[:k]
            records = [dict(expression=str(e), reward=self.cache[str(e)]["reward"], mode="reward_only")
                       for e in selected]
        else:
            selected, total, utility = [], torch.zeros_like(self.target), 0.0
            max_corr = {str(expr): 0.0 for expr in remaining}
            weights = torch.tensor([self.args.alpha, self.args.beta, self.args.gamma,
                                    -self.args.cost_weight], dtype=torch.float64)
            while len(selected) < k:
                self.check_time()
                rows, scored, utilities = [], [], []
                for expr in remaining:
                    entry = self.cache[str(expr)]
                    try:
                        u = self.score_signal((total + entry["signal"].nan_to_num()) / (len(selected) + 1))
                    except ValueError as exc:
                        self.log("invalid_combination", expression=str(expr), error=str(exc))
                        continue
                    scored.append(expr)
                    utilities.append(u)
                    rows.append([entry["reward"], u - utility, 1 - max_corr[str(expr)], entry["cost"]])
                if not scored:
                    raise ValueError(f"No evaluable pool extension at size {len(selected)}")
                raw = torch.tensor(rows, dtype=torch.float64)
                norm = (raw - raw.amin(0)) / (raw.amax(0) - raw.amin(0)).clamp_min(1e-12)
                scores = norm @ weights
                index = int(scores.argmax())
                winner = scored[index]
                remaining.remove(winner)
                selected.append(winner)
                total += self.signal(winner).nan_to_num()
                utility = utilities[index]
                records.append(dict(position=len(selected), expression=str(winner), reward=rows[index][0],
                                    marginal=rows[index][1], diversity=rows[index][2], cost=rows[index][3],
                                    score=scores[index].item(), utility=utility))
                if len(selected) < k:
                    for expr in list(remaining):
                        try:
                            max_corr[str(expr)] = max(max_corr[str(expr)], self.correlation(expr, winner))
                        except ValueError as exc:
                            remaining.remove(expr)
                            self.log("invalid_diversity", expression=str(expr), error=str(exc))
        self.selection_log = records
        for record in records:
            self.log("select", **record)
        self.log("selection_done", size=k, seconds=time.monotonic() - started,
                 pool_scores=self.pool_scores, pair_scores=self.pair_scores, pair_hits=self.pair_hits)
        return selected

    def update(self, offspring):
        if offspring:
            try:
                self.exprs = self.select(self.exprs + offspring, self.args.pool_capacity)
            except ValueError as exc:
                self.log("pool_update_rejected", error=str(exc), size=len(self.exprs))
        self.keep(self.exprs)

    def to_dict(self):
        return dict(exprs=[str(e) for e in self.exprs], weights=[1 / len(self.exprs)] * len(self.exprs),
                    metrics=[self.metrics(self.cache[str(e)]) for e in self.exprs],
                    eval_cnt=self.eval_cnt, pool_scores=self.pool_scores,
                    pair_scores=self.pair_scores, pair_hits=self.pair_hits, selection=self.selection_log)
