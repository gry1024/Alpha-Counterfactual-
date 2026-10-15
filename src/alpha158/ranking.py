"""Spearman correlation with linear-memory average-tie ranks."""
import torch
from alphagen.utils.correlation import (
    _mask_either_nan, _batch_pearsonr_given_mask, batch_spearmanr as original_spearmanr)



def rank_data(x, nan_mask):
    values, order = x.sort(dim=1)
    positions = torch.arange(x.shape[1], device=x.device).expand_as(order)
    starts = torch.ones_like(order, dtype=torch.bool)
    starts[:, 1:] = values[:, 1:] != values[:, :-1]
    ends = torch.ones_like(starts)
    ends[:, :-1] = starts[:, 1:]
    first = torch.where(starts, positions, 0).cummax(dim=1).values
    last = torch.where(ends, positions, x.shape[1] - 1)
    last = last.flip(1).cummin(dim=1).values.flip(1)
    sorted_ranks = (first.float() + last.float()) / 2
    ranks = torch.empty_like(sorted_ranks).scatter_(1, order, sorted_ranks)
    return ranks.masked_fill(nan_mask, 0.)


def batch_spearmanr(x, y):
    x, y, n, mask = _mask_either_nan(x, y)
    result = _batch_pearsonr_given_mask(rank_data(x, mask), rank_data(y, mask), n, mask)
    # Preserve the original floating-point behavior on constant/degenerate rows.
    def constant(values):
        minimum = values.masked_fill(mask, float("inf")).min(dim=1).values
        maximum = values.masked_fill(mask, -float("inf")).max(dim=1).values
        return minimum == maximum
    degenerate = constant(x) | constant(y) | (n <= 1)
    if degenerate.any():
        result[degenerate] = original_spearmanr(x[degenerate], y[degenerate])
    return result
