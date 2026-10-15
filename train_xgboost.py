"""AlphaSAGE XGBoost baseline using the shared OHLCV features and backtest."""
import numpy as np
import xgboost as xgb
from train_lightgbm import main


class CachedBatches(xgb.DataIter):
    def __init__(self, cache, mask, labels):
        super().__init__()
        self.cache = cache
        self.indices = np.flatnonzero(mask)
        self.labels = labels
        self.offset = 0

    def reset(self):
        self.offset = 0

    def next(self, input_data):
        if self.offset >= len(self.indices):
            return False
        end = min(self.offset + 16384, len(self.indices))
        input_data(data=self.cache[self.indices[self.offset:end]], label=self.labels[self.offset:end])
        self.offset = end
        return True


def train(cache, train_mask, y_train, valid_mask, y_valid, args):
    train_data = xgb.QuantileDMatrix(CachedBatches(cache, train_mask, y_train), nthread=args.threads)
    valid_data = xgb.QuantileDMatrix(CachedBatches(cache, valid_mask, y_valid),
                                   ref=train_data, nthread=args.threads)
    params = dict(objective="reg:squarederror", eval_metric="rmse", colsample_bytree=0.8879,
                  learning_rate=0.01, subsample=0.8789, reg_lambda=205.6999, reg_alpha=580.9768,
                  max_depth=4, tree_method="hist", nthread=args.threads, seed=args.seed)
    print(f"[{args.instruments}] XGBoost train rows={train_mask.sum()}, valid rows={valid_mask.sum()}", flush=True)
    model = xgb.train(params, train_data, num_boost_round=100,
                      evals=[(train_data, "train"), (valid_data, "valid")],
                      early_stopping_rounds=100, verbose_eval=10)
    return model, train_data, valid_data, params


if __name__ == "__main__":
    main("xgboost")
