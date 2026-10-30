"""Plot recorded Test metrics without rerunning evaluation."""
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def plot_metrics(history, path):
    rows = history[history["dataset"] == "Test"].drop_duplicates("round", keep="last").sort_values("round")
    figure = Figure(figsize=(15, 10), layout="constrained")
    metrics = ("ic", "ric", "ret", "icir", "ricir", "ret_sharpe",
               "ic_std", "ric_std", "ret_std", "retir", "ret_mdd")
    axes = figure.subplots(4, 3).flat
    for axis, metric in zip(axes, metrics):
        axis.plot(rows["round"], rows[metric], marker="o", label="Test")
        axis.set(xlabel="Round", ylabel=metric.upper())
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.3)
        axis.legend()
    figure.delaxes(figure.axes[-1])
    figure.savefig(path, dpi=150)
