from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(arr: np.ndarray, center: str = "mean", ax: Optional[plt.Axes] = None) -> plt.Axes:
    assert center in ["zero", "mean", "range"]
    assert arr.size > 0, "Cannot plot an empty tensor."

    hist_color = "deepskyblue"
    normal_color = "tab:blue"
    hist_alpha = 0.9

    a = arr[np.isfinite(arr)].astype(np.float64)
    a_min, a_max = a.min(), a.max()
    a_mean, a_std = np.mean(a), np.std(a)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2), constrained_layout=True)
    fig = ax.figure

    if center == "range":
        x_min, x_max = a_min, a_max
    elif center == "mean":
        max_diff = max(a_mean - a_min, a_max - a_mean)
        x_min, x_max = a_mean - max_diff, a_mean + max_diff
    elif center == "range":
        abs_max = max(abs(a_min), abs(a_max))
        x_min, x_max = -abs_max, abs_max
    else:
        raise NotImplementedError("")

    n_sigmas = int(max((a_mean - a_min) / a_std, (a_max - a_mean) / a_std))

    x_range = x_max - x_min
    expand_coeff = 0.02
    x_min -= x_range * expand_coeff
    x_max += x_range * expand_coeff

    n_bins = int(np.clip(a.size / 50, 10, 100))

    bin_edges = np.linspace(x_min, x_max, num=n_bins)
    bar_width = bin_edges[1] - bin_edges[0]

    hist, bin_edges = np.histogram(a, bins=bin_edges, density=True)
    ax: plt.Axes
    ax.bar(bin_edges[:-1], hist, width=bar_width, color=hist_color, alpha=hist_alpha, align="edge", zorder=4)

    xs = np.linspace(x_min, x_max, num=100)
    density = _normal_pdf(xs, a_mean, a_std)
    ax.plot(xs, density, color=normal_color, zorder=5)

    y_lim = max(hist.max(), density.max()) * 1.3

    # Make text bank part of the line under it
    bbox = dict(boxstyle="round", fc="white", edgecolor="none")

    for s in range(-n_sigmas, n_sigmas + 1):
        x_pos = a_mean + s * a_std
        if x_min < x_pos < x_max:
            greek = ["-σ", "μ", "+σ"][s + 1] if -1 <= s <= 1 else f"{s:+}σ"
            weight = "bold" if not s else None
            ax.axvline(x_pos, 0, 1, c="black")
            ax.text(x_pos, y_lim * 0.95, greek, ha="center", va="top", bbox=bbox, zorder=5, weight=weight)

    # lines for min and max values
    ax.annotate(
        f"min={_pretty_str(a_min)}",
        (a_min, y_lim / 2),
        xytext=(-1, 0),
        textcoords="offset points",
        bbox=bbox,
        rotation=90,
        ha="right",
        va="center",
    )

    ax.annotate(
        f"max={_pretty_str(a_max)}",
        (a_max, y_lim / 2),
        xytext=(2, 0),
        textcoords="offset points",
        bbox=bbox,
        rotation=90,
        ha="left",
        va="center",
    )

    ax.axvline(a_min, 0, 1, c="red", zorder=4)
    ax.axvline(a_max, 0, 1, c="red", zorder=4)

    a_str = _arr_summary(arr)
    ax.text(x_min, y_lim * 1.05, s=a_str)
    ax.set_ylim(0, y_lim)
    ax.set_yticks([])

    ax.set_xlim(x_min, x_max)

    return ax


def _normal_pdf(x, mean, std) -> float:
    z = (x - mean) / std
    denom = std * np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z ** 2) / denom


def _pretty_str(x):
    if isinstance(x, int):
        return "{}".format(x)

    if isinstance(x, float):
        if x == 0.0:
            return "0."

        if abs(x) < 100:
            return "{:.2f}".format(x)

        return "{:.2e}".format(x)

    raise NotImplementedError("")


_dtnames = {
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
}


def _arr_summary(x: np.ndarray) -> str:
    shape = str(list(x.shape)) if x.ndim > 0 else ""
    type_str = "{}".format(shape)

    size = "n = {}".format(x.size)

    gx = x[np.isfinite(x)]
    minmax = "x∈[{}, {}]".format(_pretty_str(gx.min()), _pretty_str(gx.max()))
    meanstd = "μ={} σ={}".format(_pretty_str(gx.mean()), _pretty_str(gx.std()))
    summary = "{} {} {}".format(size, minmax, meanstd)

    dtype = _dtnames.get(x.dtype.name, str(x.dtype)[6:])

    res = " ".join([type_str, dtype, summary])
    return res
