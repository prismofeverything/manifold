"""Plotting helpers for simulation outputs."""

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(
    x: Sequence,
    series: dict,
    xlabel: str = "time",
    ylabel: str = "level",
    title: str = "plot",
    out_dir: str = "out",
) -> str:
    """Plot {label: y_series} against x. Saves to out_dir/{title}.png."""
    Path(out_dir).mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    for label, y in series.items():
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    path = f"{out_dir}/{title}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_complex(
    x: Sequence,
    series: dict,
    xlabel: str = "time",
    title: str = "plot",
    out_dir: str = "out",
) -> str:
    """Plot complex-valued series in two stacked panels: real and imaginary."""
    Path(out_dir).mkdir(exist_ok=True)
    fig, (ax_re, ax_im) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    for label, y in series.items():
        re = [complex(v).real for v in y]
        im = [complex(v).imag for v in y]
        ax_re.plot(x, re, label=label)
        ax_im.plot(x, im, label=label)
    ax_re.set_ylabel("real")
    ax_re.legend()
    ax_im.set_ylabel("imag")
    ax_im.set_xlabel(xlabel)
    fig.suptitle(title)
    path = f"{out_dir}/{title}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_trajectory(
    values,
    xlabel: str = "real",
    ylabel: str = "imag",
    title: str = "trajectory",
    out_dir: str = "out",
    extra_series: dict | None = None,
) -> str:
    """Plot a sequence of complex values as a trajectory in the complex plane.

    Marks the start (green dot) and end (red dot). Optional extra_series
    is {label: complex_iterable} for overlaying other trajectories on
    the same plot — useful for showing two coupled oscillators together.
    """
    Path(out_dir).mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    re = [complex(v).real for v in values]
    im = [complex(v).imag for v in values]
    ax.plot(re, im, color="tab:blue", alpha=0.6, linewidth=0.7, label="trajectory")
    ax.scatter([re[0]], [im[0]], color="tab:green", s=40, label="start", zorder=5)
    ax.scatter([re[-1]], [im[-1]], color="tab:red", s=40, label="end", zorder=5)

    if extra_series:
        for label, vals in extra_series.items():
            re2 = [complex(v).real for v in vals]
            im2 = [complex(v).imag for v in vals]
            ax.plot(re2, im2, alpha=0.6, linewidth=0.7, label=label)
            ax.scatter([re2[0]], [im2[0]], s=40, marker="^", zorder=5)
            ax.scatter([re2[-1]], [im2[-1]], s=40, marker="s", zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    path = f"{out_dir}/{title}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
