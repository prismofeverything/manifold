"""Drawing helpers for the book of diagrams.

Each book entry is a figure with three sections: a schematic showing
nodes and channels, the equations specifying the dynamics, and a
behavior plot from a small simulation.

Channel kinds (visual conventions):
  - excite: solid black arrow, normal arrowhead
  - inhibit: solid black line ending in a small open circle
  - plastic: dashed line, arrowhead
  - self: small curved loop attached to the node
  - input: green arrow from outside the diagram
"""

from typing import Iterable, Optional, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def draw_node(ax, pos, label, color="#cce5ff", radius=0.18, label_fontsize=10):
    """Filled circle with a centered label."""
    circle = mpatches.Circle(pos, radius, facecolor=color, edgecolor="black",
                             linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(pos[0], pos[1], label, ha="center", va="center",
            fontsize=label_fontsize, zorder=4)


def _arrow_endpoints(src, dst, src_radius=0.18, dst_radius=0.18):
    """Trim arrow endpoints so they sit on circle boundaries, not centers."""
    src = np.array(src, dtype=float)
    dst = np.array(dst, dtype=float)
    delta = dst - src
    dist = np.linalg.norm(delta)
    if dist < 1e-9:
        return src, dst
    unit = delta / dist
    return src + unit * src_radius, dst - unit * dst_radius


def draw_excite(ax, src, dst, label="", curvature=0.0):
    """Excitatory arrow (solid, normal head)."""
    a, b = _arrow_endpoints(src, dst)
    arrow = mpatches.FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=14, color="black",
        connectionstyle=f"arc3,rad={curvature}", linewidth=1.5, zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mid = (np.array(a) + np.array(b)) / 2 + np.array([0, 0.06])
        ax.text(mid[0], mid[1], label, fontsize=8, ha="center", va="bottom")


def draw_inhibit(ax, src, dst, label="", curvature=0.0):
    """Inhibitory arrow (solid, ends in small open circle ●—○)."""
    a, b = _arrow_endpoints(src, dst, dst_radius=0.22)
    line = mpatches.FancyArrowPatch(
        a, b, arrowstyle="-", color="black",
        connectionstyle=f"arc3,rad={curvature}", linewidth=1.5, zorder=2,
    )
    ax.add_patch(line)
    cap = mpatches.Circle(b, 0.04, facecolor="white", edgecolor="black",
                          linewidth=1.5, zorder=3)
    ax.add_patch(cap)
    if label:
        mid = (np.array(a) + np.array(b)) / 2 + np.array([0, 0.06])
        ax.text(mid[0], mid[1], label, fontsize=8, ha="center", va="bottom")


def draw_plastic(ax, src, dst, label="", curvature=0.0):
    """Plastic arrow (dashed, normal head)."""
    a, b = _arrow_endpoints(src, dst)
    arrow = mpatches.FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=14, color="black",
        connectionstyle=f"arc3,rad={curvature}", linewidth=1.5,
        linestyle=(0, (4, 2)), zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mid = (np.array(a) + np.array(b)) / 2 + np.array([0, 0.06])
        ax.text(mid[0], mid[1], label, fontsize=8, ha="center", va="bottom")


def draw_self_loop(ax, pos, label="", side="top", radius=0.10, kind="excite"):
    """Self-loop drawn as a small circular curve attached to one side of node."""
    if side == "top":
        center = (pos[0], pos[1] + 0.30)
        anchor = (pos[0], pos[1] + 0.18)
    elif side == "right":
        center = (pos[0] + 0.30, pos[1])
        anchor = (pos[0] + 0.18, pos[1])
    elif side == "bottom":
        center = (pos[0], pos[1] - 0.30)
        anchor = (pos[0], pos[1] - 0.18)
    else:
        center = (pos[0] - 0.30, pos[1])
        anchor = (pos[0] - 0.18, pos[1])
    loop = mpatches.Circle(center, radius, facecolor="none", edgecolor="black",
                           linewidth=1.5, zorder=2,
                           linestyle="-" if kind != "plastic" else (0, (4, 2)))
    ax.add_patch(loop)
    if kind == "inhibit":
        # End cap
        cap = mpatches.Circle(anchor, 0.03, facecolor="white", edgecolor="black",
                              linewidth=1.5, zorder=3)
        ax.add_patch(cap)
    if label:
        if side == "top":
            ax.text(center[0], center[1] + 0.16, label, fontsize=8, ha="center")
        elif side == "right":
            ax.text(center[0] + 0.16, center[1], label, fontsize=8, va="center")
        elif side == "bottom":
            ax.text(center[0], center[1] - 0.18, label, fontsize=8, ha="center")
        else:
            ax.text(center[0] - 0.16, center[1], label, fontsize=8, ha="center", va="center")


def draw_external_input(ax, dst, label="x", offset=(-0.55, 0)):
    """Green arrow from outside the diagram into a node."""
    src = (dst[0] + offset[0], dst[1] + offset[1])
    a, b = _arrow_endpoints(src, dst, src_radius=0.0)
    arrow = mpatches.FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=14, color="#2ca02c",
        linewidth=1.5, zorder=2,
    )
    ax.add_patch(arrow)
    ax.text(src[0] - 0.05, src[1], label, fontsize=10, color="#2ca02c",
            ha="right", va="center")


def setup_schematic_ax(ax, xlim=(-1.0, 1.0), ylim=(-0.7, 0.7), title=""):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)


def make_entry_figure(
    title: str,
    schematic_fn,
    equations: str,
    behavior_fn,
    classification: str = "",
    notes: str = "",
    figsize=(11, 8),
):
    """Common 3-section layout: schematic on top, equations + classification
    in the middle band, behavior plot at the bottom."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 0.8, 1.4],
                          width_ratios=[1, 1], hspace=0.35, wspace=0.15)
    ax_schema = fig.add_subplot(gs[0, :])
    ax_eqns = fig.add_subplot(gs[1, 0])
    ax_meta = fig.add_subplot(gs[1, 1])
    ax_behavior = fig.add_subplot(gs[2, :])

    schematic_fn(ax_schema)
    ax_eqns.text(0.02, 0.97, equations, fontsize=9, family="monospace",
                 va="top", ha="left", transform=ax_eqns.transAxes)
    ax_eqns.axis("off")
    meta_text = ""
    if classification:
        meta_text += f"behavior:  {classification}\n\n"
    if notes:
        meta_text += f"notes:\n{notes}"
    ax_meta.text(0.02, 0.97, meta_text, fontsize=9, va="top", ha="left",
                 transform=ax_meta.transAxes)
    ax_meta.axis("off")

    behavior_fn(ax_behavior)
    fig.suptitle(title, fontsize=13, y=0.99)
    return fig
