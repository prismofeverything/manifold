"""Reusable three-layer animation.

Extracts the 4-panel visual layout from 10h_three_layer_animation.py into
a function that can be applied to any three-layer experiment whose
observation history follows the standard schema:

  - L1_{v}_{combo}: complex state of L1 detector (any of `arm_combos`)
                    at vertex v
  - L2_f{v}, L2_b{v}: complex state of L2 front/back detectors at vertex v
  - I_A, I_B: complex state of the two L3 interpretation nodes

Works for both `sigmoid_activity` experiments (states real-valued, imag=0)
and `polar_sigmoid` experiments (full complex states): we use `abs()`
on states throughout for amplitude visualization, so the same code path
handles both regimes.

Layout (matches 10h):
  Panel 1 — L1 rich detectors per cube vertex, drawn as 8 mini 3-arm
            corner icons inside a hex tile. Icon brightness = detector
            amplitude. Hex background opacity = mean amplitude at vertex.
  Panel 2 — L2 simple detectors (front/back per vertex). Marker size +
            opacity = amplitude.
  Panel 3 — L3 two cube wireframes, one per interpretation. Whole-cube
            opacity = the I node's amplitude.
  Panel 4 — HSL composite over the cube wireframe: hue = winning L1
            detector index at each vertex, saturation = L2 front/back
            contrast, lightness = which L3 interpretation is winning.
"""

import itertools
from typing import Sequence

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def cube_2d(v):
    """Standard Necker projection: z=1 face offset up-right of z=0 face."""
    x, y, z = v
    return (x + z * 0.45, y + z * 0.35)


def _hex_corners(center, radius):
    angles = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
    return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]


def _arm_2d_directions(v, neighbor_fn):
    here = np.array(cube_2d(v))
    dirs = []
    for axis in (0, 1, 2):
        n = neighbor_fn(v, axis)
        d = np.array(cube_2d(n)) - here
        dirs.append(d / (np.linalg.norm(d) + 1e-12))
    return dirs


def _draw_corner_icon(ax, center, arm_dirs, depths, scale=0.04, alpha=1.0, lw=1.5):
    artists = []
    for direction, d in zip(arm_dirs, depths):
        end = (center[0] + scale * direction[0], center[1] + scale * direction[1])
        col = "#1f77b4" if d == 0 else "#d62728"
        line, = ax.plot(
            [center[0], end[0]], [center[1], end[1]],
            color=col, alpha=alpha, linewidth=lw, solid_capstyle="round",
        )
        artists.append(line)
    return artists


def animate_three_layer_history(
    history: dict,
    vertices: Sequence,
    arm_combos: Sequence,
    n_steps: int,
    sample_every: int = 25,
    output_path: str = "out/animation.gif",
    title_prefix: str = "Three-layer hierarchy",
    fps: int = 15,
    figsize=(14, 13),
    n_features_per_l1_tile: int = 8,
):
    """Generate a 4-panel animation from a three-layer history.

    Expects history keys per the schema in the module docstring. Returns
    the path to the saved GIF.
    """
    frames = list(range(0, n_steps, sample_every))

    # Cube edges for wireframe drawing
    edges = (
        [((0, y, z), (1, y, z)) for y in (0, 1) for z in (0, 1)]
        + [((x, 0, z), (x, 1, z)) for x in (0, 1) for z in (0, 1)]
        + [((x, y, 0), (x, y, 1)) for x in (0, 1) for y in (0, 1)]
    )

    def neighbor(v, axis):
        return tuple(1 - x if i == axis else x for i, x in enumerate(v))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.18)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_aspect("equal"); ax.axis("off")

    ax1.set_title("L1 — rich features (3-arm corners × depth combos per vertex)")
    ax2.set_title("L2 — middle layer (front / back per vertex)")
    ax3.set_title("L3 — top layer (two interpretations, opacity = lock-in)")
    ax4.set_title("Composite (HSL): hue=L1 winner, sat=L2 contrast, light=L3 winner")

    vertex_pos = {v: cube_2d(v) for v in vertices}

    HEX_R = 0.36
    ICON_R = 0.20

    # ---- Panel 1 ----
    p1_hex_artists, p1_icon_lines = {}, {}
    for v in vertices:
        center = vertex_pos[v]
        hex_poly = mpatches.Polygon(_hex_corners(center, HEX_R), closed=True,
                                    facecolor="lightgray", edgecolor="black",
                                    alpha=0.2, linewidth=0.5)
        ax1.add_patch(hex_poly)
        p1_hex_artists[v] = hex_poly
        arm_dirs = _arm_2d_directions(v, neighbor)
        for i, combo in enumerate(arm_combos):
            angle = 2 * np.pi * i / n_features_per_l1_tile
            ic = (center[0] + ICON_R * np.cos(angle), center[1] + ICON_R * np.sin(angle))
            lines = _draw_corner_icon(ax1, ic, arm_dirs, combo,
                                      scale=0.07, alpha=0.15, lw=1.2)
            p1_icon_lines[(v, combo)] = lines
    ax1.set_xlim(-0.6, 2.0); ax1.set_ylim(-0.6, 1.9)

    # ---- Panel 2 ----
    p2_hex_artists, p2_front_dots, p2_back_dots = {}, {}, {}
    for v in vertices:
        center = vertex_pos[v]
        hex_poly = mpatches.Polygon(_hex_corners(center, HEX_R), closed=True,
                                    facecolor="lightgray", edgecolor="black",
                                    alpha=0.2, linewidth=0.5)
        ax2.add_patch(hex_poly)
        p2_hex_artists[v] = hex_poly
        f_dot = ax2.scatter(center[0] - 0.10, center[1], s=80, c="#1f77b4", alpha=0.3)
        b_dot = ax2.scatter(center[0] + 0.10, center[1], s=80, c="#d62728", alpha=0.3)
        p2_front_dots[v] = f_dot; p2_back_dots[v] = b_dot
    ax2.set_xlim(-0.6, 2.0); ax2.set_ylim(-0.6, 1.9)

    # ---- Panel 3 ----
    cube_a_offset, cube_b_offset = (0.0, 0.0), (2.2, 0.0)
    def cube_pos(v, off):
        p = cube_2d(v); return (p[0] + off[0], p[1] + off[1])
    p3_a_lines, p3_b_lines = [], []
    p3_a_dots, p3_b_dots = {}, {}
    for u, vv in edges:
        la, = ax3.plot(*zip(cube_pos(u, cube_a_offset), cube_pos(vv, cube_a_offset)),
                       color="black", alpha=0.3, linewidth=1.2)
        lb, = ax3.plot(*zip(cube_pos(u, cube_b_offset), cube_pos(vv, cube_b_offset)),
                       color="black", alpha=0.3, linewidth=1.2)
        p3_a_lines.append(la); p3_b_lines.append(lb)
    for v in vertices:
        col_a = "#1f77b4" if v[2] == 0 else "#d62728"
        col_b = "#d62728" if v[2] == 0 else "#1f77b4"
        pa, pb = cube_pos(v, cube_a_offset), cube_pos(v, cube_b_offset)
        p3_a_dots[v] = ax3.scatter([pa[0]], [pa[1]], s=120, c=col_a, alpha=0.3, zorder=5)
        p3_b_dots[v] = ax3.scatter([pb[0]], [pb[1]], s=120, c=col_b, alpha=0.3, zorder=5)
    ax3.text(cube_a_offset[0] + 0.7, -0.5, "Interp A (z=0 front)", ha="center", fontsize=10)
    ax3.text(cube_b_offset[0] + 0.7, -0.5, "Interp B (z=0 back)",  ha="center", fontsize=10)
    ax3.set_xlim(-0.6, 4.4); ax3.set_ylim(-0.7, 1.9)

    # ---- Panel 4 ----
    p4_lines = []
    for u, vv in edges:
        line, = ax4.plot(*zip(cube_2d(u), cube_2d(vv)), color="black", alpha=0.3, linewidth=1.2)
        p4_lines.append(line)
    p4_dots = {v: ax4.scatter([cube_2d(v)[0]], [cube_2d(v)[1]], s=200, c="gray", alpha=0.6, zorder=5)
               for v in vertices}
    ax4.set_xlim(-0.6, 2.0); ax4.set_ylim(-0.6, 1.9)

    suptitle = fig.suptitle(f"{title_prefix} — t=0", fontsize=14)

    n_combos = len(arm_combos)

    def update(frame_idx):
        t = frames[frame_idx]
        artists = []

        # Compute per-vertex L1 amplitudes (uses abs to handle complex states)
        l1_amps = {}
        for v in vertices:
            for c in arm_combos:
                key = f"L1_{v}_{c}"
                l1_amps[(v, c)] = abs(complex(history[key][t]))
        l2_f = {v: abs(complex(history[f"L2_f{v}"][t])) for v in vertices}
        l2_b = {v: abs(complex(history[f"L2_b{v}"][t])) for v in vertices}
        I_A = abs(complex(history["I_A"][t]))
        I_B = abs(complex(history["I_B"][t]))

        # Panel 1
        for v in vertices:
            total = sum(l1_amps[(v, c)] for c in arm_combos) / n_combos
            p1_hex_artists[v].set_alpha(0.1 + 0.45 * min(1.0, total))
            for c in arm_combos:
                a = max(0.05, min(1.0, l1_amps[(v, c)]))
                for line in p1_icon_lines[(v, c)]:
                    line.set_alpha(a); line.set_linewidth(0.8 + 1.4 * a)
            artists.append(p1_hex_artists[v])

        # Panel 2
        for v in vertices:
            total = (l2_f[v] + l2_b[v]) / 2
            p2_hex_artists[v].set_alpha(0.1 + 0.45 * min(1.0, total))
            af = max(0.1, min(1.0, l2_f[v])); ab = max(0.1, min(1.0, l2_b[v]))
            p2_front_dots[v].set_alpha(af); p2_front_dots[v].set_sizes([60 + 200 * af])
            p2_back_dots[v].set_alpha(ab);  p2_back_dots[v].set_sizes([60 + 200 * ab])
            artists.extend([p2_hex_artists[v], p2_front_dots[v], p2_back_dots[v]])

        # Panel 3
        a_alpha = 0.15 + 0.85 * min(1.0, I_A)
        b_alpha = 0.15 + 0.85 * min(1.0, I_B)
        for line in p3_a_lines: line.set_alpha(a_alpha * 0.6)
        for line in p3_b_lines: line.set_alpha(b_alpha * 0.6)
        for v in vertices:
            p3_a_dots[v].set_alpha(a_alpha); p3_b_dots[v].set_alpha(b_alpha)
            artists.extend([p3_a_dots[v], p3_b_dots[v]])

        # Panel 4: HSL composite
        for v in vertices:
            l1_acts = [l1_amps[(v, c)] for c in arm_combos]
            h = (int(np.argmax(l1_acts)) / n_combos) if any(a > 0 for a in l1_acts) else 0.0
            s_val = abs(l2_f[v] - l2_b[v]) / (l2_f[v] + l2_b[v] + 1e-3)
            s_val = max(0.15, min(1.0, s_val))
            ll = 0.35 + 0.30 * (I_B / (I_A + I_B + 1e-3))
            rgb = mcolors.hsv_to_rgb([h, s_val, ll])
            p4_dots[v].set_color(rgb)
            p4_dots[v].set_alpha(0.5 + 0.5 * max(0.1, sum(l1_acts) / n_combos))
            artists.append(p4_dots[v])

        suptitle.set_text(f"{title_prefix} — t={t}    I_A={I_A:.2f}  I_B={I_B:.2f}")
        artists.append(suptitle)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    return output_path
