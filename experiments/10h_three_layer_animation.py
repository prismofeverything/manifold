"""Animate the three-layer hierarchy (10g) with feature icons.

Four panels showing the layers and their activity over time:

  Panel 1 (L1 rich):  8 hex tiles at the 2D Necker projection of each
                      cube vertex. Inside each hex: 8 small 3-arm
                      corner icons, one per (X-arm, Y-arm, Z-arm) depth
                      combo. Each arm drawn as a line segment colored
                      by depth (blue = 0/front, red = 1/back). Active
                      detectors brighten; tile background opacity tracks
                      total activity at that vertex.

  Panel 2 (L2 middle): Same hex layout. Each tile holds two markers,
                      'front' (blue) and 'back' (red). Marker size +
                      opacity track activity.

  Panel 3 (L3 top):   Two cube wireframes — interpretation A and B.
                      Each cube's overall opacity tracks the I node's
                      activity. The vertices in each cube are colored
                      by their depth assignment under that interpretation.

  Panel 4 (HSL view): A single cube wireframe with each vertex shown
                      in HSL where:
                        H (hue)        = winning L1 detector at vertex
                        S (saturation) = L2 front/back contrast
                        L (lightness)  = which L3 interpretation wins
"""

import itertools

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from manifold import (
    Channel, Constant, Node, Noise, PlasticChannel, WeightNormalizer,
    hebbian, homeostatic_feedback, run, sigmoid_activity, tracker,
)


VERTICES = list(itertools.product([0, 1], repeat=3))
ARM_COMBOS = list(itertools.product([0, 1], repeat=3))

# Reuse parameters from 10g (the working three-layer setup)
IMAGE_DRIVE_L1 = 0.5
INHIB_INTRATILE_L1 = 1.5
EXCITE_CROSS_L1 = 0.2
INHIB_CROSS_L1 = 0.6
ADAPT_FEEDBACK_L1 = 1.0
RATE_ADAPT_L1 = 0.005
RATE_FAST_L1 = 0.1
GAIN_L1 = 8.0
THRESHOLD_L1 = 2.5

IMAGE_DRIVE_L2 = 0.5
INHIB_INTRATILE_L2 = 1.5
EXCITE_CROSS_L2 = 0.4
INHIB_CROSS_L2 = 0.4
ADAPT_FEEDBACK_L2 = 1.5
RATE_FAST_L2 = 0.1
RATE_ADAPT_L2 = 0.005
GAIN_L2 = 8.0
THRESHOLD_L2 = 2.5

INHIB_HI = 2.0
ADAPT_FEEDBACK_HI = 2.0
RATE_FAST_HI = 0.1
RATE_ADAPT_HI = 0.003
RATE_HOMEO_HI = 0.0003
GAIN_HI = 6.0
THRESHOLD_HI = 1.5
HOMEO_TARGET = 0.4
HOMEO_GAIN = 4.0
NOISE_STD = 0.04

ETA = 0.002
DECAY_PLASTIC = 0.002
TARGET_L1_TO_L2_SUM = 1.0
TARGET_L2_TO_L3_BU_SUM = 2.0
TARGET_L3_TO_L2_TD_SUM = 2.0
L1_TO_L2_GAIN = 1.5
L3_TO_L2_TD_GAIN = 1.0

N_STEPS = 8000      # shorter for animation speed
SAMPLE_EVERY = 25
SEED = 42


# Layout helpers
def cube_2d(v):
    """Standard Necker cube projection: z=1 face offset up-right of z=0."""
    x, y, z = v
    return (x + z * 0.45, y + z * 0.35)


def neighbor(v, axis):
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def hex_corners(center, radius):
    """6 corners of a regular hexagon (flat-top)."""
    angles = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
    return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]


def arm_2d_directions(v):
    """For vertex v, return 2D unit vectors along its X, Y, Z arms."""
    here = np.array(cube_2d(v))
    dirs = []
    for axis in (0, 1, 2):
        n = neighbor(v, axis)
        d = np.array(cube_2d(n)) - here
        dirs.append(d / (np.linalg.norm(d) + 1e-12))
    return dirs


# Build and run the three-layer network
def build_and_run():
    image = Constant(IMAGE_DRIVE_L1)
    image_l2 = Constant(IMAGE_DRIVE_L2)
    rng = np.random.default_rng(SEED)

    l1 = {v: {} for v in VERTICES}
    l1_mems = {v: {} for v in VERTICES}
    for v in VERTICES:
        for combo in ARM_COMBOS:
            f, g = sigmoid_activity(rate=RATE_FAST_L1, gain=GAIN_L1, threshold=THRESHOLD_L1)
            l1[v][combo] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
            mf, mg = tracker(rate=RATE_ADAPT_L1)
            l1_mems[v][combo] = Node(state=0 + 0j, dynamics=mf, output=mg)

    for v in VERTICES:
        for combo in ARM_COMBOS:
            d_x, d_y, d_z = combo
            det = l1[v][combo]
            mem = l1_mems[v][combo]
            det.add_channel(Channel(image))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK_L1 * y))
            mem.add_channel(Channel(det))
            for other in ARM_COMBOS:
                if other == combo:
                    continue
                det.add_channel(Channel(l1[v][other], transform=lambda y: -INHIB_INTRATILE_L1 * y))
            x_nbr = neighbor(v, 0)
            for c2 in ARM_COMBOS:
                w = EXCITE_CROSS_L1 if c2[0] == d_x else -INHIB_CROSS_L1
                det.add_channel(Channel(l1[x_nbr][c2], transform=lambda y, w=w: w * y))
            y_nbr = neighbor(v, 1)
            for c2 in ARM_COMBOS:
                w = EXCITE_CROSS_L1 if c2[1] == d_y else -INHIB_CROSS_L1
                det.add_channel(Channel(l1[y_nbr][c2], transform=lambda y, w=w: w * y))
            z_nbr = neighbor(v, 2)
            for c2 in ARM_COMBOS:
                w = EXCITE_CROSS_L1 if c2[2] == 1 - d_z else -INHIB_CROSS_L1
                det.add_channel(Channel(l1[z_nbr][c2], transform=lambda y, w=w: w * y))

    front, back = {}, {}
    front_mem, back_mem = {}, {}
    for v in VERTICES:
        f, g = sigmoid_activity(rate=RATE_FAST_L2, gain=GAIN_L2, threshold=THRESHOLD_L2)
        front[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        back[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        mf, mg = tracker(rate=RATE_ADAPT_L2)
        front_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)
        back_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)

    learn = hebbian(eta=ETA, decay=DECAY_PLASTIC)
    l1_to_l2_chs = {}
    for v in VERTICES:
        for det, sib, mem in [(front[v], back[v], front_mem[v]),
                              (back[v], front[v], back_mem[v])]:
            det.add_channel(Channel(image_l2))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK_L2 * y))
            mem.add_channel(Channel(det))
            det.add_channel(Channel(sib, transform=lambda y: -INHIB_INTRATILE_L2 * y))
            ch_list = []
            for combo in ARM_COMBOS:
                ch = PlasticChannel(l1[v][combo], dest=det,
                                    weight=0.05 * rng.random(), learn=learn,
                                    transform=lambda y: L1_TO_L2_GAIN * y)
                det.add_channel(ch)
                ch_list.append(ch)
            l1_to_l2_chs[id(det)] = ch_list
        for axis in (0, 1, 2):
            n = neighbor(v, axis)
            if axis in (0, 1):
                front[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS_L2 * y))
                front[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS_L2 * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS_L2 * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS_L2 * y))
            else:
                front[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS_L2 * y))
                front[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS_L2 * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS_L2 * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS_L2 * y))

    fI_a, gI_a = sigmoid_activity(rate=RATE_FAST_HI, gain=GAIN_HI, threshold=THRESHOLD_HI)
    fI_b, gI_b = sigmoid_activity(rate=RATE_FAST_HI, gain=GAIN_HI, threshold=THRESHOLD_HI)
    I_a = Node(state=0.2 + 0j, dynamics=fI_a, output=gI_a)
    I_b = Node(state=0.2 + 0j, dynamics=fI_b, output=gI_b)
    fmem_a, gmem_a = tracker(rate=RATE_ADAPT_HI)
    fmem_b, gmem_b = tracker(rate=RATE_ADAPT_HI)
    Imem_a = Node(state=0 + 0j, dynamics=fmem_a, output=gmem_a)
    Imem_b = Node(state=0 + 0j, dynamics=fmem_b, output=gmem_b)
    fhomeo_a, ghomeo_a = tracker(rate=RATE_HOMEO_HI)
    fhomeo_b, ghomeo_b = tracker(rate=RATE_HOMEO_HI)
    Ihomeo_a = Node(state=HOMEO_TARGET + 0j, dynamics=fhomeo_a, output=ghomeo_a)
    Ihomeo_b = Node(state=HOMEO_TARGET + 0j, dynamics=fhomeo_b, output=ghomeo_b)
    noise_a = Noise(std=NOISE_STD, seed=SEED + 1)
    noise_b = Noise(std=NOISE_STD, seed=SEED + 2)

    homeo_xform = homeostatic_feedback(target=HOMEO_TARGET, gain=HOMEO_GAIN)
    I_a.add_channel(Channel(I_b, transform=lambda y: -INHIB_HI * y))
    I_a.add_channel(Channel(Imem_a, transform=lambda y: -ADAPT_FEEDBACK_HI * y))
    I_a.add_channel(Channel(Ihomeo_a, transform=homeo_xform))
    I_a.add_channel(Channel(noise_a))
    I_b.add_channel(Channel(I_a, transform=lambda y: -INHIB_HI * y))
    I_b.add_channel(Channel(Imem_b, transform=lambda y: -ADAPT_FEEDBACK_HI * y))
    I_b.add_channel(Channel(Ihomeo_b, transform=homeo_xform))
    I_b.add_channel(Channel(noise_b))
    Imem_a.add_channel(Channel(I_a))
    Imem_b.add_channel(Channel(I_b))
    Ihomeo_a.add_channel(Channel(I_a))
    Ihomeo_b.add_channel(Channel(I_b))

    bu_a, bu_b, td_a, td_b = [], [], [], []
    for v in VERTICES:
        for label, det in [("front", front[v]), ("back", back[v])]:
            ch_bu_a = PlasticChannel(det, dest=I_a, weight=0.05 * rng.random(), learn=learn)
            ch_bu_b = PlasticChannel(det, dest=I_b, weight=0.05 * rng.random(), learn=learn)
            I_a.add_channel(ch_bu_a)
            I_b.add_channel(ch_bu_b)
            bu_a.append((v, label, ch_bu_a))
            bu_b.append((v, label, ch_bu_b))
            ch_td_a = PlasticChannel(I_a, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y: L3_TO_L2_TD_GAIN * y)
            ch_td_b = PlasticChannel(I_b, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y: L3_TO_L2_TD_GAIN * y)
            det.add_channel(ch_td_a)
            det.add_channel(ch_td_b)
            td_a.append((v, label, ch_td_a))
            td_b.append((v, label, ch_td_b))

    norms = []
    for v in VERTICES:
        for det in (front[v], back[v]):
            norms.append(WeightNormalizer(l1_to_l2_chs[id(det)], target_sum=TARGET_L1_TO_L2_SUM))
    norms.append(WeightNormalizer([ch for _, _, ch in bu_a], target_sum=TARGET_L2_TO_L3_BU_SUM))
    norms.append(WeightNormalizer([ch for _, _, ch in bu_b], target_sum=TARGET_L2_TO_L3_BU_SUM))
    norms.append(WeightNormalizer([ch for _, _, ch in td_a], target_sum=TARGET_L3_TO_L2_TD_SUM))
    norms.append(WeightNormalizer([ch for _, _, ch in td_b], target_sum=TARGET_L3_TO_L2_TD_SUM))

    sources = []
    for v in VERTICES:
        for combo in ARM_COMBOS:
            sources.append(l1[v][combo])
            sources.append(l1_mems[v][combo])
        sources.extend([front[v], back[v], front_mem[v], back_mem[v]])
    sources.extend([I_a, I_b, Imem_a, Imem_b, Ihomeo_a, Ihomeo_b, noise_a, noise_b])
    sources.extend(norms)

    obs = {
        "I_A": lambda: I_a.state.real,
        "I_B": lambda: I_b.state.real,
    }
    for v in VERTICES:
        obs[f"L2_f{v}"] = (lambda v=v: front[v].state.real)
        obs[f"L2_b{v}"] = (lambda v=v: back[v].state.real)
        for combo in ARM_COMBOS:
            obs[f"L1_{v}_{combo}"] = (lambda v=v, combo=combo: l1[v][combo].state.real)

    print("Running simulation...")
    return run(sources=sources, n_steps=N_STEPS, observers=obs)


# Drawing helpers
def draw_corner_icon(ax, center, arm_dirs, depths, scale=0.04, alpha=1.0, lw=1.5):
    """3-arm corner icon. Each arm is a line segment; color by depth."""
    artists = []
    for direction, d in zip(arm_dirs, depths):
        end = (center[0] + scale * direction[0], center[1] + scale * direction[1])
        col = "#1f77b4" if d == 0 else "#d62728"  # blue=front, red=back
        line, = ax.plot(
            [center[0], end[0]], [center[1], end[1]],
            color=col, alpha=alpha, linewidth=lw, solid_capstyle="round",
        )
        artists.append(line)
    return artists


def winning_combo_hue(activities_by_combo):
    """Map the winning ARM_COMBO index to a hue in [0,1]."""
    if not activities_by_combo:
        return 0.0
    idx = int(np.argmax(activities_by_combo))
    return idx / len(ARM_COMBOS)


# Edges of the cube for wireframe drawing
EDGES = (
    [((0, y, z), (1, y, z)) for y in (0, 1) for z in (0, 1)]
    + [((x, 0, z), (x, 1, z)) for x in (0, 1) for z in (0, 1)]
    + [((x, y, 0), (x, y, 1)) for x in (0, 1) for y in (0, 1)]
)


def main():
    history = build_and_run()
    print("Building animation...")

    frames = list(range(0, N_STEPS, SAMPLE_EVERY))

    # Pre-compute layouts. The four panels share screen space — each gets a
    # 2D coordinate frame centered around its content.
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_aspect("equal")
        ax.axis("off")

    ax1.set_title("L1 — rich features (3-arm corners × depth combos per vertex)")
    ax2.set_title("L2 — middle layer (front / back per vertex)")
    ax3.set_title("L3 — top layer (two interpretations, opacity = lock-in)")
    ax4.set_title("Composite (HSL): hue=L1 winner, sat=L2 contrast, light=L3 winner")

    vertex_pos = {v: cube_2d(v) for v in VERTICES}

    # ---- Panel 1 setup: hex tile + 8 corner icons per vertex ----
    HEX_R = 0.36
    ICON_R = 0.20  # radius from tile center to icon center
    p1_hex_artists = {}
    p1_icon_lines = {}  # per (vertex, combo) -> list of line objects

    for v in VERTICES:
        center = vertex_pos[v]
        # Hex background
        hex_poly = mpatches.Polygon(hex_corners(center, HEX_R), closed=True,
                                    facecolor="lightgray", edgecolor="black",
                                    alpha=0.2, linewidth=0.5)
        ax1.add_patch(hex_poly)
        p1_hex_artists[v] = hex_poly

        # 8 corner icons arranged in a circle inside the hex
        arm_dirs = arm_2d_directions(v)
        for i, combo in enumerate(ARM_COMBOS):
            angle = 2 * np.pi * i / 8
            ic = (center[0] + ICON_R * np.cos(angle), center[1] + ICON_R * np.sin(angle))
            lines = draw_corner_icon(ax1, ic, arm_dirs, combo,
                                     scale=0.07, alpha=0.15, lw=1.2)
            p1_icon_lines[(v, combo)] = lines

    ax1.set_xlim(-0.6, 2.0)
    ax1.set_ylim(-0.6, 1.9)

    # ---- Panel 2 setup: hex tile + front/back markers per vertex ----
    p2_hex_artists = {}
    p2_front_dots = {}
    p2_back_dots = {}
    for v in VERTICES:
        center = vertex_pos[v]
        hex_poly = mpatches.Polygon(hex_corners(center, HEX_R), closed=True,
                                    facecolor="lightgray", edgecolor="black",
                                    alpha=0.2, linewidth=0.5)
        ax2.add_patch(hex_poly)
        p2_hex_artists[v] = hex_poly
        # front marker on left, back on right
        f_dot = ax2.scatter(center[0] - 0.10, center[1], s=80, c="#1f77b4", alpha=0.3)
        b_dot = ax2.scatter(center[0] + 0.10, center[1], s=80, c="#d62728", alpha=0.3)
        p2_front_dots[v] = f_dot
        p2_back_dots[v] = b_dot

    ax2.set_xlim(-0.6, 2.0)
    ax2.set_ylim(-0.6, 1.9)

    # ---- Panel 3 setup: two cubes, each showing one interpretation ----
    # Lay out: cube A on left, cube B on right
    cube_a_offset = (0.0, 0.0)
    cube_b_offset = (2.2, 0.0)

    def cube_pos_offset(v, offset):
        p = cube_2d(v)
        return (p[0] + offset[0], p[1] + offset[1])

    p3_a_lines = []
    p3_b_lines = []
    p3_a_dots = {}
    p3_b_dots = {}

    for u, vv in EDGES:
        line_a, = ax3.plot(*zip(cube_pos_offset(u, cube_a_offset), cube_pos_offset(vv, cube_a_offset)),
                           color="black", alpha=0.3, linewidth=1.2)
        line_b, = ax3.plot(*zip(cube_pos_offset(u, cube_b_offset), cube_pos_offset(vv, cube_b_offset)),
                           color="black", alpha=0.3, linewidth=1.2)
        p3_a_lines.append(line_a)
        p3_b_lines.append(line_b)

    for v in VERTICES:
        # Under interpretation A (z=0 front): z=0 vertices blue (front), z=1 red (back)
        col_a = "#1f77b4" if v[2] == 0 else "#d62728"
        col_b = "#d62728" if v[2] == 0 else "#1f77b4"
        pa = cube_pos_offset(v, cube_a_offset)
        pb = cube_pos_offset(v, cube_b_offset)
        p3_a_dots[v] = ax3.scatter([pa[0]], [pa[1]], s=120, c=col_a, alpha=0.3, zorder=5)
        p3_b_dots[v] = ax3.scatter([pb[0]], [pb[1]], s=120, c=col_b, alpha=0.3, zorder=5)

    ax3.text(cube_a_offset[0] + 0.7, -0.5, "Interp A (z=0 front)", ha="center", fontsize=10)
    ax3.text(cube_b_offset[0] + 0.7, -0.5, "Interp B (z=0 back)",  ha="center", fontsize=10)
    ax3.set_xlim(-0.6, 4.4)
    ax3.set_ylim(-0.7, 1.9)

    # ---- Panel 4 setup: cube wireframe with HSL vertex colors ----
    p4_lines = []
    for u, vv in EDGES:
        line, = ax4.plot(*zip(cube_2d(u), cube_2d(vv)), color="black", alpha=0.3, linewidth=1.2)
        p4_lines.append(line)
    p4_dots = {v: ax4.scatter([cube_2d(v)[0]], [cube_2d(v)[1]], s=200, c="gray", alpha=0.6, zorder=5)
               for v in VERTICES}
    ax4.set_xlim(-0.6, 2.0)
    ax4.set_ylim(-0.6, 1.9)

    suptitle = fig.suptitle("Three-layer Necker cube — t=0", fontsize=14)

    # ---- Animate ----
    def update(frame_idx):
        t = frames[frame_idx]
        all_artists = []

        # Per-vertex aggregates
        l1_act_by_v_combo = {(v, c): history[f"L1_{v}_{c}"][t] for v in VERTICES for c in ARM_COMBOS}
        l2_f = {v: history[f"L2_f{v}"][t] for v in VERTICES}
        l2_b = {v: history[f"L2_b{v}"][t] for v in VERTICES}
        I_A = history["I_A"][t]
        I_B = history["I_B"][t]

        # Panel 1: hex bgs by total L1 activity at vertex; icons by individual activity
        for v in VERTICES:
            total = sum(l1_act_by_v_combo[(v, c)] for c in ARM_COMBOS) / len(ARM_COMBOS)
            p1_hex_artists[v].set_alpha(0.1 + 0.45 * min(1.0, total))
            for c in ARM_COMBOS:
                a = max(0.05, min(1.0, l1_act_by_v_combo[(v, c)]))
                for line in p1_icon_lines[(v, c)]:
                    line.set_alpha(a)
                    line.set_linewidth(0.8 + 1.4 * a)
            all_artists.append(p1_hex_artists[v])

        # Panel 2: hex bg by total L2 activity; markers by individual activity
        for v in VERTICES:
            total = (l2_f[v] + l2_b[v]) / 2
            p2_hex_artists[v].set_alpha(0.1 + 0.45 * min(1.0, total))
            af = max(0.1, min(1.0, l2_f[v]))
            ab = max(0.1, min(1.0, l2_b[v]))
            p2_front_dots[v].set_alpha(af)
            p2_front_dots[v].set_sizes([60 + 200 * af])
            p2_back_dots[v].set_alpha(ab)
            p2_back_dots[v].set_sizes([60 + 200 * ab])
            all_artists.extend([p2_hex_artists[v], p2_front_dots[v], p2_back_dots[v]])

        # Panel 3: cube A opacity = I_A; cube B = I_B
        a_alpha = 0.15 + 0.85 * I_A
        b_alpha = 0.15 + 0.85 * I_B
        for line in p3_a_lines:
            line.set_alpha(a_alpha * 0.6)
        for line in p3_b_lines:
            line.set_alpha(b_alpha * 0.6)
        for v in VERTICES:
            p3_a_dots[v].set_alpha(a_alpha)
            p3_b_dots[v].set_alpha(b_alpha)
            all_artists.extend([p3_a_dots[v], p3_b_dots[v]])

        # Panel 4: HSL composite per vertex
        for v in VERTICES:
            l1_acts = [l1_act_by_v_combo[(v, c)] for c in ARM_COMBOS]
            h = winning_combo_hue(l1_acts)
            # saturation: how strong the L2 front-vs-back contrast is
            s_val = abs(l2_f[v] - l2_b[v]) / (l2_f[v] + l2_b[v] + 1e-3)
            s_val = max(0.15, min(1.0, s_val))
            # lightness: which top interpretation wins (0.35 if I_A wins, 0.65 if I_B wins)
            ll = 0.35 + 0.30 * (I_B / (I_A + I_B + 1e-3))
            rgb = mcolors.hsv_to_rgb([h, s_val, ll])
            p4_dots[v].set_color(rgb)
            p4_dots[v].set_alpha(0.5 + 0.5 * max(0.1, sum(l1_acts) / len(l1_acts)))
            all_artists.append(p4_dots[v])

        suptitle.set_text(
            f"Three-layer Necker cube — t={t}    I_A={I_A:.2f}  I_B={I_B:.2f}"
        )
        all_artists.append(suptitle)
        return all_artists

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=60, blit=False)
    print("Saving GIF...")
    anim.save("out/10h_three_layer_animation.gif", writer="pillow", fps=15)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
