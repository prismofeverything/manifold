"""Phase 10n: unified three-layer hierarchy WITH topology learning at L2.

Combines two threads:
  - 10m: polar_sigmoid throughout — both amplitude AND phase active
  - 10i: L2's intra-layer connectivity learned from co-activation, not
         hand-wired

Architecture changes from 10m:
  - Remove L2's hand-wired cross-tile constraints (the same/different-depth
    excite/inhibit edges that encoded cube topology)
  - Add all-to-all PLASTIC intra-layer connections at L2 (with hebbian)
  - Keep small η/decay so plastic feedback can't run away
  - Keep vertex-local mutual inhibition front⊥back (the local validity
    rule — minimal hand-wired structure)

The cycling signal that drives Hebbian comes from the surrounding
hierarchy: L1's chaotic input + L3's plastic top-down both drive L2 to
oscillate between A and B patterns. The intra-layer Hebbian then captures
which L2 nodes co-activate, building up the cube topology from co-
activation statistics.

Hypothesis: even though L1 is rich/chaotic and L3 starts undifferentiated,
the small bias from random initial top-down + the L1 plastic feedforward
creates enough cycling for L2 plastic intra-layer to learn the cube edge
structure (face same-depth → high; Z-edge cross-depth → high; non-cube
pairs → low).

If it works: full self-organization across all three layers, with phase
signatures throughout.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from manifold import (
    Channel, Constant, Node, Noise, PlasticChannel, WeightNormalizer,
    add_plastic_lateral, hebbian, homeostatic_feedback, polar_sigmoid,
    run, tracker,
)


VERTICES = list(itertools.product([0, 1], repeat=3))
ARM_COMBOS = list(itertools.product([0, 1], repeat=3))

# L1 (rich) — same as 10m
IMAGE_DRIVE_L1 = 0.5
INHIB_INTRATILE_L1 = 1.5
EXCITE_CROSS_L1 = 0.2
INHIB_CROSS_L1 = 0.6
ADAPT_FEEDBACK_L1 = 1.0
RATE_ADAPT_L1 = 0.005
RATE_FAST_L1 = 0.1
GAIN_L1 = 8.0
THRESHOLD_L1 = 2.5
OMEGA_L1 = 0.06
OMEGA_JITTER_L1 = 0.015

# L2 (middle) — NO hand-wired cross-tile; plastic intra-layer instead
IMAGE_DRIVE_L2 = 0.5
INHIB_INTRATILE_L2 = 1.5
ADAPT_FEEDBACK_L2 = 1.5
RATE_FAST_L2 = 0.1
RATE_ADAPT_L2 = 0.005
GAIN_L2 = 8.0
THRESHOLD_L2 = 2.5
OMEGA_L2 = 0.05
OMEGA_JITTER_L2 = 0.015

# L3 (top) — same as 10m
INHIB_HI = 2.0
ADAPT_FEEDBACK_HI = 2.0
RATE_FAST_HI = 0.1
RATE_ADAPT_HI = 0.003
RATE_HOMEO_HI = 0.0003
GAIN_HI = 6.0
THRESHOLD_HI = 1.5
OMEGA_HI = 0.03
HOMEO_TARGET = 0.4
HOMEO_GAIN = 4.0
NOISE_STD = 0.04

# Plasticity
ETA = 0.002
DECAY_PLASTIC = 0.002
TARGET_L1_TO_L2_SUM = 1.0
TARGET_L2_TO_L3_BU_SUM = 2.0
TARGET_L3_TO_L2_TD_SUM = 2.0
L1_TO_L2_GAIN = 1.5
L3_TO_L2_TD_GAIN = 1.0
COUPLING_ALL = 1.0

# L2 intra-layer plasticity (the new piece). η/decay << 1 caps weights
# at small values so the plastic feedback doesn't break L2 bistability.
ETA_L2_INTRA = 0.0005
DECAY_L2_INTRA = 0.003

N_STEPS = 16000
SEED = 42


def neighbor(v, axis):
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def init_random_complex(rng, max_amp=0.3):
    r = max_amp * rng.random()
    th = 2 * np.pi * rng.random()
    return r * np.exp(1j * th)


def main():
    image = Constant(IMAGE_DRIVE_L1)
    image_l2 = Constant(IMAGE_DRIVE_L2)
    rng = np.random.default_rng(SEED)

    # ---- L1 (rich, same as 10m) ----
    l1 = {v: {} for v in VERTICES}
    l1_mems = {v: {} for v in VERTICES}
    for v in VERTICES:
        for combo in ARM_COMBOS:
            omega = OMEGA_L1 + OMEGA_JITTER_L1 * rng.normal()
            f, g = polar_sigmoid(rate=RATE_FAST_L1, gain=GAIN_L1, threshold=THRESHOLD_L1,
                                 omega=omega, coupling=COUPLING_ALL)
            l1[v][combo] = Node(state=init_random_complex(rng), dynamics=f, output=g)
            mf, mg = tracker(rate=RATE_ADAPT_L1)
            l1_mems[v][combo] = Node(state=0 + 0j, dynamics=mf, output=mg)

    for v in VERTICES:
        for combo in ARM_COMBOS:
            d_x, d_y, d_z = combo
            det = l1[v][combo]; mem = l1_mems[v][combo]
            det.add_channel(Channel(image))
            mem.add_channel(Channel(det, transform=lambda y: complex(abs(y), 0)))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK_L1 * y))
            for other in ARM_COMBOS:
                if other == combo: continue
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

    # ---- L2 (middle) — only vertex-local hand-wiring; cross-tile is plastic ----
    front, back = {}, {}
    front_mem, back_mem = {}, {}
    for v in VERTICES:
        omega_f = OMEGA_L2 + OMEGA_JITTER_L2 * rng.normal()
        omega_b = OMEGA_L2 + OMEGA_JITTER_L2 * rng.normal()
        f_f, g_f = polar_sigmoid(rate=RATE_FAST_L2, gain=GAIN_L2, threshold=THRESHOLD_L2,
                                 omega=omega_f, coupling=COUPLING_ALL)
        f_b, g_b = polar_sigmoid(rate=RATE_FAST_L2, gain=GAIN_L2, threshold=THRESHOLD_L2,
                                 omega=omega_b, coupling=COUPLING_ALL)
        front[v] = Node(state=init_random_complex(rng), dynamics=f_f, output=g_f)
        back[v] = Node(state=init_random_complex(rng), dynamics=f_b, output=g_b)
        mf, mg = tracker(rate=RATE_ADAPT_L2)
        front_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)
        back_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)

    learn = hebbian(eta=ETA, decay=DECAY_PLASTIC)
    learn_intra = hebbian(eta=ETA_L2_INTRA, decay=DECAY_L2_INTRA)

    l1_to_l2_chs = {}
    for v in VERTICES:
        for det, sib, mem in [(front[v], back[v], front_mem[v]),
                              (back[v], front[v], back_mem[v])]:
            det.add_channel(Channel(image_l2))
            mem.add_channel(Channel(det, transform=lambda y: complex(abs(y), 0)))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK_L2 * y))
            det.add_channel(Channel(sib, transform=lambda y: -INHIB_INTRATILE_L2 * y))
            ch_list = []
            for combo in ARM_COMBOS:
                ch = PlasticChannel(l1[v][combo], dest=det,
                                    weight=0.05 * rng.random(), learn=learn,
                                    transform=lambda y: L1_TO_L2_GAIN * y)
                det.add_channel(ch)
                ch_list.append(ch)
            l1_to_l2_chs[id(det)] = ch_list
        # NO hand-wired cross-tile constraints at L2 in 10n — those are
        # what we want plastic intra-layer to discover.

    # All-to-all plastic intra-layer at L2 (the new piece)
    upper_nodes = []
    upper_labels = []
    for v in VERTICES:
        upper_nodes.append(front[v]); upper_labels.append((v, "front"))
        upper_nodes.append(back[v]);  upper_labels.append((v, "back"))
    intra_channels = add_plastic_lateral(
        upper_nodes, learn=learn_intra,
        init_weight=0.05, init_random=True, seed=SEED + 3,
    )

    # ---- L3 (top, same as 10m) ----
    f_a, g_a = polar_sigmoid(rate=RATE_FAST_HI, gain=GAIN_HI, threshold=THRESHOLD_HI,
                             omega=OMEGA_HI - 0.005, coupling=COUPLING_ALL)
    f_b, g_b = polar_sigmoid(rate=RATE_FAST_HI, gain=GAIN_HI, threshold=THRESHOLD_HI,
                             omega=OMEGA_HI + 0.005, coupling=COUPLING_ALL)
    I_a = Node(state=0.2 * np.exp(1j * 0.0), dynamics=f_a, output=g_a)
    I_b = Node(state=0.2 * np.exp(1j * np.pi), dynamics=f_b, output=g_b)
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
    Imem_a.add_channel(Channel(I_a, transform=lambda y: complex(abs(y), 0)))
    Imem_b.add_channel(Channel(I_b, transform=lambda y: complex(abs(y), 0)))
    Ihomeo_a.add_channel(Channel(I_a, transform=lambda y: complex(abs(y), 0)))
    Ihomeo_b.add_channel(Channel(I_b, transform=lambda y: complex(abs(y), 0)))

    bu_a, bu_b, td_a, td_b = [], [], [], []
    for v in VERTICES:
        for label, det in [("front", front[v]), ("back", back[v])]:
            ch_bu_a = PlasticChannel(det, dest=I_a, weight=0.05 * rng.random(), learn=learn)
            ch_bu_b = PlasticChannel(det, dest=I_b, weight=0.05 * rng.random(), learn=learn)
            I_a.add_channel(ch_bu_a); I_b.add_channel(ch_bu_b)
            bu_a.append((v, label, ch_bu_a)); bu_b.append((v, label, ch_bu_b))
            ch_td_a = PlasticChannel(I_a, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y: L3_TO_L2_TD_GAIN * y)
            ch_td_b = PlasticChannel(I_b, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y: L3_TO_L2_TD_GAIN * y)
            det.add_channel(ch_td_a); det.add_channel(ch_td_b)
            td_a.append((v, label, ch_td_a)); td_b.append((v, label, ch_td_b))

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
            sources.append(l1[v][combo]); sources.append(l1_mems[v][combo])
        sources.extend([front[v], back[v], front_mem[v], back_mem[v]])
    sources.extend([I_a, I_b, Imem_a, Imem_b, Ihomeo_a, Ihomeo_b, noise_a, noise_b])
    sources.extend(norms)

    obs = {"I_A": lambda: I_a.state, "I_B": lambda: I_b.state}
    for v in VERTICES:
        obs[f"L2_f{v}"] = (lambda v=v: front[v].state)
        obs[f"L2_b{v}"] = (lambda v=v: back[v].state)
        for combo in ARM_COMBOS:
            obs[f"L1_{v}_{combo}"] = (lambda v=v, combo=combo: l1[v][combo].state)

    print("Running 10n (unified + L2 topology learning)...")
    history = run(sources=sources, n_steps=N_STEPS, observers=obs)
    times = list(range(N_STEPS))

    # ---- Analysis: did L2's plastic intra-layer learn cube topology? ----
    label_of = {id(node): lbl for node, lbl in zip(upper_nodes, upper_labels)}
    cats = {
        "same-z same-depth (face — cube edge)": [],
        "same-z opp-depth (cross-face — NOT cube)": [],
        "cross-z same-depth (cross-cube — NOT cube)": [],
        "cross-z opp-depth (Z-edge — cube edge)": [],
    }
    for ch in intra_channels:
        v_src, d_src = label_of[id(ch.source)]
        v_dest, d_dest = label_of[id(ch.dest)]
        if v_src == v_dest:
            continue
        same_z = v_src[2] == v_dest[2]
        same_d = d_src == d_dest
        if same_z and same_d:
            cats["same-z same-depth (face — cube edge)"].append(ch.weight.real)
        elif same_z and not same_d:
            cats["same-z opp-depth (cross-face — NOT cube)"].append(ch.weight.real)
        elif not same_z and same_d:
            cats["cross-z same-depth (cross-cube — NOT cube)"].append(ch.weight.real)
        else:
            cats["cross-z opp-depth (Z-edge — cube edge)"].append(ch.weight.real)

    print()
    print("Final L2 intra-layer weights (mean ± std):")
    for k, ws in cats.items():
        print(f"  {k:48s}  {np.mean(ws):+.4f} ± {np.std(ws):.4f}  (n={len(ws)})")

    # Plot per-pattern amplitudes + L2 weight distribution
    l2_a_keys = [f"L2_f{v}" if v[2] == 0 else f"L2_b{v}" for v in VERTICES]
    l2_b_keys = [f"L2_b{v}" if v[2] == 0 else f"L2_f{v}" for v in VERTICES]
    l2_a_amp = [float(np.mean([abs(complex(history[k][t])) for k in l2_a_keys])) for t in range(N_STEPS)]
    l2_b_amp = [float(np.mean([abs(complex(history[k][t])) for k in l2_b_keys])) for t in range(N_STEPS)]
    L3_A = [abs(complex(history["I_A"][t])) for t in range(N_STEPS)]
    L3_B = [abs(complex(history["I_B"][t])) for t in range(N_STEPS)]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].plot(times, l2_a_amp, color="tab:cyan", linewidth=0.6, label="L2 A subset", alpha=0.8)
    axes[0].plot(times, l2_b_amp, color="tab:red", linewidth=0.6, label="L2 B subset", alpha=0.8)
    axes[0].plot(times, L3_A, color="tab:green", linewidth=1.2, label="L3 I_A")
    axes[0].plot(times, L3_B, color="tab:olive", linewidth=1.2, label="L3 I_B")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("10n — Unified + L2 topology learning: amplitude bistability")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(alpha=0.3)

    cat_names = list(cats.keys())
    means = [np.mean(cats[k]) for k in cat_names]
    stds = [np.std(cats[k]) for k in cat_names]
    colors = ["tab:blue", "tab:gray", "tab:gray", "tab:blue"]
    bars = axes[1].bar(range(len(cat_names)), means, yerr=stds, color=colors,
                       edgecolor="black", alpha=0.75, capsize=6)
    for i, m in enumerate(means):
        axes[1].text(i, m + abs(stds[i]) + 0.001, f"{m:+.4f}", ha="center",
                     fontsize=9, fontweight="bold")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xticks(range(len(cat_names)))
    axes[1].set_xticklabels(cat_names, fontsize=8)
    axes[1].set_ylabel("learned L2 intra-layer weight")
    axes[1].set_title("Did L2 intra-layer plasticity learn cube topology?")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("out/10n_unified_topology_summary.png", dpi=110)
    plt.close(fig)
    print("Saved out/10n_unified_topology_summary.png")

    # Save history for animation generation downstream
    return history


if __name__ == "__main__":
    main()
