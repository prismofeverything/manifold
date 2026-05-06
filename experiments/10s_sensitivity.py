"""Sensitivity analysis on 10n (the furthest-developed system: unified
amplitude+phase, three-layer hierarchy with L2 topology learning).

Sweeps each of several key parameters around its default value, runs the
full network with JAX (so each run is ~3 seconds), and measures three
key metrics:

  1. **L3 alternation period** — how often I_A and I_B switch dominance.
     Computed from sign-changes in (|I_A| - |I_B|) over the trace.
  2. **Cube-edge weight separation** — mean of cube-edge plastic weights
     (face + Z-edge) minus mean of non-cube weights. Positive = topology
     was learned; near-zero = uniform = topology not learned.
  3. **Mean L3 amplitude** — `(|I_A| + |I_B|) / 2` averaged over time.
     Tells us if the system stays "alive" (high) or collapses (near 0).

Plots: one panel per parameter, each showing all three metrics on
shared x-axis (parameter value). Vertical dashed line marks the default.
"""

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np

from manifold import (
    Channel, Constant, Node, Noise, PlasticChannel, WeightNormalizer,
    add_plastic_lateral, hebbian, polar_sigmoid, run_compiled, tracker,
)
from manifold.dynamics import homeostatic_feedback


VERTICES = list(itertools.product([0, 1], repeat=3))
ARM_COMBOS = list(itertools.product([0, 1], repeat=3))


def neighbor(v, axis):
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def init_random_complex(rng, max_amp=0.3):
    r = max_amp * rng.random()
    th = 2 * np.pi * rng.random()
    return r * np.exp(1j * th)


# Defaults — match 10n's slow-frequency variant
DEFAULTS = dict(
    # L1
    IMAGE_DRIVE_L1=0.5,
    INHIB_INTRATILE_L1=1.5,
    EXCITE_CROSS_L1=0.2,
    INHIB_CROSS_L1=0.6,
    ADAPT_FEEDBACK_L1=1.0,
    RATE_ADAPT_L1=0.001,
    RATE_FAST_L1=0.1,
    GAIN_L1=8.0,
    THRESHOLD_L1=2.5,
    OMEGA_L1=0.012,
    OMEGA_JITTER_L1=0.003,
    # L2
    IMAGE_DRIVE_L2=0.5,
    INHIB_INTRATILE_L2=1.5,
    ADAPT_FEEDBACK_L2=1.5,
    RATE_FAST_L2=0.1,
    RATE_ADAPT_L2=0.0008,
    GAIN_L2=8.0,
    THRESHOLD_L2=2.5,
    OMEGA_L2=0.010,
    OMEGA_JITTER_L2=0.003,
    # L3
    INHIB_HI=2.0,
    ADAPT_FEEDBACK_HI=2.0,
    RATE_FAST_HI=0.1,
    RATE_ADAPT_HI=0.0004,
    RATE_HOMEO_HI=0.00004,
    GAIN_HI=6.0,
    THRESHOLD_HI=1.5,
    OMEGA_HI=0.006,
    HOMEO_TARGET=0.4,
    HOMEO_GAIN=4.0,
    NOISE_STD=0.01,
    # Plasticity
    ETA=0.002,
    DECAY_PLASTIC=0.002,
    TARGET_L1_TO_L2_SUM=1.0,
    TARGET_L2_TO_L3_BU_SUM=2.0,
    TARGET_L3_TO_L2_TD_SUM=2.0,
    L1_TO_L2_GAIN=1.5,
    L3_TO_L2_TD_GAIN=1.0,
    COUPLING_ALL=1.0,
    ETA_L2_INTRA=0.0005,
    DECAY_L2_INTRA=0.003,
)


def build_network(p, seed=42):
    """Build 10n's network with parameter overrides p. Returns
    (sources, I_a, I_b, intra_channels)."""
    image = Constant(p["IMAGE_DRIVE_L1"])
    image_l2 = Constant(p["IMAGE_DRIVE_L2"])
    rng = np.random.default_rng(seed)

    # L1
    l1, l1_mems = {v: {} for v in VERTICES}, {v: {} for v in VERTICES}
    for v in VERTICES:
        for combo in ARM_COMBOS:
            omega = p["OMEGA_L1"] + p["OMEGA_JITTER_L1"] * rng.normal()
            f, g = polar_sigmoid(rate=p["RATE_FAST_L1"], gain=p["GAIN_L1"],
                                 threshold=p["THRESHOLD_L1"], omega=omega,
                                 coupling=p["COUPLING_ALL"])
            l1[v][combo] = Node(state=init_random_complex(rng), dynamics=f, output=g)
            mf, mg = tracker(rate=p["RATE_ADAPT_L1"])
            l1_mems[v][combo] = Node(state=0+0j, dynamics=mf, output=mg)

    for v in VERTICES:
        for combo in ARM_COMBOS:
            d_x, d_y, d_z = combo
            det = l1[v][combo]; mem = l1_mems[v][combo]
            det.add_channel(Channel(image))
            mem.add_channel(Channel(det, transform=lambda y: complex(abs(y), 0)))
            det.add_channel(Channel(mem, transform=lambda y, k=p["ADAPT_FEEDBACK_L1"]: -k * y))
            for other in ARM_COMBOS:
                if other == combo: continue
                det.add_channel(Channel(l1[v][other], transform=lambda y, k=p["INHIB_INTRATILE_L1"]: -k * y))
            x_nbr = neighbor(v, 0)
            for c2 in ARM_COMBOS:
                w = p["EXCITE_CROSS_L1"] if c2[0] == d_x else -p["INHIB_CROSS_L1"]
                det.add_channel(Channel(l1[x_nbr][c2], transform=lambda y, w=w: w * y))
            y_nbr = neighbor(v, 1)
            for c2 in ARM_COMBOS:
                w = p["EXCITE_CROSS_L1"] if c2[1] == d_y else -p["INHIB_CROSS_L1"]
                det.add_channel(Channel(l1[y_nbr][c2], transform=lambda y, w=w: w * y))
            z_nbr = neighbor(v, 2)
            for c2 in ARM_COMBOS:
                w = p["EXCITE_CROSS_L1"] if c2[2] == 1 - d_z else -p["INHIB_CROSS_L1"]
                det.add_channel(Channel(l1[z_nbr][c2], transform=lambda y, w=w: w * y))

    # L2
    front, back, front_mem, back_mem = {}, {}, {}, {}
    for v in VERTICES:
        omega_f = p["OMEGA_L2"] + p["OMEGA_JITTER_L2"] * rng.normal()
        omega_b = p["OMEGA_L2"] + p["OMEGA_JITTER_L2"] * rng.normal()
        f_f, g_f = polar_sigmoid(rate=p["RATE_FAST_L2"], gain=p["GAIN_L2"],
                                 threshold=p["THRESHOLD_L2"], omega=omega_f,
                                 coupling=p["COUPLING_ALL"])
        f_b, g_b = polar_sigmoid(rate=p["RATE_FAST_L2"], gain=p["GAIN_L2"],
                                 threshold=p["THRESHOLD_L2"], omega=omega_b,
                                 coupling=p["COUPLING_ALL"])
        front[v] = Node(state=init_random_complex(rng), dynamics=f_f, output=g_f)
        back[v] = Node(state=init_random_complex(rng), dynamics=f_b, output=g_b)
        mf, mg = tracker(rate=p["RATE_ADAPT_L2"])
        front_mem[v] = Node(state=0+0j, dynamics=mf, output=mg)
        back_mem[v] = Node(state=0+0j, dynamics=mf, output=mg)

    learn = hebbian(eta=p["ETA"], decay=p["DECAY_PLASTIC"])
    learn_intra = hebbian(eta=p["ETA_L2_INTRA"], decay=p["DECAY_L2_INTRA"])

    l1_to_l2_chs = {}
    for v in VERTICES:
        for det, sib, mem in [(front[v], back[v], front_mem[v]),
                              (back[v], front[v], back_mem[v])]:
            det.add_channel(Channel(image_l2))
            mem.add_channel(Channel(det, transform=lambda y: complex(abs(y), 0)))
            det.add_channel(Channel(mem, transform=lambda y, k=p["ADAPT_FEEDBACK_L2"]: -k * y))
            det.add_channel(Channel(sib, transform=lambda y, k=p["INHIB_INTRATILE_L2"]: -k * y))
            ch_list = []
            for combo in ARM_COMBOS:
                ch = PlasticChannel(l1[v][combo], dest=det,
                                    weight=0.05 * rng.random(), learn=learn,
                                    transform=lambda y, k=p["L1_TO_L2_GAIN"]: k * y)
                det.add_channel(ch)
                ch_list.append(ch)
            l1_to_l2_chs[id(det)] = ch_list

    upper_nodes = []
    upper_labels = []
    for v in VERTICES:
        upper_nodes.append(front[v]); upper_labels.append((v, "front"))
        upper_nodes.append(back[v]);  upper_labels.append((v, "back"))
    intra_channels = add_plastic_lateral(
        upper_nodes, learn=learn_intra,
        init_weight=0.05, init_random=True, seed=seed + 3,
    )

    # L3
    f_a, g_a = polar_sigmoid(rate=p["RATE_FAST_HI"], gain=p["GAIN_HI"],
                             threshold=p["THRESHOLD_HI"], omega=p["OMEGA_HI"] - 0.005,
                             coupling=p["COUPLING_ALL"])
    f_b, g_b = polar_sigmoid(rate=p["RATE_FAST_HI"], gain=p["GAIN_HI"],
                             threshold=p["THRESHOLD_HI"], omega=p["OMEGA_HI"] + 0.005,
                             coupling=p["COUPLING_ALL"])
    I_a = Node(state=0.2 * np.exp(1j * 0.0), dynamics=f_a, output=g_a)
    I_b = Node(state=0.2 * np.exp(1j * np.pi), dynamics=f_b, output=g_b)
    fmem_a, gmem_a = tracker(rate=p["RATE_ADAPT_HI"])
    fmem_b, gmem_b = tracker(rate=p["RATE_ADAPT_HI"])
    Imem_a = Node(state=0+0j, dynamics=fmem_a, output=gmem_a)
    Imem_b = Node(state=0+0j, dynamics=fmem_b, output=gmem_b)
    fhomeo_a, ghomeo_a = tracker(rate=p["RATE_HOMEO_HI"])
    fhomeo_b, ghomeo_b = tracker(rate=p["RATE_HOMEO_HI"])
    Ihomeo_a = Node(state=p["HOMEO_TARGET"]+0j, dynamics=fhomeo_a, output=ghomeo_a)
    Ihomeo_b = Node(state=p["HOMEO_TARGET"]+0j, dynamics=fhomeo_b, output=ghomeo_b)
    noise_a = Noise(std=p["NOISE_STD"], seed=seed + 1)
    noise_b = Noise(std=p["NOISE_STD"], seed=seed + 2)

    homeo_xform = homeostatic_feedback(target=p["HOMEO_TARGET"], gain=p["HOMEO_GAIN"])
    I_a.add_channel(Channel(I_b, transform=lambda y, k=p["INHIB_HI"]: -k * y))
    I_a.add_channel(Channel(Imem_a, transform=lambda y, k=p["ADAPT_FEEDBACK_HI"]: -k * y))
    I_a.add_channel(Channel(Ihomeo_a, transform=homeo_xform))
    I_a.add_channel(Channel(noise_a))
    I_b.add_channel(Channel(I_a, transform=lambda y, k=p["INHIB_HI"]: -k * y))
    I_b.add_channel(Channel(Imem_b, transform=lambda y, k=p["ADAPT_FEEDBACK_HI"]: -k * y))
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
                                     transform=lambda y, k=p["L3_TO_L2_TD_GAIN"]: k * y)
            ch_td_b = PlasticChannel(I_b, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y, k=p["L3_TO_L2_TD_GAIN"]: k * y)
            det.add_channel(ch_td_a); det.add_channel(ch_td_b)
            td_a.append((v, label, ch_td_a)); td_b.append((v, label, ch_td_b))

    norms = []
    for v in VERTICES:
        for det in (front[v], back[v]):
            norms.append(WeightNormalizer(l1_to_l2_chs[id(det)], target_sum=p["TARGET_L1_TO_L2_SUM"]))
    norms.append(WeightNormalizer([ch for _, _, ch in bu_a], target_sum=p["TARGET_L2_TO_L3_BU_SUM"]))
    norms.append(WeightNormalizer([ch for _, _, ch in bu_b], target_sum=p["TARGET_L2_TO_L3_BU_SUM"]))
    norms.append(WeightNormalizer([ch for _, _, ch in td_a], target_sum=p["TARGET_L3_TO_L2_TD_SUM"]))
    norms.append(WeightNormalizer([ch for _, _, ch in td_b], target_sum=p["TARGET_L3_TO_L2_TD_SUM"]))

    sources = []
    for v in VERTICES:
        for combo in ARM_COMBOS:
            sources.append(l1[v][combo]); sources.append(l1_mems[v][combo])
        sources.extend([front[v], back[v], front_mem[v], back_mem[v]])
    sources.extend([I_a, I_b, Imem_a, Imem_b, Ihomeo_a, Ihomeo_b, noise_a, noise_b])
    sources.extend(norms)

    return sources, I_a, I_b, intra_channels, upper_nodes, upper_labels


def measure(history_array, node_to_idx, I_a, I_b, intra_channels, upper_nodes, upper_labels):
    i_a_idx = node_to_idx[id(I_a)]
    i_b_idx = node_to_idx[id(I_b)]
    a_amp = np.abs(history_array[:, i_a_idx])
    b_amp = np.abs(history_array[:, i_b_idx])

    # Alternation period from sign-changes of (a_amp - b_amp)
    diff = a_amp - b_amp
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) >= 2:
        period = float(2 * np.mean(np.diff(sign_changes)))
    else:
        period = float("nan")

    # Mean amplitude (system is alive when this is high)
    mean_amp = float((a_amp.mean() + b_amp.mean()) / 2)

    # Cube edge separation from final plastic weights
    label_of = {id(n): lbl for n, lbl in zip(upper_nodes, upper_labels)}
    cube_w, non_cube_w = [], []
    for ch in intra_channels:
        v_src, d_src = label_of[id(ch.source)]
        v_dest, d_dest = label_of[id(ch.dest)]
        if v_src == v_dest:
            continue
        same_z = v_src[2] == v_dest[2]
        same_d = d_src == d_dest
        is_cube_edge = (same_z and same_d) or (not same_z and not same_d)
        target = cube_w if is_cube_edge else non_cube_w
        target.append(ch.weight.real)
    separation = float(np.mean(cube_w) - np.mean(non_cube_w)) if cube_w and non_cube_w else float("nan")

    return {"period": period, "mean_amp": mean_amp, "separation": separation}


def run_one(params, n_steps=10000, seed=42):
    full_params = {**DEFAULTS, **params}
    sources, I_a, I_b, intra_channels, upper_nodes, upper_labels = build_network(full_params, seed=seed)
    history_array, node_to_idx, _ = run_compiled(sources, n_steps=n_steps, seed=seed, writeback=True)
    return measure(history_array, node_to_idx, I_a, I_b, intra_channels, upper_nodes, upper_labels)


def main():
    n_steps = 10000
    sweeps = {
        "RATE_ADAPT_HI":     ("log",    np.geomspace(5e-5, 5e-3, 8)),
        "ADAPT_FEEDBACK_HI": ("linear", np.linspace(0.5, 4.0, 8)),
        "INHIB_HI":          ("linear", np.linspace(0.5, 4.0, 8)),
        "NOISE_STD":         ("log",    np.geomspace(0.001, 0.1, 8)),
        "L1_TO_L2_GAIN":     ("linear", np.linspace(0.0, 3.0, 8)),
        "L3_TO_L2_TD_GAIN":  ("linear", np.linspace(0.0, 3.0, 8)),
        "ETA_L2_INTRA":      ("log",    np.geomspace(1e-4, 5e-3, 8)),
        "OMEGA_HI":          ("linear", np.linspace(0.001, 0.05, 8)),
    }

    print(f"Running sensitivity sweep — {sum(len(vs) for _, vs in sweeps.values())} runs total")
    results = {}
    for param_name, (scale, values) in sweeps.items():
        print(f"\nSweeping {param_name} ({scale}):")
        runs = []
        for v in values:
            t0 = time.time()
            metrics = run_one({param_name: float(v)}, n_steps=n_steps)
            dt = time.time() - t0
            runs.append((float(v), metrics))
            print(f"  {v:.5g}: period={metrics['period']:>7.0f}  "
                  f"sep={metrics['separation']:+.4f}  amp={metrics['mean_amp']:.3f}  "
                  f"({dt:.1f}s)")
        results[param_name] = (scale, runs)

    # Plot: 4×2 grid, one panel per parameter, three metrics on twinned axes
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for ax, (name, (scale, runs)) in zip(axes.flat, results.items()):
        xs = np.array([v for v, _ in runs])
        periods = np.array([r["period"] for _, r in runs])
        seps = np.array([r["separation"] for _, r in runs])
        amps = np.array([r["mean_amp"] for _, r in runs])

        ax.plot(xs, periods, "o-", color="tab:blue", label="alternation period (steps)")
        ax.set_xlabel(name)
        ax.set_ylabel("period (steps)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        if scale == "log":
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.axvline(DEFAULTS[name], color="black", linestyle=":", alpha=0.4, label="default")

        ax2 = ax.twinx()
        ax2.plot(xs, seps, "s-", color="tab:red", label="cube-edge separation")
        ax2.plot(xs, amps, "^-", color="tab:green", label="mean L3 amp")
        ax2.set_ylabel("separation (red) / amplitude (green)")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Combined legend in upper-right
        h1, l1_ = ax.get_legend_handles_labels()
        h2, l2_ = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1_ + l2_, fontsize=7, loc="upper right")
        ax.set_title(name, fontsize=10)

    fig.suptitle("10s — Sensitivity analysis on 10n parameters\n"
                 "(period: blue, cube-edge separation: red, mean amplitude: green)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("out/10s_sensitivity.png", dpi=110)
    plt.close(fig)
    print("\nSaved out/10s_sensitivity.png")


if __name__ == "__main__":
    main()
