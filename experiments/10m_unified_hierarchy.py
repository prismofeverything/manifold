"""Phase 10m: three-layer hierarchy with both amplitude AND phase at every layer.

The 10g hierarchy used `sigmoid_activity` (amplitude only) at all three
layers. This experiment replaces every node-class with `polar_sigmoid`
so each node has both axes of complex state actively evolving:

  - amplitude: bistability + competition (same engine as 10g)
  - phase: intrinsic rotation + Kuramoto coupling on the same channels
           (positive transforms → in-phase, negative → anti-phase)

Wiring is unchanged from 10g — same cross-tile constraints, same
adaptation memories, same plastic L1→L2 and L2↔L3 channels, same
homeostatic + noise mechanisms at L3. The only changes are:

  1. Replace `sigmoid_activity(...)` with `polar_sigmoid(... omega=ω, coupling=1)`
  2. Random initial phases for every node
  3. Heterogeneous ω per node (small Gaussian jitter)

What we measure: amplitude bistability at each layer (same as 10g) plus
phase signatures — within-subset coherence R, and the phase offset
between A and B subsets at each layer (expected: anti-phase locked at
~±π, as in 10l).

Plus an interesting question: do the layers' phase signatures align?
L3's two interpretations are anti-phase to each other; L2's two
interpretations should also be anti-phase; L1's various populations should
align with their L2 parent's phase via the plastic feedforward (which
uses Hebbian on complex states, so weights grow only when source and
dest are *in-phase* co-active).
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from manifold import (
    Channel, Constant, Node, Noise, PlasticChannel, WeightNormalizer,
    hebbian, history_as_dict, homeostatic_feedback, polar_sigmoid,
    run_compiled, tracker,
)


VERTICES = list(itertools.product([0, 1], repeat=3))
ARM_COMBOS = list(itertools.product([0, 1], repeat=3))

# L1 (rich)
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

# L2 (middle)
IMAGE_DRIVE_L2 = 0.5
INHIB_INTRATILE_L2 = 1.5
EXCITE_CROSS_L2 = 0.4
INHIB_CROSS_L2 = 0.4
ADAPT_FEEDBACK_L2 = 1.5
RATE_FAST_L2 = 0.1
RATE_ADAPT_L2 = 0.005
GAIN_L2 = 8.0
THRESHOLD_L2 = 2.5
OMEGA_L2 = 0.05
OMEGA_JITTER_L2 = 0.015

# L3 (top)
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

ETA = 0.002
DECAY_PLASTIC = 0.002
TARGET_L1_TO_L2_SUM = 1.0
TARGET_L2_TO_L3_BU_SUM = 2.0
TARGET_L3_TO_L2_TD_SUM = 2.0
L1_TO_L2_GAIN = 1.5
L3_TO_L2_TD_GAIN = 1.0
COUPLING_ALL = 1.0          # polar_sigmoid coupling — channel weights set actual strengths

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

    # ---- L1 (rich) — polar_sigmoid per detector ----
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
            det = l1[v][combo]
            mem = l1_mems[v][combo]
            det.add_channel(Channel(image))
            # Adaptation memory tracks AMPLITUDE not real-part (would oscillate)
            mem.add_channel(Channel(det, transform=lambda y: complex(abs(y), 0)))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK_L1 * y))
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

    # ---- L2 (middle) — polar_sigmoid ----
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

    # ---- L3 (top) — polar_sigmoid ----
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

    # JAX-compiled execution (~70-90× faster than slow Python)
    print("Running (JAX-compiled)...")
    import time as _time
    t0 = _time.time()
    history_array, node_to_idx, _final = run_compiled(
        sources, n_steps=N_STEPS, dt=1.0, seed=42,
    )
    elapsed = _time.time() - t0
    print(f"  ran {N_STEPS} steps in {elapsed:.2f}s ({N_STEPS / elapsed:,.0f} steps/s)")

    # Observations: full complex state at every level
    obs_spec = {
        "I_A": (I_a, "complex"),
        "I_B": (I_b, "complex"),
    }
    for v in VERTICES:
        obs_spec[f"L2_f{v}"] = (front[v], "complex")
        obs_spec[f"L2_b{v}"] = (back[v], "complex")
        for combo in ARM_COMBOS:
            obs_spec[f"L1_{v}_{combo}"] = (l1[v][combo], "complex")
    history = history_as_dict(history_array, node_to_idx, obs_spec)
    times = list(range(N_STEPS))

    # ---- Analysis ----
    def is_a_consistent(v, combo):
        _, _, c = v
        d_x, d_y, d_z = combo
        return d_x == c and d_y == c and d_z == 1 - c

    def is_b_consistent(v, combo):
        _, _, c = v
        d_x, d_y, d_z = combo
        return d_x == 1 - c and d_y == 1 - c and d_z == c

    # L1 subsets
    l1_a_keys = [f"L1_{v}_{c}" for v in VERTICES for c in ARM_COMBOS if is_a_consistent(v, c)]
    l1_b_keys = [f"L1_{v}_{c}" for v in VERTICES for c in ARM_COMBOS if is_b_consistent(v, c)]
    # L2 subsets
    l2_a_keys = [f"L2_f{v}" if v[2] == 0 else f"L2_b{v}" for v in VERTICES]
    l2_b_keys = [f"L2_b{v}" if v[2] == 0 else f"L2_f{v}" for v in VERTICES]

    def phasor(s):
        r = abs(s)
        return s / r if r > 1e-9 else 1.0 + 0.0j

    def wrap(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def stats(t, keys):
        states = [history[k][t] for k in keys]
        amp = float(np.mean([abs(s) for s in states]))
        ph = np.array([phasor(s) for s in states])
        mean_ph = np.mean(ph)
        return amp, float(abs(mean_ph)), float(np.angle(mean_ph))

    L1_A_amp, L1_A_R, L1_A_ph = [], [], []
    L1_B_amp, L1_B_R, L1_B_ph = [], [], []
    L2_A_amp, L2_A_R, L2_A_ph = [], [], []
    L2_B_amp, L2_B_R, L2_B_ph = [], [], []
    L3_A_amp, L3_A_ph = [], []
    L3_B_amp, L3_B_ph = [], []
    for t in range(N_STEPS):
        a1, r1a, p1a = stats(t, l1_a_keys); L1_A_amp.append(a1); L1_A_R.append(r1a); L1_A_ph.append(p1a)
        b1, r1b, p1b = stats(t, l1_b_keys); L1_B_amp.append(b1); L1_B_R.append(r1b); L1_B_ph.append(p1b)
        a2, r2a, p2a = stats(t, l2_a_keys); L2_A_amp.append(a2); L2_A_R.append(r2a); L2_A_ph.append(p2a)
        b2, r2b, p2b = stats(t, l2_b_keys); L2_B_amp.append(b2); L2_B_R.append(r2b); L2_B_ph.append(p2b)
        ia = history["I_A"][t]; ib = history["I_B"][t]
        L3_A_amp.append(abs(ia)); L3_A_ph.append(np.angle(phasor(ia)))
        L3_B_amp.append(abs(ib)); L3_B_ph.append(np.angle(phasor(ib)))

    # Cross-population phase offset at each layer
    L1_offset = [wrap(L1_A_ph[t] - L1_B_ph[t]) for t in range(N_STEPS)]
    L2_offset = [wrap(L2_A_ph[t] - L2_B_ph[t]) for t in range(N_STEPS)]
    L3_offset = [wrap(L3_A_ph[t] - L3_B_ph[t]) for t in range(N_STEPS)]

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    axes[0].plot(times, L1_A_amp, color="tab:blue",   label="L1 A subset", linewidth=0.8)
    axes[0].plot(times, L1_B_amp, color="tab:orange", label="L1 B subset", linewidth=0.8)
    axes[0].plot(times, L2_A_amp, color="tab:cyan",   label="L2 A subset", linewidth=0.8)
    axes[0].plot(times, L2_B_amp, color="tab:red",    label="L2 B subset", linewidth=0.8)
    axes[0].plot(times, L3_A_amp, color="tab:green",  label="L3 I_A",       linewidth=1.2)
    axes[0].plot(times, L3_B_amp, color="tab:olive",  label="L3 I_B",       linewidth=1.2)
    axes[0].set_ylabel("amplitude (mean |s|)")
    axes[0].set_title("Three-layer hierarchy with polar_sigmoid: amplitude bistability per layer")
    axes[0].legend(fontsize=8, loc="upper right", ncol=3)
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, L1_A_R, color="tab:blue",   label="R within L1 A", linewidth=0.8)
    axes[1].plot(times, L1_B_R, color="tab:orange", label="R within L1 B", linewidth=0.8)
    axes[1].plot(times, L2_A_R, color="tab:cyan",   label="R within L2 A", linewidth=0.8)
    axes[1].plot(times, L2_B_R, color="tab:red",    label="R within L2 B", linewidth=0.8)
    axes[1].set_ylabel("within-subset R")
    axes[1].set_title("Phase coherence within each L1/L2 subset")
    axes[1].legend(fontsize=8, loc="lower right", ncol=2)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].plot(times, L1_offset, color="tab:purple", label="L1 A−B phase offset", linewidth=0.6)
    axes[2].plot(times, L2_offset, color="tab:brown",  label="L2 A−B phase offset", linewidth=0.8)
    axes[2].plot(times, L3_offset, color="tab:gray",   label="L3 A−B phase offset", linewidth=1.0)
    for y in (np.pi, -np.pi):
        axes[2].axhline(y, color="black", linestyle=":", alpha=0.3)
    axes[2].set_ylabel("phase offset (rad)")
    axes[2].set_title("Cross-population phase offset at each layer (target ≈ ±π)")
    axes[2].set_ylim(-np.pi - 0.4, np.pi + 0.4)
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(alpha=0.3)

    # Phase alignment between layers: L1 A vs L2 A vs L3 I_A
    # Subtract pairwise phases to see if higher layers' phases align with lower layers'.
    L1_L2_A = [wrap(L1_A_ph[t] - L2_A_ph[t]) for t in range(N_STEPS)]
    L2_L3_A = [wrap(L2_A_ph[t] - L3_A_ph[t]) for t in range(N_STEPS)]
    axes[3].plot(times, L1_L2_A, color="tab:blue",  label="L1 A − L2 A phase offset", linewidth=0.6)
    axes[3].plot(times, L2_L3_A, color="tab:green", label="L2 A − L3 A phase offset", linewidth=0.6)
    axes[3].axhline(0, color="black", linestyle=":", alpha=0.3)
    axes[3].set_ylabel("inter-layer phase offset (rad)")
    axes[3].set_xlabel("time step")
    axes[3].set_title("Inter-layer phase alignment (lower vs higher A subset)")
    axes[3].set_ylim(-np.pi - 0.4, np.pi + 0.4)
    axes[3].legend(fontsize=8, loc="upper right")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("out/10m_unified_hierarchy_summary.png", dpi=110)
    plt.close(fig)
    print("Saved out/10m_unified_hierarchy_summary.png")


if __name__ == "__main__":
    main()
