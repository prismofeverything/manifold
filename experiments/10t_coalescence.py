"""10t — Coalescence dynamics: capturing the Shimaoka 2010 transient PLV peak at switch.

Coarse-grained whole-brain model. Three regions (~ occipital / parietal / frontal).
Each region has TWO subsystems, fully decoupled:

  (1) Activity: 2 competing populations (A vs B) with polar_sigmoid + adaptation
      + mutual inhibition. Bistable winner-take-all with slow adaptation-driven
      flips. Reuses the proven L3 circuit from 10n. The "percept" of region X
      is whichever of A/B is high.

  (2) Phase oscillator: ONE Kuramoto unit on the unit circle at intrinsic ω_X.
      Receives ONLY cross-region phase coupling channels (no drive, no inhibition,
      no adaptation). With weak base coupling and big Δω, oscillators drift apart.

Cross-region phase coupling is gated by a switch detector: 4·m_L3_A·(1-m_L3_A)
low-passed. Peaks during the moment that m_L3_A is mid-range — i.e., during a
transition.

The decoupling is critical: activity nodes have many real-valued inputs (drive,
inhibition, adaptation) which would pull a polar_sigmoid's phase to 0. Keeping
the phase oscillator separate avoids this.

Predictions (Shimaoka et al. 2010):
  - Each region's activity (A/B) flips slowly under adaptation.
  - Phase oscillators free-run at ω_X between switches → low cross-region PLV.
  - At a switch, gate fires → cross-region coupling boosts → phases briefly
    lock → cross-region PLV transiently peaks.
  - Removing the gate at fixed strong coupling → constant PLV (no coalescence).
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from manifold import (
    Channel, Constant, Node, Noise, homeostatic_feedback, kuramoto, polar_sigmoid,
    run, tracker,
)


# ---- Phase oscillator frequencies (spread wide so layers don't auto-lock) ----
OMEGA_L1 = 0.060   # ~9.5 Hz at dt=1ms (alpha-ish)
OMEGA_L2 = 0.030   # ~4.8 Hz (theta)
OMEGA_L3 = 0.012   # ~1.9 Hz (delta-ish)

# ---- Activity bistable circuit (per region) ----
# Borrowed from 10n's L3 settings — proven bistable behavior.
RATE_FAST = 0.1
RATE_ADAPT = 0.0005
GAIN = 6.0
THRESHOLD = 1.5
INHIB = 2.0
ADAPT_FEEDBACK = 2.0
DRIVE = 0.5
RATE_HOMEO = 0.00005
HOMEO_TARGET = 0.4
HOMEO_GAIN = 4.0
NOISE_STD = 0.01
# polar_sigmoid couples phase via Kuramoto, but we're using it just for amplitude
# bistability — set coupling=0 so phase isn't pulled around. (We don't observe
# the phase of these activity nodes anyway.)
ACT_PHASE_COUPLING = 0.0
ACT_OMEGA = 0.0  # no rotation needed for the activity nodes

# ---- Switch detector ----
RATE_SWITCH = 0.02   # τ=50: post-switch tail outlives the transition itself
SWITCH_THRESH = 0.1  # both pops above this → "co-active" → in transition

# ---- Cross-layer phase coupling ----
PHASE_CROSS_BASE = 0.005   # weak baseline — Δω wins, layers drift apart
PHASE_CROSS_BOOST = 1.5    # transient boost during switch (strongly locks)
PHASE_CROSS_UNGATED = 0.2  # ungated control: matched-strong constant
PHASE_COUPLING = 1.0       # kuramoto factory's coupling param

# ---- PLV window (analysis only) ----
PLV_WINDOW = 100  # shorter window → transient locks show as peaks not dips

N_STEPS = 30000
SEED = 42
DT = 1.0


def make_activity_pair(rng, init_a_high: bool):
    """Two competing populations forming a bistable circuit with adaptation."""
    fa, ga = polar_sigmoid(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD,
                           omega=ACT_OMEGA, coupling=ACT_PHASE_COUPLING)
    fb, gb = polar_sigmoid(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD,
                           omega=ACT_OMEGA, coupling=ACT_PHASE_COUPLING)
    init_a = (0.6 if init_a_high else 0.05) + 0.05 * rng.random()
    init_b = (0.05 if init_a_high else 0.6) + 0.05 * rng.random()
    A = Node(state=complex(init_a, 0), dynamics=fa, output=ga)
    B = Node(state=complex(init_b, 0), dynamics=fb, output=gb)
    return A, B


def make_phase_node(omega, init_phase):
    f, g = kuramoto(omega=omega, coupling=PHASE_COUPLING)
    return Node(state=np.exp(1j * init_phase), dynamics=f, output=g)


def make_tracker_node(rate=RATE_ADAPT, init=0+0j):
    f, g = tracker(rate=rate)
    return Node(state=init, dynamics=f, output=g)


def wire_bistable_region(A, B, mem_a, mem_b, drive, noise_a, noise_b,
                          homeo_a, homeo_b):
    """Standard polar_sigmoid bistable circuit with adaptation + homeostasis."""
    homeo_x = homeostatic_feedback(target=HOMEO_TARGET, gain=HOMEO_GAIN)
    for det, sib, mem, noise, homeo in [(A, B, mem_a, noise_a, homeo_a),
                                          (B, A, mem_b, noise_b, homeo_b)]:
        det.add_channel(Channel(drive))
        mem.add_channel(Channel(det, transform=lambda y: complex(abs(y), 0)))
        det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK * y))
        det.add_channel(Channel(sib, transform=lambda y: -INHIB * y))
        det.add_channel(Channel(noise))
        homeo.add_channel(Channel(det, transform=lambda y: complex(abs(y), 0)))
        det.add_channel(Channel(homeo, transform=homeo_x))


def build_and_run(use_gate: bool, save_path: str):
    rng = np.random.default_rng(SEED)
    drive = Constant(DRIVE)

    # ---- Three regions, each with bistable A/B activity ----
    L1_A, L1_B = make_activity_pair(rng, init_a_high=True)
    L2_A, L2_B = make_activity_pair(rng, init_a_high=False)
    L3_A, L3_B = make_activity_pair(rng, init_a_high=True)
    activity_pops = [L1_A, L1_B, L2_A, L2_B, L3_A, L3_B]

    Mems = {n: make_tracker_node() for n in activity_pops}
    Homeos = {n: make_tracker_node(rate=RATE_HOMEO, init=HOMEO_TARGET + 0j)
              for n in activity_pops}

    # Noise sources for each population
    noise_sources = []
    noise_for = {}
    for i, n in enumerate(activity_pops):
        ns = Noise(std=NOISE_STD, seed=SEED + 100 + i)
        noise_for[n] = ns
        noise_sources.append(ns)

    for (A, B) in [(L1_A, L1_B), (L2_A, L2_B), (L3_A, L3_B)]:
        wire_bistable_region(A, B, Mems[A], Mems[B], drive,
                             noise_for[A], noise_for[B], Homeos[A], Homeos[B])

    # ---- Phase oscillators (one per region; only cross-region inputs) ----
    P_L1 = make_phase_node(OMEGA_L1, 0.0)
    P_L2 = make_phase_node(OMEGA_L2, 1.5)
    P_L3 = make_phase_node(OMEGA_L3, 3.0)
    phase_pops = [P_L1, P_L2, P_L3]

    # ---- Switch detector: fires when BOTH L3_A and L3_B are above threshold,
    # i.e., during the brief co-active window of a percept transition. Steady
    # state has one near 1 and one near 0 → product is 0. Crossing has both
    # at moderate values → product peaks.
    def switch_dyn(s, xs, dt):
        a = abs(xs[0]) if len(xs) > 0 else 0.0
        b = abs(xs[1]) if len(xs) > 1 else 0.0
        return complex(max(0.0, a - SWITCH_THRESH) * max(0.0, b - SWITCH_THRESH)
                       / (0.5 - SWITCH_THRESH) ** 2, 0.0)  # normalize to ~1 peak
    def switch_out(s, xs):
        return s
    SwitchRaw = Node(state=0 + 0j, dynamics=switch_dyn, output=switch_out)
    SwitchRaw.add_channel(Channel(L3_A))
    SwitchRaw.add_channel(Channel(L3_B))
    Switch = make_tracker_node(rate=RATE_SWITCH)
    Switch.add_channel(Channel(SwitchRaw))

    # ---- Cross-region phase coupling, gated or constant ----
    if use_gate:
        def cross_xform(y, _S=Switch):
            return (PHASE_CROSS_BASE + PHASE_CROSS_BOOST * abs(_S.read())) * y
    else:
        def cross_xform(y):
            return PHASE_CROSS_UNGATED * y

    cross_pairs = [(P_L1, P_L2), (P_L2, P_L3), (P_L1, P_L3)]
    for src, dst in cross_pairs:
        dst.add_channel(Channel(src, transform=cross_xform))
        src.add_channel(Channel(dst, transform=cross_xform))

    # ---- Sources ----
    sources = (activity_pops + phase_pops + list(Mems.values()) + list(Homeos.values())
               + [SwitchRaw, Switch] + noise_sources)

    observers = {
        "L1_A": lambda: L1_A.read(), "L1_B": lambda: L1_B.read(),
        "L2_A": lambda: L2_A.read(), "L2_B": lambda: L2_B.read(),
        "L3_A": lambda: L3_A.read(), "L3_B": lambda: L3_B.read(),
        "m_L1_A": lambda: Mems[L1_A].read(), "m_L1_B": lambda: Mems[L1_B].read(),
        "m_L2_A": lambda: Mems[L2_A].read(), "m_L2_B": lambda: Mems[L2_B].read(),
        "m_L3_A": lambda: Mems[L3_A].read(), "m_L3_B": lambda: Mems[L3_B].read(),
        "P_L1": lambda: P_L1.read(), "P_L2": lambda: P_L2.read(), "P_L3": lambda: P_L3.read(),
        "Switch": lambda: Switch.read(),
    }

    print(f"Running 10t (gate={'on' if use_gate else 'off'})...")
    t0 = time.time()
    history = run(sources, n_steps=N_STEPS, dt=DT, observers=observers)
    elapsed = time.time() - t0
    print(f"  ran {N_STEPS} steps in {elapsed:.1f}s ({N_STEPS/elapsed:,.0f} steps/s)")

    H = {k: np.array(v) for k, v in history.items()}
    return analyze_and_plot(H, use_gate, save_path)


def sliding_plv(phase_a, phase_b, window=PLV_WINDOW):
    diff = phase_a - phase_b
    cos_d = np.cos(diff)
    sin_d = np.sin(diff)
    kernel = np.ones(window) / window
    c = np.convolve(cos_d, kernel, mode="same")
    s = np.convolve(sin_d, kernel, mode="same")
    return np.sqrt(c**2 + s**2)


def analyze_and_plot(H, use_gate, save_path):
    times = np.arange(len(H["L1_A"]))
    act_amp = {k: np.abs(H[k]) for k in ["L1_A", "L1_B", "L2_A", "L2_B", "L3_A", "L3_B"]}
    phase = {k: np.angle(H[k]) for k in ["P_L1", "P_L2", "P_L3"]}

    # Slow switch detection from L3 adaptation memory
    mem_a = np.abs(H["m_L3_A"])
    mem_b = np.abs(H["m_L3_B"])
    diff = mem_a - mem_b
    sign = np.sign(diff)
    sign[sign == 0] = 1
    zc = np.where(np.diff(sign) != 0)[0]
    if len(zc) > 1:
        zc = zc[np.concatenate(([True], np.diff(zc) > 200))]
    print(f"  L3 percept switches: {len(zc)}")

    # Cross-region PLV across all 3 phase pairs
    plv_l1_l2 = sliding_plv(phase["P_L1"], phase["P_L2"])
    plv_l2_l3 = sliding_plv(phase["P_L2"], phase["P_L3"])
    plv_l1_l3 = sliding_plv(phase["P_L1"], phase["P_L3"])
    plv_mean = (plv_l1_l2 + plv_l2_l3 + plv_l1_l3) / 3

    fig, axes = plt.subplots(4, 1, figsize=(11, 12))

    # Panel 1: activity (slow bistable envelope)
    axes[0].plot(times, act_amp["L1_A"], 'C0', linewidth=0.8, label="L1_A")
    axes[0].plot(times, act_amp["L1_B"], 'C0', linestyle="--", linewidth=0.8, label="L1_B")
    axes[0].plot(times, act_amp["L2_A"], 'C1', linewidth=0.8, label="L2_A")
    axes[0].plot(times, act_amp["L2_B"], 'C1', linestyle="--", linewidth=0.8, label="L2_B")
    axes[0].plot(times, act_amp["L3_A"], 'C2', linewidth=1.2, label="L3_A")
    axes[0].plot(times, act_amp["L3_B"], 'C2', linestyle="--", linewidth=1.2, label="L3_B")
    axes[0].set_ylabel("activity")
    axes[0].set_title(
        f"10t coalescence — coarse-grained whole-brain "
        f"({'gated cross-region (Shimaoka prediction)' if use_gate else 'ungated control'})")
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, N_STEPS)

    axes[1].plot(times, np.abs(H["Switch"]), 'k-', linewidth=1, label="switch detector")
    for x in zc:
        axes[1].axvline(x, color="red", alpha=0.3, linewidth=0.5)
    axes[1].set_ylabel("switch signal")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlim(0, N_STEPS)

    axes[2].plot(times, plv_l1_l2, label="L1-L2", alpha=0.5)
    axes[2].plot(times, plv_l2_l3, label="L2-L3", alpha=0.5)
    axes[2].plot(times, plv_l1_l3, label="L1-L3", alpha=0.5)
    axes[2].plot(times, plv_mean, 'k-', linewidth=2, label="mean cross-region PLV")
    for x in zc:
        axes[2].axvline(x, color="red", alpha=0.2, linewidth=0.5)
    axes[2].set_ylabel("cross-region PLV (200-step)")
    axes[2].set_xlabel("time step")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlim(0, N_STEPS)

    if len(zc) > 2:
        zc_use = zc[1:-1]
        win = 1000
        aligned = []
        for x in zc_use:
            lo, hi = x - win, x + win
            if lo >= 0 and hi < len(times):
                aligned.append(plv_mean[lo:hi])
        if aligned:
            aligned = np.array(aligned)
            mean_aligned = aligned.mean(axis=0)
            std_aligned = aligned.std(axis=0)
            t_aligned = np.arange(-win, win)
            axes[3].plot(t_aligned, mean_aligned, 'k-', linewidth=2,
                         label=f"mean over {len(aligned)} switches")
            axes[3].fill_between(t_aligned, mean_aligned - std_aligned,
                                 mean_aligned + std_aligned, alpha=0.3, color='gray',
                                 label="±1 SD")
            axes[3].axvline(0, color="red", linestyle="--", alpha=0.7, label="switch")
            axes[3].set_xlabel("time relative to switch (steps)")
            axes[3].set_ylabel("mean PLV")
            axes[3].set_title("Switch-aligned cross-region PLV — does it peak at switch?")
            axes[3].legend(loc="upper right", fontsize=8)
            axes[3].grid(alpha=0.3)
            axes[3].set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"  saved {save_path}")

    plv_at_zc = (np.mean([plv_mean[x] for x in zc]) if len(zc) else 0.0)
    if len(zc):
        far_idx = []
        for t in range(0, len(times), 200):
            min_dist = min(abs(t - z) for z in zc)
            if min_dist > 800:
                far_idx.append(t)
        plv_baseline = (np.mean([plv_mean[t] for t in far_idx]) if far_idx
                        else float("nan"))
    else:
        plv_baseline = float(plv_mean.mean())

    return {
        "n_switches": len(zc),
        "plv_mean_avg": float(plv_mean.mean()),
        "plv_at_switches": float(plv_at_zc),
        "plv_baseline": float(plv_baseline),
    }


def main():
    print("=" * 64)
    print("Run 1: gate ON (Shimaoka-style transient cross-region locking)")
    print("=" * 64)
    r_on = build_and_run(use_gate=True, save_path="out/10t_coalescence_gated.png")
    print(r_on)

    print()
    print("=" * 64)
    print("Run 2: gate OFF (control — constant cross-region coupling)")
    print("=" * 64)
    r_off = build_and_run(use_gate=False, save_path="out/10t_coalescence_ungated.png")
    print(r_off)

    print()
    print("=" * 64)
    print("Comparison")
    print("=" * 64)
    for tag, r in [("gated  ", r_on), ("ungated", r_off)]:
        ratio = r["plv_at_switches"] / max(r["plv_baseline"], 1e-6)
        print(f"  {tag}: PLV at switch = {r['plv_at_switches']:.3f}, "
              f"baseline = {r['plv_baseline']:.3f}, ratio = {ratio:.2f} "
              f"(N switches = {r['n_switches']})")


if __name__ == "__main__":
    main()
