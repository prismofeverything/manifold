"""Phase 10l: unified experiment — amplitude AND phase both at work.

Same Necker cube architecture as `10c_necker_simple.py` (16 detectors,
front/back per vertex, hand-wired cross-tile constraints, adaptation
memories), but with `polar_sigmoid` dynamics. Each detector now has both
axes of complex state actively evolving:

  - Amplitude r ∈ [0,1]: bistable activity (Pattern A vs B alternation,
    driven by sigmoidal input + slow adaptation, as in 10c_simple).
  - Phase θ ∈ [0, 2π): rotates at intrinsic ω, pulled by Kuramoto
    coupling from neighbors. Excitatory channels → in-phase coupling;
    inhibitory channels → anti-phase coupling (because arg(-x) = arg(x)+π).

We measure both signatures:

  • amplitude: mean |s| over the A-target subset and B-target subset
    (the bistability — same as 10c_simple result)
  • phase coherence: order parameter R = |mean(exp(iφ_i))| over each
    subset. R near 1 = synchronized; R near 0 = drifting/random.
    Expected: R high in the *active* subset, lower in the suppressed
    subset (where amplitudes are near zero, coupling is weak, phases
    drift independently).

If this works as predicted, the Necker cube perception now has *two*
distinct signatures of which interpretation is winning: amplitude
(which features fire) AND phase coherence (which features bind into
a synchronized population). For static input these are redundant; for
dynamic input the phase signature carries additional temporal info.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from manifold import (
    Channel, Constant, Node, plot, polar_sigmoid, run, tracker,
)


VERTICES = list(itertools.product([0, 1], repeat=3))

IMAGE_DRIVE = 0.5
INHIB_INTRATILE = 1.5
EXCITE_CROSS = 0.4
INHIB_CROSS = 0.4
ADAPT_FEEDBACK = 1.5
RATE_FAST = 0.1
RATE_ADAPT = 0.005
GAIN = 8.0
THRESHOLD = 2.5

# New: phase parameters
OMEGA = 0.05            # intrinsic rotation (mean)
OMEGA_JITTER = 0.015    # per-node Gaussian jitter — heterogeneous frequencies
COUPLING = 0.4          # Kuramoto coupling strength

# Why jitter ω: with all nodes at the same ω, suppressed nodes (low
# amplitude, weak coupling) still rotate together at the shared intrinsic
# rate, preserving their phase relationship from when they were active.
# The order parameter R stays high for *both* subsets, hiding the
# active-vs-suppressed contrast. With heterogeneous ω, suppressed nodes
# drift apart at their own frequencies, so R decays in the suppressed
# subset — making the active/suppressed distinction visible in phase.

N_STEPS = 4000
SEED = 42


def neighbor(v, axis):
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def main():
    image = Constant(IMAGE_DRIVE)
    rng = np.random.default_rng(SEED)

    front, back, front_mem, back_mem = {}, {}, {}, {}
    for v in VERTICES:
        # Heterogeneous ω per node so suppressed nodes truly drift apart
        omega_f = OMEGA + OMEGA_JITTER * rng.normal()
        omega_b = OMEGA + OMEGA_JITTER * rng.normal()
        f_f, g_f = polar_sigmoid(
            rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD,
            omega=omega_f, coupling=COUPLING,
        )
        f_b, g_b = polar_sigmoid(
            rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD,
            omega=omega_b, coupling=COUPLING,
        )
        # Below uses _f and _b separately; alias the shared loop var
        f, g = f_f, g_f
        # Random initial amplitude AND phase
        r0_f = 0.3 * rng.random(); th_f = 2 * np.pi * rng.random()
        r0_b = 0.3 * rng.random(); th_b = 2 * np.pi * rng.random()
        front[v] = Node(state=r0_f * np.exp(1j * th_f), dynamics=f_f, output=g_f)
        back[v] = Node(state=r0_b * np.exp(1j * th_b), dynamics=f_b, output=g_b)
        mf, mg = tracker(rate=RATE_ADAPT)
        front_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)
        back_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)

    for v in VERTICES:
        for det, sib, mem in [(front[v], back[v], front_mem[v]),
                              (back[v], front[v], back_mem[v])]:
            det.add_channel(Channel(image))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK * y))
            mem.add_channel(Channel(det))
            det.add_channel(Channel(sib, transform=lambda y: -INHIB_INTRATILE * y))

        for axis in (0, 1, 2):
            n = neighbor(v, axis)
            if axis in (0, 1):
                front[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS * y))
                front[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS * y))
            else:
                front[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS * y))
                front[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS * y))

    sources = []
    for v in VERTICES:
        sources.extend([front[v], back[v], front_mem[v], back_mem[v]])

    obs = {}
    for v in VERTICES:
        obs[f"f{v}"] = (lambda v=v: front[v].state)
        obs[f"b{v}"] = (lambda v=v: back[v].state)

    print("Running...")
    history = run(sources=sources, n_steps=N_STEPS, observers=obs)
    times = list(range(N_STEPS))

    # Subsets:
    #   Pattern A active set = front for z=0 vertices, back for z=1 vertices
    #   Pattern B active set = back for z=0,        front for z=1
    a_keys = [f"f{v}" if v[2] == 0 else f"b{v}" for v in VERTICES]
    b_keys = [f"b{v}" if v[2] == 0 else f"f{v}" for v in VERTICES]

    pa_amp, pb_amp = [], []
    pa_R, pb_R = [], []
    a_mean_phase, b_mean_phase = [], []  # for each subset
    ab_phase_diff = []                    # signed difference, wrapped to (-π, π]

    def phasor(s):
        r = abs(s)
        return s / r if r > 1e-9 else 1.0 + 0.0j

    def wrap(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    for t in range(N_STEPS):
        a_states = [history[k][t] for k in a_keys]
        b_states = [history[k][t] for k in b_keys]

        a_amp = float(np.mean([abs(s) for s in a_states]))
        b_amp = float(np.mean([abs(s) for s in b_states]))
        pa_amp.append(a_amp); pb_amp.append(b_amp)

        a_phasors = np.array([phasor(s) for s in a_states])
        b_phasors = np.array([phasor(s) for s in b_states])

        a_mean = np.mean(a_phasors)
        b_mean = np.mean(b_phasors)
        pa_R.append(float(abs(a_mean)))
        pb_R.append(float(abs(b_mean)))
        a_mean_phase.append(float(np.angle(a_mean)))
        b_mean_phase.append(float(np.angle(b_mean)))
        ab_phase_diff.append(wrap(np.angle(a_mean) - np.angle(b_mean)))

    # Three-panel summary: amplitude bistability, within-subset coherence,
    # phase OFFSET between subsets (the actually-meaningful signature here).
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(times, pa_amp, color="tab:blue", label="|s| Pattern A subset")
    axes[0].plot(times, pb_amp, color="tab:orange", label="|s| Pattern B subset")
    axes[0].set_ylabel("amplitude (mean |s|)")
    axes[0].set_title("Unified amplitude + phase: amplitude carries the bistability")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, pa_R, color="tab:blue", linestyle="-", label="R — within A subset")
    axes[1].plot(times, pb_R, color="tab:orange", linestyle="-", label="R — within B subset")
    axes[1].set_ylabel("within-subset R")
    axes[1].set_title("Phase coherence within each subset stays high (both populations stay bound)")
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    # The interesting signature: phase offset between A's mean phase
    # and B's mean phase. If the populations are anti-phase locked, this
    # sits near ±π. That's the binding signature distinguishing the two
    # populations beyond their amplitude.
    axes[2].plot(times, ab_phase_diff, color="tab:purple", linewidth=0.7)
    axes[2].axhline(np.pi, color="black", linestyle=":", alpha=0.4, label="±π (anti-phase)")
    axes[2].axhline(-np.pi, color="black", linestyle=":", alpha=0.4)
    axes[2].set_ylabel("arg(<A>) − arg(<B>) (rad)")
    axes[2].set_xlabel("time step")
    axes[2].set_title("Phase offset between A and B mean phasors — locks at ~±π (anti-phase)")
    axes[2].set_ylim(-np.pi - 0.4, np.pi + 0.4)
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("out/10l_unified_summary.png", dpi=110)
    plt.close(fig)

    # Companion zoomed view of first 1500 steps where transient sync is visible
    fig2, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    win = slice(0, 1500)
    axes[0].plot(times[win], pa_amp[win], color="tab:blue", label="|s| Pattern A")
    axes[0].plot(times[win], pb_amp[win], color="tab:orange", label="|s| Pattern B")
    axes[0].plot(times[win], pa_R[win], color="tab:blue", linestyle="--", alpha=0.7, label="R — A subset")
    axes[0].plot(times[win], pb_R[win], color="tab:orange", linestyle="--", alpha=0.7, label="R — B subset")
    axes[0].set_ylabel("amplitude (solid)  /  R (dashed)")
    axes[0].set_title("Zoomed view: first 1500 steps — amplitude bistability + phase coherence in active set")
    axes[0].legend(fontsize=8, loc="center right")
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.10)

    # phase trajectories: a few sample nodes' arg(s) over time, colored by subset
    for v in [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)]:
        f_states = history[f"f{v}"][:1500]
        f_phs = np.unwrap([np.angle(s) if abs(s) > 1e-9 else 0.0 for s in f_states])
        col = "tab:blue" if v[2] == 0 else "tab:red"
        axes[1].plot(times[win], f_phs, color=col, alpha=0.7,
                     label=f"front{v}" if v in [(0, 0, 0), (0, 0, 1)] else None,
                     linewidth=0.7)
    axes[1].set_ylabel("arg(s) unwrapped (rad)")
    axes[1].set_xlabel("time step")
    axes[1].set_title("Sample node phases (blue = z=0 fronts, red = z=1 fronts)")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("out/10l_unified_zoomed.png", dpi=110)
    plt.close(fig2)
    print("Saved out/10l_unified_summary.png and out/10l_unified_zoomed.png")


if __name__ == "__main__":
    main()
