"""Replot 10m result with clearer per-layer breakdown."""

import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "experiments")
mod = importlib.import_module("10m_unified_hierarchy")


def main():
    captured = {}
    orig_main = mod.main
    real_run = mod.run  # the name that's actually called inside main()

    def patched_run(*args, **kwargs):
        h = real_run(*args, **kwargs)
        captured["history"] = h
        return h

    mod.run = patched_run
    try:
        orig_main()
    finally:
        mod.run = real_run

    history = captured["history"]
    N_STEPS = mod.N_STEPS
    VERTICES = mod.VERTICES
    ARM_COMBOS = mod.ARM_COMBOS

    def is_a_consistent(v, combo):
        _, _, c = v
        d_x, d_y, d_z = combo
        return d_x == c and d_y == c and d_z == 1 - c

    def is_b_consistent(v, combo):
        _, _, c = v
        d_x, d_y, d_z = combo
        return d_x == 1 - c and d_y == 1 - c and d_z == c

    l1_a_keys = [f"L1_{v}_{c}" for v in VERTICES for c in ARM_COMBOS if is_a_consistent(v, c)]
    l1_b_keys = [f"L1_{v}_{c}" for v in VERTICES for c in ARM_COMBOS if is_b_consistent(v, c)]
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
        m = np.mean(ph)
        return amp, float(abs(m)), float(np.angle(m))

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

    L2_offset = [wrap(L2_A_ph[t] - L2_B_ph[t]) for t in range(N_STEPS)]
    L3_offset = [wrap(L3_A_ph[t] - L3_B_ph[t]) for t in range(N_STEPS)]
    L2_L3_A = [wrap(L2_A_ph[t] - L3_A_ph[t]) for t in range(N_STEPS)]

    times = list(range(N_STEPS))
    # Late window where learning has settled
    win = slice(8000, 12000)
    times_w = times[win]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(times_w, L2_A_amp[win], color="tab:cyan",   label="L2 A subset", linewidth=1.0)
    axes[0].plot(times_w, L2_B_amp[win], color="tab:red",    label="L2 B subset", linewidth=1.0)
    axes[0].plot(times_w, L3_A_amp[win], color="tab:green",  label="L3 I_A",       linewidth=1.4)
    axes[0].plot(times_w, L3_B_amp[win], color="tab:olive",  label="L3 I_B",       linewidth=1.4)
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Late-window detail: L2 and L3 amplitude bistability (zoomed t=8000–12000)")
    axes[0].legend(fontsize=9, loc="upper right", ncol=2)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(times_w, L2_A_R[win], color="tab:cyan",  label="R within L2 A", linewidth=1.0)
    axes[1].plot(times_w, L2_B_R[win], color="tab:red",   label="R within L2 B", linewidth=1.0)
    axes[1].set_ylabel("within-subset R (L2)")
    axes[1].set_title("L2 phase coherence within each subset")
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].plot(times_w, L2_offset[win], color="tab:brown", label="L2 A−B phase offset", linewidth=0.8)
    axes[2].plot(times_w, L3_offset[win], color="tab:gray",  label="L3 A−B phase offset", linewidth=1.0)
    axes[2].axhline(np.pi, color="black", linestyle=":", alpha=0.4)
    axes[2].axhline(-np.pi, color="black", linestyle=":", alpha=0.4)
    axes[2].set_ylabel("phase offset (rad)")
    axes[2].set_title("Cross-population phase offset (target ≈ ±π)")
    axes[2].set_ylim(-np.pi - 0.4, np.pi + 0.4)
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].grid(alpha=0.3)

    axes[3].plot(times_w, L2_L3_A[win], color="tab:green", label="L2 A − L3 I_A phase offset", linewidth=0.6)
    axes[3].axhline(0, color="black", linestyle=":", alpha=0.4)
    axes[3].set_ylabel("inter-layer offset")
    axes[3].set_xlabel("time step")
    axes[3].set_title("Inter-layer phase alignment: L2 active subset vs L3 winner")
    axes[3].set_ylim(-np.pi - 0.4, np.pi + 0.4)
    axes[3].legend(fontsize=9, loc="upper right")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("out/10m_unified_hierarchy_zoomed.png", dpi=110)
    plt.close(fig)
    print("Saved out/10m_unified_hierarchy_zoomed.png")


if __name__ == "__main__":
    main()
