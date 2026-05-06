"""Phase 6: three-node bridge — direct reference to the Sompolinsky/V1 paper.

Three nodes A, B, C in a chain topology:

    A <--> B <--> C            (no direct edge A <--> C)

Question from the paper: can two nodes that are *not* directly connected
nonetheless synchronize through a common intermediary? We instantiate
the bridge with Kuramoto coupling and matched intrinsic frequencies,
start the three nodes at evenly-spaced phases (0, 2pi/3, 4pi/3), and
watch.

We also run a variant where the bridge node B has a *different*
intrinsic frequency from A and C: does the bridge still work? (B's
phase is pulled by both A and C while drifting on its own — see the
plots.)
"""

import numpy as np

from manifold import Channel, Node, plot, plot_trajectory, polar, run


N_STEPS = 800
OMEGA_MATCHED = 0.05
OMEGA_BRIDGE_OFFSET = 0.02   # for the variant
COUPLING = 0.05
THETA_0 = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]


def build_run(omega_a: float, omega_b: float, omega_c: float) -> dict:
    f_a, g_a = polar(rate=0.0, omega=omega_a, coupling=COUPLING)
    f_b, g_b = polar(rate=0.0, omega=omega_b, coupling=COUPLING)
    f_c, g_c = polar(rate=0.0, omega=omega_c, coupling=COUPLING)

    a = Node(state=np.exp(1j * THETA_0[0]), dynamics=f_a, output=g_a)
    b = Node(state=np.exp(1j * THETA_0[1]), dynamics=f_b, output=g_b)
    c = Node(state=np.exp(1j * THETA_0[2]), dynamics=f_c, output=g_c)

    # B is the bridge — wired to both A and C. A and C see only B.
    a.add_channel(Channel(b))
    c.add_channel(Channel(b))
    b.add_channel(Channel(a))
    b.add_channel(Channel(c))

    return run(
        sources=[a, b, c],
        n_steps=N_STEPS,
        observers={
            "a": lambda: a.state,
            "b": lambda: b.state,
            "c": lambda: c.state,
        },
    )


def diffs(history: dict) -> dict:
    a = np.unwrap([np.angle(v) for v in history["a"]])
    b = np.unwrap([np.angle(v) for v in history["b"]])
    c = np.unwrap([np.angle(v) for v in history["c"]])
    return {
        "a - b": list(a - b),
        "a - c (indirect)": list(a - c),
        "b - c": list(b - c),
    }


def main():
    times = list(range(N_STEPS))

    # 6a: all matched. A and C should sync through B.
    matched = build_run(OMEGA_MATCHED, OMEGA_MATCHED, OMEGA_MATCHED)
    plot(times, diffs(matched),
         ylabel="phase difference (rad)",
         title="06a_bridge_matched_phase_diffs")
    plot_trajectory(
        matched["a"],
        title="06a_bridge_matched_trajectory",
        extra_series={"b": matched["b"], "c": matched["c"]},
    )

    # 6b: bridge B has a different omega. Can A and C still sync?
    mismatched = build_run(
        OMEGA_MATCHED,
        OMEGA_MATCHED + OMEGA_BRIDGE_OFFSET,
        OMEGA_MATCHED,
    )
    plot(times, diffs(mismatched),
         ylabel="phase difference (rad)",
         title="06b_bridge_mismatched_phase_diffs")
    plot_trajectory(
        mismatched["a"],
        title="06b_bridge_mismatched_trajectory",
        extra_series={"b": mismatched["b"], "c": mismatched["c"]},
    )


if __name__ == "__main__":
    main()
