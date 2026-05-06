"""Phase 4: two coupled Kuramoto oscillators — start of the book of diagrams.

Two nodes A and B on the unit circle (rate=0, fixed amplitude), each
with an intrinsic frequency, mutually coupled via a single Channel each
(reading the other's state). Coupling enters through the Kuramoto term
in `polar`'s phase update:

    dtheta_a/dt = omega_a + K * sin(arg(b) - theta_a)
    dtheta_b/dt = omega_b + K * sin(arg(a) - theta_b)

Steady-state phase difference delta = theta_b - theta_a satisfies:
    d(delta)/dt = (omega_b - omega_a) - 2*K*sin(delta)
A fixed point exists iff |omega_b - omega_a| <= 2|K|.

We run four canonical configurations and overlay their phase differences:

  4a: matched omegas, K = +0.05      -> in-phase lock (delta -> 0)
  4b: matched omegas, K = -0.05      -> anti-phase lock (delta -> pi)
  4c: mismatched (dw=0.02), K = +0.005  -> drift (no lock; |dw| > 2|K|)
  4d: mismatched (dw=0.02), K = +0.05   -> locked at offset (sin(delta) = dw/2K)
"""

import numpy as np

from manifold import Channel, Node, plot, plot_trajectory, polar, run


N_STEPS = 600


def two_node_run(
    omega_a: float,
    omega_b: float,
    coupling: float,
    theta_a0: float = 0.0,
    theta_b0: float = np.pi / 2,
) -> dict:
    f_a, g_a = polar(rate=0.0, omega=omega_a, coupling=coupling)
    f_b, g_b = polar(rate=0.0, omega=omega_b, coupling=coupling)

    node_a = Node(state=np.exp(1j * theta_a0), dynamics=f_a, output=g_a)
    node_b = Node(state=np.exp(1j * theta_b0), dynamics=f_b, output=g_b)

    node_a.add_channel(Channel(node_b))
    node_b.add_channel(Channel(node_a))

    return run(
        sources=[node_a, node_b],
        n_steps=N_STEPS,
        observers={
            "a": lambda: node_a.state,
            "b": lambda: node_b.state,
        },
    )


def phase_diff(history: dict) -> list:
    a = np.unwrap([np.angle(v) for v in history["a"]])
    b = np.unwrap([np.angle(v) for v in history["b"]])
    return list(b - a)


def main():
    configs = [
        ("4a matched, K=+0.05",   dict(omega_a=0.05, omega_b=0.05, coupling=+0.05)),
        ("4b matched, K=-0.05",   dict(omega_a=0.05, omega_b=0.05, coupling=-0.05)),
        ("4c mismatch, K=+0.005", dict(omega_a=0.05, omega_b=0.07, coupling=+0.005)),
        ("4d mismatch, K=+0.05",  dict(omega_a=0.05, omega_b=0.07, coupling=+0.05)),
    ]

    times = list(range(N_STEPS))
    diffs = {}
    histories = {}
    for label, kwargs in configs:
        h = two_node_run(**kwargs)
        diffs[label] = phase_diff(h)
        histories[label] = h

    plot(
        times,
        diffs,
        ylabel="phase difference (b - a, rad)",
        title="04_two_oscillator_phase_diffs",
    )

    # Trajectory of the in-phase-sync configuration: both nodes converge
    # to the same point on the unit circle, rotating together.
    h_4a = histories["4a matched, K=+0.05"]
    plot_trajectory(
        h_4a["a"],
        title="04a_trajectory_in_phase",
        extra_series={"b": h_4a["b"]},
    )

    # Trajectory for anti-phase: nodes settle to opposite points.
    h_4b = histories["4b matched, K=-0.05"]
    plot_trajectory(
        h_4b["a"],
        title="04b_trajectory_anti_phase",
        extra_series={"b": h_4b["b"]},
    )


if __name__ == "__main__":
    main()
