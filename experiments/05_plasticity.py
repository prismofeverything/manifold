"""Phase 5: plastic connections via Hebbian learning.

Two matched-frequency oscillators, started 45 degrees out of phase,
mutually coupled through `PlasticChannel`s with Hebbian rule:

    dw/dt = eta * Re(source * conj(dest)) - decay * w

For unit-amplitude oscillators on the unit circle, Re(s*conj(d)) =
cos(arg(s) - arg(d)). Initial phase difference of pi/4 gives cos = 0.71
> 0, so weights grow. As weights grow, coupling strengthens (polar uses
magnitude-weighted coupling), which accelerates sync, which keeps the
correlation high, which grows weights further. Equilibrium near
w ~ eta/decay when cos(delta) -> 1.

Compare against a fixed-weight baseline at the same starting weight.
With weights stuck at 0.005, coupling K = 0.005 per direction is far
weaker, so synchronization is much slower (or doesn't happen on this
timescale).
"""

import numpy as np

from manifold import Channel, Node, PlasticChannel, hebbian, plot, polar, run


N_STEPS = 1500
OMEGA = 0.05
W0 = 0.005
ETA = 0.005
DECAY = 0.005
THETA_A0 = 0.0
THETA_B0 = np.pi / 4


def build_run(plastic: bool) -> dict:
    f_a, g_a = polar(rate=0.0, omega=OMEGA, coupling=1.0)
    f_b, g_b = polar(rate=0.0, omega=OMEGA, coupling=1.0)

    node_a = Node(state=np.exp(1j * THETA_A0), dynamics=f_a, output=g_a)
    node_b = Node(state=np.exp(1j * THETA_B0), dynamics=f_b, output=g_b)

    obs = {
        "a": lambda: node_a.state,
        "b": lambda: node_b.state,
    }

    if plastic:
        learn = hebbian(eta=ETA, decay=DECAY)
        ch_b_to_a = PlasticChannel(node_b, dest=node_a, weight=W0, learn=learn)
        ch_a_to_b = PlasticChannel(node_a, dest=node_b, weight=W0, learn=learn)
        node_a.add_channel(ch_b_to_a)
        node_b.add_channel(ch_a_to_b)
        # Capture each channel via default arg so we observe THIS one.
        obs["w_BA"] = (lambda c=ch_b_to_a: c.weight)
        obs["w_AB"] = (lambda c=ch_a_to_b: c.weight)
    else:
        node_a.add_channel(Channel(node_b, transform=lambda y: W0 * y))
        node_b.add_channel(Channel(node_a, transform=lambda y: W0 * y))

    return run(
        sources=[node_a, node_b],
        n_steps=N_STEPS,
        observers=obs,
    )


def phase_diff(history: dict) -> list:
    a = np.unwrap([np.angle(v) for v in history["a"]])
    b = np.unwrap([np.angle(v) for v in history["b"]])
    return list(b - a)


def main():
    plastic = build_run(plastic=True)
    fixed = build_run(plastic=False)
    times = list(range(N_STEPS))

    plot(
        times,
        {
            "delta (plastic)":    phase_diff(plastic),
            "delta (fixed weak)": phase_diff(fixed),
        },
        ylabel="phase difference (rad)",
        title="05_plasticity_phase_diff",
    )

    plot(
        times,
        {
            "w_BA": [v.real for v in plastic["w_BA"]],
            "w_AB": [v.real for v in plastic["w_AB"]],
        },
        ylabel="weight",
        title="05_plasticity_weights",
    )


if __name__ == "__main__":
    main()
