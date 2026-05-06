"""Phase 3: complex state with intrinsic phase rotation.

Same step-input schedule as phase 1, but the node now has an intrinsic
angular frequency omega. The amplitude |s| still tracks the input
magnitude (adaptation preserved on |s|), while the state rotates at
omega in the complex plane.

Plots:
  03_oscillator_magnitude  — |s| vs time vs |input|
  03_oscillator_phase      — arg(s) vs time, unwrapped (linear ramp at omega)
  03_oscillator_trajectory — state trajectory in the complex plane
                             (spirals out, circles, then spirals in)

The trajectory plot is the most informative: with rate=0.05 the magnitude
adaptation is slow relative to the rotation period (2*pi/omega ~ 63 steps),
so we see ~3 full revolutions during each input phase.
"""

import numpy as np

from manifold import Channel, Environment, Node, plot, plot_trajectory, polar, run


SENSITIVITY = 1.0
RATE = 0.05
OMEGA = 0.1
INTERVAL = 200


def step_schedule(t: float) -> float:
    if t < INTERVAL:
        return 0.0
    if t < 2 * INTERVAL:
        return 1.0
    return 0.0


def main():
    env = Environment(step_schedule)
    f, g = polar(sensitivity=SENSITIVITY, rate=RATE, omega=OMEGA)
    # Small nonzero init so phase rotation has somewhere to act from t=0.
    node = Node(state=0.001 + 0j, dynamics=f, output=g)
    node.add_channel(Channel(env))

    history = run(
        sources=[env, node],
        n_steps=3 * INTERVAL,
        observers={
            "s":     lambda: node.state,
            "input": lambda: env.read(),
        },
    )

    times = list(range(len(history["s"])))
    states = history["s"]
    inputs = history["input"]

    plot(
        times,
        {
            "|s|":     [abs(v) for v in states],
            "|input|": [abs(v) for v in inputs],
        },
        ylabel="magnitude",
        title="03_oscillator_magnitude",
    )

    phases = np.unwrap([np.angle(v) for v in states])
    plot(
        times,
        {"arg(s) (unwrapped)": list(phases)},
        ylabel="phase (rad)",
        title="03_oscillator_phase",
    )

    plot_trajectory(states, title="03_oscillator_trajectory")


if __name__ == "__main__":
    main()
