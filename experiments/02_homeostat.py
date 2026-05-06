"""Phase 2: closed-loop environment.

A homeostat. The Environment has its own state x — a physical quantity
subject to a constant disturbance (`drift`). Without control, x grows
linearly. The Node senses x via a Channel; the Environment closes the
loop by reading the Node's accumulated state (its low-pass memory, the
compensation built up by adaptation) and subtracting it from each update:

    x   <- x + dt * (drift - gain * node.state)
    s   <- s + dt * rate * (sensitivity * x - s)

The adaptation node's *output* (high-pass error) signals change; its
*state* (low-pass memory) is the running compensation. Both are produced
by the same node; the env taps into state via a `StateView` source.

Equilibrium: ds/dt = 0 -> x = s; dx/dt = 0 -> s = D/k. With D = 0.01
and k = 0.05, x converges to 0.2 instead of growing unbounded. With
alpha = 0.1 the loop is slightly under-damped (eigenvalues -0.05 +/- 0.05i,
period ~ 130 steps).

We run two trials (gain=0 vs gain=k) and overlay them.
"""

from manifold import Channel, Node, Source, StateView, adaptation, plot, run


SENSITIVITY = 1.0
RATE = 0.1
DRIFT = 0.01
GAIN = 0.05
N_STEPS = 1000


class Homeostat(Source):
    """1-D physical environment with constant disturbance and a controller
    that subtracts from each update.

    state(t+dt) = state(t) + dt * (drift - gain * controller.read())
    """

    def __init__(self, drift: float, gain: float, controller: Source, x0: float = 0.0):
        self.drift = drift
        self.gain = gain
        self.controller = controller
        self.state: complex = complex(x0)
        self._next_state: complex = self.state

    def read(self) -> complex:
        return self.state

    def compute(self, dt: float = 1.0) -> None:
        u = self.controller.read()
        self._next_state = self.state + dt * (self.drift - self.gain * u)

    def commit(self) -> None:
        self.state = self._next_state


def run_trial(gain: float) -> dict:
    f, g = adaptation(sensitivity=SENSITIVITY, rate=RATE)
    node = Node(state=0 + 0j, dynamics=f, output=g)

    controller = StateView(node)
    env = Homeostat(drift=DRIFT, gain=gain, controller=controller)

    node.add_channel(Channel(env))

    return run(
        sources=[env, node],
        n_steps=N_STEPS,
        observers={
            "x": lambda: env.read(),
            "s": lambda: node.state,
            "y": lambda: node.read(),
        },
    )


def main():
    open_loop = run_trial(gain=0.0)
    closed_loop = run_trial(gain=GAIN)

    times = list(range(N_STEPS))

    plot(
        times,
        {
            "x (open loop)":   [v.real for v in open_loop["x"]],
            "x (closed loop)": [v.real for v in closed_loop["x"]],
            "s (closed loop)": [v.real for v in closed_loop["s"]],
        },
        ylabel="value",
        title="02_homeostat",
    )

    plot(
        times,
        {
            "x (env state)":               [v.real for v in closed_loop["x"]],
            "s (node state, compensation)": [v.real for v in closed_loop["s"]],
            "y (node output, error)":      [v.real for v in closed_loop["y"]],
        },
        ylabel="value",
        title="02_homeostat_closed_loop",
    )


if __name__ == "__main__":
    main()
