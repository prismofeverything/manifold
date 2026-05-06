"""Phase 1: reproduce the original adaptation behavior with the new
channel/node/environment abstraction.

A single Node has two Channels:
  - external: reads the Environment value (raw, no transform).
  - self:     reads the Node's own output, scaled by the adaptation rate.

The Node's output function reads the external input and subtracts the
state. The Node's dynamics function takes the self-channel reading as
the entire ds:

    g(s, [x_ext, y_self]) = sensitivity * x_ext - s
    f(s, [x_ext, y_self]) = y_self                       # = rate * y_{t-1}

In continuous time this is exactly ds/dt = rate * (sensitivity * x - s),
the same leaky integrator as the original `manifold.py`. In discrete time
the self-loop carries a one-step lag (state at step t is updated using
the node's output from step t-1); for a slow rate this is invisible in
the plot.

The point of the self-channel framing is architectural: every input to
the node — including its own state via feedback — flows through the
same `Channel` mechanism, so connection-level concepts like plasticity
and phase-dependent gain compose uniformly across self-loops and
inter-node connections.
"""

from manifold import Channel, Environment, Node, plot, plot_complex, run


SENSITIVITY = 0.1
RATE = 0.1
INTERVAL = 111


def step_schedule(t: float) -> float:
    """0 -> 1 -> 0 step, switching at t=INTERVAL and t=2*INTERVAL."""
    if t < INTERVAL:
        return 0.0
    if t < 2 * INTERVAL:
        return 1.0
    return 0.0


def main():
    env = Environment(step_schedule)

    def f(s, xs, dt):
        return s + dt * xs[1]                       # state += dt * self-channel reading

    def g(s, xs):
        return SENSITIVITY * xs[0] - s              # output = ext - state

    node = Node(state=0 + 0j, dynamics=f, output=g)
    node.add_channel(Channel(env))                              # xs[0]: external
    node.add_channel(Channel(node, lambda y: RATE * y))         # xs[1]: self-loop

    history = run(
        sources=[env, node],
        n_steps=3 * INTERVAL,
        observers={
            "error":  lambda: node.read(),
            "memory": lambda: node.state,
            "input":  lambda: env.read(),
        },
    )

    times = list(range(len(history["error"])))
    plot(
        times,
        {
            "error":  [v.real for v in history["error"]],
            "memory": [v.real for v in history["memory"]],
        },
        title="01_adaptation",
    )
    plot_complex(
        times,
        {"state": history["memory"], "output": history["error"]},
        title="01_adaptation_complex",
    )


if __name__ == "__main__":
    main()
