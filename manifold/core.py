"""Core abstractions for adaptive networks.

A `Source` is anything with a `.read()` method that returns a complex value.
A `Channel` wires a `Source` through a transform applied at the point of contact.
A `Node` has complex state plus a list of channels, with `f` (dynamics) and
`g` (output) callables.
An `Environment` is a time-driven `Source` whose value follows a schedule
(and can be extended with its own dynamics for closed-loop setups).

State is complex throughout: the real axis carries level/amplitude (the
adaptation regime) and the imaginary axis is reserved for phase (the
synchronization regime). A model that only uses one axis still lives in
the same substrate as a model that uses both.

Update model: each step is two-phase. Every source's `compute(dt)` runs
first (computing next state/output without exposing it), then every
source's `commit()` runs (promoting next -> current). This way nodes
that read each other see a consistent previous-step snapshot.
"""

from typing import Callable, List, Optional, Sequence, Union

import numpy as np


Number = Union[complex, float, int]
# Dynamics returns the NEXT state (not the derivative). Each dynamics
# function is responsible for its own integration: Euler for linear
# dynamics, exponential for rotations, RK4 if you want it. Receives dt
# so the same function can be driven at any timestep.
Dynamics = Callable[[complex, List[complex], float], complex]
Output = Callable[[complex, List[complex]], complex]


class Source:
    """Anything readable by a Channel. Subclasses override `read`; sources
    that have their own internal dynamics also override `compute`/`commit`."""

    def read(self) -> complex:
        raise NotImplementedError

    def compute(self, dt: float = 1.0) -> None:
        pass

    def commit(self) -> None:
        pass


class Channel:
    """A wire from a Source through a transform.

    The transform is the function applied at the point of contact — for
    example a sensitivity scaling, a sign inversion, or a fixed coupling
    strength. For weights that change over time, use `PlasticChannel`.

    Channels have compute/commit so subclasses can carry their own state
    (a learnable weight, a delay buffer, etc.). The base class is purely
    functional — its compute/commit are no-ops.
    """

    def __init__(
        self,
        source: Source,
        transform: Optional[Callable[[complex], complex]] = None,
    ):
        self.source = source
        self.transform = transform if transform is not None else (lambda y: y)

    def read(self) -> complex:
        return self.transform(self.source.read())

    def compute(self, dt: float = 1.0) -> None:
        pass

    def commit(self) -> None:
        pass


class PlasticChannel(Channel):
    """A channel with a learnable scalar weight.

    Each step: read returns `weight * transform(source.read())`, and
    `learn(weight, source_value, dest_state, dt)` returns the new weight
    based on whatever rule the user supplies (Hebbian, STDP, etc.).

    The destination node is needed so the rule can compare source and
    destination (e.g. for spike-timing or phase-coincidence rules).
    Within Node.compute() the channel's compute() runs first, so the
    plastic update sees the previous-step source and dest values; the
    new weight only takes effect after commit.
    """

    def __init__(
        self,
        source: Source,
        dest: "Node",
        weight: Number = 1.0,
        learn: Optional[Callable[[complex, complex, complex, float], complex]] = None,
        transform: Optional[Callable[[complex], complex]] = None,
    ):
        super().__init__(source, transform)
        self.dest = dest
        self.weight: complex = complex(weight)
        self._next_weight: complex = self.weight
        self.learn = learn if learn is not None else (lambda w, sv, dv, dt: w)

    def read(self) -> complex:
        return self.weight * self.transform(self.source.read())

    def compute(self, dt: float = 1.0) -> None:
        sv = self.transform(self.source.read())
        dv = self.dest.state
        self._next_weight = complex(self.learn(self.weight, sv, dv, dt))

    def commit(self) -> None:
        self.weight = self._next_weight


class Node(Source):
    """A dynamical node with complex state.

    Each step:
      inputs       = [c.read() for c in channels]
      next_output  = output(state, inputs)               # uses CURRENT state
      next_state   = dynamics(state, inputs, dt)         # full discrete update

    Output is computed from the pre-update state so that a sudden change
    in input shows up immediately in the output (matching the high-pass
    behavior of the original adaptation model).

    Dynamics returns the next state directly (not a derivative). This
    means each dynamics function picks its own integration scheme — Euler
    is fine for linear adaptation, but a rotating oscillator should use
    polar/exponential update to preserve magnitude.
    """

    def __init__(
        self,
        state: Number = 0 + 0j,
        dynamics: Optional[Dynamics] = None,
        output: Optional[Output] = None,
        channels: Optional[Sequence[Channel]] = None,
    ):
        self.state: complex = complex(state)
        self.channels: List[Channel] = list(channels) if channels else []
        # Default dynamics: state stays put.
        self.dynamics: Dynamics = dynamics if dynamics is not None else (lambda s, xs, dt: s)
        self.output_fn: Output = output if output is not None else (lambda s, xs: s)

        # Output starts at 0; the first compute() fills it from output_fn.
        # This avoids needing all channels (including self-loops) wired at
        # construction time.
        self._output: complex = 0 + 0j
        self._next_state: complex = self.state
        self._next_output: complex = self._output

    def read(self) -> complex:
        return self._output

    def add_channel(self, channel: Channel) -> "Node":
        self.channels.append(channel)
        return self

    def compute(self, dt: float = 1.0) -> None:
        # Step plastic channels first so their weight updates see this
        # step's pre-commit source and dest values. Channel reads below
        # still use the OLD weights (commit hasn't happened yet).
        for channel in self.channels:
            channel.compute(dt)
        inputs = [channel.read() for channel in self.channels]
        self._next_output = complex(self.output_fn(self.state, inputs))
        self._next_state = complex(self.dynamics(self.state, inputs, dt))

    def commit(self) -> None:
        for channel in self.channels:
            channel.commit()
        self.state = self._next_state
        self._output = self._next_output


class Environment(Source):
    """A time-varying source driven by a schedule(t) -> value callable.

    Subclass and override `compute`/`commit` for closed-loop environments
    that read from the network and evolve their own internal state.
    """

    def __init__(self, schedule: Callable[[float], Number], t0: float = 0.0):
        self.schedule = schedule
        self.t: float = t0
        self._output: complex = complex(schedule(t0))
        self._next_output: complex = self._output

    def read(self) -> complex:
        return self._output

    def compute(self, dt: float = 1.0) -> None:
        self.t += dt
        self._next_output = complex(self.schedule(self.t))

    def commit(self) -> None:
        self._output = self._next_output


class Constant(Source):
    """A fixed-value source. Useful as a placeholder or test stub."""

    def __init__(self, value: Number = 0 + 0j):
        self._value: complex = complex(value)

    def read(self) -> complex:
        return self._value


class StateView(Source):
    """Read a Node's internal state directly, rather than its output.

    Useful when feedback should tap the low-pass tracker (memory) rather
    than the high-pass output. During compute(), returns the state from
    the previous commit — same lag semantics as reading any other Source.
    """

    def __init__(self, node: "Node"):
        self.node = node

    def read(self) -> complex:
        return self.node.state


class Noise(Source):
    """Gaussian noise source — fresh random complex value each step.

    Adds stochastic exploration to the network. Place a `Channel` from
    this source into any node that should receive noise. Real-only by
    default to avoid coupling phase dynamics; pass `imag_std=...` for
    complex noise.
    """

    def __init__(self, std: float = 0.01, imag_std: float = 0.0, seed: Optional[int] = None):
        self.std = std
        self.imag_std = imag_std
        self.rng = np.random.default_rng(seed)
        self._output: complex = 0 + 0j
        self._next_output: complex = 0 + 0j

    def read(self) -> complex:
        return self._output

    def compute(self, dt: float = 1.0) -> None:
        scale = float(np.sqrt(dt))
        re = self.rng.normal(0.0, self.std * scale) if self.std > 0 else 0.0
        im = self.rng.normal(0.0, self.imag_std * scale) if self.imag_std > 0 else 0.0
        self._next_output = complex(re, im)

    def commit(self) -> None:
        self._output = self._next_output


class WeightNormalizer(Source):
    """Synaptic-scaling utility: rescales a group of `PlasticChannel` weights
    so the sum of |weight| matches `target_sum`.

    Acts as a `Source` (read returns 0 — it doesn't participate in the
    signal flow). Its commit step rescales weights — so place it in the
    `run` sources list AFTER the channels' owning node, so the rescaling
    sees the post-Hebbian-update weights for that step.

    Synaptic scaling is the standard implementation of the Turrigiano-style
    homeostatic-plasticity insight: keep total synaptic input bounded
    while letting Hebbian rebalance among individual weights.
    """

    def __init__(self, channels: Sequence["PlasticChannel"], target_sum: float = 1.0):
        self.channels = list(channels)
        self.target_sum = target_sum

    def read(self) -> complex:
        return 0 + 0j

    def compute(self, dt: float = 1.0) -> None:
        pass

    def commit(self) -> None:
        total = sum(abs(c.weight) for c in self.channels)
        if total > 1e-12:
            scale = self.target_sum / total
            for c in self.channels:
                c.weight = c.weight * scale


def run(
    sources: Sequence[Source],
    n_steps: int,
    dt: float = 1.0,
    observers: Optional[dict] = None,
) -> dict:
    """Drive a list of sources for n_steps with two-phase updates.

    `observers` is an optional dict of {name: callable} read after each
    commit; results accumulate as lists and are returned keyed by name.
    """
    history: dict = {name: [] for name in (observers or {})}
    for _ in range(n_steps):
        for src in sources:
            src.compute(dt)
        for src in sources:
            src.commit()
        for name, fn in (observers or {}).items():
            history[name].append(fn())
    return history
