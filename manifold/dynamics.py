"""Reusable dynamics (f) and output (g) functions for nodes.

Each builder returns a `(dynamics, output)` pair suitable for
`Node(dynamics=f, output=g)`. They operate on complex state — for the
adaptation regime only the real axis is exercised, but the same function
runs on the imaginary axis if the input has a phase component.
"""

from typing import Callable, Tuple

import numpy as np


def adaptation(
    sensitivity: float = 1.0,
    rate: float = 0.1,
) -> Tuple[Callable, Callable]:
    """Leaky-integrator adaptation.

      s(t+dt) = s + dt * rate * (sensitivity * sum(inputs) - s)
      output  = sensitivity * sum(inputs) - s

    The state is a low-pass of the (scaled) input; the output is the
    high-pass complement. With a step input the output spikes and decays
    back to zero as the state catches up.
    """

    def f(s: complex, xs, dt: float):
        x = sensitivity * sum(xs)
        return s + dt * rate * (x - s)

    def g(s: complex, xs):
        x = sensitivity * sum(xs)
        return x - s

    return f, g


def polar(
    sensitivity: float = 1.0,
    rate: float = 0.1,
    omega: float = 0.0,
    coupling: float = 0.0,
) -> Tuple[Callable, Callable]:
    """Adaptive oscillator in polar form.

    State s = r * exp(i * theta) evolves as:
      dr/dt     = rate * (sensitivity * |sum(inputs)| - r)
      dtheta/dt = omega + coupling * sum_j sin(arg(input_j) - theta)

    Magnitude follows leaky-integrator adaptation on the *magnitude* of
    the input sum. Phase rotates at intrinsic omega plus Kuramoto-style
    pull toward each input's phase.

    Limit cases:
      omega = coupling = 0           -> magnitude-only adaptation
      rate  = 0, coupling != 0       -> pure-phase Kuramoto on a fixed
                                        amplitude circle (use `kuramoto`)

    Output is the state itself.

    Singularity: at s = 0 the phase is undefined. Initialize state with
    a small nonzero magnitude (e.g. 0.001+0j) when omega != 0 so the
    rotation has somewhere to act.
    """

    def f(s: complex, xs, dt: float):
        r = abs(s)
        theta = np.angle(s) if r > 1e-12 else 0.0
        x = sensitivity * sum(xs)
        dr = rate * (abs(x) - r)
        if coupling and xs:
            # Magnitude-weighted Kuramoto: each input contributes
            # |xi| * sin(arg(xi) - theta). For unit-magnitude inputs this
            # is the standard Kuramoto term; for plastic channels the
            # weight magnitude scales the coupling strength (matches the
            # Sompolinsky J(r,r') formulation).
            coupling_term = sum(abs(xi) * np.sin(np.angle(xi) - theta) for xi in xs)
        else:
            coupling_term = 0.0
        dtheta = omega + coupling * coupling_term
        # Polar Euler: update r and theta separately, recombine. This
        # preserves magnitude exactly under pure rotation (whereas a
        # Cartesian Euler step on (dr + i*r*dtheta) would inflate it).
        r_new = max(0.0, r + dt * dr)
        theta_new = theta + dt * dtheta
        return r_new * np.exp(1j * theta_new)

    def g(s: complex, xs):
        return s

    return f, g


def kuramoto(omega: float = 0.0, coupling: float = 0.1) -> Tuple[Callable, Callable]:
    """Pure-phase Kuramoto oscillator: amplitude is fixed (rate=0), only
    phase evolves under intrinsic omega + coupling toward input phases.

    Initialize state on the unit circle (e.g. exp(1j*theta0)) so the
    amplitude stays at 1 throughout.
    """
    return polar(sensitivity=1.0, rate=0.0, omega=omega, coupling=coupling)


def hebbian(eta: float = 0.001, decay: float = 0.0) -> Callable:
    """Hebbian learning rule for a `PlasticChannel`.

      dw/dt = eta * Re(source * conj(dest)) - decay * w

    For unit-amplitude oscillators on the unit circle, Re(s*conj(d)) =
    cos(arg(s) - arg(d)): aligned phases -> positive update, anti-aligned
    -> negative. Decay term keeps weights bounded.

    Returns a `learn(w, sv, dv, dt) -> new_w` callable.
    """
    def learn(w: complex, sv: complex, dv: complex, dt: float) -> complex:
        update = (sv * np.conj(dv)).real
        return w + dt * (eta * update - decay * w)
    return learn


def sigmoid_activity(
    rate: float = 0.1,
    gain: float = 4.0,
    threshold: float = 0.0,
) -> Tuple[Callable, Callable]:
    """Sigmoidal amplitude dynamics for competitive networks.

      r(t+dt) = max(0, r + dt * rate * (sigmoid(gain*drive - threshold) - r))
      drive   = sum(xi.real for xi in inputs)

    Activity is bounded to [0,1] via the sigmoid; the floor is enforced
    after the update. Inputs contribute additively in their real part —
    excitatory channels with positive transforms drive activity up,
    inhibitory channels with negative transforms drive it down.

    Imag axis is preserved unchanged so this can compose with phase
    dynamics (e.g., a downstream phase-coupling layer).

    Bistability via mutual inhibition: when two such units inhibit each
    other and the gain is steep enough that the slope at the symmetric
    fixed point exceeds 1/inhibition, the symmetric solution becomes
    unstable and the asymmetric (winner-take-all) solutions are stable.
    """
    def f(s: complex, xs, dt: float) -> complex:
        drive = sum(xi.real for xi in xs)
        target = 1.0 / (1.0 + np.exp(-(gain * drive - threshold)))
        r = s.real
        r_new = r + dt * rate * (target - r)
        return complex(max(0.0, r_new), s.imag)
    def g(s: complex, xs):
        return s
    return f, g


def polar_sigmoid(
    rate: float = 0.1,
    gain: float = 8.0,
    threshold: float = 2.5,
    omega: float = 0.0,
    coupling: float = 1.0,
) -> Tuple[Callable, Callable]:
    """Sigmoidal amplitude + Kuramoto phase — both axes at work.

    Combines `sigmoid_activity` (bounded amplitude with sigmoidal target,
    used for bistable competition) with `polar` (intrinsic rotation +
    magnitude-weighted Kuramoto coupling on phase). State is complex:

      r_target = sigmoid(gain * sum(xi.real) - threshold)
      r_new    = max(0, r + dt * rate * (r_target - r))
      dtheta   = omega + coupling * sum |xi| * sin(arg(xi) - theta)
      s_new    = r_new * exp(i * (theta + dt*dtheta))

    Same channels do dual work:
      - positive transform → excite (r↑) AND in-phase coupling
      - negative transform → inhibit (r↓) AND anti-phase coupling (since
        arg(-x) = arg(x) + π, so sin(arg + π - theta) is the anti-phase
        Kuramoto term)

    Use this when an experiment should produce *both* a bistable
    amplitude pattern (which features are active) AND a phase-coherence
    pattern (how the active features are bound together). The unified
    substrate the project's complex-state choice was set up for.

    Singularity: at s = 0 the phase is undefined; we default theta = 0.
    Initialize state with a small nonzero magnitude when ω != 0 so the
    phase has somewhere to rotate from.
    """
    def f(s: complex, xs, dt: float) -> complex:
        r = abs(s)
        theta = np.angle(s) if r > 1e-12 else 0.0

        drive = sum(xi.real for xi in xs)
        target_r = 1.0 / (1.0 + np.exp(-(gain * drive - threshold)))
        r_new = r + dt * rate * (target_r - r)
        r_new = max(0.0, r_new)

        if coupling and xs:
            coupling_term = sum(abs(xi) * np.sin(np.angle(xi) - theta) for xi in xs)
        else:
            coupling_term = 0.0
        theta_new = theta + dt * (omega + coupling * coupling_term)

        return r_new * np.exp(1j * theta_new)

    def g(s: complex, xs):
        return s

    return f, g


def tracker(rate: float = 0.01) -> Tuple[Callable, Callable]:
    """Slow leaky integrator: state tracks the sum of its real-part inputs.
    Output = state (so downstream channels read the integrated value).

    Used as the adaptation memory for `sigmoid_activity` nodes. Pair with
    a slow rate (rate << activity rate) and wire its output back into the
    activity node with a negative transform — that's the relaxation-
    oscillator engine.
    """
    def f(s: complex, xs, dt: float) -> complex:
        target = sum(xi.real for xi in xs)
        r_new = s.real + dt * rate * (target - s.real)
        return complex(r_new, s.imag)
    def g(s: complex, xs):
        return s
    return f, g


def homeostatic_feedback(target: float = 0.5, gain: float = 5.0) -> Callable:
    """Channel transform turning a tracker's output into homeostatic feedback.

      transform(y) = -gain * (Re(y) - target)

    Wire as a `Channel` from a slow tracker of a node's activity back into
    the node itself. When the tracker is above target, the feedback is
    negative (suppressing drive); below target, positive (boosting drive).
    Activity is regulated toward `target` on the tracker's timescale.

    Composes with adaptation memory and mutual inhibition; provides the
    *slow* regulation that keeps a competing population in its operating
    range without depending on precise parameter tuning.
    """
    def transform(y: complex) -> complex:
        return complex(-gain * (y.real - target), 0.0)
    return transform


def gated_hebbian(eta_max: float = 0.003, decay: float = 0.003) -> Callable:
    """Hebbian learning with mid-activity gating.

      dw/dt = eta_max * gate(dv) * Re(sv * conj(dv)) - decay * w
      gate(dv) = max(0, 4 * Re(dv) * (1 - Re(dv)))

    The 4*x*(1-x) gating peaks at dv=0.5 and is zero at dv=0 or 1. Means
    learning happens only when the destination is in a *learnable* range:
    saturated and silent units don't entrench their current state via
    runaway plasticity. This is one of the four robustness mechanisms
    (homeostasis, weight normalization, noise, gated plasticity) that
    turn fine-tuned models into robust ones.
    """
    def learn(w: complex, sv: complex, dv: complex, dt: float) -> complex:
        d = dv.real
        gate = 4.0 * d * (1.0 - d) if 0.0 <= d <= 1.0 else 0.0
        if gate < 0.0:
            gate = 0.0
        update = (sv * np.conj(dv)).real
        return w + dt * (eta_max * gate * update - decay * w)
    return learn


def stdp_sin(eta: float = 0.001, decay: float = 0.0) -> Callable:
    """Spike-timing-dependent plasticity rule using phase difference.

      dw/dt = eta * sin(arg(source) - arg(dest)) - decay * w

    Source leading by ~pi/2 strengthens (positive update); source lagging
    weakens. Anti-symmetric and biphasic, like the biological STDP kernel
    when phases stand in for spike times.

    Returns a `learn(w, sv, dv, dt) -> new_w` callable.
    """
    def learn(w: complex, sv: complex, dv: complex, dt: float) -> complex:
        if abs(sv) < 1e-12 or abs(dv) < 1e-12:
            update = 0.0
        else:
            update = np.sin(np.angle(sv) - np.angle(dv))
        return w + dt * (eta * update - decay * w)
    return learn
