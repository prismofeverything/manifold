"""Marked transform helpers for JAX compilation.

Each helper returns a callable that ALSO carries `_meta` describing its
shape and parameters. The slow Python path uses them as plain callables;
the JAX compiler reads the metadata to generate vectorized kernels.

Existing experiments using bare lambdas (e.g., `lambda y: -1.5 * y`)
still work in the slow path — the compiler tries to infer "linear" via
probing if `_meta` is absent. Use these helpers when authoring new
experiments to make compilation explicit and unambiguous.
"""

from typing import Any


def linear(w):
    """y → w · y, where w is a static (real or complex) scalar."""
    fn = lambda y: w * y
    fn._meta = {"kind": "linear", "params": {"w": complex(w)}}
    return fn


def identity():
    """y → y. Equivalent to linear(1)."""
    return linear(1)


def real_only_feedback(target=0.0, gain=1.0):
    """y → -gain · (y.real − target). Used for adaptation (target=0) or
    homeostasis (target>0). Output is real-valued (imag = 0)."""
    fn = lambda y: complex(-gain * (y.real - target), 0)
    fn._meta = {"kind": "real_only_feedback",
                "params": {"target": float(target), "gain": float(gain)}}
    return fn


def abs_to_real():
    """y → complex(|y|, 0). Used to feed an amplitude tracker the
    magnitude of a polar_sigmoid node (otherwise it would track the
    oscillating real part)."""
    fn = lambda y: complex(abs(y), 0)
    fn._meta = {"kind": "abs_to_real", "params": {}}
    return fn


def infer_transform_meta(transform) -> dict:
    """Best-effort metadata inference for unmarked transforms.

    Probes the transform with a few test inputs to see if it's linear.
    Returns the meta dict, or raises if it cannot be inferred.
    """
    if hasattr(transform, "_meta"):
        return transform._meta

    try:
        a = transform(complex(1.0, 0.0))
        b = transform(complex(2.0, 0.0))
        c = transform(complex(0.0, 0.0))
    except Exception as e:
        raise ValueError(f"Could not probe transform {transform}: {e}")

    a, b, c = complex(a), complex(b), complex(c)
    # Linear iff f(0)=0 and f(2) = 2·f(1)
    if abs(c) < 1e-9 and abs(b - 2 * a) < 1e-9:
        return {"kind": "linear", "params": {"w": a}}

    raise ValueError(
        f"Cannot infer transform shape for {transform}. "
        f"f(0)={c}, f(1)={a}, f(2)={b}. "
        f"Use one of the marked helpers in manifold.transforms."
    )
