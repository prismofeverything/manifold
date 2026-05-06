"""Validate compile_to_jax MVP on 10c_simple's network.

Builds the same OOP network as `10c_necker_simple.py` (same 16 activity
nodes + 16 trackers, same cross-tile constraints), then:
  1. Runs the slow Python path (the existing `manifold.run`) — reference.
  2. Compiles the same network via `compile_to_jax` and runs with `lax.scan`.
  3. Times both and compares the resulting Pattern A / Pattern B traces.

If the two paths produce qualitatively the same bistable cycling, the
compiler is correct on this subset of primitives.
"""

import itertools
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from manifold import (
    Channel, Constant, Node, run, sigmoid_activity, tracker,
)
from manifold.jax_compile import compile_to_jax


VERTICES = list(itertools.product([0, 1], repeat=3))

IMAGE_DRIVE = 0.5
INHIB_INTRATILE = 1.5
EXCITE_CROSS = 0.4
INHIB_CROSS = 0.4
ADAPT_FEEDBACK = 1.5
RATE_FAST = 0.1
RATE_ADAPT = 0.005
GAIN = 8.0
THRESHOLD = 2.5
N_STEPS = 6000
SEED = 42


def neighbor(v, axis):
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def build_network():
    """Same construction as 10c_necker_simple, returns sources list +
    handles to the activity nodes (for measurement)."""
    image = Constant(IMAGE_DRIVE)
    rng = np.random.default_rng(SEED)

    front, back, fmem, bmem = {}, {}, {}, {}
    for v in VERTICES:
        f, g = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD)
        front[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        back[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        mf, mg = tracker(rate=RATE_ADAPT)
        fmem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)
        bmem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)

    for v in VERTICES:
        for det, sib, mem in [(front[v], back[v], fmem[v]),
                              (back[v], front[v], bmem[v])]:
            det.add_channel(Channel(image))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK * y))
            mem.add_channel(Channel(det))
            det.add_channel(Channel(sib, transform=lambda y: -INHIB_INTRATILE * y))
        for axis in (0, 1, 2):
            n = neighbor(v, axis)
            if axis in (0, 1):
                front[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS * y))
                front[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS * y))
            else:
                front[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS * y))
                front[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS * y))

    sources = [image]
    for v in VERTICES:
        sources.extend([front[v], back[v], fmem[v], bmem[v]])

    return sources, front, back


def main():
    sources, front, back = build_network()

    # Slow Python reference run
    print("=== Slow Python reference ===")
    obs = {f"f{v}": (lambda v=v: front[v].state.real) for v in VERTICES}
    obs.update({f"b{v}": (lambda v=v: back[v].state.real) for v in VERTICES})
    t0 = time.time()
    history_slow = run(sources=sources, n_steps=N_STEPS, observers=obs)
    t_slow = time.time() - t0
    print(f"  ran {N_STEPS} steps in {t_slow:.3f} s ({N_STEPS/t_slow:,.0f} steps/s)")

    pa_slow = np.mean([
        [history_slow[f"f{v}"][t] if v[2] == 0 else history_slow[f"b{v}"][t] for v in VERTICES]
        for t in range(N_STEPS)
    ], axis=1)
    pb_slow = np.mean([
        [history_slow[f"b{v}"][t] if v[2] == 0 else history_slow[f"f{v}"][t] for v in VERTICES]
        for t in range(N_STEPS)
    ], axis=1)

    # Re-build network (the slow run mutated state) for the JAX path
    sources, front, back = build_network()

    print("\n=== Compiled JAX path ===")
    t0 = time.time()
    init_state, step = compile_to_jax(sources, dt=1.0)
    t_compile = time.time() - t0
    print(f"  compiled in {t_compile:.3f} s")

    # Find which compiled-state indices correspond to which Node
    # (compile_to_jax flattens all Nodes; index order = source list order, Nodes only)
    nodes_in_order = [s for s in sources if isinstance(s, Node)]
    node_idx = {id(n): i for i, n in enumerate(nodes_in_order)}
    a_idxs = [node_idx[id(front[v]) if v[2] == 0 else id(back[v])] for v in VERTICES]
    b_idxs = [node_idx[id(back[v]) if v[2] == 0 else id(front[v])] for v in VERTICES]
    a_idxs = jnp.array(a_idxs)
    b_idxs = jnp.array(b_idxs)

    # First scan triggers JIT compile + trace
    t0 = time.time()
    final_state, history_jax = jax.lax.scan(step, init_state, xs=None, length=N_STEPS)
    history_jax.block_until_ready()
    t_first = time.time() - t0
    print(f"  first scan (with JIT compilation): {t_first:.3f} s")

    # Second run hits the compile cache
    t0 = time.time()
    final_state, history_jax = jax.lax.scan(step, init_state, xs=None, length=N_STEPS)
    history_jax.block_until_ready()
    t_jax = time.time() - t0
    print(f"  second scan (compiled): {t_jax:.4f} s ({N_STEPS/t_jax:,.0f} steps/s)")
    print(f"\nSpeedup vs slow Python: {t_slow / t_jax:.0f}x")

    # Compare patterns
    history_jax_np = np.asarray(history_jax)  # (N_STEPS, n_nodes)
    pa_jax = history_jax_np[:, a_idxs].mean(axis=1)
    pb_jax = history_jax_np[:, b_idxs].mean(axis=1)

    times = np.arange(N_STEPS)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(times, pa_slow, color="tab:blue", label="slow Python A", linewidth=1.0)
    axes[0].plot(times, pb_slow, color="tab:orange", label="slow Python B", linewidth=1.0)
    axes[0].set_ylabel("amplitude (slow)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Slow Python:  {t_slow:.2f} s")

    axes[1].plot(times, pa_jax, color="tab:blue", label="JAX compiled A", linewidth=1.0)
    axes[1].plot(times, pb_jax, color="tab:orange", label="JAX compiled B", linewidth=1.0)
    axes[1].set_ylabel("amplitude (JAX)")
    axes[1].set_xlabel("time step")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_title(f"JAX compiled:  {t_jax:.4f} s  ({t_slow / t_jax:.0f}× faster)")

    fig.tight_layout()
    fig.savefig("out/10q_jax_compile_validation.png", dpi=110)
    plt.close(fig)
    print("Saved out/10q_jax_compile_validation.png")


if __name__ == "__main__":
    main()
