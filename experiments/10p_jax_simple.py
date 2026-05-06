"""Phase 10p: JAX port of 10c_simple — proof-of-concept speedup.

Direct JAX implementation of the same 16-node bistable Necker cube as
10c_necker_simple. Hand-coded (no auto-compilation from OOP network);
the goal here is to measure the speedup we can expect once we build a
full compiler later.

Architecture:
  - 16 activity nodes laid out as a flat array, indexed [front_v_idx ... back_v_idx]
  - 16 memory nodes (adaptation trackers), parallel array
  - Channel topology baked into static index arrays (src_indices, dst_indices, weights)
  - One jit-compiled step function: read channels, aggregate per-dest, apply sigmoid, update memory
  - jax.lax.scan over N_STEPS for the run loop

Same parameters as 10c_simple. Validates: amplitude bistability appears (Pattern A vs B
alternating), period set by adaptation rate. Times: compare to slow Python version.
"""

import itertools
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


VERTICES = list(itertools.product([0, 1], repeat=3))
N_VERTICES = len(VERTICES)              # 8
N_NODES = 2 * N_VERTICES                # 16: front + back per vertex

# Same params as 10c_necker_simple
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
DT = 1.0
SEED = 42


def vertex_idx(v, side):
    """Flat node index. side=0 → front, side=1 → back."""
    vi = VERTICES.index(v)
    return side * N_VERTICES + vi


def neighbor(v, axis):
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def build_channel_arrays():
    """Build static (src, dst, weight) arrays for all channels.

    Returns three numpy arrays. The step kernel uses these to compute
    per-channel inputs and aggregate by destination via segment_sum.
    """
    srcs, dsts, weights = [], [], []

    for v in VERTICES:
        for side, sib_side in [(0, 1), (1, 0)]:  # 0=front, 1=back
            self_idx = vertex_idx(v, side)
            sib_idx = vertex_idx(v, sib_side)

            # Within-vertex mutual inhibition
            srcs.append(sib_idx); dsts.append(self_idx); weights.append(-INHIB_INTRATILE)

            # Cross-tile constraints
            for axis in (0, 1, 2):
                n = neighbor(v, axis)
                # Each axis: front[v] gets EXCITE from same-side neighbor (X,Y)
                # OR from opposite-side neighbor (Z), and INHIB from the other.
                if axis in (0, 1):  # X, Y face edges: same-depth excite
                    same_idx = vertex_idx(n, side)
                    opp_idx  = vertex_idx(n, sib_side)
                    srcs.append(same_idx); dsts.append(self_idx); weights.append(EXCITE_CROSS)
                    srcs.append(opp_idx);  dsts.append(self_idx); weights.append(-INHIB_CROSS)
                else:  # Z edge: opposite-depth excite
                    same_idx = vertex_idx(n, side)
                    opp_idx  = vertex_idx(n, sib_side)
                    srcs.append(opp_idx);  dsts.append(self_idx); weights.append(EXCITE_CROSS)
                    srcs.append(same_idx); dsts.append(self_idx); weights.append(-INHIB_CROSS)

    return jnp.array(srcs), jnp.array(dsts), jnp.array(weights, dtype=jnp.float32)


def main():
    rng = np.random.default_rng(SEED)

    # Initial activity: small random per node (real-valued; phase axis unused here)
    a0 = jnp.array(0.3 * rng.random(N_NODES), dtype=jnp.float32)
    m0 = jnp.zeros(N_NODES, dtype=jnp.float32)

    src_idx, dst_idx, w_arr = build_channel_arrays()
    n_channels = src_idx.shape[0]
    print(f"Network: {N_NODES} nodes, {n_channels} channels")

    image_drive = jnp.full((N_NODES,), IMAGE_DRIVE, dtype=jnp.float32)

    @jax.jit
    def step(state, _):
        a, m = state
        # Per-channel input: weight * source_activity
        ch_inputs = w_arr * a[src_idx]
        # Aggregate per destination
        cross_drive = jax.ops.segment_sum(ch_inputs, dst_idx, num_segments=N_NODES)
        # Total drive: image + cross-tile - adaptation feedback
        drive = image_drive + cross_drive - ADAPT_FEEDBACK * m
        # Sigmoid-bounded amplitude target
        target = jax.nn.sigmoid(GAIN * drive - THRESHOLD)
        a_new = a + DT * RATE_FAST * (target - a)
        a_new = jnp.maximum(a_new, 0.0)
        m_new = m + DT * RATE_ADAPT * (a_new - m)
        return (a_new, m_new), a_new

    init_state = (a0, m0)

    # JIT warmup (first call compiles)
    print("JIT-compiling and running...")
    t0 = time.time()
    final_state, history = jax.lax.scan(step, init_state, xs=None, length=N_STEPS)
    history.block_until_ready()  # wait for async XLA compute
    t1 = time.time()
    print(f"  total wall time (incl. compilation): {t1 - t0:.3f} s")

    # Run again to time without compilation
    t0 = time.time()
    final_state, history = jax.lax.scan(step, init_state, xs=None, length=N_STEPS)
    history.block_until_ready()
    t1 = time.time()
    print(f"  total wall time (compiled, second run): {t1 - t0:.3f} s")
    print(f"  steps/sec: {N_STEPS / (t1 - t0):,.0f}")

    # Convert to numpy for plotting
    history_np = np.asarray(history)  # shape (N_STEPS, N_NODES)

    # Pattern A and B subset means
    a_idxs = [vertex_idx(v, 0 if v[2] == 0 else 1) for v in VERTICES]
    b_idxs = [vertex_idx(v, 1 if v[2] == 0 else 0) for v in VERTICES]
    pa = history_np[:, a_idxs].mean(axis=1)
    pb = history_np[:, b_idxs].mean(axis=1)

    times = list(range(N_STEPS))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, pa, label="Pattern A subset", color="tab:blue")
    ax.plot(times, pb, label="Pattern B subset", color="tab:orange")
    ax.set_xlabel("time step")
    ax.set_ylabel("mean amplitude")
    ax.set_title("10p — JAX port of 10c_simple (same dynamics, vectorized)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("out/10p_jax_simple.png", dpi=110)
    plt.close(fig)
    print("Saved out/10p_jax_simple.png")


if __name__ == "__main__":
    main()
