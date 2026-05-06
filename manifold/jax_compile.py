"""compile_to_jax — turn an OOP `Source`/`Channel`/`Node` network into a
(initial_state, jit-compiled step_fn) pair for fast JAX execution.

Coverage:
  Sources:    Constant, Node, Noise, StateView (resolved to its node),
              WeightNormalizer (post-step rescale, not signal-flow source)
  Dynamics:   sigmoid_activity, tracker, polar_sigmoid, polar, adaptation
  Transforms: linear (probed from bare lambdas if unmarked),
              real_only_feedback, abs_to_real
  Plasticity: hebbian, gated_hebbian, stdp_sin

State is a pytree:
  state = {
    "node_states":      complex64 array, shape (n_nodes,)
    "plastic_weights":  float32 array,   shape (n_plastic,)
    "key":              PRNGKey for noise (advanced each step)
  }

Adding a new primitive: tag the factory with `_meta`, add a kernel
branch in `_make_step`, and (if it's a transform) add a probe rule to
`transforms.infer_transform_meta`.
"""

import collections
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .core import (
    Channel, Constant, Node, Noise, PlasticChannel, Source, StateView,
    WeightNormalizer,
)
from .transforms import infer_transform_meta


class CompileError(Exception):
    pass


SUPPORTED_DYNAMICS = {
    "sigmoid_activity",
    "tracker",
    "polar_sigmoid",
    "polar",
    "adaptation",
}
SUPPORTED_TRANSFORMS = {"linear", "real_only_feedback", "abs_to_real"}
SUPPORTED_LEARN = {"hebbian", "gated_hebbian", "stdp_sin"}


def _resolve_source(src):
    """If src is a StateView, follow to underlying node (only valid for
    identity-output nodes)."""
    if isinstance(src, StateView):
        node = src.node
        out_meta = getattr(node.output_fn, "_meta", None)
        if out_meta is None or out_meta["kind"] != "identity_output":
            raise CompileError(
                f"StateView only supported for nodes with identity_output (g(s,xs)=s). "
                f"Node has output_fn meta: {out_meta}"
            )
        return node
    return src


def _packed_param_arrays(items, param_keys):
    """Items is list of (idx, params dict). Return dict mapping each key
    to an array of values, ordered by item order."""
    out = {}
    for k in param_keys:
        out[k] = jnp.array([p[k] for _, p in items], dtype=jnp.float32)
    return out


def compile_to_jax(sources, dt: float = 1.0, seed: int = 0):
    """Compile an OOP network into (init_state, step_fn).

    The returned `step_fn(state, _)` takes the state pytree and an
    unused per-step input (for `lax.scan`), and returns
    (new_state, observed_node_states). Observed_node_states is the
    complex node states array of shape (n_nodes,) for that step.
    """
    # Categorize
    nodes = [s for s in sources if isinstance(s, Node)]
    constants = [s for s in sources if isinstance(s, Constant)]
    noises = [s for s in sources if isinstance(s, Noise)]
    normalizers = [s for s in sources if isinstance(s, WeightNormalizer)]
    others = [s for s in sources
              if not isinstance(s, (Node, Constant, Noise, WeightNormalizer))]
    if others:
        kinds = {type(s).__name__ for s in others}
        raise CompileError(f"Unsupported source types: {sorted(kinds)}")

    # Auto-discover Constants and Noise referenced by channels but not
    # in the sources list (common pattern: experiments only register Nodes).
    seen_const_ids = {id(c) for c in constants}
    seen_noise_ids = {id(n) for n in noises}
    for node in nodes:
        for ch in node.channels:
            src = _resolve_source(ch.source)
            if isinstance(src, Constant) and id(src) not in seen_const_ids:
                constants.append(src)
                seen_const_ids.add(id(src))
            elif isinstance(src, Noise) and id(src) not in seen_noise_ids:
                noises.append(src)
                seen_noise_ids.add(id(src))

    n_nodes = len(nodes)
    if n_nodes == 0:
        raise CompileError("No Node sources to compile.")

    node_to_idx = {id(n): i for i, n in enumerate(nodes)}
    const_to_idx = {id(c): i for i, c in enumerate(constants)}
    noise_to_idx = {id(n): i for i, n in enumerate(noises)}

    # ---- Group nodes by dynamics kind ----
    dynamics_groups: dict = collections.OrderedDict()
    for i, node in enumerate(nodes):
        meta = getattr(node.dynamics, "_meta", None)
        if meta is None:
            raise CompileError(f"Node {i} has unmarked dynamics.")
        kind = meta["kind"]
        if kind not in SUPPORTED_DYNAMICS:
            raise CompileError(f"Unsupported dynamics kind: {kind}")
        dynamics_groups.setdefault(kind, []).append((i, meta["params"]))

    # ---- Walk channels: classify by transform kind, separate plastic ----
    # Store as lists of dicts; convert to arrays at the end.
    static_chs = collections.defaultdict(list)  # kind -> list of (src_kind, src_idx, dst_idx, params)
    plastic_chs = []                            # list of (src_kind, src_idx, dst_idx, init_weight, learn_meta)

    for dst_i, node in enumerate(nodes):
        for ch in node.channels:
            src = _resolve_source(ch.source)
            # Determine source kind + idx
            if id(src) in node_to_idx:
                src_kind, src_idx = "node", node_to_idx[id(src)]
            elif id(src) in const_to_idx:
                src_kind, src_idx = "constant", const_to_idx[id(src)]
            elif id(src) in noise_to_idx:
                src_kind, src_idx = "noise", noise_to_idx[id(src)]
            else:
                raise CompileError(
                    f"Channel into node {dst_i} has source not in compiled set: {src}"
                )

            # Plastic channel
            if isinstance(ch, PlasticChannel):
                # plastic transform must be linear-like (we'll assume the
                # plastic stores `weight` and the transform may also scale)
                # For now: composite weight = ch.weight * (transform's effective scale)
                t_meta = infer_transform_meta(ch.transform)
                if t_meta["kind"] != "linear":
                    raise CompileError(
                        f"Plastic channel transform must be linear; got {t_meta['kind']}"
                    )
                # The static `transform_w` factor multiplies into the weight effective.
                transform_w = t_meta["params"]["w"].real
                learn_meta = getattr(ch.learn, "_meta", None)
                if learn_meta is None:
                    raise CompileError(
                        f"PlasticChannel into node {dst_i} has unmarked learn rule."
                    )
                if learn_meta["kind"] not in SUPPORTED_LEARN:
                    raise CompileError(
                        f"Unsupported plasticity rule: {learn_meta['kind']}"
                    )
                plastic_chs.append({
                    "src_kind": src_kind, "src_idx": src_idx, "dst_idx": dst_i,
                    "init_weight": float(ch.weight.real),
                    "transform_w": float(transform_w),
                    "learn_kind": learn_meta["kind"],
                    "learn_params": learn_meta["params"],
                    "channel_obj": ch,
                })
                continue

            # Static channel — classify by transform kind
            try:
                t_meta = infer_transform_meta(ch.transform)
            except ValueError as e:
                raise CompileError(f"Channel into node {dst_i}: {e}") from e
            t_kind = t_meta["kind"]
            if t_kind not in SUPPORTED_TRANSFORMS:
                raise CompileError(f"Unsupported transform kind: {t_kind}")
            static_chs[t_kind].append({
                "src_kind": src_kind, "src_idx": src_idx, "dst_idx": dst_i,
                "params": t_meta["params"],
            })

    # ---- Build channel arrays per transform kind ----
    def _arr(values, dtype=jnp.int32):
        return jnp.array(values, dtype=dtype) if values else jnp.zeros(0, dtype=dtype)

    # Linear channels split by source kind (so we can index into the
    # right state array)
    lin_node = static_chs.get("linear", [])
    lin_node_node = [c for c in lin_node if c["src_kind"] == "node"]
    lin_node_const = [c for c in lin_node if c["src_kind"] == "constant"]
    lin_node_noise = [c for c in lin_node if c["src_kind"] == "noise"]

    lin_n_src = _arr([c["src_idx"] for c in lin_node_node])
    lin_n_dst = _arr([c["dst_idx"] for c in lin_node_node])
    lin_n_w = _arr([c["params"]["w"].real for c in lin_node_node], dtype=jnp.float32)
    lin_n_w_imag = _arr([c["params"]["w"].imag for c in lin_node_node], dtype=jnp.float32)

    lin_c_src = _arr([c["src_idx"] for c in lin_node_const])
    lin_c_dst = _arr([c["dst_idx"] for c in lin_node_const])
    lin_c_w = _arr([c["params"]["w"].real for c in lin_node_const], dtype=jnp.float32)

    lin_no_src = _arr([c["src_idx"] for c in lin_node_noise])
    lin_no_dst = _arr([c["dst_idx"] for c in lin_node_noise])
    lin_no_w = _arr([c["params"]["w"].real for c in lin_node_noise], dtype=jnp.float32)

    # Real-only feedback channels (always node source in our experiments)
    rof = static_chs.get("real_only_feedback", [])
    rof_src = _arr([c["src_idx"] for c in rof])
    rof_dst = _arr([c["dst_idx"] for c in rof])
    rof_gain = _arr([c["params"]["gain"] for c in rof], dtype=jnp.float32)
    rof_target = _arr([c["params"]["target"] for c in rof], dtype=jnp.float32)

    # Abs-to-real channels
    atr = static_chs.get("abs_to_real", [])
    atr_src = _arr([c["src_idx"] for c in atr])
    atr_dst = _arr([c["dst_idx"] for c in atr])

    # Plastic channels (always linear, node source — verified above)
    pl_src = _arr([c["src_idx"] for c in plastic_chs])
    pl_dst = _arr([c["dst_idx"] for c in plastic_chs])
    pl_transform_w = _arr([c["transform_w"] for c in plastic_chs], dtype=jnp.float32)
    pl_init_weights = jnp.array([c["init_weight"] for c in plastic_chs],
                                dtype=jnp.float32) if plastic_chs else jnp.zeros(0, dtype=jnp.float32)
    pl_learn_kinds = [c["learn_kind"] for c in plastic_chs]
    pl_eta = _arr([c["learn_params"].get("eta", c["learn_params"].get("eta_max", 0.0))
                   for c in plastic_chs], dtype=jnp.float32)
    pl_decay = _arr([c["learn_params"]["decay"] for c in plastic_chs], dtype=jnp.float32)
    # Mask per learn kind (since different rules apply different update formulas)
    pl_kind_hebbian = jnp.array(
        [k == "hebbian" for k in pl_learn_kinds], dtype=jnp.float32,
    ) if pl_learn_kinds else jnp.zeros(0, dtype=jnp.float32)
    pl_kind_gated = jnp.array(
        [k == "gated_hebbian" for k in pl_learn_kinds], dtype=jnp.float32,
    ) if pl_learn_kinds else jnp.zeros(0, dtype=jnp.float32)
    pl_kind_stdp = jnp.array(
        [k == "stdp_sin" for k in pl_learn_kinds], dtype=jnp.float32,
    ) if pl_learn_kinds else jnp.zeros(0, dtype=jnp.float32)

    # Map plastic channel objects to indices for normalizer wiring
    plastic_obj_to_idx = {id(c["channel_obj"]): i for i, c in enumerate(plastic_chs)}

    # WeightNormalizer specs
    norm_specs = []
    for n in normalizers:
        idxs = []
        for c in n.channels:
            if id(c) not in plastic_obj_to_idx:
                raise CompileError(
                    "WeightNormalizer references a channel that wasn't compiled "
                    "(probably attached to a node not in the sources list)."
                )
            idxs.append(plastic_obj_to_idx[id(c)])
        norm_specs.append((jnp.array(idxs, dtype=jnp.int32), float(n.target_sum)))

    # ---- Dynamics dispatch tables (per-node param arrays) ----
    dynamics_dispatch = []
    for kind, items in dynamics_groups.items():
        idxs = jnp.array([i for i, _ in items], dtype=jnp.int32)
        if kind == "sigmoid_activity":
            params = _packed_param_arrays(items, ["rate", "gain", "threshold"])
        elif kind == "tracker":
            params = _packed_param_arrays(items, ["rate"])
        elif kind == "polar_sigmoid":
            params = _packed_param_arrays(items, ["rate", "gain", "threshold", "omega", "coupling"])
        elif kind == "polar":
            params = _packed_param_arrays(items, ["sensitivity", "rate", "omega", "coupling"])
        elif kind == "adaptation":
            params = _packed_param_arrays(items, ["sensitivity", "rate"])
        else:
            raise CompileError(f"Unsupported dynamics: {kind}")
        dynamics_dispatch.append((kind, idxs, params))

    # ---- Initial state ----
    init_node_states = jnp.array(
        [complex(n.state) for n in nodes], dtype=jnp.complex64,
    )

    # Constants: real-valued (we pass through linear-with-w on real values)
    constant_values = jnp.array(
        [float(c.read().real) for c in constants], dtype=jnp.float32,
    ) if constants else jnp.zeros(0, dtype=jnp.float32)

    # Noise stds (one std per noise source; imag_std unused for now)
    noise_stds = jnp.array(
        [float(n.std) for n in noises], dtype=jnp.float32,
    ) if noises else jnp.zeros(0, dtype=jnp.float32)
    n_noises = len(noises)

    init_state = {
        "node_states": init_node_states,
        "plastic_weights": pl_init_weights,
        "key": jax.random.PRNGKey(seed),
    }

    @jax.jit
    def step(state, _):
        node_states = state["node_states"]
        plastic_weights = state["plastic_weights"]
        key = state["key"]

        # Generate noise values for this step
        if n_noises > 0:
            key, subkey = jax.random.split(key)
            # std of the per-step value scales with sqrt(dt) like in our Noise class
            noise_vals = jax.random.normal(subkey, (n_noises,)) * noise_stds * jnp.sqrt(dt)
        else:
            noise_vals = jnp.zeros(0, dtype=jnp.float32)

        # ----- Channel reads -----
        # Compose into one big (dst, xi_complex) list, then aggregate.
        all_dst = []
        all_xi = []

        # Linear from nodes (complex weights * complex node state)
        if lin_n_src.shape[0] > 0:
            w_complex = lin_n_w + 1j * lin_n_w_imag
            xi_lin = w_complex * node_states[lin_n_src]
            all_dst.append(lin_n_dst); all_xi.append(xi_lin)

        # Linear from constants (real weight * real value, cast to complex)
        if lin_c_src.shape[0] > 0:
            xi_lin_c = (lin_c_w * constant_values[lin_c_src]).astype(jnp.complex64)
            all_dst.append(lin_c_dst); all_xi.append(xi_lin_c)

        # Linear from noise
        if lin_no_src.shape[0] > 0:
            xi_lin_no = (lin_no_w * noise_vals[lin_no_src]).astype(jnp.complex64)
            all_dst.append(lin_no_dst); all_xi.append(xi_lin_no)

        # Real-only feedback (always from nodes; output is real)
        if rof_src.shape[0] > 0:
            src_real = node_states[rof_src].real
            xi_rof = (-rof_gain * (src_real - rof_target)).astype(jnp.complex64)
            all_dst.append(rof_dst); all_xi.append(xi_rof)

        # Abs-to-real
        if atr_src.shape[0] > 0:
            xi_atr = jnp.abs(node_states[atr_src]).astype(jnp.complex64)
            all_dst.append(atr_dst); all_xi.append(xi_atr)

        # Plastic (always linear, from nodes; weight is state)
        if pl_src.shape[0] > 0:
            effective_w = (plastic_weights * pl_transform_w).astype(jnp.complex64)
            xi_pl = effective_w * node_states[pl_src]
            all_dst.append(pl_dst); all_xi.append(xi_pl)

        # Aggregate (concat then segment-sum)
        if all_xi:
            cat_xi = jnp.concatenate(all_xi)
            cat_dst = jnp.concatenate(all_dst)
        else:
            cat_xi = jnp.zeros(0, dtype=jnp.complex64)
            cat_dst = jnp.zeros(0, dtype=jnp.int32)

        # Two drives: amplitude (real-part sum) and phase coupling
        amp_drive = jnp.zeros(n_nodes, dtype=jnp.float32).at[cat_dst].add(cat_xi.real)

        # Phase coupling: |xi| * sin(arg(xi) - theta_dst). Need theta per channel.
        if cat_xi.shape[0] > 0:
            ch_mag = jnp.abs(cat_xi)
            ch_arg = jnp.angle(cat_xi)
            theta_per_ch = jnp.angle(node_states[cat_dst])
            coupling_per_ch = ch_mag * jnp.sin(ch_arg - theta_per_ch)
            coupling_drive = jnp.zeros(n_nodes, dtype=jnp.float32).at[cat_dst].add(coupling_per_ch)
        else:
            coupling_drive = jnp.zeros(n_nodes, dtype=jnp.float32)

        # Also: |xi| sum per dst (used by polar/polar_sigmoid amplitude)
        if cat_xi.shape[0] > 0:
            mag_drive = jnp.zeros(n_nodes, dtype=jnp.float32).at[cat_dst].add(jnp.abs(cat_xi))
        else:
            mag_drive = jnp.zeros(n_nodes, dtype=jnp.float32)

        # ----- Apply dynamics per group -----
        new_states = node_states
        for kind, idxs, params in dynamics_dispatch:
            old = node_states[idxs]
            d_amp = amp_drive[idxs]
            d_coup = coupling_drive[idxs]
            d_mag = mag_drive[idxs]

            if kind == "sigmoid_activity":
                target = jax.nn.sigmoid(params["gain"] * d_amp - params["threshold"])
                old_real = old.real
                new_real = old_real + dt * params["rate"] * (target - old_real)
                new_real = jnp.maximum(new_real, 0.0)
                new = (new_real + 1j * old.imag).astype(jnp.complex64)

            elif kind == "tracker":
                old_real = old.real
                new_real = old_real + dt * params["rate"] * (d_amp - old_real)
                new = (new_real + 1j * old.imag).astype(jnp.complex64)

            elif kind == "polar_sigmoid":
                r = jnp.abs(old)
                theta = jnp.where(r > 1e-12, jnp.angle(old), 0.0)
                target_r = jax.nn.sigmoid(params["gain"] * d_amp - params["threshold"])
                r_new = r + dt * params["rate"] * (target_r - r)
                r_new = jnp.maximum(r_new, 0.0)
                dtheta = params["omega"] + params["coupling"] * d_coup
                theta_new = theta + dt * dtheta
                new = (r_new * jnp.exp(1j * theta_new)).astype(jnp.complex64)

            elif kind == "polar":
                r = jnp.abs(old)
                theta = jnp.where(r > 1e-12, jnp.angle(old), 0.0)
                # Linear amplitude target = sensitivity * |sum xs| ~ d_mag * sensitivity
                # (matches the slow `polar` factory: dr = rate * (sensitivity*|sum| - r))
                target_r = params["sensitivity"] * d_mag
                r_new = r + dt * params["rate"] * (target_r - r)
                r_new = jnp.maximum(r_new, 0.0)
                dtheta = params["omega"] + params["coupling"] * d_coup
                theta_new = theta + dt * dtheta
                new = (r_new * jnp.exp(1j * theta_new)).astype(jnp.complex64)

            elif kind == "adaptation":
                # ds/dt = rate * (sensitivity * sum(xs) - s); state is real-valued
                target = params["sensitivity"] * d_amp
                old_real = old.real
                new_real = old_real + dt * params["rate"] * (target - old_real)
                new = (new_real + 1j * old.imag).astype(jnp.complex64)

            else:
                raise RuntimeError(f"unreachable dynamics kind: {kind}")

            new_states = new_states.at[idxs].set(new)

        # ----- Plastic weight updates -----
        new_plastic = plastic_weights
        if pl_src.shape[0] > 0:
            sv = node_states[pl_src]
            dv = node_states[pl_dst]
            re_sv_conjdv = (sv * jnp.conj(dv)).real

            # Hebbian (and gated_hebbian uses the same Re(...) update with a gate)
            heb_update = re_sv_conjdv

            # Gated: 4 * dv.real * (1 - dv.real), clamped to [0,1]
            d_real = dv.real
            gate = jnp.clip(4.0 * d_real * (1.0 - d_real), 0.0, 1.0)
            gated_update = gate * re_sv_conjdv

            # STDP-sin: sin(arg(sv) - arg(dv)), zero if either |sv|, |dv| ~ 0
            sv_mag = jnp.abs(sv); dv_mag = jnp.abs(dv)
            stdp_safe = (sv_mag > 1e-12) & (dv_mag > 1e-12)
            stdp_update = jnp.where(
                stdp_safe,
                jnp.sin(jnp.angle(sv) - jnp.angle(dv)),
                0.0,
            )

            # Combine via masks (one-hot kind selectors)
            update = (pl_kind_hebbian * heb_update
                      + pl_kind_gated * gated_update
                      + pl_kind_stdp * stdp_update)
            dw = pl_eta * update - pl_decay * plastic_weights
            new_plastic = plastic_weights + dt * dw

        # ----- Weight normalizers -----
        for ch_idxs, target_sum in norm_specs:
            current_sum = jnp.sum(jnp.abs(new_plastic[ch_idxs]))
            scale = target_sum / jnp.maximum(current_sum, 1e-12)
            new_plastic = new_plastic.at[ch_idxs].set(new_plastic[ch_idxs] * scale)

        new_state = {
            "node_states": new_states,
            "plastic_weights": new_plastic,
            "key": key,
        }
        return new_state, new_states

    return init_state, step


def run_compiled(
    sources,
    n_steps: int,
    dt: float = 1.0,
    seed: int = 0,
    writeback: bool = True,
):
    """Compile a network and run it for n_steps. Convenience wrapper.

    Returns:
      history_array: numpy array of shape (n_steps, n_nodes), complex dtype.
                     history_array[t, i] is node i's state at step t+1.
      node_to_idx:   dict mapping id(node) → column index in the array.
                     Extract per-node observation:
                         h = history_array[:, node_to_idx[id(my_node)]]
      final_state:   the final pytree (node_states, plastic_weights, key).

    If `writeback=True` (default), the OOP `Node.state` and
    `PlasticChannel.weight` fields are updated to reflect the JAX-run's
    final values, so post-run analysis code that reads them sees the
    right values.
    """
    init_state, step = compile_to_jax(sources, dt=dt, seed=seed)
    final_state, history = jax.lax.scan(step, init_state, xs=None, length=n_steps)
    history.block_until_ready()
    history_np = np.asarray(history)

    nodes = [s for s in sources if isinstance(s, Node)]
    node_to_idx = {id(n): i for i, n in enumerate(nodes)}

    if writeback:
        final_node_states = np.asarray(final_state["node_states"])
        final_plastic_weights = np.asarray(final_state["plastic_weights"])
        for i, node in enumerate(nodes):
            node.state = complex(final_node_states[i])
        plastic_idx = 0
        for node in nodes:
            for ch in node.channels:
                if isinstance(ch, PlasticChannel):
                    node_idx_for_dst = node_to_idx[id(node)]
                    ch.weight = complex(float(final_plastic_weights[plastic_idx]), 0)
                    plastic_idx += 1

    return history_np, node_to_idx, final_state


def history_as_dict(history_array, node_to_idx, observers_spec):
    """Convert (history_array, node_to_idx) into a dict matching the slow
    `run()` API.

    `observers_spec` is a dict {name: (node, attr)} where attr is one of:
      'real', 'imag', 'complex', 'abs'.
    """
    out = {}
    for name, (node, attr) in observers_spec.items():
        col = history_array[:, node_to_idx[id(node)]]
        if attr == "real":
            out[name] = list(col.real)
        elif attr == "imag":
            out[name] = list(col.imag)
        elif attr == "complex":
            out[name] = list(col.astype(complex))
        elif attr == "abs":
            out[name] = list(np.abs(col))
        else:
            raise ValueError(f"Unknown attr {attr!r} for observer {name}")
    return out
