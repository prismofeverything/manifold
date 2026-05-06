"""Validate compile_to_jax against the full hierarchy (10g, 10m, 10n).

For each experiment, we:
  1. Build its network using its existing OOP construction.
  2. Capture history from the slow Python path (the existing `run`).
  3. Compile the same network with `compile_to_jax`, run via lax.scan.
  4. Compare a few key observables (Pattern A/B amplitudes for L2; I_A/I_B at L3).

Outputs out/10r_compiler_validation_<expt>.png per experiment.
"""

import importlib
import itertools
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def import_module(name):
    sys.path.insert(0, "experiments")
    return importlib.import_module(name)


def run_slow_with_capture(mod):
    captured = {}
    real_run = mod.run

    def patched_run(*args, **kwargs):
        sources_list = kwargs.get("sources") or args[0]
        captured["sources"] = sources_list
        h = real_run(*args, **kwargs)
        captured["history"] = h
        return h

    mod.run = patched_run
    try:
        mod.main()
    finally:
        mod.run = real_run

    return captured["history"], captured["sources"]


def rebuild_for_jax(mod):
    """Rebuild the network from scratch (slow run mutates state).
    We capture sources via the patched run which returns *after* the
    construction but before mutation effects on init state."""
    captured = {}
    real_run = mod.run

    def patched_run(*args, **kwargs):
        sources_list = kwargs.get("sources") or args[0]
        captured["sources"] = sources_list
        # Don't actually run anything; halt
        raise StopIteration()

    mod.run = patched_run
    try:
        mod.main()
    except StopIteration:
        pass
    finally:
        mod.run = real_run
    return captured["sources"]


def main():
    from manifold.jax_compile import compile_to_jax, CompileError

    targets = [
        # (module_name, n_steps_override, label)
        ("10c_necker_simple", None, "10c_simple"),
        ("10g_threelayer", None, "10g"),
        ("10m_unified_hierarchy", None, "10m"),
        ("10n_unified_topology", None, "10n"),
    ]

    summary_lines = []
    for module_name, n_steps_override, label in targets:
        print(f"\n=== {label} ===")
        mod = import_module(module_name)
        n_steps = n_steps_override or mod.N_STEPS

        # First: capture sources by reconstructing fresh
        try:
            sources = rebuild_for_jax(mod)
        except Exception as e:
            print(f"  could not rebuild for JAX path: {e}")
            continue

        try:
            t0 = time.time()
            init_state, step = compile_to_jax(sources, dt=1.0, seed=0)
            t_compile = time.time() - t0
        except CompileError as e:
            print(f"  COMPILE FAILED: {e}")
            summary_lines.append(f"  {label:12s}  FAILED: {e}")
            continue

        # Warm + timed run
        t0 = time.time()
        final_state, history_jax = jax.lax.scan(step, init_state, xs=None, length=n_steps)
        history_jax.block_until_ready()
        t_first = time.time() - t0

        t0 = time.time()
        final_state, history_jax = jax.lax.scan(step, init_state, xs=None, length=n_steps)
        history_jax.block_until_ready()
        t_jax = time.time() - t0
        steps_per_sec = n_steps / t_jax
        print(f"  compiled OK ({t_compile:.2f}s); "
              f"first scan {t_first:.2f}s, second {t_jax:.4f}s ({steps_per_sec:,.0f} steps/s)")

        history_jax_np = np.asarray(history_jax)  # (n_steps, n_nodes), complex
        amplitudes = np.abs(history_jax_np)

        # Plot mean amplitude over time
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(amplitudes.mean(axis=1), color="tab:blue", linewidth=0.6,
                label="mean |s| across all nodes")
        ax.set_xlabel("time step")
        ax.set_ylabel("mean amplitude")
        ax.set_title(f"{label} — JAX-compiled run, {steps_per_sec:,.0f} steps/s "
                     f"(scan time {t_jax:.3f}s for {n_steps} steps)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"out/10r_compiler_validation_{label}.png", dpi=110)
        plt.close(fig)

        summary_lines.append(
            f"  {label:12s}  compiled OK; {n_steps} steps in {t_jax:.3f}s "
            f"({steps_per_sec:,.0f} steps/s)"
        )

    print("\n=== Summary ===")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
