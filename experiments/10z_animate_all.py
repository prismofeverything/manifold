"""Run animations for all three three-layer experiments using the
generalized animate_three_layer_history function.

Each experiment is imported and re-run (no pickled history yet); we patch
its `run` import to capture the resulting history dict, then feed it to
the animation function.

Outputs three GIFs:
  out/10g_animation.gif (basic three-layer, sigmoid_activity)
  out/10m_animation.gif (unified — polar_sigmoid throughout)
  out/10n_animation.gif (unified + L2 topology learning)
"""

import importlib
import itertools
import sys


def run_with_capture(module_name):
    """Import a module, patch its run() to capture history, run main()."""
    sys.path.insert(0, "experiments")
    mod = importlib.import_module(module_name)
    captured = {}
    real_run = mod.run

    def patched_run(*args, **kwargs):
        h = real_run(*args, **kwargs)
        captured["history"] = h
        return h

    mod.run = patched_run
    try:
        mod.main()
    finally:
        mod.run = real_run

    return captured["history"], mod


def main():
    from manifold.animate import animate_three_layer_history

    VERTICES = list(itertools.product([0, 1], repeat=3))
    ARM_COMBOS = list(itertools.product([0, 1], repeat=3))

    targets = [
        # (module_name, output_path, title, sample_every)
        ("10g_threelayer", "out/10g_animation.gif",
         "10g — three-layer (sigmoid_activity, hand-wired cube)", 50),
        ("10m_unified_hierarchy", "out/10m_animation.gif",
         "10m — three-layer unified (polar_sigmoid, hand-wired cube)", 50),
        ("10n_unified_topology", "out/10n_animation.gif",
         "10n — unified + L2 topology learning", 50),
    ]

    for module_name, out_path, title, sample_every in targets:
        print(f"\n=== {module_name} ===")
        history, mod = run_with_capture(module_name)
        n_steps = mod.N_STEPS
        print(f"  history captured ({n_steps} steps); generating {out_path} ...")
        animate_three_layer_history(
            history=history,
            vertices=VERTICES,
            arm_combos=ARM_COMBOS,
            n_steps=n_steps,
            sample_every=sample_every,
            output_path=out_path,
            title_prefix=title,
            fps=15,
        )
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
