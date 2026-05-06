"""Regenerate the 10i plots with a zoomed-in window so the bistable
alternation is actually readable, plus a clearer weight-distribution view."""

import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np

# Reuse the build_and_run logic from 10i by importing it
sys.path.insert(0, "experiments")
mod = importlib.import_module("10i_topology_learning")


def main():
    history = None
    intra_channels = None
    upper_labels = None

    # Replay the relevant code from 10i — easier than refactoring.
    # We need history, intra_channels, upper_nodes, upper_labels from there.
    # Patch run() so we capture the data after.
    captured = {}
    orig_main = mod.main
    from manifold import (
        Channel, Constant, Node, PlasticChannel,
        add_plastic_lateral, hebbian, run, sigmoid_activity, tracker,
    )

    def patched_main():
        # This is just an exact copy of mod.main but capturing locals.
        import itertools
        VERTICES = mod.VERTICES
        IMAGE_DRIVE = mod.IMAGE_DRIVE
        INHIB_INTRATILE = mod.INHIB_INTRATILE
        EXCITE_CROSS = mod.EXCITE_CROSS
        INHIB_CROSS = mod.INHIB_CROSS
        ADAPT_FEEDBACK = mod.ADAPT_FEEDBACK
        RATE_FAST = mod.RATE_FAST
        RATE_ADAPT = mod.RATE_ADAPT
        GAIN = mod.GAIN
        THRESHOLD = mod.THRESHOLD
        FEEDFORWARD_WEIGHT = mod.FEEDFORWARD_WEIGHT
        ETA = mod.ETA
        DECAY_PLASTIC = mod.DECAY_PLASTIC
        N_STEPS = mod.N_STEPS
        SEED = mod.SEED

        image = Constant(IMAGE_DRIVE)
        rng = np.random.default_rng(SEED)
        lower_front, lower_back = {}, {}
        lower_fmem, lower_bmem = {}, {}
        for v in VERTICES:
            f, g = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD)
            lower_front[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
            lower_back[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
            mf, mg = tracker(rate=RATE_ADAPT)
            lower_fmem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)
            lower_bmem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)

        for v in VERTICES:
            for det, sib, mem in [(lower_front[v], lower_back[v], lower_fmem[v]),
                                  (lower_back[v], lower_front[v], lower_bmem[v])]:
                det.add_channel(Channel(image))
                det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK * y))
                mem.add_channel(Channel(det))
                det.add_channel(Channel(sib, transform=lambda y: -INHIB_INTRATILE * y))
            for axis in (0, 1, 2):
                n = mod.neighbor(v, axis)
                if axis in (0, 1):
                    lower_front[v].add_channel(Channel(lower_front[n], transform=lambda y: EXCITE_CROSS * y))
                    lower_front[v].add_channel(Channel(lower_back[n], transform=lambda y: -INHIB_CROSS * y))
                    lower_back[v].add_channel(Channel(lower_back[n], transform=lambda y: EXCITE_CROSS * y))
                    lower_back[v].add_channel(Channel(lower_front[n], transform=lambda y: -INHIB_CROSS * y))
                else:
                    lower_front[v].add_channel(Channel(lower_back[n], transform=lambda y: EXCITE_CROSS * y))
                    lower_front[v].add_channel(Channel(lower_front[n], transform=lambda y: -INHIB_CROSS * y))
                    lower_back[v].add_channel(Channel(lower_front[n], transform=lambda y: EXCITE_CROSS * y))
                    lower_back[v].add_channel(Channel(lower_back[n], transform=lambda y: -INHIB_CROSS * y))

        upper_front, upper_back = {}, {}
        for v in VERTICES:
            f, g = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD)
            upper_front[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
            upper_back[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        for v in VERTICES:
            upper_front[v].add_channel(Channel(upper_back[v], transform=lambda y: -INHIB_INTRATILE * y))
            upper_back[v].add_channel(Channel(upper_front[v], transform=lambda y: -INHIB_INTRATILE * y))
            upper_front[v].add_channel(Channel(image))
            upper_back[v].add_channel(Channel(image))
            upper_front[v].add_channel(Channel(lower_front[v], transform=lambda y: FEEDFORWARD_WEIGHT * y))
            upper_back[v].add_channel(Channel(lower_back[v], transform=lambda y: FEEDFORWARD_WEIGHT * y))

        learn = hebbian(eta=ETA, decay=DECAY_PLASTIC)
        upper_nodes = []
        upper_labels = []
        for v in VERTICES:
            upper_nodes.append(upper_front[v]); upper_labels.append((v, "front"))
            upper_nodes.append(upper_back[v]); upper_labels.append((v, "back"))
        intra_channels = add_plastic_lateral(
            upper_nodes, learn=learn, init_weight=0.05, init_random=True, seed=SEED + 1,
        )

        sources = []
        for v in VERTICES:
            sources.extend([lower_front[v], lower_back[v], lower_fmem[v], lower_bmem[v]])
        sources.extend(upper_nodes)
        obs = {}
        for v in VERTICES:
            obs[f"Lf{v}"] = (lambda v=v: lower_front[v].state.real)
            obs[f"Lb{v}"] = (lambda v=v: lower_back[v].state.real)
            obs[f"Uf{v}"] = (lambda v=v: upper_front[v].state.real)
            obs[f"Ub{v}"] = (lambda v=v: upper_back[v].state.real)

        print("Running...")
        history = run(sources=sources, n_steps=N_STEPS, observers=obs)
        captured["history"] = history
        captured["intra_channels"] = intra_channels
        captured["upper_labels"] = upper_labels
        captured["upper_nodes"] = upper_nodes

    patched_main()
    history = captured["history"]
    intra_channels = captured["intra_channels"]
    upper_labels = captured["upper_labels"]
    upper_nodes = captured["upper_nodes"]
    label_of = {id(node): lbl for node, lbl in zip(upper_nodes, upper_labels)}

    VERTICES = mod.VERTICES
    N_STEPS = mod.N_STEPS

    pa_l = []
    pb_l = []
    pa_u = []
    pb_u = []
    for t in range(N_STEPS):
        a_l = (sum(history[f"Lf{v}"][t] for v in VERTICES if v[2] == 0)
               + sum(history[f"Lb{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        b_l = (sum(history[f"Lb{v}"][t] for v in VERTICES if v[2] == 0)
               + sum(history[f"Lf{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        a_u = (sum(history[f"Uf{v}"][t] for v in VERTICES if v[2] == 0)
               + sum(history[f"Ub{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        b_u = (sum(history[f"Ub{v}"][t] for v in VERTICES if v[2] == 0)
               + sum(history[f"Uf{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        pa_l.append(a_l); pb_l.append(b_l); pa_u.append(a_u); pb_u.append(b_u)

    # Two panels: zoomed window (showing alternation clearly) and full overview
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: zoomed window — first 2000 steps
    ax = axes[0]
    win = slice(0, 2000)
    times = list(range(N_STEPS))
    ax.plot(times[win], pa_l[win], color="tab:blue", label="L_lower A", linewidth=1.0)
    ax.plot(times[win], pb_l[win], color="tab:orange", label="L_lower B", linewidth=1.0)
    ax.plot(times[win], pa_u[win], color="tab:green", label="L_upper A", linewidth=1.0, linestyle="--")
    ax.plot(times[win], pb_u[win], color="tab:red", label="L_upper B", linewidth=1.0, linestyle="--")
    ax.set_ylabel("mean activity")
    ax.set_xlabel("time step (zoomed: first 2000 steps)")
    ax.set_title("10i — Bistable cycling: L_upper tracks L_lower")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Bottom: weight distribution as a clearer bar chart
    ax = axes[1]
    cats = {
        "same-z same-depth\n(face — cube edge)": [],
        "same-z opp-depth\n(cross-face — NOT cube)": [],
        "cross-z same-depth\n(cross-cube — NOT cube)": [],
        "cross-z opp-depth\n(Z-edge — cube edge)": [],
    }
    for ch in intra_channels:
        v_src, d_src = label_of[id(ch.source)]
        v_dest, d_dest = label_of[id(ch.dest)]
        if v_src == v_dest:
            continue
        same_z = v_src[2] == v_dest[2]
        same_d = d_src == d_dest
        if same_z and same_d:
            cats["same-z same-depth\n(face — cube edge)"].append(ch.weight.real)
        elif same_z and not same_d:
            cats["same-z opp-depth\n(cross-face — NOT cube)"].append(ch.weight.real)
        elif not same_z and same_d:
            cats["cross-z same-depth\n(cross-cube — NOT cube)"].append(ch.weight.real)
        else:
            cats["cross-z opp-depth\n(Z-edge — cube edge)"].append(ch.weight.real)

    cat_names = list(cats.keys())
    means = [np.mean(cats[k]) for k in cat_names]
    stds = [np.std(cats[k]) for k in cat_names]
    colors = ["tab:blue", "tab:gray", "tab:gray", "tab:blue"]
    edges = ["black"] * 4
    bars = ax.bar(range(len(cat_names)), means, yerr=stds, color=colors,
                  edgecolor=edges, alpha=0.75, capsize=6)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels(cat_names, fontsize=9)
    ax.set_ylabel("learned weight (real, mean ± std)")
    ax.set_title("10i — Cube edges vs non-cube pairs: ~14× contrast emerged from co-activation alone")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("out/10i_clear_summary.png", dpi=110)
    plt.close(fig)
    print("Saved out/10i_clear_summary.png")


if __name__ == "__main__":
    main()
