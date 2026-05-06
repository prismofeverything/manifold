"""Phase 10i: Topology learning — L_upper discovers cube graph from cycling.

The architecture from 10c_simple has the cube topology hand-wired into
its cross-tile constraints. This experiment asks: can a layer with NO
hand-wired cross-tile structure *learn* the cube topology purely from
the activity statistics of a layer below it that does cycle between A
and B interpretations?

Recipe (see `project_topology_learning.md` memory):
  - L_lower: 16-detector simple cube with hand-wired cross-tile constraints
             + own adaptation. Cycles between A and B autonomously.
  - L_upper: 16 detectors (front, back per vertex). Mutual inhibition
             front ⊥ back per vertex (the only hand-wired structure —
             vertex-local validity). All-to-all plastic intra-layer
             connections. Hand-wired feedforward L_lower → L_upper from
             same-vertex same-label nodes (so the spatial mapping is
             given but the cross-tile structure must emerge).

Hypothesis: after running, L_upper's plastic intra-layer weights should
mirror the cube edge structure — same-depth weights grow on X/Y edges
and on intra-face vertex pairs (face equivalence classes), opposite-
depth weights grow on Z edges, weights between unrelated vertex/depth
pairs decay.
"""

import itertools

import numpy as np

from manifold import (
    Channel, Constant, Node, PlasticChannel, WeightNormalizer,
    add_lateral_inhibition, add_plastic_lateral,
    hebbian, plot, run, sigmoid_activity, tracker,
)


VERTICES = list(itertools.product([0, 1], repeat=3))

# L_lower: same parameters as 10c_simple
IMAGE_DRIVE = 0.5
INHIB_INTRATILE = 1.5
EXCITE_CROSS = 0.4
INHIB_CROSS = 0.4
ADAPT_FEEDBACK = 1.5
RATE_FAST = 0.1
RATE_ADAPT = 0.005
GAIN = 8.0
THRESHOLD = 2.5

# Feedforward L_lower -> L_upper
FEEDFORWARD_WEIGHT = 1.5     # same-vertex, same-label connections (hand-wired)

# L_upper: plastic intra-layer.
# eta/decay << 1 caps weights at small values so the plastic feedback
# can't run away and break L_upper's bistable cycling. Equilibrium
# weight = (eta/decay) * <sv*dv>: ~0.05 for co-active pairs, 0 for never.
ETA = 0.0005
DECAY_PLASTIC = 0.003

N_STEPS = 24000
SEED = 42


def neighbor(v, axis):
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def main():
    image = Constant(IMAGE_DRIVE)
    rng = np.random.default_rng(SEED)

    # --- L_lower (hand-wired bistable cube, like 10c_simple) ---
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
            n = neighbor(v, axis)
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

    # --- L_upper (vertex-local validity hand-wired; cross-tile must be learned) ---
    upper_front, upper_back = {}, {}
    for v in VERTICES:
        f, g = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD)
        upper_front[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        upper_back[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)

    # Hand-wired structure at L_upper: only vertex-local validity (front ⊥ back).
    for v in VERTICES:
        upper_front[v].add_channel(Channel(upper_back[v], transform=lambda y: -INHIB_INTRATILE * y))
        upper_back[v].add_channel(Channel(upper_front[v], transform=lambda y: -INHIB_INTRATILE * y))
        # Image baseline drive
        upper_front[v].add_channel(Channel(image))
        upper_back[v].add_channel(Channel(image))
        # Hand-wired feedforward: L_lower same-position drives L_upper same-position
        upper_front[v].add_channel(
            Channel(lower_front[v], transform=lambda y: FEEDFORWARD_WEIGHT * y))
        upper_back[v].add_channel(
            Channel(lower_back[v], transform=lambda y: FEEDFORWARD_WEIGHT * y))

    # Plastic intra-layer at L_upper: all-to-all between every L_upper node.
    learn = hebbian(eta=ETA, decay=DECAY_PLASTIC)
    upper_nodes = []
    upper_labels = []
    for v in VERTICES:
        upper_nodes.append(upper_front[v]); upper_labels.append((v, "front"))
        upper_nodes.append(upper_back[v]);  upper_labels.append((v, "back"))

    intra_channels = add_plastic_lateral(
        upper_nodes, learn=learn, init_weight=0.05, init_random=True, seed=SEED + 1,
    )

    # --- Run --- (no weight normalizer; let Hebbian/decay equilibrate naturally)
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
    times = list(range(N_STEPS))

    # Sanity: lower and upper layers cycling together.
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

    plot(times,
         {"L_lower A": pa_l, "L_lower B": pb_l,
          "L_upper A": pa_u, "L_upper B": pb_u},
         ylabel="mean activity",
         title="10i_lower_and_upper_patterns")

    # --- Analyze learned intra-layer weights at L_upper ---
    # For each pair (src_label, dest_label), categorize:
    #   - same-z, same-depth: cube X/Y edge or face-equivalent (should be HIGH)
    #   - same-z, opposite-depth: vertex-local pair (already inhibited; should be LOW)
    #   - cross-z, same-depth: should be LOW (different face under any interp)
    #   - cross-z, opposite-depth: cube Z edge structure (should be HIGH)
    #   - same vertex (any depth): excluded; it's the hand-wired vertex inhibition
    cats = {
        "same-z same-depth (face)": [],
        "same-z opp-depth (cross-face)": [],
        "cross-z same-depth (cross-cube)": [],
        "cross-z opp-depth (Z-edge structure)": [],
    }
    label_of = {id(node): lbl for node, lbl in zip(upper_nodes, upper_labels)}
    for ch in intra_channels:
        src_label = label_of[id(ch.source)]
        dest_label = label_of[id(ch.dest)]
        v_src, d_src = src_label
        v_dest, d_dest = dest_label
        if v_src == v_dest:
            continue   # vertex-local pair, hand-wired
        same_z = v_src[2] == v_dest[2]
        same_d = d_src == d_dest
        if same_z and same_d:
            cats["same-z same-depth (face)"].append(ch.weight.real)
        elif same_z and not same_d:
            cats["same-z opp-depth (cross-face)"].append(ch.weight.real)
        elif not same_z and same_d:
            cats["cross-z same-depth (cross-cube)"].append(ch.weight.real)
        else:
            cats["cross-z opp-depth (Z-edge structure)"].append(ch.weight.real)

    print()
    print(f"Final L_upper intra-layer weights (no normalization; "
          f"natural eta/decay equilibrium):")
    print(f"  category                                 mean    n")
    for k, ws in cats.items():
        print(f"  {k:42s}  {np.mean(ws):.3f}   {len(ws)}")

    # Plot weight histogram per category
    fig_data = {k: ws for k, ws in cats.items()}
    # Save histogram figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(max(ws) for ws in cats.values() if ws) * 1.1, 30)
    for k, ws in cats.items():
        ax.hist(ws, bins=bins, alpha=0.5, label=k)
    ax.set_xlabel("learned weight (real part)")
    ax.set_ylabel("count")
    ax.set_title("10i — L_upper intra-layer weight distribution by category")
    ax.legend(fontsize=8)
    fig.savefig("out/10i_weight_histogram.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
