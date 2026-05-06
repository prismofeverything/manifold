"""Phase 10g: three-layer hierarchy — V1 → V2/V4 → IT for the Necker cube.

Building on the user's insight: when the rich feature representation is too
chaotic for an interpretive top to learn from directly, insert a structurally
cleaner *intermediate* layer that has native bistability and lets each layer
above it find a clean substrate to organize on.

Three layers:

  L1 (rich, V1-like): 64 detectors per cube — 8 (X-arm, Y-arm, Z-arm) depth
       combos × 8 vertices. Cross-tile constraints + within-tile inhibition
       + image. No own adaptation.

  L2 (middle, V2/V4-like): 16 detectors (front, back per vertex). Own
       cross-tile bistability (the 10c_simple substrate). Per-vertex
       plastic Hebbian inputs from L1's 8 detectors at that vertex —
       L2 *discovers* which L1 configs co-activate with its front vs
       back identity at each vertex. No globally-rich-to-middle wiring;
       biologically the V2 receptive fields are local.

  L3 (top, IT-like): 2 plastic I nodes with full robustness mechanisms
       (homeostasis + weight normalization + noise + adaptation). Plastic
       Hebbian both directions to/from L2.

What we expect to observe:
  - L1: chaotic local cycling at each vertex (no global coherence on its own).
  - L2: clean A/B bistability driven by its own cross-tile constraints.
  - L3: I_A and I_B self-organize, each specializing on one L2 pattern.
  - L1→L2 weights: each L2 detector ends up with high weight to the
       *locally valid* L1 detector that implies its identity (front-implying
       configs for "front" L2 detectors; back-implying for "back").
"""

import itertools

import numpy as np

from manifold import (
    Channel, Constant, Node, Noise, PlasticChannel, WeightNormalizer,
    hebbian, history_as_dict, homeostatic_feedback, plot, run_compiled,
    sigmoid_activity, tracker,
)


VERTICES = list(itertools.product([0, 1], repeat=3))
ARM_COMBOS = list(itertools.product([0, 1], repeat=3))

# L1 (rich) — with own adaptation so it cycles through local states rather
# than locking into the first one it finds. Otherwise L1 biases L2 toward
# whichever interpretation it happened to settle in.
IMAGE_DRIVE_L1 = 0.5
INHIB_INTRATILE_L1 = 1.5
EXCITE_CROSS_L1 = 0.2
INHIB_CROSS_L1 = 0.6
ADAPT_FEEDBACK_L1 = 1.0
RATE_ADAPT_L1 = 0.005
RATE_FAST_L1 = 0.1
GAIN_L1 = 8.0
THRESHOLD_L1 = 2.5

# L2 (middle, simple)
IMAGE_DRIVE_L2 = 0.5
INHIB_INTRATILE_L2 = 1.5
EXCITE_CROSS_L2 = 0.4
INHIB_CROSS_L2 = 0.4
ADAPT_FEEDBACK_L2 = 1.5
RATE_FAST_L2 = 0.1
RATE_ADAPT_L2 = 0.005
GAIN_L2 = 8.0
THRESHOLD_L2 = 2.5

# L3 (top)
INHIB_HI = 2.0
ADAPT_FEEDBACK_HI = 2.0
RATE_FAST_HI = 0.1
RATE_ADAPT_HI = 0.003
RATE_HOMEO_HI = 0.0003
GAIN_HI = 6.0
THRESHOLD_HI = 1.5
HOMEO_TARGET = 0.4
HOMEO_GAIN = 4.0
NOISE_STD = 0.04

# Plasticity
ETA = 0.002
DECAY_PLASTIC = 0.002
TARGET_L1_TO_L2_SUM = 1.0       # per L2 detector, sum of weights from its 8 L1 detectors
TARGET_L2_TO_L3_BU_SUM = 2.0
TARGET_L3_TO_L2_TD_SUM = 2.0
L1_TO_L2_GAIN = 1.5             # how much L1 input drives L2
L3_TO_L2_TD_GAIN = 1.0

N_STEPS = 24000
SEED = 42


def neighbor(v, axis):
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def main():
    image = Constant(IMAGE_DRIVE_L1)
    image_l2 = Constant(IMAGE_DRIVE_L2)
    rng = np.random.default_rng(SEED)

    # ---- L1 (rich) ----
    l1 = {v: {} for v in VERTICES}
    l1_mems = {v: {} for v in VERTICES}
    for v in VERTICES:
        for combo in ARM_COMBOS:
            f, g = sigmoid_activity(rate=RATE_FAST_L1, gain=GAIN_L1, threshold=THRESHOLD_L1)
            l1[v][combo] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
            mf, mg = tracker(rate=RATE_ADAPT_L1)
            l1_mems[v][combo] = Node(state=0 + 0j, dynamics=mf, output=mg)

    for v in VERTICES:
        for combo in ARM_COMBOS:
            d_x, d_y, d_z = combo
            det = l1[v][combo]
            mem = l1_mems[v][combo]
            det.add_channel(Channel(image))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK_L1 * y))
            mem.add_channel(Channel(det))
            for other in ARM_COMBOS:
                if other == combo:
                    continue
                det.add_channel(Channel(l1[v][other], transform=lambda y: -INHIB_INTRATILE_L1 * y))
            x_nbr = neighbor(v, 0)
            for c2 in ARM_COMBOS:
                w = EXCITE_CROSS_L1 if c2[0] == d_x else -INHIB_CROSS_L1
                det.add_channel(Channel(l1[x_nbr][c2], transform=lambda y, w=w: w * y))
            y_nbr = neighbor(v, 1)
            for c2 in ARM_COMBOS:
                w = EXCITE_CROSS_L1 if c2[1] == d_y else -INHIB_CROSS_L1
                det.add_channel(Channel(l1[y_nbr][c2], transform=lambda y, w=w: w * y))
            z_nbr = neighbor(v, 2)
            for c2 in ARM_COMBOS:
                w = EXCITE_CROSS_L1 if c2[2] == 1 - d_z else -INHIB_CROSS_L1
                det.add_channel(Channel(l1[z_nbr][c2], transform=lambda y, w=w: w * y))

    # ---- L2 (middle) ----
    front, back = {}, {}
    front_mem, back_mem = {}, {}
    for v in VERTICES:
        f, g = sigmoid_activity(rate=RATE_FAST_L2, gain=GAIN_L2, threshold=THRESHOLD_L2)
        front[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        back[v]  = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        mf, mg = tracker(rate=RATE_ADAPT_L2)
        front_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)
        back_mem[v]  = Node(state=0 + 0j, dynamics=mf, output=mg)

    learn = hebbian(eta=ETA, decay=DECAY_PLASTIC)
    l1_to_l2_channels_per_dest: dict = {}   # dest_node -> list of plastic channels

    for v in VERTICES:
        for det, sib, mem in [(front[v], back[v], front_mem[v]),
                              (back[v], front[v], back_mem[v])]:
            det.add_channel(Channel(image_l2))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK_L2 * y))
            mem.add_channel(Channel(det))
            det.add_channel(Channel(sib, transform=lambda y: -INHIB_INTRATILE_L2 * y))

            # Plastic Hebbian inputs from all 8 L1 detectors at this vertex.
            ch_list = []
            for combo in ARM_COMBOS:
                ch = PlasticChannel(
                    l1[v][combo], dest=det,
                    weight=0.05 * rng.random(), learn=learn,
                    transform=lambda y: L1_TO_L2_GAIN * y,
                )
                det.add_channel(ch)
                ch_list.append(ch)
            l1_to_l2_channels_per_dest[id(det)] = ch_list

        # Cross-tile constraints (the bistability mechanism)
        for axis in (0, 1, 2):
            n = neighbor(v, axis)
            if axis in (0, 1):
                front[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS_L2 * y))
                front[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS_L2 * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS_L2 * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS_L2 * y))
            else:
                front[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS_L2 * y))
                front[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS_L2 * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS_L2 * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS_L2 * y))

    # ---- L3 (top) ----
    fI_a, gI_a = sigmoid_activity(rate=RATE_FAST_HI, gain=GAIN_HI, threshold=THRESHOLD_HI)
    fI_b, gI_b = sigmoid_activity(rate=RATE_FAST_HI, gain=GAIN_HI, threshold=THRESHOLD_HI)
    I_a = Node(state=0.2 + 0j, dynamics=fI_a, output=gI_a)
    I_b = Node(state=0.2 + 0j, dynamics=fI_b, output=gI_b)

    fmem_a, gmem_a = tracker(rate=RATE_ADAPT_HI)
    fmem_b, gmem_b = tracker(rate=RATE_ADAPT_HI)
    Imem_a = Node(state=0 + 0j, dynamics=fmem_a, output=gmem_a)
    Imem_b = Node(state=0 + 0j, dynamics=fmem_b, output=gmem_b)

    fhomeo_a, ghomeo_a = tracker(rate=RATE_HOMEO_HI)
    fhomeo_b, ghomeo_b = tracker(rate=RATE_HOMEO_HI)
    Ihomeo_a = Node(state=HOMEO_TARGET + 0j, dynamics=fhomeo_a, output=ghomeo_a)
    Ihomeo_b = Node(state=HOMEO_TARGET + 0j, dynamics=fhomeo_b, output=ghomeo_b)

    noise_a = Noise(std=NOISE_STD, seed=SEED + 1)
    noise_b = Noise(std=NOISE_STD, seed=SEED + 2)

    homeo_xform = homeostatic_feedback(target=HOMEO_TARGET, gain=HOMEO_GAIN)
    I_a.add_channel(Channel(I_b, transform=lambda y: -INHIB_HI * y))
    I_a.add_channel(Channel(Imem_a, transform=lambda y: -ADAPT_FEEDBACK_HI * y))
    I_a.add_channel(Channel(Ihomeo_a, transform=homeo_xform))
    I_a.add_channel(Channel(noise_a))
    I_b.add_channel(Channel(I_a, transform=lambda y: -INHIB_HI * y))
    I_b.add_channel(Channel(Imem_b, transform=lambda y: -ADAPT_FEEDBACK_HI * y))
    I_b.add_channel(Channel(Ihomeo_b, transform=homeo_xform))
    I_b.add_channel(Channel(noise_b))
    Imem_a.add_channel(Channel(I_a))
    Imem_b.add_channel(Channel(I_b))
    Ihomeo_a.add_channel(Channel(I_a))
    Ihomeo_b.add_channel(Channel(I_b))

    # L2 -> L3 (bottom-up) + L3 -> L2 (top-down)
    bu_a, bu_b, td_a, td_b = [], [], [], []
    for v in VERTICES:
        for label, det in [("front", front[v]), ("back", back[v])]:
            ch_bu_a = PlasticChannel(det, dest=I_a, weight=0.05 * rng.random(), learn=learn)
            ch_bu_b = PlasticChannel(det, dest=I_b, weight=0.05 * rng.random(), learn=learn)
            I_a.add_channel(ch_bu_a)
            I_b.add_channel(ch_bu_b)
            bu_a.append((v, label, ch_bu_a))
            bu_b.append((v, label, ch_bu_b))

            ch_td_a = PlasticChannel(I_a, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y: L3_TO_L2_TD_GAIN * y)
            ch_td_b = PlasticChannel(I_b, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y: L3_TO_L2_TD_GAIN * y)
            det.add_channel(ch_td_a)
            det.add_channel(ch_td_b)
            td_a.append((v, label, ch_td_a))
            td_b.append((v, label, ch_td_b))

    # Weight normalizers
    normalizers = []
    # L1 -> L2 per dest
    for v in VERTICES:
        for det in (front[v], back[v]):
            normalizers.append(WeightNormalizer(
                l1_to_l2_channels_per_dest[id(det)], target_sum=TARGET_L1_TO_L2_SUM))
    # L2 -> L3 BU, L3 -> L2 TD
    normalizers.append(WeightNormalizer([ch for _, _, ch in bu_a], target_sum=TARGET_L2_TO_L3_BU_SUM))
    normalizers.append(WeightNormalizer([ch for _, _, ch in bu_b], target_sum=TARGET_L2_TO_L3_BU_SUM))
    normalizers.append(WeightNormalizer([ch for _, _, ch in td_a], target_sum=TARGET_L3_TO_L2_TD_SUM))
    normalizers.append(WeightNormalizer([ch for _, _, ch in td_b], target_sum=TARGET_L3_TO_L2_TD_SUM))

    # Sources order: L1, L2, L3, normalizers last
    sources = []
    for v in VERTICES:
        for combo in ARM_COMBOS:
            sources.append(l1[v][combo])
            sources.append(l1_mems[v][combo])
        sources.extend([front[v], back[v], front_mem[v], back_mem[v]])
    sources.extend([I_a, I_b, Imem_a, Imem_b, Ihomeo_a, Ihomeo_b, noise_a, noise_b])
    sources.extend(normalizers)

    # JAX-compiled execution path (~500-850x faster than the slow Python framework).
    print("Running (JAX-compiled)...")
    import time as _time
    t0 = _time.time()
    history_array, node_to_idx, _final = run_compiled(
        sources, n_steps=N_STEPS, dt=1.0, seed=42,
    )
    elapsed = _time.time() - t0
    print(f"  ran {N_STEPS} steps in {elapsed:.2f}s ({N_STEPS / elapsed:,.0f} steps/s)")

    # Build the observation dict using the same names as the old slow path.
    obs_spec = {
        "I_A": (I_a, "real"),
        "I_B": (I_b, "real"),
    }
    for v in VERTICES:
        obs_spec[f"L2_f{v}"] = (front[v], "real")
        obs_spec[f"L2_b{v}"] = (back[v], "real")
        for combo in ARM_COMBOS:
            obs_spec[f"L1_{v}_{combo}"] = (l1[v][combo], "real")
    history = history_as_dict(history_array, node_to_idx, obs_spec)
    times = list(range(N_STEPS))

    # L2 patterns
    l2_pa = []
    l2_pb = []
    for t in range(N_STEPS):
        a = (sum(history[f"L2_f{v}"][t] for v in VERTICES if v[2] == 0)
             + sum(history[f"L2_b{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        b = (sum(history[f"L2_b{v}"][t] for v in VERTICES if v[2] == 0)
             + sum(history[f"L2_f{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        l2_pa.append(a)
        l2_pb.append(b)

    plot(times,
         {"I_A": history["I_A"], "I_B": history["I_B"],
          "L2 Pattern A": l2_pa, "L2 Pattern B": l2_pb},
         ylabel="activity",
         title="10g_threelayer_top_and_middle")

    # Sample L1 detector to show chaotic activity
    v0 = (0, 0, 0)
    plot(times,
         {f"L1 {v0} {c}": history[f"L1_{v0}_{c}"][:] for c in [(0,0,1), (1,1,0), (0,1,1)]},
         ylabel="activity",
         title="10g_L1_sample_at_vertex_000")

    # What did each L2 detector learn? Check L1->L2 weights at one vertex
    print()
    print("Final L1->L2 weights at vertex (0,0,0) (after normalization):")
    print("  combo (X,Y,Z)   →  weight to L2 'front'   weight to L2 'back'")
    front_chs = l1_to_l2_channels_per_dest[id(front[v0])]
    back_chs = l1_to_l2_channels_per_dest[id(back[v0])]
    for i, combo in enumerate(ARM_COMBOS):
        wf = front_chs[i].weight.real
        wb = back_chs[i].weight.real
        # vertex (0,0,0) at depth 0 = "front" implies arm config (0,0,1) under interpretation A.
        a_consistent = combo == (0, 0, 1)
        b_consistent = combo == (1, 1, 0)
        marker = "  <-- A-consistent (front-implying)" if a_consistent else (
                 "  <-- B-consistent (back-implying)"  if b_consistent else "")
        print(f"  {combo}              {wf:.3f}                    {wb:.3f}{marker}")

    # L2->L3 weights
    print()
    print("Final L2->L3 weights (mean):")
    def is_a(v, label):
        return (v[2] == 0 and label == "front") or (v[2] == 1 and label == "back")
    def is_b(v, label):
        return (v[2] == 0 and label == "back") or (v[2] == 1 and label == "front")
    def mean_w(channels, predicate):
        ws = [ch.weight.real for v, l, ch in channels if predicate(v, l)]
        return float(np.mean(ws)) if ws else float("nan")
    print(f"  L2 A-pattern → I_A: {mean_w(bu_a, is_a):.3f}")
    print(f"  L2 B-pattern → I_A: {mean_w(bu_a, is_b):.3f}")
    print(f"  L2 A-pattern → I_B: {mean_w(bu_b, is_a):.3f}")
    print(f"  L2 B-pattern → I_B: {mean_w(bu_b, is_b):.3f}")


if __name__ == "__main__":
    main()
