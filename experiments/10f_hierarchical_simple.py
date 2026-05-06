"""Phase 10f: hierarchical higher layer on the SIMPLE bistable lower layer.

The rich-substrate version (10e_hierarchical.py) shows partial self-
organization: I_A reliably learns one of the two cube interpretations,
but I_B doesn't, because once I_A is locked into pattern B, the
"complementary" pattern A doesn't naturally exist in the rich lower
layer for I_B to learn from. Catch-22 at the second-attractor level.

This experiment uses the SIMPLE 16-detector lower layer (10c_simple)
which DOES have both A and B as natural attractors that the substrate
cycles between under its own adaptation. The hierarchical higher layer
should now find both — each I node learning one of the two patterns
that the lower layer naturally produces.

Same robustness mechanisms as 10e: homeostasis, weight normalization,
noise, plasticity. Random init.
"""

import itertools

import numpy as np

from manifold import (
    Channel, Constant, Node, Noise, PlasticChannel, WeightNormalizer,
    hebbian, homeostatic_feedback, plot, run, sigmoid_activity, tracker,
)


VERTICES = list(itertools.product([0, 1], repeat=3))

# Lower layer (same as 10c_simple — the bistable substrate)
IMAGE_DRIVE = 0.5
INHIB_INTRATILE = 1.5
EXCITE_CROSS = 0.4
INHIB_CROSS = 0.4
ADAPT_FEEDBACK_LO = 1.5
RATE_FAST_LO = 0.1
RATE_ADAPT_LO = 0.005
GAIN_LO = 8.0
THRESHOLD_LO = 2.5

# Higher layer
INHIB_HI = 2.0
ADAPT_FEEDBACK_HI = 2.0
RATE_FAST_HI = 0.1
RATE_ADAPT_HI = 0.003
RATE_HOMEO_HI = 0.0003
GAIN_HI = 6.0
THRESHOLD_HI = 1.5

# Robustness regulators
HOMEO_TARGET = 0.4
HOMEO_GAIN = 4.0
NOISE_STD = 0.04
TARGET_BU_SUM = 2.0     # 16 detectors so sum/n ~0.125 mean weight
TARGET_TD_SUM = 2.0
TD_GAIN = 1.0

# Plasticity
ETA = 0.002
DECAY_PLASTIC = 0.002

N_STEPS = 24000
SEED = 42


def main():
    image = Constant(IMAGE_DRIVE)
    rng = np.random.default_rng(SEED)

    # ---- Lower layer: 16 detectors (front, back per vertex) ----
    front, back = {}, {}
    front_mem, back_mem = {}, {}
    for v in VERTICES:
        f, g = sigmoid_activity(rate=RATE_FAST_LO, gain=GAIN_LO, threshold=THRESHOLD_LO)
        front[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        back[v]  = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        mf, mg = tracker(rate=RATE_ADAPT_LO)
        front_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)
        back_mem[v]  = Node(state=0 + 0j, dynamics=mf, output=mg)

    def neighbor(v, axis):
        return tuple(1 - x if i == axis else x for i, x in enumerate(v))

    for v in VERTICES:
        for det, sib, mem in [
            (front[v], back[v], front_mem[v]),
            (back[v], front[v], back_mem[v]),
        ]:
            det.add_channel(Channel(image))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK_LO * y))
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

    # ---- Higher layer ----
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

    # Plastic bottom-up + top-down (purely random init)
    learn = hebbian(eta=ETA, decay=DECAY_PLASTIC)
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
                                     transform=lambda y: TD_GAIN * y)
            ch_td_b = PlasticChannel(I_b, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y: TD_GAIN * y)
            det.add_channel(ch_td_a)
            det.add_channel(ch_td_b)
            td_a.append((v, label, ch_td_a))
            td_b.append((v, label, ch_td_b))

    norm_bu_a = WeightNormalizer([ch for _, _, ch in bu_a], target_sum=TARGET_BU_SUM)
    norm_bu_b = WeightNormalizer([ch for _, _, ch in bu_b], target_sum=TARGET_BU_SUM)
    norm_td_a = WeightNormalizer([ch for _, _, ch in td_a], target_sum=TARGET_TD_SUM)
    norm_td_b = WeightNormalizer([ch for _, _, ch in td_b], target_sum=TARGET_TD_SUM)

    sources = []
    for v in VERTICES:
        sources.extend([front[v], back[v], front_mem[v], back_mem[v]])
    sources.extend([I_a, I_b, Imem_a, Imem_b, Ihomeo_a, Ihomeo_b, noise_a, noise_b])
    sources.extend([norm_bu_a, norm_bu_b, norm_td_a, norm_td_b])

    obs = {
        "I_A": lambda: I_a.state.real,
        "I_B": lambda: I_b.state.real,
        "homeo_A": lambda: Ihomeo_a.state.real,
        "homeo_B": lambda: Ihomeo_b.state.real,
    }
    for v in VERTICES:
        obs[f"f{v}"] = (lambda v=v: front[v].state.real)
        obs[f"b{v}"] = (lambda v=v: back[v].state.real)

    print("Running...")
    history = run(sources=sources, n_steps=N_STEPS, observers=obs)
    times = list(range(N_STEPS))

    # Pattern A = z=0 vertices front + z=1 back
    pa = []
    pb = []
    for t in range(N_STEPS):
        a = (sum(history[f"f{v}"][t] for v in VERTICES if v[2] == 0)
             + sum(history[f"b{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        b = (sum(history[f"b{v}"][t] for v in VERTICES if v[2] == 0)
             + sum(history[f"f{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        pa.append(a)
        pb.append(b)

    plot(times,
         {"I_A": history["I_A"], "I_B": history["I_B"],
          "homeo_A": history["homeo_A"], "homeo_B": history["homeo_B"]},
         ylabel="activity",
         title="10f_hier_simple_higher")

    plot(times,
         {"Pattern A (z=0 front)": pa, "Pattern B (z=0 back)": pb,
          "I_A": history["I_A"], "I_B": history["I_B"]},
         ylabel="activity",
         title="10f_hier_simple_lower_vs_higher")

    # Did each I learn ONE pattern's detectors more strongly?
    # A-pattern detectors = front for z=0 vertices, back for z=1 vertices.
    def is_a(v, label):
        return (v[2] == 0 and label == "front") or (v[2] == 1 and label == "back")
    def is_b(v, label):
        return (v[2] == 0 and label == "back") or (v[2] == 1 and label == "front")

    def mean_w(channels, predicate):
        ws = [ch.weight.real for v, l, ch in channels if predicate(v, l)]
        return float(np.mean(ws)) if ws else float("nan")

    print()
    print("Final BU weights (mean):")
    print(f"  A-pattern → I_A: {mean_w(bu_a, is_a):.3f}")
    print(f"  B-pattern → I_A: {mean_w(bu_a, is_b):.3f}")
    print(f"  A-pattern → I_B: {mean_w(bu_b, is_a):.3f}")
    print(f"  B-pattern → I_B: {mean_w(bu_b, is_b):.3f}")
    print()
    print("Final TD weights (mean):")
    print(f"  I_A → A-pattern: {mean_w(td_a, is_a):.3f}")
    print(f"  I_A → B-pattern: {mean_w(td_a, is_b):.3f}")
    print(f"  I_B → A-pattern: {mean_w(td_b, is_a):.3f}")
    print(f"  I_B → B-pattern: {mean_w(td_b, is_b):.3f}")


if __name__ == "__main__":
    main()
