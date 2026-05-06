"""Phase 10e (robust): hierarchical Necker cube with self-organization.

Earlier version of this file used pure plasticity with hand-tuned bias to
break symmetry. That fell into the Catch-22 of mutual cause: the lower
layer needed top-down bias to lock into a consistent pattern; the higher
layer needed sustained lower patterns to learn from. Random init
collapsed the dynamics — either one I saturated forever or both died.

This version applies all four robustness mechanisms (see
project_robust_vs_finetuned.md memory):

  1. **Homeostatic feedback** on each I node — a slow tracker pushes
     activity toward `HOMEO_TARGET`, so neither I can saturate forever
     and neither can stay silent forever.
  2. **Weight normalization** (synaptic scaling) on each I's bottom-up
     and top-down channels — total |weight| stays at a target, so
     Hebbian rebalances among individual weights without runaway.
  3. **Noise** on each I — small stochastic kick that breaks symmetry
     and lets the system find its operating point through fluctuations.
  4. **Gated Hebbian** plasticity — learning rate peaks at mid-activity
     and is zero at saturation/silence, so whoever wins early can't
     entrench irreversibly.

Adaptation memory remains as the fast switching mechanism. The four new
mechanisms are slow regulators around it.

Initial weights are now PURE random (no hand-wired bias). The system
must self-organize from scratch.
"""

import itertools

import numpy as np

from manifold import (
    Channel, Constant, Node, Noise, PlasticChannel, WeightNormalizer,
    hebbian, homeostatic_feedback, plot, run, sigmoid_activity, tracker,
)


VERTICES = list(itertools.product([0, 1], repeat=3))
ARM_COMBOS = list(itertools.product([0, 1], repeat=3))

# Lower layer (same constraint structure as before, plus per-detector
# adaptation so the substrate cycles through patterns on ~200-step
# timescale — gives the higher layer temporal structure to lock onto).
IMAGE_DRIVE = 0.5
INHIB_INTRATILE = 1.5
EXCITE_CROSS = 0.2
INHIB_CROSS = 0.6
RATE_FAST_LO = 0.1
GAIN_LO = 8.0
THRESHOLD_LO = 2.5
ADAPT_FEEDBACK_LO = 1.0
RATE_ADAPT_LO = 0.005

# Higher layer
INHIB_HI = 2.0
ADAPT_FEEDBACK_HI = 2.0
RATE_FAST_HI = 0.1
RATE_ADAPT_HI = 0.003          # adaptation timescale (~330 steps)
RATE_HOMEO_HI = 0.0003         # homeostasis timescale (~3000 steps)
GAIN_HI = 6.0
THRESHOLD_HI = 1.5

# Robustness regulators
HOMEO_TARGET = 0.4
HOMEO_GAIN = 4.0
NOISE_STD = 0.04
TARGET_BU_SUM = 4.0
TARGET_TD_SUM = 4.0
TD_GAIN = 1.0                  # top-down boost factor (stronger now)

# Plasticity (plain Hebbian; weight cap from synaptic scaling, not gating)
ETA = 0.002
DECAY_PLASTIC = 0.002

N_STEPS = 24000
SEED = 42


def neighbor(v, axis):
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def is_a_consistent(v, combo):
    _, _, c = v
    d_x, d_y, d_z = combo
    return d_x == c and d_y == c and d_z == 1 - c


def is_b_consistent(v, combo):
    _, _, c = v
    d_x, d_y, d_z = combo
    return d_x == 1 - c and d_y == 1 - c and d_z == c


def main():
    image = Constant(IMAGE_DRIVE)
    rng = np.random.default_rng(SEED)

    # ---- Lower layer ----
    detectors = {v: {} for v in VERTICES}
    lo_memories = {v: {} for v in VERTICES}
    for v in VERTICES:
        for combo in ARM_COMBOS:
            f, g = sigmoid_activity(rate=RATE_FAST_LO, gain=GAIN_LO, threshold=THRESHOLD_LO)
            detectors[v][combo] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
            mf, mg = tracker(rate=RATE_ADAPT_LO)
            lo_memories[v][combo] = Node(state=0 + 0j, dynamics=mf, output=mg)

    # Cross-tile + within-tile + image + lower-layer adaptation
    for v in VERTICES:
        for combo in ARM_COMBOS:
            d_x, d_y, d_z = combo
            det = detectors[v][combo]
            mem = lo_memories[v][combo]
            det.add_channel(Channel(image))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK_LO * y))
            mem.add_channel(Channel(det))
            for other in ARM_COMBOS:
                if other == combo:
                    continue
                det.add_channel(Channel(detectors[v][other],
                                       transform=lambda y: -INHIB_INTRATILE * y))
            x_nbr = neighbor(v, 0)
            for c2 in ARM_COMBOS:
                w = EXCITE_CROSS if c2[0] == d_x else -INHIB_CROSS
                det.add_channel(Channel(detectors[x_nbr][c2], transform=lambda y, w=w: w * y))
            y_nbr = neighbor(v, 1)
            for c2 in ARM_COMBOS:
                w = EXCITE_CROSS if c2[1] == d_y else -INHIB_CROSS
                det.add_channel(Channel(detectors[y_nbr][c2], transform=lambda y, w=w: w * y))
            z_nbr = neighbor(v, 2)
            for c2 in ARM_COMBOS:
                w = EXCITE_CROSS if c2[2] == 1 - d_z else -INHIB_CROSS
                det.add_channel(Channel(detectors[z_nbr][c2], transform=lambda y, w=w: w * y))

    # ---- Higher layer ----
    fI_a, gI_a = sigmoid_activity(rate=RATE_FAST_HI, gain=GAIN_HI, threshold=THRESHOLD_HI)
    fI_b, gI_b = sigmoid_activity(rate=RATE_FAST_HI, gain=GAIN_HI, threshold=THRESHOLD_HI)
    I_a = Node(state=0.2 + 0j, dynamics=fI_a, output=gI_a)
    I_b = Node(state=0.2 + 0j, dynamics=fI_b, output=gI_b)

    # Adaptation memory (fast switching)
    fmem_a, gmem_a = tracker(rate=RATE_ADAPT_HI)
    fmem_b, gmem_b = tracker(rate=RATE_ADAPT_HI)
    Imem_a = Node(state=0 + 0j, dynamics=fmem_a, output=gmem_a)
    Imem_b = Node(state=0 + 0j, dynamics=fmem_b, output=gmem_b)

    # Homeostatic tracker (slow regulation toward target)
    fhomeo_a, ghomeo_a = tracker(rate=RATE_HOMEO_HI)
    fhomeo_b, ghomeo_b = tracker(rate=RATE_HOMEO_HI)
    Ihomeo_a = Node(state=HOMEO_TARGET + 0j, dynamics=fhomeo_a, output=ghomeo_a)
    Ihomeo_b = Node(state=HOMEO_TARGET + 0j, dynamics=fhomeo_b, output=ghomeo_b)

    # Noise source per I
    noise_a = Noise(std=NOISE_STD, seed=SEED + 1)
    noise_b = Noise(std=NOISE_STD, seed=SEED + 2)

    # Mutual inhibition + adaptation + homeostatic feedback + noise
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

    # Plastic bottom-up — pure random init (no hand bias)
    learn = hebbian(eta=ETA, decay=DECAY_PLASTIC)
    bu_a, bu_b = [], []
    for v in VERTICES:
        for combo in ARM_COMBOS:
            det = detectors[v][combo]
            ch_a = PlasticChannel(det, dest=I_a, weight=0.05 * rng.random(), learn=learn)
            ch_b = PlasticChannel(det, dest=I_b, weight=0.05 * rng.random(), learn=learn)
            I_a.add_channel(ch_a)
            I_b.add_channel(ch_b)
            bu_a.append((v, combo, ch_a))
            bu_b.append((v, combo, ch_b))

    # Plastic top-down — pure random init
    td_a, td_b = [], []
    for v in VERTICES:
        for combo in ARM_COMBOS:
            det = detectors[v][combo]
            ch_td_a = PlasticChannel(I_a, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y: TD_GAIN * y)
            ch_td_b = PlasticChannel(I_b, dest=det, weight=0.05 * rng.random(), learn=learn,
                                     transform=lambda y: TD_GAIN * y)
            det.add_channel(ch_td_a)
            det.add_channel(ch_td_b)
            td_a.append((v, combo, ch_td_a))
            td_b.append((v, combo, ch_td_b))

    # Weight normalizers — keep total |weight| at target sum per I per direction
    norm_bu_a = WeightNormalizer([ch for _, _, ch in bu_a], target_sum=TARGET_BU_SUM)
    norm_bu_b = WeightNormalizer([ch for _, _, ch in bu_b], target_sum=TARGET_BU_SUM)
    norm_td_a = WeightNormalizer([ch for _, _, ch in td_a], target_sum=TARGET_TD_SUM)
    norm_td_b = WeightNormalizer([ch for _, _, ch in td_b], target_sum=TARGET_TD_SUM)

    # ---- Sources order ----
    sources = []
    for v in VERTICES:
        for combo in ARM_COMBOS:
            sources.append(detectors[v][combo])
            sources.append(lo_memories[v][combo])
    sources.extend([I_a, I_b, Imem_a, Imem_b, Ihomeo_a, Ihomeo_b, noise_a, noise_b])
    # Normalizers LAST so they see post-Hebbian-update weights this step
    sources.extend([norm_bu_a, norm_bu_b, norm_td_a, norm_td_b])

    obs = {
        "I_A": lambda: I_a.state.real,
        "I_B": lambda: I_b.state.real,
        "homeo_A": lambda: Ihomeo_a.state.real,
        "homeo_B": lambda: Ihomeo_b.state.real,
    }
    for v in VERTICES:
        for combo in ARM_COMBOS:
            obs[f"{v}_{combo}"] = (lambda v=v, combo=combo: detectors[v][combo].state.real)

    print("Running...")
    history = run(sources=sources, n_steps=N_STEPS, observers=obs)
    times = list(range(N_STEPS))

    plot(times,
         {"I_A": history["I_A"], "I_B": history["I_B"],
          "homeo_A (avg target=%.1f)" % HOMEO_TARGET: history["homeo_A"],
          "homeo_B": history["homeo_B"]},
         ylabel="activity",
         title="10e_robust_higher_layer")

    a_keys = [(v, c) for v in VERTICES for c in ARM_COMBOS if is_a_consistent(v, c)]
    b_keys = [(v, c) for v in VERTICES for c in ARM_COMBOS if is_b_consistent(v, c)]
    other_keys = [(v, c) for v in VERTICES for c in ARM_COMBOS
                  if not is_a_consistent(v, c) and not is_b_consistent(v, c)]
    pa = [sum(history[f"{v}_{c}"][t] for v, c in a_keys) / len(a_keys) for t in range(N_STEPS)]
    pb = [sum(history[f"{v}_{c}"][t] for v, c in b_keys) / len(b_keys) for t in range(N_STEPS)]
    po = [sum(history[f"{v}_{c}"][t] for v, c in other_keys) / len(other_keys) for t in range(N_STEPS)]
    plot(times,
         {"Pattern A": pa, "Pattern B": pb, "Inconsistent": po},
         ylabel="mean activity",
         title="10e_robust_lower_layer")

    def mean_w(channels, predicate):
        ws = [ch.weight.real for v, c, ch in channels if predicate(v, c)]
        return float(np.mean(ws)) if ws else float("nan")

    print()
    print("Final BU weights (mean by category, normalized):")
    print(f"  A-consistent → I_A: {mean_w(bu_a, is_a_consistent):.3f}")
    print(f"  A-consistent → I_B: {mean_w(bu_b, is_a_consistent):.3f}")
    print(f"  B-consistent → I_A: {mean_w(bu_a, is_b_consistent):.3f}")
    print(f"  B-consistent → I_B: {mean_w(bu_b, is_b_consistent):.3f}")
    print(f"  Inconsistent → I_A: {mean_w(bu_a, lambda v, c: not is_a_consistent(v, c) and not is_b_consistent(v, c)):.3f}")
    print(f"  Inconsistent → I_B: {mean_w(bu_b, lambda v, c: not is_a_consistent(v, c) and not is_b_consistent(v, c)):.3f}")
    print()
    print("Final TD weights (mean by category, normalized):")
    print(f"  I_A → A-consistent: {mean_w(td_a, is_a_consistent):.3f}")
    print(f"  I_A → B-consistent: {mean_w(td_a, is_b_consistent):.3f}")
    print(f"  I_B → A-consistent: {mean_w(td_b, is_a_consistent):.3f}")
    print(f"  I_B → B-consistent: {mean_w(td_b, is_b_consistent):.3f}")


if __name__ == "__main__":
    main()
