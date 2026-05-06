"""Phase 10c: Necker cube as constraint resonance.

Each cube vertex (8 of them) is a tile containing 8 feature detectors —
one per (X-arm-depth, Y-arm-depth, Z-arm-depth) combination, with depth
binary (front/back, simplified from the 3-depth general framing). The
arm-depth at vertex u along axis a encodes "the depth of u's neighbor
along axis a."

The image excites all 64 detectors equally — *the image is depth-
ambiguous*, so every depth combo fires initially. Two mechanisms then
prune the activity to a globally consistent interpretation:

  1. Within-tile mutual inhibition: only one depth combo wins at each
     vertex.
  2. Cross-tile excitation between detectors that *agree* on shared arm
     depths:
       - X edges (same-face): matching arm-X depths excite each other
       - Y edges (same-face): matching arm-Y depths
       - Z edges (front-back axis): OPPOSITE arm-Z depths excite each other
         (because Z-arm at one end encodes the OTHER end's depth, and the
         two ends of a Z edge are at different depths in any cube
         interpretation)

For the cube structure, only TWO globally consistent patterns of arm
depths exist — the two Necker interpretations. Adaptation memory on
each detector causes the dominant pattern to fatigue, allowing the
alternative pattern to take over. Periodic switching with no hand-
wired "interpretation A vs B" nodes.

The two consistent patterns at each vertex (a,b,c):
  Pattern A (z=0 face is front): (d_x=c, d_y=c, d_z=1-c)
  Pattern B (z=1 face is front): (d_x=1-c, d_y=1-c, d_z=c)
"""

import itertools

import numpy as np

from manifold import Channel, Constant, Node, plot, run, sigmoid_activity, tracker


VERTICES = list(itertools.product([0, 1], repeat=3))   # 8 cube vertices
ARM_COMBOS = list(itertools.product([0, 1], repeat=3)) # 8 (d_x, d_y, d_z)
X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2

IMAGE_DRIVE = 0.5
INHIB_INTRATILE = 1.5
EXCITE_CROSS = 0.2     # weak enough that adaptation can crash the dominant pattern
INHIB_CROSS = 0.6      # strong enough to suppress the alternative while a pattern rules
ADAPT_FEEDBACK = 1.5
RATE_FAST = 0.1
RATE_SLOW = 0.005
GAIN = 8.0
THRESHOLD = 2.5
N_STEPS = 6000
SEED = 42


def neighbor(v: tuple, axis: int) -> tuple:
    return tuple(1 - x if i == axis else x for i, x in enumerate(v))


def is_a_consistent(v: tuple, combo: tuple) -> bool:
    """Pattern A: vertex (a,b,c) at depth c. Arm-depths follow."""
    _, _, c = v
    d_x, d_y, d_z = combo
    return d_x == c and d_y == c and d_z == 1 - c


def is_b_consistent(v: tuple, combo: tuple) -> bool:
    _, _, c = v
    d_x, d_y, d_z = combo
    return d_x == 1 - c and d_y == 1 - c and d_z == c


def main():
    image = Constant(IMAGE_DRIVE)
    rng = np.random.default_rng(SEED)

    # Create detector and memory nodes: detectors[v][combo]
    detectors = {v: {} for v in VERTICES}
    memories = {v: {} for v in VERTICES}
    for v in VERTICES:
        for combo in ARM_COMBOS:
            f, g = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD)
            init = 0.3 * rng.random()
            detectors[v][combo] = Node(state=init + 0j, dynamics=f, output=g)
            mf, mg = tracker(rate=RATE_SLOW)
            memories[v][combo] = Node(state=0 + 0j, dynamics=mf, output=mg)

    # Wire each detector
    for v in VERTICES:
        for combo in ARM_COMBOS:
            d_x, d_y, d_z = combo
            det = detectors[v][combo]
            mem = memories[v][combo]

            # Image input + adaptation
            det.add_channel(Channel(image))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK * y))
            mem.add_channel(Channel(det))

            # Within-tile mutual inhibition (7 siblings)
            for other_combo in ARM_COMBOS:
                if other_combo == combo:
                    continue
                det.add_channel(Channel(
                    detectors[v][other_combo],
                    transform=lambda y: -INHIB_INTRATILE * y,
                ))

            # Cross-tile: matching arm-depths excite, non-matching inhibit.
            # X-neighbor: matching d_x excites, non-matching d_x inhibits.
            x_nbr = neighbor(v, X_AXIS)
            for c2 in ARM_COMBOS:
                if c2[X_AXIS] == d_x:
                    det.add_channel(Channel(
                        detectors[x_nbr][c2],
                        transform=lambda y: EXCITE_CROSS * y,
                    ))
                else:
                    det.add_channel(Channel(
                        detectors[x_nbr][c2],
                        transform=lambda y: -INHIB_CROSS * y,
                    ))
            # Y-neighbor: matching d_y excites, non-matching inhibits.
            y_nbr = neighbor(v, Y_AXIS)
            for c2 in ARM_COMBOS:
                if c2[Y_AXIS] == d_y:
                    det.add_channel(Channel(
                        detectors[y_nbr][c2],
                        transform=lambda y: EXCITE_CROSS * y,
                    ))
                else:
                    det.add_channel(Channel(
                        detectors[y_nbr][c2],
                        transform=lambda y: -INHIB_CROSS * y,
                    ))
            # Z-neighbor: OPPOSITE d_z excites (Z-edges link different depths),
            # SAME d_z inhibits.
            z_nbr = neighbor(v, Z_AXIS)
            for c2 in ARM_COMBOS:
                if c2[Z_AXIS] == 1 - d_z:
                    det.add_channel(Channel(
                        detectors[z_nbr][c2],
                        transform=lambda y: EXCITE_CROSS * y,
                    ))
                else:
                    det.add_channel(Channel(
                        detectors[z_nbr][c2],
                        transform=lambda y: -INHIB_CROSS * y,
                    ))

    # Run
    sources = []
    for v in VERTICES:
        for combo in ARM_COMBOS:
            sources.append(detectors[v][combo])
            sources.append(memories[v][combo])

    obs = {}
    for v in VERTICES:
        for combo in ARM_COMBOS:
            obs[f"{v}_{combo}"] = (
                lambda v=v, combo=combo: detectors[v][combo].state.real
            )

    history = run(sources=sources, n_steps=N_STEPS, observers=obs)
    times = list(range(N_STEPS))

    # Aggregate: mean activity by consistency class
    a_keys = [(v, c) for v in VERTICES for c in ARM_COMBOS if is_a_consistent(v, c)]
    b_keys = [(v, c) for v in VERTICES for c in ARM_COMBOS if is_b_consistent(v, c)]
    other_keys = [(v, c) for v in VERTICES for c in ARM_COMBOS
                  if not is_a_consistent(v, c) and not is_b_consistent(v, c)]

    pattern_a = [
        sum(history[f"{v}_{c}"][t] for v, c in a_keys) / len(a_keys)
        for t in range(N_STEPS)
    ]
    pattern_b = [
        sum(history[f"{v}_{c}"][t] for v, c in b_keys) / len(b_keys)
        for t in range(N_STEPS)
    ]
    other = [
        sum(history[f"{v}_{c}"][t] for v, c in other_keys) / len(other_keys)
        for t in range(N_STEPS)
    ]

    plot(
        times,
        {
            "Pattern A (mean of 8 detectors)":  pattern_a,
            "Pattern B (mean of 8 detectors)":  pattern_b,
            "Inconsistent (mean of 48)":         other,
        },
        ylabel="mean detector activity",
        title="10c_necker_pattern_dominance",
    )

    # Per-vertex breakdown for vertex (0,0,0) — show all 8 detectors
    v0 = (0, 0, 0)
    series = {}
    for combo in ARM_COMBOS:
        label = f"{combo}"
        if is_a_consistent(v0, combo):
            label += " [A-consistent]"
        if is_b_consistent(v0, combo):
            label += " [B-consistent]"
        series[label] = history[f"{v0}_{combo}"]
    plot(
        times,
        series,
        ylabel="activity",
        title="10c_necker_vertex_000_detectors",
    )


if __name__ == "__main__":
    main()
