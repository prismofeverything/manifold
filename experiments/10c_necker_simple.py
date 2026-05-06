"""Phase 10c (simple): Necker cube bistability with vertex-depth detectors only.

Companion to 10c_necker_constraint.py. There we used the user's richer
framing — 8 detectors per vertex, one per (X-arm, Y-arm, Z-arm) depth
combo. The rich version shows constraint dynamics but doesn't reliably
collapse to the 2 globally consistent patterns because 6 of the 8
local detectors at each vertex don't satisfy the cube's face structure
(X-arm = Y-arm = NOT Z-arm), and our cross-tile constraints don't
sufficiently distinguish them from the 2 valid configs.

Here we keep only the 2 *locally valid* configs per vertex (= the front
and back depth labels). 8 vertices × 2 detectors = 16 detector nodes.
Same constraint logic but on a much smaller solution space, so the
basin structure is clean and the system reliably finds A or B and
switches via adaptation.

This is the minimal demonstration that the constraint-resonance
mechanism works for the Necker cube. The richer 64-detector version
shows what *would* work with stronger constraints / a higher-order
biasing layer / plastic top-down — that's a future experiment.
"""

import itertools

import numpy as np

from manifold import Channel, Constant, Node, plot, run, sigmoid_activity, tracker


VERTICES = list(itertools.product([0, 1], repeat=3))   # 8 cube vertices

IMAGE_DRIVE = 0.5
INHIB_INTRATILE = 1.5     # front ⊥ back at the same vertex
EXCITE_CROSS = 0.4
INHIB_CROSS = 0.4
ADAPT_FEEDBACK = 1.5
RATE_FAST = 0.1
RATE_SLOW = 0.005
GAIN = 8.0
THRESHOLD = 2.5
N_STEPS = 6000
SEED = 42


def main():
    image = Constant(IMAGE_DRIVE)
    rng = np.random.default_rng(SEED)

    # 2 detectors per vertex: index 0 = "front", index 1 = "back"
    front = {}
    back = {}
    front_mem = {}
    back_mem = {}
    for v in VERTICES:
        f, g = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD)
        front[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        back[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        mf, mg = tracker(rate=RATE_SLOW)
        front_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)
        back_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)

    # Helper: cube neighbor along axis
    def neighbor(v, axis):
        return tuple(1 - x if i == axis else x for i, x in enumerate(v))

    for v in VERTICES:
        for sib_pair in [(front[v], back[v], front_mem[v]), (back[v], front[v], back_mem[v])]:
            det, sibling, mem = sib_pair
            det.add_channel(Channel(image))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK * y))
            mem.add_channel(Channel(det))
            det.add_channel(Channel(sibling, transform=lambda y: -INHIB_INTRATILE * y))

        # Cross-tile constraints
        for axis in (0, 1, 2):
            n = neighbor(v, axis)
            if axis in (0, 1):  # X or Y edge: same-depth excites, different inhibits
                front[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS * y))
                front[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS * y))
            else:  # Z edge: opposite-depth excites, same inhibits
                front[v].add_channel(Channel(back[n], transform=lambda y: EXCITE_CROSS * y))
                front[v].add_channel(Channel(front[n], transform=lambda y: -INHIB_CROSS * y))
                back[v].add_channel(Channel(front[n], transform=lambda y: EXCITE_CROSS * y))
                back[v].add_channel(Channel(back[n], transform=lambda y: -INHIB_CROSS * y))

    sources = []
    for v in VERTICES:
        sources.extend([front[v], back[v], front_mem[v], back_mem[v]])

    obs = {}
    for v in VERTICES:
        obs[f"f{v}"] = (lambda v=v: front[v].state.real)
        obs[f"b{v}"] = (lambda v=v: back[v].state.real)

    history = run(sources=sources, n_steps=N_STEPS, observers=obs)
    times = list(range(N_STEPS))

    # Pattern A: z=0 vertices front, z=1 vertices back
    # Pattern B: z=0 vertices back, z=1 vertices front
    pattern_a = []
    pattern_b = []
    for t in range(N_STEPS):
        a = (sum(history[f"f{v}"][t] for v in VERTICES if v[2] == 0)
             + sum(history[f"b{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        b = (sum(history[f"b{v}"][t] for v in VERTICES if v[2] == 0)
             + sum(history[f"f{v}"][t] for v in VERTICES if v[2] == 1)) / 8
        pattern_a.append(a)
        pattern_b.append(b)

    plot(
        times,
        {"Pattern A (z=0 front)": pattern_a,
         "Pattern B (z=0 back)":  pattern_b},
        ylabel="mean detector activity",
        title="10c_simple_necker_pattern_dominance",
    )

    # Per-vertex front/back for vertex (0,0,0)
    plot(
        times,
        {"front (0,0,0)": history["f(0, 0, 0)"],
         "back  (0,0,0)": history["b(0, 0, 0)"]},
        ylabel="activity",
        title="10c_simple_necker_vertex_000",
    )


if __name__ == "__main__":
    main()
