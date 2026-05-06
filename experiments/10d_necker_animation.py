"""Animate the simple Necker cube perception flipping over time.

Uses the same network as 10c_necker_simple (16 detectors: front/back per
vertex). The 2D projection shows the cube wireframe with each vertex
colored by its dominant depth-detector — blue when "front" dominates,
red when "back" dominates. Marker size scales with the winner's activity.

When Pattern A is active: the four z=0 vertices are blue (front), the
four z=1 vertices are red (back). In Pattern B these flip. The
animation shows the system alternating between the two interpretations
under adaptation.
"""

import itertools

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from manifold import Channel, Constant, Node, run, sigmoid_activity, tracker


# Same parameters as 10c_necker_simple
VERTICES = list(itertools.product([0, 1], repeat=3))
EDGES = (
    [((0, y, z), (1, y, z)) for y in (0, 1) for z in (0, 1)]
    + [((x, 0, z), (x, 1, z)) for x in (0, 1) for z in (0, 1)]
    + [((x, y, 0), (x, y, 1)) for x in (0, 1) for y in (0, 1)]
)

IMAGE_DRIVE = 0.5
INHIB_INTRATILE = 1.5
EXCITE_CROSS = 0.4
INHIB_CROSS = 0.4
ADAPT_FEEDBACK = 1.5
RATE_FAST = 0.1
RATE_SLOW = 0.005
GAIN = 8.0
THRESHOLD = 2.5
N_STEPS = 4000
SEED = 42

# 2D projection of the cube: z=1 face offset up-right of the z=0 face
def cube_2d(v):
    x, y, z = v
    return (x + z * 0.45, y + z * 0.35)


def build_and_run():
    image = Constant(IMAGE_DRIVE)
    rng = np.random.default_rng(SEED)

    front, back, front_mem, back_mem = {}, {}, {}, {}
    for v in VERTICES:
        f, g = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD)
        front[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        back[v] = Node(state=0.3 * rng.random() + 0j, dynamics=f, output=g)
        mf, mg = tracker(rate=RATE_SLOW)
        front_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)
        back_mem[v] = Node(state=0 + 0j, dynamics=mf, output=mg)

    def neighbor(v, axis):
        return tuple(1 - x if i == axis else x for i, x in enumerate(v))

    for v in VERTICES:
        for det, sib, mem in [
            (front[v], back[v], front_mem[v]),
            (back[v], front[v], back_mem[v]),
        ]:
            det.add_channel(Channel(image))
            det.add_channel(Channel(mem, transform=lambda y: -ADAPT_FEEDBACK * y))
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

    sources = []
    for v in VERTICES:
        sources.extend([front[v], back[v], front_mem[v], back_mem[v]])

    obs = {}
    for v in VERTICES:
        obs[f"f{v}"] = (lambda v=v: front[v].state.real)
        obs[f"b{v}"] = (lambda v=v: back[v].state.real)

    return run(sources=sources, n_steps=N_STEPS, observers=obs)


def main():
    print("Running simulation...")
    history = build_and_run()
    print("Building animation...")

    SAMPLE_EVERY = 20
    frames = list(range(0, N_STEPS, SAMPLE_EVERY))
    positions = {v: cube_2d(v) for v in VERTICES}

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw edges (static)
    for u, v in EDGES:
        px = [positions[u][0], positions[v][0]]
        py = [positions[u][1], positions[v][1]]
        ax.plot(px, py, color="gray", alpha=0.4, linewidth=1.5, zorder=1)

    # Initial vertex scatter
    xs = [positions[v][0] for v in VERTICES]
    ys = [positions[v][1] for v in VERTICES]
    scatter = ax.scatter(
        xs, ys, s=[200] * 8, c=["gray"] * 8,
        edgecolors="black", linewidths=1.0, zorder=2,
    )

    ax.set_xlim(-0.3, 1.7)
    ax.set_ylim(-0.3, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    title = ax.set_title("t = 0", fontsize=14)

    # Per-vertex annotation (which depth label)
    annotations = []
    for v in VERTICES:
        ann = ax.annotate("", positions[v], xytext=(8, 8),
                          textcoords="offset points", fontsize=8, color="black")
        annotations.append((v, ann))

    def update(frame_idx):
        t = frames[frame_idx]
        sizes = []
        colors = []
        for v in VERTICES:
            f = history[f"f{v}"][t]
            b = history[f"b{v}"][t]
            if f >= b:
                # front dominant: blue
                colors.append((0.1, 0.3, 0.9, min(1.0, f + 0.2)))
                sizes.append(100 + f * 600)
            else:
                colors.append((0.9, 0.2, 0.2, min(1.0, b + 0.2)))
                sizes.append(100 + b * 600)
        scatter.set_sizes(sizes)
        scatter.set_color(colors)
        title.set_text(f"t = {t}     (blue = front, red = back)")
        return [scatter, title]

    print("Saving GIF...")
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    anim.save("out/10d_necker_animation.gif", writer="pillow", fps=20)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
