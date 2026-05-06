"""Phase 10b: smallest bistable rivalry — the relaxation-oscillator engine.

Two `sigmoid_activity` nodes A and B that mutually inhibit. Each has a
slow `tracker` as its adaptation memory, tracking its own activity and
subtracting from its own drive. Constant symmetric input.

Without adaptation: winner-take-all settles permanently.
With adaptation: the winner fatigues, the loser takes over, alternation
indefinitely.

This is the substrate for the Necker cube (phase 10c) — and importantly,
it's a *reuse* of the same `adaptation` mechanism from phase 1. There the
adaptation acts on a high-pass error to return it to baseline; here it
acts at a slow rate on amplitude to release the dominant interpretation.

Wiring per population (A, mirrored for B):
    A_act ← +1·input
    A_act ← −inhib·B_act
    A_act ← −1·A_mem        (adaptation feedback)
    A_mem ← +1·A_act        (memory tracks activity, slowly)

Plot:
  10b_rivalry_activities — the two activity traces alternating
  10b_rivalry_full       — activities and memories together (relaxation)
"""

from manifold import Channel, Constant, Node, plot, run, sigmoid_activity, tracker


RATE_FAST = 0.1
RATE_SLOW = 0.005
GAIN = 6.0
THRESHOLD = 3.0
INHIB = 1.5
INPUT_LEVEL = 1.0
N_STEPS = 4000


def main():
    input_src = Constant(INPUT_LEVEL)

    a_dyn, a_out = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD)
    b_dyn, b_out = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THRESHOLD)
    a_mem_dyn, a_mem_out = tracker(rate=RATE_SLOW)
    b_mem_dyn, b_mem_out = tracker(rate=RATE_SLOW)

    # Symmetry-breaking initial conditions
    a_act = Node(state=0.6 + 0j, dynamics=a_dyn, output=a_out)
    a_mem = Node(state=0.0 + 0j, dynamics=a_mem_dyn, output=a_mem_out)
    b_act = Node(state=0.0 + 0j, dynamics=b_dyn, output=b_out)
    b_mem = Node(state=0.0 + 0j, dynamics=b_mem_dyn, output=b_mem_out)

    # A's activity wiring
    a_act.add_channel(Channel(input_src))
    a_act.add_channel(Channel(b_act, transform=lambda y: -INHIB * y))
    a_act.add_channel(Channel(a_mem, transform=lambda y: -1.0 * y))

    # B's activity wiring (mirror)
    b_act.add_channel(Channel(input_src))
    b_act.add_channel(Channel(a_act, transform=lambda y: -INHIB * y))
    b_act.add_channel(Channel(b_mem, transform=lambda y: -1.0 * y))

    # Memories track activities
    a_mem.add_channel(Channel(a_act))
    b_mem.add_channel(Channel(b_act))

    history = run(
        sources=[a_act, a_mem, b_act, b_mem],
        n_steps=N_STEPS,
        observers={
            "A activity": lambda: a_act.state.real,
            "A memory":   lambda: a_mem.state.real,
            "B activity": lambda: b_act.state.real,
            "B memory":   lambda: b_mem.state.real,
        },
    )

    times = list(range(N_STEPS))

    plot(
        times,
        {
            "A activity": history["A activity"],
            "B activity": history["B activity"],
        },
        ylabel="activity",
        title="10b_rivalry_activities",
    )

    plot(
        times,
        {
            "A act": history["A activity"],
            "A mem": history["A memory"],
            "B act": history["B activity"],
            "B mem": history["B memory"],
        },
        ylabel="value",
        title="10b_rivalry_full",
    )


if __name__ == "__main__":
    main()
