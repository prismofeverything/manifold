"""The book of diagrams — first three levels.

Generates one figure per canonical n-node network for n = 1, 2, 3.
Each entry: schematic (graph), equations, behavior plot, classification,
notes about composition. See `project_book_of_diagrams.md` memory for
the per-entry structure rationale.

Output: `out/book/L{n}_{nletter}_{name}.png` for each entry.
"""

import os

import numpy as np

from manifold import (
    Channel, Constant, Node, PlasticChannel, hebbian, polar, run,
    sigmoid_activity, tracker, Source,
)
from manifold.book import (
    draw_excite, draw_external_input, draw_inhibit, draw_node, draw_plastic,
    draw_self_loop, make_entry_figure, setup_schematic_ax,
)


OUT_DIR = "out/book"


# -------------------- Level 1 --------------------------------------------

def entry_L1_1a_adaptation():
    """Adaptation: leaky integrator + high-pass output. Phase-1 primitive."""
    SENS = 0.1
    RATE = 0.1
    N = 333

    def schedule(t):
        if t < 111: return 0.0
        if t < 222: return 1.0
        return 0.0

    def schematic(ax):
        setup_schematic_ax(ax, title="schematic — adaptation node with self-channel and step input")
        node_pos = (0.0, 0.0)
        draw_node(ax, node_pos, "s", color="#cce5ff")
        draw_external_input(ax, node_pos, label="x", offset=(-0.55, 0))
        draw_self_loop(ax, node_pos, label="self: y → α·y", side="top", kind="excite")
        ax.text(0.0, -0.45, "output:  y = β·x − s", fontsize=9, ha="center", style="italic")

    def behavior(ax):
        from manifold import Environment
        env = Environment(schedule)

        def f(s, xs, dt): return s + dt * xs[1]
        def g(s, xs): return SENS * xs[0] - s

        node = Node(state=0+0j, dynamics=f, output=g)
        node.add_channel(Channel(env))
        node.add_channel(Channel(node, lambda y: RATE * y))
        h = run([env, node], n_steps=N,
                observers={"error": lambda: node.read(), "memory": lambda: node.state})
        t = list(range(N))
        ax.plot(t, [v.real for v in h["error"]], label="output (error)", color="tab:blue")
        ax.plot(t, [v.real for v in h["memory"]], label="state (memory)", color="tab:orange")
        ax.plot(t, [schedule(i) for i in t], label="input x", color="tab:green", linestyle=":")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — output spikes at input change, memory tracks input level")

    eq = (
        "state s ∈ ℂ\n"
        "channel c1: ext input x\n"
        "channel c2: self, transform = α·y\n\n"
        "f(s, [x, y_self], dt)  = s + dt·y_self\n"
        "g(s, [x, y_self])       = β·x − s\n\n"
        "params: β = 0.1, α = 0.1"
    )
    fig = make_entry_figure(
        title="L1.1a — Adaptation (leaky integrator + high-pass output)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="returns to baseline (high-pass error to zero)",
        notes=("• phase-1 primitive\n"
               "• leaky low-pass on state, high-pass on output\n"
               "• same engine produces 10b rivalry switching\n"
               "  when memory is fed with mutual inhib at slow rate"),
    )
    fig.savefig(f"{OUT_DIR}/L1_1a_adaptation.png", dpi=110, bbox_inches="tight")


def entry_L1_1b_tracker():
    """Tracker: state = low-pass of input, output = state. The memory primitive."""
    RATE = 0.05
    N = 600

    def schedule(t):
        if t < 100: return 0.0
        if t < 350: return 1.0
        return 0.3

    def schematic(ax):
        setup_schematic_ax(ax, title="schematic — pure tracker, output = state")
        node_pos = (0.0, 0.0)
        draw_node(ax, node_pos, "s")
        draw_external_input(ax, node_pos, label="x", offset=(-0.55, 0))
        ax.text(0.0, -0.45, "output:  y = s   (no high-pass)", fontsize=9, ha="center", style="italic")

    def behavior(ax):
        from manifold import Environment
        env = Environment(schedule)
        f, g = tracker(rate=RATE)
        node = Node(state=0+0j, dynamics=f, output=g)
        node.add_channel(Channel(env))
        h = run([env, node], n_steps=N, observers={"s": lambda: node.state.real})
        t = list(range(N))
        ax.plot(t, h["s"], label="state s = output", color="tab:orange")
        ax.plot(t, [schedule(i) for i in t], label="input x", color="tab:green", linestyle=":")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — state lags input (low-pass), no high-pass component")

    eq = (
        "state s ∈ ℂ\n"
        "channel c1: ext input x\n\n"
        "f(s, [x], dt)  = s + dt·rate·(x − s)\n"
        "g(s, [x])       = s\n\n"
        "params: rate = 0.05"
    )
    fig = make_entry_figure(
        title="L1.1b — Tracker (low-pass integrator)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="settles to setpoint = input mean",
        notes=("• used as the 'memory' tracker for adaptation circuits\n"
               "• building block for homeostatic feedback loops"),
    )
    fig.savefig(f"{OUT_DIR}/L1_1b_tracker.png", dpi=110, bbox_inches="tight")


def entry_L1_1c_oscillator():
    """Pure Kuramoto oscillator — single node, intrinsic ω, no input."""
    OMEGA = 0.1
    N = 600

    def schematic(ax):
        setup_schematic_ax(ax, title="schematic — pure oscillator, no input, intrinsic ω")
        node_pos = (0.0, 0.0)
        draw_node(ax, node_pos, "s")
        draw_self_loop(ax, node_pos, label="phase rotation: ω", side="top", kind="excite")
        ax.text(0.0, -0.45, "state on unit circle, rotates at ω", fontsize=9, ha="center", style="italic")

    def behavior(ax):
        f, g = polar(rate=0.0, omega=OMEGA, coupling=0.0)
        node = Node(state=np.exp(1j * 0.0), dynamics=f, output=g)
        h = run([node], n_steps=N, observers={"s": lambda: node.state})
        t = list(range(N))
        re = [v.real for v in h["s"]]
        im = [v.imag for v in h["s"]]
        ax.plot(t, re, label="Re(s)", color="tab:blue")
        ax.plot(t, im, label="Im(s)", color="tab:orange")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — sinusoidal Re/Im at intrinsic ω, |s| stays at 1")

    eq = (
        "state s ∈ ℂ, |s| = 1\n\n"
        "f(s, [], dt)  = s · exp(i·dt·ω)\n"
        "g(s, [])       = s\n\n"
        "params: ω = 0.1"
    )
    fig = make_entry_figure(
        title="L1.1c — Pure oscillator (Kuramoto, single node)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="oscillates at intrinsic ω",
        notes="• building block for synchronization networks (phases 4, 9)",
    )
    fig.savefig(f"{OUT_DIR}/L1_1c_oscillator.png", dpi=110, bbox_inches="tight")


def entry_L1_1d_adaptive_oscillator():
    """Polar with rate>0 and ω>0 — amplitude tracks input, phase rotates."""
    OMEGA = 0.1
    RATE = 0.05
    N = 600

    def schedule(t):
        if t < 200: return 0.0
        if t < 400: return 1.0
        return 0.0

    def schematic(ax):
        setup_schematic_ax(ax, title="schematic — adaptive oscillator (amplitude + phase)")
        node_pos = (0.0, 0.0)
        draw_node(ax, node_pos, "s")
        draw_external_input(ax, node_pos, label="x", offset=(-0.55, 0))
        draw_self_loop(ax, node_pos, label="ω + amplitude track", side="top", kind="excite")
        ax.text(0.0, -0.45, "|s| tracks |x|;  arg(s) rotates at ω", fontsize=9, ha="center", style="italic")

    def behavior(ax):
        from manifold import Environment
        env = Environment(schedule)
        f, g = polar(sensitivity=1.0, rate=RATE, omega=OMEGA, coupling=0.0)
        node = Node(state=0.001+0j, dynamics=f, output=g)
        node.add_channel(Channel(env))
        h = run([env, node], n_steps=N, observers={"s": lambda: node.state})
        t = list(range(N))
        mags = [abs(v) for v in h["s"]]
        ax.plot(t, mags, label="|s| (amplitude)", color="tab:orange")
        ax.plot(t, [v.real for v in h["s"]], label="Re(s)", color="tab:blue", linewidth=0.7, alpha=0.7)
        ax.plot(t, [schedule(i) for i in t], label="|input|", color="tab:green", linestyle=":")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — magnitude tracks input; Re(s) shows the rotation")

    eq = (
        "state s = r·exp(i·θ) ∈ ℂ\n\n"
        "polar Euler:\n"
        "  dr/dt = rate · (sensitivity·|sum xs| − r)\n"
        "  dθ/dt = ω + coupling·Σ |xⱼ|·sin(arg xⱼ − θ)\n"
        "  s_new = r_new · exp(i·θ_new)\n\n"
        "params: rate = 0.05, ω = 0.1"
    )
    fig = make_entry_figure(
        title="L1.1d — Adaptive oscillator (amplitude + phase)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="amplitude adapts; phase rotates at ω",
        notes=("• combines adaptation (1a) with rotation (1c)\n"
               "• base unit for the Sompolinsky V1 layer"),
    )
    fig.savefig(f"{OUT_DIR}/L1_1d_adaptive_oscillator.png", dpi=110, bbox_inches="tight")


# -------------------- Level 2 --------------------------------------------

def entry_L2_2a_mutual_excitation():
    """Two nodes with positive mutual coupling → synchronize."""
    OMEGA = 0.05
    K = 0.05
    N = 500

    def schematic(ax):
        setup_schematic_ax(ax, xlim=(-1.0, 1.0), title="schematic — mutual excitation (Kuramoto coupling)")
        a, b = (-0.4, 0.0), (0.4, 0.0)
        draw_node(ax, a, "A")
        draw_node(ax, b, "B")
        draw_excite(ax, a, b, label="K", curvature=-0.25)
        draw_excite(ax, b, a, label="K", curvature=-0.25)

    def behavior(ax):
        f1, g1 = polar(rate=0.0, omega=OMEGA, coupling=K)
        f2, g2 = polar(rate=0.0, omega=OMEGA, coupling=K)
        a = Node(state=np.exp(1j*0.0), dynamics=f1, output=g1)
        b = Node(state=np.exp(1j*np.pi/2), dynamics=f2, output=g2)
        a.add_channel(Channel(b))
        b.add_channel(Channel(a))
        h = run([a, b], n_steps=N, observers={"a": lambda: a.state, "b": lambda: b.state})
        t = list(range(N))
        a_ph = np.unwrap([np.angle(v) for v in h["a"]])
        b_ph = np.unwrap([np.angle(v) for v in h["b"]])
        ax.plot(t, list(b_ph - a_ph), color="tab:purple", label="phase diff arg(b)−arg(a)")
        ax.axhline(0.0, color="black", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("phase diff (rad)")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — phase difference → 0 (in-phase synchronization)")

    eq = (
        "two phase oscillators on unit circle\n\n"
        "for each node i, with channels reading the other:\n"
        "  dθᵢ/dt = ωᵢ + K·sin(θⱼ − θᵢ)\n\n"
        "params: ω = 0.05, K = +0.05 (positive)"
    )
    fig = make_entry_figure(
        title="L2.2a — Mutual excitation (in-phase synchronization)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="phase-locks at zero offset",
        notes="• cf. phase-4 matched-coupling configurations",
    )
    fig.savefig(f"{OUT_DIR}/L2_2a_mutual_excitation.png", dpi=110, bbox_inches="tight")


def entry_L2_2b_mutual_inhibition():
    """Two nodes with mutual inhibition → winner-take-all."""
    GAIN = 6.0
    THR = 3.0
    INHIB = 1.5
    N = 800
    INPUT = 1.0

    def schematic(ax):
        setup_schematic_ax(ax, xlim=(-1.0, 1.0), title="schematic — mutual inhibition, no adaptation")
        a, b = (-0.4, 0.0), (0.4, 0.0)
        draw_node(ax, a, "A")
        draw_node(ax, b, "B")
        draw_external_input(ax, a, label="x", offset=(-0.45, 0.0))
        draw_external_input(ax, b, label="x", offset=(0.45, 0.0))
        draw_inhibit(ax, a, b, label="−w", curvature=-0.25)
        draw_inhibit(ax, b, a, label="−w", curvature=-0.25)

    def behavior(ax):
        env = Constant(INPUT)
        fa, ga = sigmoid_activity(rate=0.1, gain=GAIN, threshold=THR)
        fb, gb = sigmoid_activity(rate=0.1, gain=GAIN, threshold=THR)
        a = Node(state=0.6+0j, dynamics=fa, output=ga)
        b = Node(state=0.4+0j, dynamics=fb, output=gb)
        a.add_channel(Channel(env))
        b.add_channel(Channel(env))
        a.add_channel(Channel(b, transform=lambda y: -INHIB * y))
        b.add_channel(Channel(a, transform=lambda y: -INHIB * y))
        h = run([a, b], n_steps=N, observers={"a": lambda: a.state.real, "b": lambda: b.state.real})
        t = list(range(N))
        ax.plot(t, h["a"], label="A", color="tab:blue")
        ax.plot(t, h["b"], label="B", color="tab:orange")
        ax.legend(loc="center right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("activity")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — winner settles permanently (no switching)")

    eq = (
        "sigmoid_activity dynamics for each:\n"
        "  drive_A = x − w·B\n"
        "  drive_B = x − w·A\n"
        "  rᵢ ← rᵢ + dt·rate·(σ(gain·drive − threshold) − rᵢ)\n\n"
        "params: w = 1.5, gain = 6, threshold = 3"
    )
    fig = make_entry_figure(
        title="L2.2b — Mutual inhibition (winner-take-all settling)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="bistable equilibrium; winner depends on init",
        notes="• stable but no switching; add adaptation → 2c rivalry",
    )
    fig.savefig(f"{OUT_DIR}/L2_2b_mutual_inhibition.png", dpi=110, bbox_inches="tight")


def entry_L2_2c_rivalry():
    """Mutual inhibition + slow adaptation = relaxation oscillator. Phase 10b."""
    N = 4000
    GAIN = 6.0
    THR = 3.0
    INHIB = 1.5
    RATE_FAST = 0.1
    RATE_SLOW = 0.005

    def schematic(ax):
        setup_schematic_ax(ax, xlim=(-1.2, 1.2), title="schematic — mutual inhibition + adaptation memory (rivalry)")
        a, b = (-0.5, 0.0), (0.5, 0.0)
        am, bm = (-0.5, -0.55), (0.5, -0.55)
        draw_node(ax, a, "A")
        draw_node(ax, b, "B")
        draw_node(ax, am, "Mₐ", color="#eeeeee")
        draw_node(ax, bm, "M_b", color="#eeeeee")
        draw_inhibit(ax, a, b, label="−w", curvature=-0.30)
        draw_inhibit(ax, b, a, label="−w", curvature=-0.30)
        draw_excite(ax, a, am, label="track", curvature=0.20)
        draw_excite(ax, b, bm, label="track", curvature=-0.20)
        draw_inhibit(ax, am, a, label="−1", curvature=-0.20)
        draw_inhibit(ax, bm, b, label="−1", curvature=0.20)
        draw_external_input(ax, a, label="x", offset=(-0.45, 0.0))
        draw_external_input(ax, b, label="x", offset=(0.45, 0.0))

    def behavior(ax):
        env = Constant(1.0)
        fa, ga = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THR)
        fb, gb = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THR)
        a = Node(state=0.6+0j, dynamics=fa, output=ga)
        b = Node(state=0.0+0j, dynamics=fb, output=gb)
        fma, gma = tracker(rate=RATE_SLOW)
        fmb, gmb = tracker(rate=RATE_SLOW)
        am_ = Node(state=0+0j, dynamics=fma, output=gma)
        bm_ = Node(state=0+0j, dynamics=fmb, output=gmb)

        a.add_channel(Channel(env))
        a.add_channel(Channel(b, transform=lambda y: -INHIB * y))
        a.add_channel(Channel(am_, transform=lambda y: -1.0 * y))
        b.add_channel(Channel(env))
        b.add_channel(Channel(a, transform=lambda y: -INHIB * y))
        b.add_channel(Channel(bm_, transform=lambda y: -1.0 * y))
        am_.add_channel(Channel(a))
        bm_.add_channel(Channel(b))

        h = run([a, am_, b, bm_], n_steps=N,
                observers={"A": lambda: a.state.real, "B": lambda: b.state.real})
        t = list(range(N))
        ax.plot(t, h["A"], label="A", color="tab:blue")
        ax.plot(t, h["B"], label="B", color="tab:orange")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("activity")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — alternates indefinitely; period set by adaptation rate")

    eq = (
        "rᵢ: activity   mᵢ: adaptation memory (slow tracker)\n\n"
        "drᵢ/dt = rate_fast · (σ(gain·(x − w·rⱼ − mᵢ) − thr) − rᵢ)\n"
        "dmᵢ/dt = rate_slow · (rᵢ − mᵢ)\n\n"
        "params: w = 1.5, rate_fast = 0.1, rate_slow = 0.005\n"
        "        gain = 6, thr = 3"
    )
    fig = make_entry_figure(
        title="L2.2c — Mutual inhibition + slow adaptation (bistable rivalry)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="relaxation oscillator (alternation)",
        notes=("• phase-1 adaptation primitive applied at slow rate to amplitude\n"
               "• same engine drives perceptual rivalry (10c, 10d)\n"
               "• minimal substrate for bistable perception"),
    )
    fig.savefig(f"{OUT_DIR}/L2_2c_rivalry.png", dpi=110, bbox_inches="tight")


def entry_L2_2d_plastic_pair():
    """Plastic Hebbian connection: weight grows with co-activation."""
    N = 1500
    OMEGA = 0.05
    W0 = 0.005
    ETA = 0.005
    DECAY = 0.005

    def schematic(ax):
        setup_schematic_ax(ax, xlim=(-1.0, 1.0), title="schematic — mutual plastic Hebbian channels")
        a, b = (-0.4, 0.0), (0.4, 0.0)
        draw_node(ax, a, "A")
        draw_node(ax, b, "B")
        draw_plastic(ax, a, b, label="w₁ (Hebbian)", curvature=-0.25)
        draw_plastic(ax, b, a, label="w₂ (Hebbian)", curvature=-0.25)

    def behavior(ax):
        fa, ga = polar(rate=0.0, omega=OMEGA, coupling=1.0)
        fb, gb = polar(rate=0.0, omega=OMEGA, coupling=1.0)
        a = Node(state=np.exp(1j*0.0), dynamics=fa, output=ga)
        b = Node(state=np.exp(1j*np.pi/4), dynamics=fb, output=gb)
        learn = hebbian(eta=ETA, decay=DECAY)
        ch_b_to_a = PlasticChannel(b, dest=a, weight=W0, learn=learn)
        ch_a_to_b = PlasticChannel(a, dest=b, weight=W0, learn=learn)
        a.add_channel(ch_b_to_a)
        b.add_channel(ch_a_to_b)

        h = run([a, b], n_steps=N,
                observers={"a": lambda: a.state, "b": lambda: b.state,
                           "w1": lambda: ch_b_to_a.weight, "w2": lambda: ch_a_to_b.weight})
        t = list(range(N))
        ap = np.unwrap([np.angle(v) for v in h["a"]])
        bp = np.unwrap([np.angle(v) for v in h["b"]])
        ax2 = ax.twinx()
        ax.plot(t, list(bp - ap), color="tab:purple", label="phase diff (b−a)")
        ax2.plot(t, [v.real for v in h["w1"]], color="tab:green", linestyle="--", label="weight w₁")
        ax2.plot(t, [v.real for v in h["w2"]], color="tab:olive", linestyle="--", label="weight w₂")
        ax.set_xlabel("time")
        ax.set_ylabel("phase diff (rad)", color="tab:purple")
        ax2.set_ylabel("weight", color="tab:green")
        ax.legend(loc="center right", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title("behavior — weights grow via Hebbian → faster sync")

    eq = (
        "two oscillators on unit circle, mutual plastic channels\n\n"
        "Hebbian update:  dw/dt = η·Re(sv·conj(dv)) − decay·w\n"
        "(equilibrium w → η/decay · time-avg co-activation)\n\n"
        "params: ω = 0.05, η = decay = 0.005, w₀ = 0.005"
    )
    fig = make_entry_figure(
        title="L2.2d — Plastic Hebbian pair (learned coupling)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="self-strengthening synchronization",
        notes="• phase-5 mechanism\n• if a pattern is consistent, weights settle to its correlation",
    )
    fig.savefig(f"{OUT_DIR}/L2_2d_plastic_pair.png", dpi=110, bbox_inches="tight")


# -------------------- Level 3 --------------------------------------------

def entry_L3_3a_chain():
    """A → B → C feedforward propagation."""
    N = 600
    RATE = 0.1

    def schedule(t):
        if t < 100: return 0.0
        if t < 250: return 1.0
        return 0.0

    def schematic(ax):
        setup_schematic_ax(ax, xlim=(-1.2, 1.2), title="schematic — feedforward chain")
        a, b, c = (-0.6, 0.0), (0.0, 0.0), (0.6, 0.0)
        draw_node(ax, a, "A")
        draw_node(ax, b, "B")
        draw_node(ax, c, "C")
        draw_external_input(ax, a, label="x", offset=(-0.45, 0.0))
        draw_excite(ax, a, b)
        draw_excite(ax, b, c)

    def behavior(ax):
        from manifold import Environment
        env = Environment(schedule)
        fa, ga = tracker(rate=RATE)
        fb, gb = tracker(rate=RATE)
        fc, gc = tracker(rate=RATE)
        a = Node(state=0+0j, dynamics=fa, output=ga)
        b = Node(state=0+0j, dynamics=fb, output=gb)
        c = Node(state=0+0j, dynamics=fc, output=gc)
        a.add_channel(Channel(env))
        b.add_channel(Channel(a))
        c.add_channel(Channel(b))
        h = run([env, a, b, c], n_steps=N,
                observers={"x": lambda: env.read().real, "A": lambda: a.state.real,
                           "B": lambda: b.state.real, "C": lambda: c.state.real})
        t = list(range(N))
        ax.plot(t, h["x"], label="input x", color="tab:green", linestyle=":")
        ax.plot(t, h["A"], label="A", color="tab:blue")
        ax.plot(t, h["B"], label="B", color="tab:orange")
        ax.plot(t, h["C"], label="C", color="tab:red")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("activity")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — successive low-pass filters; latency increases per stage")

    eq = (
        "three trackers chained\n\n"
        "drᵢ/dt = rate · (input from upstream − rᵢ)\n"
        "  upstream(A) = x;  upstream(B) = A;  upstream(C) = B\n\n"
        "params: rate = 0.1"
    )
    fig = make_entry_figure(
        title="L3.3a — Feedforward chain (cascaded filtering)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="latency increases per stage; cumulative low-pass",
        notes="• simplest 3-node graph; the 'pipeline'",
    )
    fig.savefig(f"{OUT_DIR}/L3_3a_chain.png", dpi=110, bbox_inches="tight")


def entry_L3_3b_bridge():
    """A↔B↔C, no A↔C edge. Phase 6: A and C sync through B."""
    N = 600
    OMEGA = 0.05
    K = 0.05

    def schematic(ax):
        setup_schematic_ax(ax, xlim=(-1.2, 1.2), title="schematic — bridge: B mediates A and C (no A↔C)")
        a, b, c = (-0.6, 0.0), (0.0, 0.0), (0.6, 0.0)
        draw_node(ax, a, "A")
        draw_node(ax, b, "B")
        draw_node(ax, c, "C")
        draw_excite(ax, a, b, label="K", curvature=-0.20)
        draw_excite(ax, b, a, label="K", curvature=-0.20)
        draw_excite(ax, b, c, label="K", curvature=-0.20)
        draw_excite(ax, c, b, label="K", curvature=-0.20)

    def behavior(ax):
        f, g = polar(rate=0.0, omega=OMEGA, coupling=K)
        a = Node(state=np.exp(1j*0.0), dynamics=f, output=g)
        b = Node(state=np.exp(1j*2*np.pi/3), dynamics=f, output=g)
        c = Node(state=np.exp(1j*4*np.pi/3), dynamics=f, output=g)
        a.add_channel(Channel(b))
        c.add_channel(Channel(b))
        b.add_channel(Channel(a))
        b.add_channel(Channel(c))
        h = run([a, b, c], n_steps=N,
                observers={"a": lambda: a.state, "b": lambda: b.state, "c": lambda: c.state})
        t = list(range(N))
        ap = np.unwrap([np.angle(v) for v in h["a"]])
        bp = np.unwrap([np.angle(v) for v in h["b"]])
        cp = np.unwrap([np.angle(v) for v in h["c"]])
        ax.plot(t, list(ap - bp), label="A−B", color="tab:blue")
        ax.plot(t, list(ap - cp), label="A−C  (indirect!)", color="tab:red")
        ax.plot(t, list(bp - cp), label="B−C", color="tab:orange")
        ax.axhline(0, color="black", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("phase diff (rad)")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — A and C lock through B even with no direct edge")

    eq = (
        "three Kuramoto oscillators, B is the bridge\n\n"
        "dθ_A/dt = ω + K·sin(θ_B − θ_A)\n"
        "dθ_B/dt = ω + K·sin(θ_A − θ_B) + K·sin(θ_C − θ_B)\n"
        "dθ_C/dt = ω + K·sin(θ_B − θ_C)\n\n"
        "params: ω = K = 0.05"
    )
    fig = make_entry_figure(
        title="L3.3b — Bridge (mediated synchronization)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="indirect synchronization through intermediary",
        notes="• phase 6 result\n• Sompolinsky V1 paper's bridge experiment",
    )
    fig.savefig(f"{OUT_DIR}/L3_3b_bridge.png", dpi=110, bbox_inches="tight")


def entry_L3_3c_triangle():
    """Three nodes, all-to-all mutual excitation (Kuramoto)."""
    N = 600
    OMEGA = 0.05
    K = 0.05

    def schematic(ax):
        setup_schematic_ax(ax, xlim=(-1.0, 1.0), ylim=(-0.7, 0.8),
                            title="schematic — triangle (all-to-all coupling)")
        a, b, c = (-0.4, 0.0), (0.4, 0.0), (0.0, 0.55)
        draw_node(ax, a, "A")
        draw_node(ax, b, "B")
        draw_node(ax, c, "C")
        draw_excite(ax, a, b, curvature=-0.20)
        draw_excite(ax, b, a, curvature=-0.20)
        draw_excite(ax, a, c, curvature=0.20)
        draw_excite(ax, c, a, curvature=0.20)
        draw_excite(ax, b, c, curvature=-0.20)
        draw_excite(ax, c, b, curvature=-0.20)

    def behavior(ax):
        f, g = polar(rate=0.0, omega=OMEGA, coupling=K)
        a = Node(state=np.exp(1j*0.0), dynamics=f, output=g)
        b = Node(state=np.exp(1j*2*np.pi/3), dynamics=f, output=g)
        c = Node(state=np.exp(1j*4*np.pi/3), dynamics=f, output=g)
        for src, dsts in [(a, [b, c]), (b, [a, c]), (c, [a, b])]:
            for d in dsts:
                src.add_channel(Channel(d))
        h = run([a, b, c], n_steps=N,
                observers={"a": lambda: a.state, "b": lambda: b.state, "c": lambda: c.state})
        t = list(range(N))
        ap = np.unwrap([np.angle(v) for v in h["a"]])
        bp = np.unwrap([np.angle(v) for v in h["b"]])
        cp = np.unwrap([np.angle(v) for v in h["c"]])
        ax.plot(t, list(ap - bp), label="A−B", color="tab:blue")
        ax.plot(t, list(ap - cp), label="A−C", color="tab:red")
        ax.plot(t, list(bp - cp), label="B−C", color="tab:orange")
        ax.axhline(0, color="black", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("phase diff (rad)")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — all three lock at zero offset (densest sync)")

    eq = (
        "all-to-all Kuramoto:\n"
        "  dθᵢ/dt = ω + K·Σⱼ sin(θⱼ − θᵢ)\n\n"
        "params: ω = K = 0.05"
    )
    fig = make_entry_figure(
        title="L3.3c — Triangle (densest all-to-all coupling)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="phase-locks all three at zero offset",
        notes="• fastest collective sync; baseline for clustering experiments",
    )
    fig.savefig(f"{OUT_DIR}/L3_3c_triangle.png", dpi=110, bbox_inches="tight")


def entry_L3_3d_inhibitory_loop():
    """A → B → C → A with one or more inhibitory edges; with adaptation,
    produces a 3-state cyclical winnerless competition."""
    N = 6000
    GAIN = 6.0
    THR = 3.0
    INHIB = 1.5
    RATE_FAST = 0.1
    RATE_SLOW = 0.005

    def schematic(ax):
        setup_schematic_ax(ax, xlim=(-1.0, 1.0), ylim=(-0.7, 0.8),
                            title="schematic — inhibitory loop with adaptation memories")
        a, b, c = (-0.45, 0.0), (0.45, 0.0), (0.0, 0.55)
        draw_node(ax, a, "A")
        draw_node(ax, b, "B")
        draw_node(ax, c, "C")
        # cyclical: A inhibits B, B inhibits C, C inhibits A
        draw_inhibit(ax, a, b, curvature=-0.20)
        draw_inhibit(ax, b, c, curvature=-0.20)
        draw_inhibit(ax, c, a, curvature=-0.20)
        # each has its own adaptation memory drawn below
        for n, pos in [("Mₐ", (-0.45, -0.5)), ("M_b", (0.45, -0.5)), ("M_c", (0.0, 0.85))]:
            draw_node(ax, pos, n, color="#eeeeee")

    def behavior(ax):
        env = Constant(1.0)
        nodes = []
        mems = []
        for i in range(3):
            f, g = sigmoid_activity(rate=RATE_FAST, gain=GAIN, threshold=THR)
            n = Node(state=0.5*np.random.rand()+0j, dynamics=f, output=g)
            mf, mg = tracker(rate=RATE_SLOW)
            m = Node(state=0+0j, dynamics=mf, output=mg)
            n.add_channel(Channel(env))
            n.add_channel(Channel(m, transform=lambda y: -1.0 * y))
            m.add_channel(Channel(n))
            nodes.append(n); mems.append(m)
        # cyclic inhibition: nodes[i] inhibited by nodes[(i-1)%3]
        for i in range(3):
            nodes[i].add_channel(Channel(nodes[(i-1) % 3], transform=lambda y: -INHIB * y))

        sources = []
        for n, m in zip(nodes, mems):
            sources.extend([n, m])
        h = run(sources, n_steps=N, observers={"A": lambda: nodes[0].state.real,
                                                "B": lambda: nodes[1].state.real,
                                                "C": lambda: nodes[2].state.real})
        t = list(range(N))
        ax.plot(t, h["A"], label="A", color="tab:blue")
        ax.plot(t, h["B"], label="B", color="tab:orange")
        ax.plot(t, h["C"], label="C", color="tab:red")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("activity")
        ax.grid(alpha=0.3)
        ax.set_title("behavior — cyclical winnerless competition (A → B → C → A)")

    eq = (
        "three sigmoid units, cyclic inhibition + per-unit adaptation\n\n"
        "drᵢ/dt = rate_f · (σ(gain·(x − w·rᵢ₋₁ − mᵢ) − thr) − rᵢ)\n"
        "dmᵢ/dt = rate_s · (rᵢ − mᵢ)\n\n"
        "params: w = 1.5, rate_s = 0.005"
    )
    fig = make_entry_figure(
        title="L3.3d — Cyclic inhibition + adaptation (3-state winnerless)",
        schematic_fn=schematic, equations=eq, behavior_fn=behavior,
        classification="cyclical alternation A → B → C → A …",
        notes=("• 3-node generalization of L2.2c rivalry\n"
               "• found in central pattern generators (CPGs)\n"
               "• discrete-state limit cycle"),
    )
    fig.savefig(f"{OUT_DIR}/L3_3d_inhibitory_loop.png", dpi=110, bbox_inches="tight")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    entries = [
        entry_L1_1a_adaptation, entry_L1_1b_tracker,
        entry_L1_1c_oscillator, entry_L1_1d_adaptive_oscillator,
        entry_L2_2a_mutual_excitation, entry_L2_2b_mutual_inhibition,
        entry_L2_2c_rivalry, entry_L2_2d_plastic_pair,
        entry_L3_3a_chain, entry_L3_3b_bridge,
        entry_L3_3c_triangle, entry_L3_3d_inhibitory_loop,
    ]
    for fn in entries:
        print(f"  building {fn.__name__}...")
        fn()
    print(f"All entries written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
