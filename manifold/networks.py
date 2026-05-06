"""Higher-level network constructors.

A `Tile` is a small subnetwork — n feature-indexed Kuramoto oscillators
wired all-to-all internally with a single coupling strength. Tiles
expose `feature(i)` for accessing individual nodes and `connect_to(other,
weight)` for wiring feature-matched between-tile connections, the
Sompolinsky V1 architecture.

Lateral-connection helpers (`add_lateral_inhibition`, `add_plastic_lateral`,
`add_mexican_hat`) systematize the connectivity patterns we kept wiring
ad-hoc in experiments. They mutate the nodes in place by adding channels.
"""

from typing import Callable, List, Optional, Sequence

import numpy as np

from .core import Channel, Node, PlasticChannel
from .dynamics import polar


class Tile:
    """A small subnetwork: n_features nodes, all-to-all internal coupling.

    Each feature i is a Kuramoto oscillator (polar with rate=0) at intrinsic
    frequency `omegas[i]` and starting phase `theta_init[i]`. Defaults
    spread phases evenly around the unit circle and use a uniform omega.
    """

    def __init__(
        self,
        n_features: int,
        coupling: float = 0.05,
        omegas: Optional[Sequence[float]] = None,
        theta_init: Optional[Sequence[float]] = None,
    ):
        self.n_features = n_features
        self.coupling = coupling

        if omegas is None:
            omegas = [0.05] * n_features
        if theta_init is None:
            theta_init = [2 * np.pi * i / n_features for i in range(n_features)]

        self.nodes: List[Node] = []
        for i in range(n_features):
            f, g = polar(rate=0.0, omega=omegas[i], coupling=1.0)
            self.nodes.append(
                Node(state=np.exp(1j * theta_init[i]), dynamics=f, output=g)
            )

        # All-to-all internal coupling. The transform carries the weight
        # so multiple connections (within-tile and inter-tile) can stack
        # cleanly on the same node.
        for i, ni in enumerate(self.nodes):
            for j, nj in enumerate(self.nodes):
                if i == j:
                    continue
                ni.add_channel(Channel(nj, transform=lambda y, k=coupling: k * y))

    def feature(self, i: int) -> Node:
        return self.nodes[i]

    def connect_to(self, other: "Tile", weight: float = 0.01) -> None:
        """Wire feature i here <-> feature i in `other`, both directions.

        Both tiles must have the same n_features. The coupling matches by
        feature *index* — in the V1 setting, feature i is the orientation
        (or color, etc.) tuning, so this implements 'same selectivity in
        different receptive fields' coupling.
        """
        assert other.n_features == self.n_features, "tile arity mismatch"
        for i in range(self.n_features):
            self.nodes[i].add_channel(
                Channel(other.nodes[i], transform=lambda y, w=weight: w * y)
            )
            other.nodes[i].add_channel(
                Channel(self.nodes[i], transform=lambda y, w=weight: w * y)
            )


# ---- Lateral connectivity helpers ----------------------------------------

def add_lateral_inhibition(nodes: Sequence[Node], strength: float = 1.0) -> None:
    """Wire mutual inhibition between every pair of nodes (no self-loops).

    The basic competitive mechanism: each node inhibits every other in the
    group, so they fight for activity. Used as winner-take-all within a
    tile, between competing populations, etc.
    """
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j:
                continue
            ni.add_channel(Channel(nj, transform=lambda y, s=strength: -s * y))


def add_lateral_excitation(nodes: Sequence[Node], strength: float = 1.0) -> None:
    """Wire mutual excitation between every pair (no self-loops).

    Used to bind a coherent population that should activate together.
    """
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j:
                continue
            ni.add_channel(Channel(nj, transform=lambda y, s=strength: s * y))


def add_plastic_lateral(
    nodes: Sequence[Node],
    learn: Callable,
    init_weight: float = 0.05,
    init_random: bool = True,
    seed: Optional[int] = None,
) -> List[PlasticChannel]:
    """All-to-all plastic intra-layer connections (returns the channels).

    Each node receives a `PlasticChannel` from every other node, with
    weights either tiny-random (default) or constant. The learning rule
    is whatever you pass — typically `hebbian` for co-activation
    learning, `stdp_sin` for phase-based, etc.

    Returned channel list is suitable for passing to a `WeightNormalizer`.

    This is the substrate for *topology learning* — letting a network
    discover its own intra-layer connectivity from co-activation
    statistics, rather than hand-wiring it. See
    `project_topology_learning.md` memory for the underlying insight.
    """
    rng = np.random.default_rng(seed)
    channels: List[PlasticChannel] = []
    for ni in nodes:
        for nj in nodes:
            if ni is nj:
                continue
            w = init_weight * (rng.random() if init_random else 1.0)
            ch = PlasticChannel(nj, dest=ni, weight=w, learn=learn)
            ni.add_channel(ch)
            channels.append(ch)
    return channels


def add_mexican_hat(
    nodes: Sequence[Node],
    positions: Sequence,
    exc: float = 1.0,
    inh: float = 0.5,
    exc_radius: float = 1.0,
    inh_radius: float = 2.0,
) -> None:
    """Distance-based lateral connectivity (Mexican-hat / center-surround).

    Each pair of nodes is wired excitatorily if their positions are
    within `exc_radius`, inhibitorily if within `inh_radius` (and beyond
    `exc_radius`), and unconnected if farther. This is the standard
    cortical local-circuit motif: short-range excitation, longer-range
    inhibition.

    `positions[i]` is the spatial position of `nodes[i]` (any tuple/array
    that works with numpy's norm).
    """
    pos = [np.asarray(p, dtype=float) for p in positions]
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < exc_radius:
                ni.add_channel(Channel(nj, transform=lambda y, s=exc: s * y))
            elif d < inh_radius:
                ni.add_channel(Channel(nj, transform=lambda y, s=inh: -s * y))
