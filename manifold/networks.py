"""Higher-level network constructors.

A `Tile` is a small subnetwork — n feature-indexed Kuramoto oscillators
wired all-to-all internally with a single coupling strength. Tiles
expose `feature(i)` for accessing individual nodes and `connect_to(other,
weight)` for wiring feature-matched between-tile connections, the
Sompolinsky V1 architecture.

Internal connections are realized as `Channel`s whose transform scales
the source value by `coupling`. The polar dynamics' magnitude-weighted
coupling sum then produces the proper weighted-Kuramoto phase update,
so per-channel strengths can vary even when the dynamics' own coupling
parameter is held at 1.
"""

from typing import List, Optional, Sequence

import numpy as np

from .core import Channel, Node
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
