"""Phase 7: tile primitive — composable subnetworks.

A `Tile` is n_features Kuramoto oscillators wired all-to-all internally
with strong coupling. Tiles compose via feature-matched between-tile
weak connections — the Sompolinsky V1 architecture.

7a: a single tile of 4 features with slightly different intrinsic omegas
    (0.04, 0.05, 0.06, 0.07) and evenly-spaced starting phases. Within-
    tile coupling 0.05 is strong enough to lock all four features into
    a common rhythm at the mean omega.

7b: two tiles connected feature-by-feature with weak (0.005) between-tile
    coupling. Each tile's internal coupling (0.05) dominates locally;
    the weak inter-tile links pull the two tiles' rhythms together.
    Plot phase of feature_0 in each tile to see the inter-tile lock.
"""

import numpy as np

from manifold import Tile, plot, run


def trial_single_tile():
    n_features = 4
    omegas = [0.04, 0.05, 0.06, 0.07]
    tile = Tile(n_features=n_features, coupling=0.05, omegas=omegas)

    obs = {f"feat_{i}": (lambda i=i: tile.feature(i).state) for i in range(n_features)}
    history = run(sources=tile.nodes, n_steps=600, observers=obs)
    times = list(range(600))

    p0 = np.unwrap([np.angle(v) for v in history["feat_0"]])
    diffs = {}
    for i in range(1, n_features):
        pi = np.unwrap([np.angle(v) for v in history[f"feat_{i}"]])
        diffs[f"feat_{i} - feat_0"] = list(pi - p0)
    plot(times, diffs,
         ylabel="phase difference within tile (rad)",
         title="07a_tile_internal_sync")


def trial_two_tiles():
    n_features = 4
    omegas_t1 = [0.04, 0.05, 0.06, 0.07]
    omegas_t2 = [0.06, 0.07, 0.08, 0.09]   # tile 2 runs faster on average

    tile1 = Tile(n_features=n_features, coupling=0.05, omegas=omegas_t1,
                 theta_init=[0.0, 0.5, 1.0, 1.5])
    tile2 = Tile(n_features=n_features, coupling=0.05, omegas=omegas_t2,
                 theta_init=[3.0, 3.5, 4.0, 4.5])
    tile1.connect_to(tile2, weight=0.02)

    obs = {
        "t1_feat0": lambda: tile1.feature(0).state,
        "t1_feat1": lambda: tile1.feature(1).state,
        "t2_feat0": lambda: tile2.feature(0).state,
        "t2_feat1": lambda: tile2.feature(1).state,
    }
    sources = list(tile1.nodes) + list(tile2.nodes)
    N_STEPS = 1200
    history = run(sources=sources, n_steps=N_STEPS, observers=obs)
    times = list(range(N_STEPS))

    # Within-tile sync: features 0 and 1 of tile 1.
    p_t1f0 = np.unwrap([np.angle(v) for v in history["t1_feat0"]])
    p_t1f1 = np.unwrap([np.angle(v) for v in history["t1_feat1"]])
    p_t2f0 = np.unwrap([np.angle(v) for v in history["t2_feat0"]])
    p_t2f1 = np.unwrap([np.angle(v) for v in history["t2_feat1"]])

    plot(
        times,
        {
            "tile1 internal (f1 - f0)": list(p_t1f1 - p_t1f0),
            "tile2 internal (f1 - f0)": list(p_t2f1 - p_t2f0),
            "between-tile  (t2_f0 - t1_f0)": list(p_t2f0 - p_t1f0),
        },
        ylabel="phase difference (rad)",
        title="07b_two_tiles_phase_diffs",
    )


def main():
    trial_single_tile()
    trial_two_tiles()


if __name__ == "__main__":
    main()
