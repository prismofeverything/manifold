"""Phase 9: Sompolinsky V1-style synchronization, the paper's bridge experiment.

Three receptive-field tiles, N=4 features each (preferred orientations
0, pi/2, pi, 3pi/2 evenly around the circle). Each tile receives a single
stimulus orientation. Each neuron's responsiveness to that stimulus is
given by the tuning curve

    V(theta_stim, theta_pref) = exp(-|delta_theta|_circle / sigma)

The Sompolinsky coupling term is J(r, r') = V(r) * W * V(r'), so only
co-activated neurons drive each other:

  - Within tile: all-to-all, weight V_self * V_other * W_S / N
  - Between tiles: matched-feature only (feat i <-> feat i),
                   weight V_self * V_other * W_L / N^2

Each node has a small omega jitter so that *inactive* feature pairs
between tiles can't lock (their effective coupling is below the locking
threshold), while *active* pairs do lock easily. This is what
distinguishes 'co-activated -> sync' from 'just connected -> drift'.

Three stimulus configurations from the paper:
  S1 = [pi, 0, pi]    — bridge: tile 0 and tile 2 both have feat_2 active;
                        tile 1's active feature is feat_0. The matched-
                        feature link feat_2(t0) <-> feat_2(t2) should
                        synchronize them; feat_0 across the same tile
                        pair should not (both inactive there).
  S2 = [0, 0, pi]     — tiles 0 & 1 share active feature, tile 2 different.
  S3 = [pi, pi, pi]   — same stimulus everywhere; feat_2 should sync
                        across all three tiles.
"""

import numpy as np

from manifold import Channel, Node, plot, polar, run


N_FEATURES = 4
PREFS = np.array([2 * np.pi * i / N_FEATURES for i in range(N_FEATURES)])
SIGMA = 0.6
W_S = 2.0
W_L = 0.5
OMEGA = 0.05
OMEGA_JITTER = 0.01
N_STEPS = 1500
N_TILES = 3
SEED = 42


def tuning(stim: float, pref: float) -> float:
    """Circular tuning curve V(stim - pref) = exp(-|delta|_circle / sigma)."""
    diff = abs(stim - pref) % (2 * np.pi)
    diff = min(diff, 2 * np.pi - diff)
    return float(np.exp(-diff / SIGMA))


def build_v1(stim_per_tile, seed=SEED):
    rng = np.random.default_rng(seed)

    tiles = []
    for t in range(N_TILES):
        tile = []
        for f in range(N_FEATURES):
            omega = OMEGA + OMEGA_JITTER * rng.normal()
            f_dyn, g = polar(rate=0.0, omega=omega, coupling=1.0)
            theta0 = 2 * np.pi * rng.random()
            node = Node(state=np.exp(1j * theta0), dynamics=f_dyn, output=g)
            tile.append(node)
        tiles.append(tile)

    # Within-tile all-to-all, weighted V_self * V_other * W_S / N
    for t in range(N_TILES):
        stim_t = stim_per_tile[t]
        for r in range(N_FEATURES):
            for j in range(N_FEATURES):
                if r == j:
                    continue
                v_self = tuning(stim_t, PREFS[r])
                v_other = tuning(stim_t, PREFS[j])
                weight = v_self * v_other * W_S / N_FEATURES
                tiles[t][r].add_channel(
                    Channel(tiles[t][j], transform=lambda y, w=weight: w * y)
                )

    # Between-tile matched-feature only, weight V_self * V_other * W_L / N^2
    for t in range(N_TILES):
        stim_t = stim_per_tile[t]
        for u in range(N_TILES):
            if t == u:
                continue
            stim_u = stim_per_tile[u]
            for f in range(N_FEATURES):
                v_self = tuning(stim_t, PREFS[f])
                v_other = tuning(stim_u, PREFS[f])
                weight = v_self * v_other * W_L / (N_FEATURES ** 2)
                tiles[t][f].add_channel(
                    Channel(tiles[u][f], transform=lambda y, w=weight: w * y)
                )

    return tiles


def run_trial(stim_per_tile, label: str) -> dict:
    tiles = build_v1(stim_per_tile)
    all_nodes = [n for tile in tiles for n in tile]

    obs = {
        f"t{t}_f{f}": (lambda t=t, f=f: tiles[t][f].state)
        for t in range(N_TILES)
        for f in range(N_FEATURES)
    }
    history = run(sources=all_nodes, n_steps=N_STEPS, observers=obs)
    times = list(range(N_STEPS))

    def get_phases(t: int, f: int):
        return np.unwrap([np.angle(v) for v in history[f"t{t}_f{f}"]])

    # Phase difference between tile 0 and tile 2 for each feature.
    # Active matched pairs (high V_0 * V_2) should lock; inactive ones drift.
    series = {}
    for f in range(N_FEATURES):
        v0 = tuning(stim_per_tile[0], PREFS[f])
        v2 = tuning(stim_per_tile[2], PREFS[f])
        diff = get_phases(0, f) - get_phases(2, f)
        series[f"f{f} (V_0={v0:.2f}, V_2={v2:.2f})"] = list(diff)

    plot(
        times, series,
        ylabel="phase diff t0 - t2 (rad)",
        title=f"09_{label}_t0_vs_t2_per_feature",
    )

    # Also plot all phases of the dominant feature (highest V product across the
    # three tiles) to see whether the active feature synchronizes globally.
    v_products = []
    for f in range(N_FEATURES):
        prod = (
            tuning(stim_per_tile[0], PREFS[f])
            * tuning(stim_per_tile[1], PREFS[f])
            * tuning(stim_per_tile[2], PREFS[f])
        )
        v_products.append(prod)
    f_dom = int(np.argmax(v_products))
    series2 = {}
    for t in range(N_TILES):
        v_t = tuning(stim_per_tile[t], PREFS[f_dom])
        series2[f"tile{t} feat{f_dom} (V={v_t:.2f})"] = list(get_phases(t, f_dom))
    plot(
        times, series2,
        ylabel=f"phase (rad, unwrapped); dominant feat={f_dom}",
        title=f"09_{label}_dominant_feature_phases",
    )

    return history


def main():
    run_trial([np.pi, 0.0, np.pi],     "S1_bridge")
    run_trial([0.0, 0.0, np.pi],       "S2_partial")
    run_trial([np.pi, np.pi, np.pi],   "S3_full")


if __name__ == "__main__":
    main()
