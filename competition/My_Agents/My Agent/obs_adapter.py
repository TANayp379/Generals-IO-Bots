"""
Bridges main.py's plain-Python Observation (lists of ints, from the wire
protocol) to the same representations used during training:

  - wire_obs_to_tensor(): (H, W, 14) float32 tensor, same channel semantics
    as generals.core.observation.Observation.as_tensor() (channel axis moved
    to the end since model.py is channels-last).
  - compute_valid_move_mask(): (H, W, 4) bool, same semantics as
    generals.core.action.compute_valid_move_mask_obs().

Deliberately has zero dependency on the `generals` package — agent.py runs
inside the competition harness, which only guarantees main.py's wire
protocol, not that the training repo is importable there.
"""
from __future__ import annotations

import numpy as np

# type_grid values
FOG, PLAIN, MOUNTAIN, CASTLE, GENERAL, STRUCTURE_IN_FOG = 0, 1, 2, 3, 4, 5
# owner_grid values
NEUTRAL_OR_UNKNOWN, ME, OPPONENT = 0, 1, 2

# Must match action.py's DIRECTIONS order exactly: UP, DOWN, LEFT, RIGHT.
DIRECTION_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def wire_obs_to_tensor(obs) -> np.ndarray:
    """obs: main.py's Observation. Returns float32 array of shape (H, W, 14)."""
    H, W = obs.H, obs.W
    type_g = np.asarray(obs.type_grid)
    owner_g = np.asarray(obs.owner_grid)
    armies = np.asarray(obs.army_grid, dtype=np.float32)

    generals = type_g == GENERAL
    castles = type_g == CASTLE
    mountains = type_g == MOUNTAIN
    fog_cells = type_g == FOG
    structures_in_fog = type_g == STRUCTURE_IN_FOG
    owned_cells = owner_g == ME
    opponent_cells = owner_g == OPPONENT
    neutral_cells = (owner_g == NEUTRAL_OR_UNKNOWN) & ~fog_cells & ~structures_in_fog

    def scalar_plane(v):
        return np.full((H, W), v, dtype=np.float32)

    channels = [
        armies,
        generals.astype(np.float32),
        castles.astype(np.float32),
        mountains.astype(np.float32),
        neutral_cells.astype(np.float32),
        owned_cells.astype(np.float32),
        opponent_cells.astype(np.float32),
        fog_cells.astype(np.float32),
        structures_in_fog.astype(np.float32),
        scalar_plane(obs.my_land),
        scalar_plane(obs.my_army),
        scalar_plane(obs.opp_land),
        scalar_plane(obs.opp_army),
        scalar_plane(obs.turn),
    ]
    return np.stack(channels, axis=-1)  # (H, W, 14)


def compute_valid_move_mask(obs) -> np.ndarray:
    """
    obs: main.py's Observation. Returns bool array (H, W, 4): mask[r, c, d]
    is True iff moving from (r, c) in direction d is legal.

    Mirrors action.compute_valid_move_mask's semantics (owned, army > 1,
    destination in bounds and passable), but conservatively also treats an
    unrevealed structure (type 5) as impassable, matching agent.py's given
    `_is_passable` — since we don't know whether it's a mountain or a
    castle, we don't let the policy blindly walk into it either.
    """
    H, W = obs.H, obs.W
    type_g = np.asarray(obs.type_grid)
    owner_g = np.asarray(obs.owner_grid)
    army_g = np.asarray(obs.army_grid)

    can_move_from = (owner_g == ME) & (army_g > 1)
    impassable = (type_g == MOUNTAIN) | (type_g == STRUCTURE_IN_FOG)

    mask = np.zeros((H, W, 4), dtype=bool)
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    for d, (dr, dc) in enumerate(DIRECTION_OFFSETS):
        di = rows + dr
        dj = cols + dc
        in_bounds = (di >= 0) & (di < H) & (dj >= 0) & (dj < W)
        di_c = np.clip(di, 0, H - 1)
        dj_c = np.clip(dj, 0, W - 1)
        dest_passable = ~impassable[di_c, dj_c]
        mask[:, :, d] = can_move_from & in_bounds & dest_passable

    return mask