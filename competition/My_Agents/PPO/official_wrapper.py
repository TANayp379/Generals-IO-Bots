# official_wrapper.py
import jax
import jax.numpy as jnp
from generals.core.observation import Observation
from generals.core.game import GameState

# 14 base channels per observation frame
NUM_BASE_CHANNELS = 14
STACK_SIZE = 4
TOTAL_IN_CHANNELS = NUM_BASE_CHANNELS * STACK_SIZE  # 56 channels

# Directions matching official JAX engine: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
OFFICIAL_DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


@jax.jit
def build_frame_tensor(obs: Observation) -> jnp.ndarray:
    """Converts an official Observation into a normalized (14, H, W) float32 tensor."""
    H, W = obs.armies.shape
    norm_army = obs.armies.astype(jnp.float32) / 200.0

    land_me = jnp.full((H, W), obs.owned_land_count / 400.0, dtype=jnp.float32)
    army_me = jnp.full((H, W), obs.owned_army_count / 1000.0, dtype=jnp.float32)
    land_opp = jnp.full((H, W), obs.opponent_land_count / 400.0, dtype=jnp.float32)
    army_opp = jnp.full((H, W), obs.opponent_army_count / 1000.0, dtype=jnp.float32)
    step_time = jnp.full((H, W), obs.timestep / 1200.0, dtype=jnp.float32)

    channels = [
        norm_army,
        obs.generals.astype(jnp.float32),
        obs.castles.astype(jnp.float32),
        obs.mountains.astype(jnp.float32),
        obs.neutral_cells.astype(jnp.float32),
        obs.owned_cells.astype(jnp.float32),
        obs.opponent_cells.astype(jnp.float32),
        obs.fog_cells.astype(jnp.float32),
        obs.structures_in_fog.astype(jnp.float32),
        land_me,
        army_me,
        land_opp,
        army_opp,
        step_time,
    ]
    return jnp.stack(channels, axis=0)


@jax.jit
def get_official_action_masks(obs: Observation):
    """Computes legal action masks directly from an official Observation frame."""
    H, W = obs.armies.shape
    
    # Source tile mask: Must own the cell and have > 1 army to move or build
    source_mask = obs.owned_cells & (obs.armies > 1)

    # Impassable tiles: Mountains or unknown structures in fog
    impassable = obs.mountains | obs.structures_in_fog

    # 4-directional move masks
    i_idx = jnp.arange(H)[:, None]
    j_idx = jnp.arange(W)[None, :]

    dest_i = i_idx[:, :, None] + OFFICIAL_DIRECTIONS[None, None, :, 0]
    dest_j = j_idx[:, :, None] + OFFICIAL_DIRECTIONS[None, None, :, 1]

    in_bounds = (dest_i >= 0) & (dest_i < H) & (dest_j >= 0) & (dest_j < W)
    safe_i = jnp.clip(dest_i, 0, H - 1)
    safe_j = jnp.clip(dest_j, 0, W - 1)

    dest_impassable = impassable[safe_i, safe_j]
    move_valid_mask = in_bounds & (~dest_impassable)

    # Castle Build Mask: Can build on owned plain land if army >= 35
    is_plain = obs.owned_cells & (~obs.castles) & (~obs.generals)
    build_valid_mask = is_plain & (obs.armies >= 35)

    target_mask = jnp.concatenate([move_valid_mask, build_valid_mask[..., None]], axis=-1)

    return source_mask, target_mask


@jax.jit
def format_network_action(dir_idx: int, selected_r: int, selected_c: int) -> jnp.ndarray:
    """Translates network's chosen action index (0..4) into official 5-int wire format."""
    is_build = (dir_idx == 4)
    kind = jnp.where(is_build, 2, 0)
    direction = jnp.where(is_build, 0, dir_idx)
    
    return jnp.array([kind, selected_r, selected_c, direction, 0], dtype=jnp.int32)