import jax
import jax.numpy as jnp
from typing import NamedTuple

# Constants from tournament rules
BUILD = 2
BASE_COST = 35
PROXIMITY_PENALTY = 10
PROXIMITY_DECAY = 2
_RADIUS = (PROXIMITY_PENALTY - 1) // PROXIMITY_DECAY

class SimState(NamedTuple):
    armies: jnp.ndarray      # (H, W) array of troop counts
    ownership: jnp.ndarray   # (2, H, W) one-hot boolean masks for Player 0 and Player 1
    type_grid: jnp.ndarray   # (H, W) mapping terrain (Mountains = 2, Castles = 3, Generals = 4)
    turn: jnp.ndarray        # Scalar turn counter


    @property
    def H(self) -> int:
        return int(self.armies.shape[0])

    @property
    def W(self) -> int:
        return int(self.armies.shape[1])

    @classmethod
    def from_obs(cls, obs):
        """Converts the environment's obs object into a JAX SimState."""
        armies = jnp.array(obs.army_grid, dtype=jnp.int32)
        type_grid = jnp.array(obs.type_grid, dtype=jnp.int32)
        turn = jnp.array(obs.turn, dtype=jnp.int32)

        owner_grid = jnp.array(obs.owner_grid)
        p0_mask = (owner_grid == 1)
        p1_mask = (owner_grid == 2)
        ownership = jnp.stack([p0_mask, p1_mask], axis=0)

        return cls(
            armies=armies,
            ownership=ownership,
            type_grid=type_grid,
            turn=turn
        )
@jax.jit
def tick_turn_clock(state: SimState) -> SimState:
    """Vectorized clock: +1 to cities/generals every 2 turns, +1 to all owned land every 50 turns."""
    new_turn = state.turn + 1
    
    is_even_turn = (new_turn % 2) == 0
    is_50th_turn = (new_turn % 50) == 0
    is_owned = (state.ownership[0] | state.ownership[1])
    is_city_or_gen = (state.type_grid == 3) | (state.type_grid == 4)
    
    city_bonus = jnp.where(is_even_turn & is_owned & is_city_or_gen, 1, 0)
    global_bonus = jnp.where(is_50th_turn & is_owned, 1, 0)
    
    new_armies = state.armies + city_bonus + global_bonus
    return state._replace(armies=new_armies, turn=new_turn)

@jax.jit
def build_cost_grid(state: SimState, player_idx: int) -> jnp.ndarray:
    """Computes the exact dynamic castle price per cell based on proximity to existing structures."""
    H, W = state.armies.shape
    own = state.ownership[player_idx]
    
    # Identify existing structures (Castles=3, Generals=4) owned by the player
    is_structure = ((state.type_grid == 3) | (state.type_grid == 4)) & own
    structures = is_structure.astype(jnp.int32)
    padded = jnp.pad(structures, _RADIUS)

    cost = jnp.full((H, W), BASE_COST, dtype=jnp.int32)
    
    # Apply proximity surcharge kernel[cite: 1]
    for di in range(-_RADIUS, _RADIUS + 1):
        for dj in range(-_RADIUS, _RADIUS + 1):
            surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                shifted = padded[_RADIUS + di:_RADIUS + di + H, _RADIUS + dj:_RADIUS + dj + W]
                cost = cost + surcharge * shifted
    return cost

@jax.jit
def apply_action(state: SimState, player_idx: int, action: jnp.ndarray) -> SimState:
    """
    Resolves a single action: [type, r, c, d, move_half].
    Handles both BUILD (type 2) and standard MOVEMENT (type 0/1).
    """
    H, W = state.armies.shape
    act_type = action[0]
    r, c = jnp.clip(action[1], 0, H - 1), jnp.clip(action[2], 0, W - 1)
    
    # --- 1. BUILD RESOLUTION ---
    is_build = (act_type == BUILD)
    owns_cell = state.ownership[player_idx, r, c]
    is_plain = (state.type_grid[r, c] != 3) & (state.type_grid[r, c] != 4) & (state.type_grid[r, c] != 2)
    
    costs = build_cost_grid(state, player_idx)
    cell_cost = costs[r, c]
    affords = state.armies[r, c] >= cell_cost
    
    valid_build = is_build & owns_cell & is_plain & affords
    
    # --- 2. MOVEMENT RESOLUTION ---
    # Directions: 0=Up, 1=Down, 2=Left, 3=Right
    dr = jnp.array([-1, 1, 0, 0])
    dc = jnp.array([0, 0, -1, 1])
    
    d = jnp.clip(action[3], 0, 3)
    move_half = action[4] == 1
    
    nr = r + dr[d]
    nc = c + dc[d]
    in_bounds = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
    
    # Clip to avoid JAX out-of-bounds indexing errors during tracing
    safe_nr = jnp.clip(nr, 0, H - 1)
    safe_nc = jnp.clip(nc, 0, W - 1)
    
    is_move = (act_type != BUILD)
    not_mountain = state.type_grid[safe_nr, safe_nc] != 2
    
    total_army = state.armies[r, c]
    moving_troops = jnp.where(move_half, total_army // 2, total_army - 1)
    has_troops = moving_troops > 0
    
    valid_move = is_move & in_bounds & not_mountain & owns_cell & has_troops
    
    # --- 3. STATE UPDATES ---
    new_armies = state.armies
    new_types = state.type_grid
    new_ownership = state.ownership
    
    # Apply Build
    new_armies = jnp.where(valid_build, new_armies.at[r, c].add(-cell_cost), new_armies)
    new_types = jnp.where(valid_build, new_types.at[r, c].set(3), new_types)
    
    # Apply Move (Subtract from source)
    new_armies = jnp.where(valid_move, new_armies.at[r, c].add(-moving_troops), new_armies)
    
    # Apply Move (Resolve destination combat)
    dest_owner = state.ownership[player_idx, safe_nr, safe_nc]
    
    # If moving to own tile, add. If moving to enemy/neutral, subtract.
    combat_delta = jnp.where(dest_owner, moving_troops, -moving_troops)
    updated_dest_army = new_armies[safe_nr, safe_nc] + combat_delta
    
    # Check if tile was captured (army drops below 0)
    captured = valid_move & (~dest_owner) & (updated_dest_army < 0)
    final_dest_army = jnp.abs(updated_dest_army)
    
    new_armies = jnp.where(valid_move, new_armies.at[safe_nr, safe_nc].set(final_dest_army), new_armies)
    
    # Update ownership if captured
    new_ownership = jnp.where(captured, new_ownership.at[player_idx, safe_nr, safe_nc].set(True), new_ownership)
    new_ownership = jnp.where(captured, new_ownership.at[1 - player_idx, safe_nr, safe_nc].set(False), new_ownership)

    return state._replace(armies=new_armies, type_grid=new_types, ownership=new_ownership)

# forward_model.py

@jax.jit
def step_sim(state: SimState, action_p0: jnp.ndarray, action_p1: jnp.ndarray) -> SimState:
    """
    Executes a complete 2-player turn step in JAX:
    1. Resolves Player 0 action
    2. Resolves Player 1 action
    3. Ticks the turn clock (army growth for cities, generals, and land)
    """
    state = apply_action(state, player_idx=0, action=action_p0)
    state = apply_action(state, player_idx=1, action=action_p1)
    state = tick_turn_clock(state)
    return state