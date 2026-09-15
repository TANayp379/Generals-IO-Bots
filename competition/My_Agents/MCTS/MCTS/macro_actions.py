# macro_actions.py
import jax.numpy as jnp
from forward_model import build_cost_grid 

def _can_see_enemy_general(sim_state, player_idx):
    # Enemy is always 1 - player_idx
    is_enemy_gen = (sim_state.type_grid == 4) & sim_state.ownership[1 - player_idx]
    return bool(jnp.any(is_enemy_gen))

def _has_threats(sim_state, player_idx):
    enemy_mask = sim_state.ownership[1 - player_idx]
    our_mask = sim_state.ownership[player_idx]
    
    # Check if enemy borders our territory
    up = jnp.pad(enemy_mask[1:, :], ((0, 1), (0, 0)))
    down = jnp.pad(enemy_mask[:-1, :], ((1, 0), (0, 0)))
    left = jnp.pad(enemy_mask[:, 1:], ((0, 0), (0, 1)))
    right = jnp.pad(enemy_mask[:, :-1], ((0, 0), (1, 0)))
    
    threat_zone = up | down | left | right
    return bool(jnp.any(our_mask & threat_zone))

def _can_afford_building_castle(sim_state, player_idx):
    our_mask = sim_state.ownership[player_idx]
    is_plain = (sim_state.type_grid != 2) & (sim_state.type_grid != 3) & (sim_state.type_grid != 4)
    costs = build_cost_grid(sim_state, player_idx=player_idx)
    affords = sim_state.armies >= costs
    return bool(jnp.any(our_mask & is_plain & affords))

def get_legal_macro_actions(sim_state, player_idx=0):
    actions = ["EXPAND_STRONG"]

    # Incorporate the Turn 800+ rule here
    if int(sim_state.turn) >= 800 and _can_see_enemy_general(sim_state, player_idx):
        actions.append("DEATHTOUCH_SNIPE")

    if _can_see_enemy_general(sim_state, player_idx):
        actions.append("ATTACK_GENERAL")

    if _has_threats(sim_state, player_idx):
        actions.append("DEFEND_BORDER")
        actions.append("GATHER_FORCE")

    if _can_afford_building_castle(sim_state, player_idx):
        actions.append("BUILD_CASTLE")

    return actions