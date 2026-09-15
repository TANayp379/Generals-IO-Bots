# heuristics.py
import jax.numpy as jnp
import numpy as np
from collections import deque
from forward_model import build_cost_grid

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PASS = [1, 0, 0, 0, 0]

# ==========================================
# 1. JAX VECTORIZED VALUE FUNCTION
# ==========================================
def evaluate_state(sim_state, player_idx=0):
    my_mask = sim_state.ownership[player_idx]
    opp_mask = sim_state.ownership[1 - player_idx]
    armies = sim_state.armies
    type_grid = sim_state.type_grid
    
    # 1. Base Game Metrics
    my_army = jnp.sum(armies * my_mask)
    opp_army = jnp.sum(armies * opp_mask)
    my_land = jnp.sum(my_mask)
    opp_land = jnp.sum(opp_mask)
    
    is_castle = (type_grid == 3)
    my_cities = jnp.sum(my_mask & is_castle)
    opp_cities = jnp.sum(opp_mask & is_castle)
    
    # 2. General Locations & Terminal Checks
    is_gen = (type_grid == 4)
    my_gen_mask = my_mask & is_gen
    opp_gen_mask = opp_mask & is_gen
    
    my_alive = jnp.sum(my_mask) > 0
    opp_alive = jnp.sum(opp_mask) > 0
    
    if not bool(my_alive): return -999999.0
    if not bool(opp_alive): return 999999.0

    # 3. Spatial Decay Fields (Threat & Pressure Gravity)
    H, W = sim_state.H, sim_state.W
    X, Y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    
    # Coordinates for our general
    my_gen_r = jnp.sum(Y * my_gen_mask)
    my_gen_c = jnp.sum(X * my_gen_mask)
    dist_to_my_gen = jnp.abs(Y - my_gen_r) + jnp.abs(X - my_gen_c)
    
    threat_field = (armies * opp_mask) / (dist_to_my_gen + 1.0)
    total_threat = jnp.sum(threat_field)
    
    # Coordinates for enemy general
    opp_gen_r = jnp.sum(Y * opp_gen_mask)
    opp_gen_c = jnp.sum(X * opp_gen_mask)
    dist_to_opp_gen = jnp.abs(Y - opp_gen_r) + jnp.abs(X - opp_gen_c)
    
    pressure_field = (armies * my_mask) / (dist_to_opp_gen + 1.0)
    total_pressure = jnp.where(opp_alive, jnp.sum(pressure_field), 0.0)

    # 4. ASYMMETRIC THREAT SCALING
    # Subtract a portion of our total army from threat score to allow strategic counter-attacks
    net_threat = jnp.maximum(0.0, total_threat - (my_army * 0.3))

    # 5. Final Spatio-Temporal Score Synthesis
    score = (
        (my_army - opp_army) * 1.5 +      # Prioritize troop advantage against Hunter
        (my_land - opp_land) * 8.0 +
        (my_cities - opp_cities) * 50.0 -
        (net_threat * 10.0) +             # Scaled threat penalty
        (total_pressure * 20.0)           # Stronger pull toward enemy general
    )
    
    return float(score)


# ==========================================
# 2. NUMPY BFS FLOW FIELD & ROLLOUT POLICY
# ==========================================
# heuristics.py

def get_fast_rollout_move(sim_state, intent="HYBRID", player_idx=0,known_gen=None):
    H, W = sim_state.H, sim_state.W
    armies = np.asarray(sim_state.armies)
    type_grid = np.asarray(sim_state.type_grid)
    
    my_mask = np.asarray(sim_state.ownership[player_idx])
    opp_mask = np.asarray(sim_state.ownership[1 - player_idx])
    
    gen_locs = list(zip(*np.where((type_grid == 4) & opp_mask)))

    # -------------------------------------------------------------
    # 1. CASTLE PLACEMENT (Short-circuit before BFS)
    # -------------------------------------------------------------
    if intent == "BUILD_CASTLE":
        costs = np.asarray(build_cost_grid(sim_state, player_idx))
        is_plain = (type_grid != 2) & (type_grid != 3) & (type_grid != 4)
        valid_builds = my_mask & is_plain & (armies >= costs)
        
        candidates = list(zip(*np.where(valid_builds)))
        if candidates:
            best_candidate = None
            best_score = -9999
            
            my_gen_locs = list(zip(*np.where((type_grid == 4) & my_mask)))
            gen_r, gen_c = my_gen_locs[0] if my_gen_locs else (H//2, W//2)
            
            for r, c in candidates:
                # Approximate choke points by counting adjacent mountains
                adj_mountains = sum(
                    1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= r+dr < H and 0 <= c+dc < W and type_grid[r+dr, c+dc] == 2
                )
                dist_to_gen = abs(r - gen_r) + abs(c - gen_c)
                
                # Formula targets 1-2 width corridors at an optimal distance of 4 from the general
                score = (adj_mountains * 10) - abs(dist_to_gen - 4)
                
                if score > best_score:
                    best_score = score
                    best_candidate = (r, c)
                    
            if best_candidate:
                # Return the specific 5-tuple action to build a castle
                return [2, int(best_candidate[0]), int(best_candidate[1]), 0, 0]
        return None

    # -------------------------------------------------------------
    # 2. TARGET SELECTION (Setting BFS Sinks)
    # -------------------------------------------------------------
    targets = []
    if intent == "DEATHTOUCH_SNIPE" and known_gen:
        targets = [known_gen]
    elif intent == "ATTACK_GENERAL" and known_gen:
        targets = [known_gen]
    if intent == "DEATHTOUCH_SNIPE" and gen_locs:
        targets = gen_locs
    elif intent == "ATTACK_GENERAL" and gen_locs:
        targets = gen_locs
    elif intent == "GATHER_FORCE":
        # Target frontline staging areas using a fast NumPy vectorized dilation
        opp_padded = np.pad(opp_mask, 1)
        borders_opp = (
            opp_padded[:-2, 1:-1] | opp_padded[2:, 1:-1] | 
            opp_padded[1:-1, :-2] | opp_padded[1:-1, 2:]
        )
        staging_areas = my_mask & borders_opp
        targets = list(zip(*np.where(staging_areas)))
        
        if not targets: 
            targets = list(zip(*np.where((~my_mask) & (type_grid != 2))))
    elif intent == "DEFEND_BORDER":
        if np.any(opp_mask):
            targets = list(zip(*np.where(opp_mask)))
            
    # Fallback
    if not targets:
        targets = list(zip(*np.where((~my_mask) & (type_grid != 2))))
        if not targets:
            targets = list(zip(*np.where(opp_mask)))
    if not targets: 
        return None

    # -------------------------------------------------------------
    # 3. BFS FLOW FIELD
    # -------------------------------------------------------------
    dist = np.full((H, W), 9999, dtype=np.int32)
    queue = deque()
    
    for r, c in targets:
        dist[r, c] = 0
        queue.append((r, c))
        
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    d_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    
    while queue:
        r, c = queue.popleft()
        d = dist[r, c]
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and type_grid[nr, nc] != 2:
                if dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    queue.append((nr, nc))

    # -------------------------------------------------------------
    # 4. SELECT STRONGEST VALID MOVE
    # -------------------------------------------------------------
    our_sources = list(zip(*np.where(my_mask & (armies > 1))))
    if not our_sources: 
        return None
        
    best_move = None
    
    if intent == "DEATHTOUCH_SNIPE":
        # Minimize distance first, then maximize army size (as a tie-breaker)
        best_score = (9999, -1) 
        for r, c in our_sources:
            current_dist = dist[r, c]
            if current_dist == 9999: continue
            my_army = armies[r, c]
            
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] < current_dist:
                    score = (current_dist, -my_army)
                    if score < best_score:
                        best_score = score
                        best_move = [0, int(r), int(c), int(d_map[(dr, dc)]), 0]
    else:
        # Standard: Maximize army moved along the shortest path
        max_army_moved = -1
        for r, c in our_sources:
            current_dist = dist[r, c]
            if current_dist == 9999: continue
            my_army = armies[r, c]
            
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] < current_dist:
                    if my_army > max_army_moved:
                        max_army_moved = my_army
                        best_move = [0, int(r), int(c), int(d_map[(dr, dc)]), 0]
                        
    return best_move