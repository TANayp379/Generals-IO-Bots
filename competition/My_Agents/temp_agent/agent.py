"""
Complete agent.py with BFS Flow Field, Memory, and Lethal Overrides.
"""
from collections import deque
import heapq

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def compute_border_severities(obs, radius):
    """
    Calculates severity using Normalized Army Ratio + Asymmetric Spatial Risk + Deathtouch Tail Risk.
    """
    severities = {}
    H, W = obs.H, obs.W

    # 1. Find our general's coordinates
    gen_r, gen_c = -1, -1
    for r in range(H):
        for c in range(W):
            if obs.type_grid[r][c] == 4 and obs.owner_grid[r][c] == 1:
                gen_r, gen_c = r, c
                break
        if gen_r != -1:
            break

    for r in range(H):
        for c in range(W):
            if obs.owner_grid[r][c] != 1:
                continue

            is_border = any(
                0 <= r + dr < H and 0 <= c + dc < W and obs.owner_grid[r + dr][c + dc] != 1
                for dr, dc in DIRECTIONS
            )
            if not is_border:
                continue

            # Calculate total enemy presence near this border cell
            opp_sum = 0
            for wr in range(max(0, r - radius), min(H, r + radius + 1)):
                for wc in range(max(0, c - radius), min(W, c + radius + 1)):
                    if obs.owner_grid[wr][wc] == 2:
                        opp_sum += obs.army_grid[wr][wc]

            if opp_sum > 0:
                my_army = obs.army_grid[r][c]
                raw_severity = (my_army - opp_sum) / max(1, my_army + opp_sum)
                
                if gen_r != -1:
                    dist = max(1, abs(r - gen_r) + abs(c - gen_c))
                    
                    # --- NEW: Deathtouch Tail Risk (Turn 800+) ---
                    # If it is past turn 800 and an enemy is within 5 tiles of the general, panic.
                    if obs.turn >= 800 and dist <= 5:
                        adjusted_severity = -999999.0
                        
                    # Standard Spatial Weighting
                    else:
                        if raw_severity < 0:
                            # Threats: Divide by distance
                            adjusted_severity = raw_severity / dist
                        else:
                            # Advantages: Multiply by distance
                            adjusted_severity = raw_severity * dist
                else:
                    adjusted_severity = raw_severity
                    
                severities[(r, c)] = adjusted_severity

    return severities

def decide_mode(severities, threshold):
    if not severities:
        return "EXPAND"
    min_severity = min(severities.values())
    if min_severity > 2 * threshold:
        return "ATTACK"
    if min_severity > threshold:
        return "EXPAND"
    return "DEFEND"

def get_expand_targets(obs, severities, k=4):
    """Target fog cells near our strongest borders."""
    if not severities:
        return [(r, c) for r in range(obs.H) for c in range(obs.W) if obs.type_grid[r][c] == 0]
        
    sorted_by_strongest = sorted(severities.items(), key=lambda item: item[1], reverse=True)
    top_k_coords = [coords for coords, severity in sorted_by_strongest[:k]]
    
    targets = []
    for r, c in top_k_coords:
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < obs.H and 0 <= nc < obs.W:
                if obs.type_grid[nr][nc] == 0:
                    targets.append((nr, nc))
                    
    if not targets:
        return [(r, c) for r in range(obs.H) for c in range(obs.W) if obs.type_grid[r][c] == 0]
    return targets

def get_defend_targets(obs, severities, k=3):
    if not severities:
        return []
    sorted_by_weakest = sorted(severities.items(), key=lambda item: item[1])
    return [coords for coords, severity in sorted_by_weakest[:k]]

def pick_any_legal_move(obs):
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] == 1 and obs.army_grid[r][c] > 1:
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < obs.H and 0 <= nc < obs.W:
                        if obs.type_grid[nr][nc] != 2 and obs.type_grid[nr][nc] != 5:
                            return (0, r, c, d, 0)
    return None

class Agent:
    def __init__(self, player_id, H, W, threshold=0, radius=1):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.threshold = threshold
        self.radius = radius
        
        self.known_mountains = set()
        self.memory_owner = {}
        self.memory_army = {}
        
        # NEW: Permanent King Tracker
        self.enemy_general = None

    def _get_lethal_move(self, obs):
        """Instant Win Override: If we can capture the general this turn, execute it."""
        if not self.enemy_general:
            return None
            
        eg_r, eg_c = self.enemy_general
        for d, (dr, dc) in enumerate(DIRECTIONS):
            # Check tiles adjacent to the enemy general for our troops
            r, c = eg_r - dr, eg_c - dc
            if 0 <= r < self.H and 0 <= c < self.W:
                if obs.owner_grid[r][c] == 1 and obs.army_grid[r][c] > 1:
                    army = obs.army_grid[r][c]
                    dest_army = obs.army_grid[eg_r][eg_c]
                    # Win condition: Deathtouch (Turn 800+) OR we have strictly more troops
                    if obs.turn >= 800 or army > dest_army + 1:
                        return (0, r, c, d, 0)
        return None

    def get_attack_targets(self, obs):
        """If we know where the general is, target ONLY the general."""
        if self.enemy_general:
            return [self.enemy_general]
            
        # Fallback if unfound: target all visible enemy cells
        targets = []
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 2:
                    targets.append((r, c))
        return targets

    def _is_passable(self, obs, r, c):
        if (r, c) in self.known_mountains:
            return False
        if obs.type_grid[r][c] == 2:
            return False
        return True

    def compute_flow_field(self, obs, targets):
        dist = [[float('inf') for _ in range(self.W)] for _ in range(self.H)]
        pq = []
        
        for r, c in targets:
            dist[r][c] = 0
            heapq.heappush(pq, (0, r, c))
            
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > dist[r][c]:
                continue
                
            cost = 1
            if obs.type_grid[r][c] != 0:
                if obs.owner_grid[r][c] == 2:
                    cost += obs.army_grid[r][c]
            else:
                mem_owner = self.memory_owner.get((r, c), 0)
                mem_army = self.memory_army.get((r, c), 0)
                if mem_owner == 2:
                    cost += mem_army
                
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    if not self._is_passable(obs, nr, nc):
                        continue
                        
                    if d + cost < dist[nr][nc]:
                        dist[nr][nc] = d + cost
                        heapq.heappush(pq, (dist[nr][nc], nr, nc))
        return dist

    def route_army_via_flow(self, obs, targets):
        if not targets:
            return None
            
        dist = self.compute_flow_field(obs, targets)
        best_move = None
        best_score = -1
        
        for r in range(self.H):
            for c in range(self.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                    
                army = obs.army_grid[r][c]
                if army <= 1:
                    continue
                    
                best_total_resistance = float('inf')
                best_d = -1
                
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.H and 0 <= nc < self.W:
                        if not self._is_passable(obs, nr, nc):
                            continue
                            
                        dest_owner = obs.owner_grid[nr][nc]
                        dest_army = obs.army_grid[nr][nc]
                        if dest_owner != 1 and army <= dest_army + 1:
                            continue
                            
                        entry_cost = 1
                        if dest_owner != 1:
                            entry_cost += dest_army
                            
                        total_resistance = dist[nr][nc] + entry_cost
                        
                        if total_resistance <= dist[r][c] and total_resistance < best_total_resistance:
                            best_total_resistance = total_resistance
                            best_d = d
                
                if best_d != -1:
                    # NEW SCORE METRIC: Penalize distance to prevent traffic jams
                    score = (army * 1000) - dist[r][c]
                        
                    if score > best_score:
                        best_score = score
                        best_move = (0, r, c, best_d, 0)
                        
        return best_move

    def act(self, obs):
        # 1. Update Persistent Memory & Track the King
        for r in range(self.H):
            for c in range(self.W):
                if obs.type_grid[r][c] != 0:
                    if obs.type_grid[r][c] == 2:
                        self.known_mountains.add((r, c))
                    if obs.type_grid[r][c] == 4 and obs.owner_grid[r][c] == 2:
                        self.enemy_general = (r, c)
                        
                    self.memory_owner[(r, c)] = obs.owner_grid[r][c]
                    self.memory_army[(r, c)] = obs.army_grid[r][c]

        # 2. Lethal Override (Instant Win)
        lethal_move = self._get_lethal_move(obs)
        if lethal_move:
            return lethal_move

        # 3. Assess the Board
        severities = compute_border_severities(obs, self.radius)
        mode = decide_mode(severities, self.threshold)

        # 4. Bloodlust Override (Base Race)
        # If we know the general's location, and we have a 50+ stack within 8 tiles,
        # ignore defense and go strictly for the throat.
        if self.enemy_general and mode == "DEFEND":
            eg_r, eg_c = self.enemy_general
            for r in range(self.H):
                for c in range(self.W):
                    if obs.owner_grid[r][c] == 1 and obs.army_grid[r][c] > 50:
                        dist_to_general = abs(r - eg_r) + abs(c - eg_c)
                        if dist_to_general <= 8:
                            mode = "ATTACK"
                            break

        # 5. Execute Routing
        move = None
        if mode == "ATTACK":
            targets = self.get_attack_targets(obs)
            move = self.route_army_via_flow(obs, targets)
        elif mode == "EXPAND":
            targets = get_expand_targets(obs, severities, k=4)
            move = self.route_army_via_flow(obs, targets)
        elif mode == "DEFEND":
            targets = get_defend_targets(obs, severities, k=3)
            move = self.route_army_via_flow(obs, targets)

        if move is None:
            move = pick_any_legal_move(obs)
            
        return move or PASS