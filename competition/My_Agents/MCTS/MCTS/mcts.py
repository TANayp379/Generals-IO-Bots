# mcts.py
import math
import time
import random
import jax.numpy as jnp

from forward_model import step_sim, SimState
from macro_actions import get_legal_macro_actions
from heuristics import get_fast_rollout_move, evaluate_state

PASS_ACTION = [1, 0, 0, 0, 0]

def _is_terminal(sim_state):
    """A state is terminal if either player has been completely eliminated (0 land)."""
    my_land = jnp.sum(sim_state.ownership[0])
    opp_land = jnp.sum(sim_state.ownership[1])
    
    return (my_land == 0) | (opp_land == 0)

# mcts.py (Helper function update)

def get_adaptive_intent(sim_state, player_idx, known_gen=None):
    """Priority-based intent selector for heavy playouts."""
    legal_intents = get_legal_macro_actions(sim_state, player_idx)
    
    # MEMORY OVERRIDE: If P0 remembers the general, relentlessly hunt it
    if player_idx == 0 and known_gen is not None:
        if int(sim_state.turn) >= 800:
            return "DEATHTOUCH_SNIPE"
        return "ATTACK_GENERAL"
    
    if "DEATHTOUCH_SNIPE" in legal_intents:
        return "DEATHTOUCH_SNIPE"
    if "ATTACK_GENERAL" in legal_intents:
        return "ATTACK_GENERAL"
    if "DEFEND_BORDER" in legal_intents:
        return "DEFEND_BORDER"
        
    return "EXPAND_STRONG"


class MCTSNode:
    def __init__(self, state, parent=None, macro_action=None):
        self.state = state
        self.parent = parent
        self.macro_action = macro_action
        
        self.children = []
        self.untried_actions = get_legal_macro_actions(state, player_idx=0)
        
        self.visits = 0
        self.value = 0.0
        self.is_terminal = _is_terminal(state)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def get_best_child_ucb1(self, exploration_constant=1.414):
        best_score = -float('inf')
        best_children = []
        
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration
                
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
                
        return random.choice(best_children)


class MCTS:
    def __init__(self, time_limit=0.035, max_depth=5):
        self.time_limit = time_limit 
        self.max_depth = max_depth

    # mcts.py (Inside the MCTS class)

    def search(self, initial_obs, known_gen=None):
        root_state = SimState.from_obs(initial_obs)
        root = MCTSNode(root_state)
        
        start_time = time.time()
        
        # Strict time margins to prevent 150ms timeout faults
        target_time_limit = 0.035   # 35 ms target search window
        hard_deadline = 0.050       # 50 ms hard cutoff

        while (time.time() - start_time) < target_time_limit:
            node = root
            if (time.time() - start_time) > hard_deadline:
                break
                
            # --- 1. SELECTION ---
            while node.is_fully_expanded() and not node.is_terminal:
                node = node.get_best_child_ucb1()
                
            # --- 2. EXPANSION ---
            if not node.is_fully_expanded() and not node.is_terminal:
                action_intent = node.untried_actions.pop()
                
                # Player 0 evaluates the popped macro action (using memory)
                raw_move_p0 = get_fast_rollout_move(node.state, action_intent, player_idx=0, known_gen=known_gen)
                if raw_move_p0 is None: raw_move_p0 = PASS_ACTION
                    
                # Player 1 evaluates adaptively based on visible threats
                opp_intent = get_adaptive_intent(node.state, player_idx=1)
                raw_move_p1 = get_fast_rollout_move(node.state, opp_intent, player_idx=1)
                if raw_move_p1 is None: raw_move_p1 = PASS_ACTION
                
                jnp_move_p0 = jnp.array(raw_move_p0, dtype=jnp.int32)
                jnp_move_p1 = jnp.array(raw_move_p1, dtype=jnp.int32)
                
                new_state = step_sim(node.state, jnp_move_p0, jnp_move_p1)
                
                child_node = MCTSNode(new_state, parent=node, macro_action=action_intent)
                node.children.append(child_node)
                node = child_node

            # --- 3. SIMULATION (HEAVY PLAYOUTS) ---
            sim_state = node.state
            depth = 0
            
            while not _is_terminal(sim_state) and depth < self.max_depth:
                # Emergency escape inside rollout loop
                if (time.time() - start_time) > target_time_limit:
                    break
                
                # Player 0 relies on memory (if available) or adaptive logic
                intent_p0 = get_adaptive_intent(sim_state, player_idx=0, known_gen=known_gen)
                raw_move_p0 = get_fast_rollout_move(sim_state, intent_p0, player_idx=0, known_gen=known_gen)
                if raw_move_p0 is None: raw_move_p0 = PASS_ACTION

                # Player 1 relies strictly on adaptive logic based on board state
                intent_p1 = get_adaptive_intent(sim_state, player_idx=1)
                raw_move_p1 = get_fast_rollout_move(sim_state, intent_p1, player_idx=1)
                if raw_move_p1 is None: raw_move_p1 = PASS_ACTION

                jnp_move_p0 = jnp.array(raw_move_p0, dtype=jnp.int32)
                jnp_move_p1 = jnp.array(raw_move_p1, dtype=jnp.int32)
                
                sim_state = step_sim(sim_state, jnp_move_p0, jnp_move_p1)
                depth += 1
                
            # --- 4. BACKPROPAGATION ---
            reward = evaluate_state(sim_state, player_idx=0)
            
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        if not root.children:
            return "EXPAND_STRONG"
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.macro_action

 