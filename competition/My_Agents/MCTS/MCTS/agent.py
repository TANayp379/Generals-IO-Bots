# agent.py
from mcts import MCTS
from heuristics import get_fast_rollout_move
from forward_model import SimState, step_sim
import numpy as np
import sys
import jax.numpy as jnp

class Agent:
    def __init__(self, player_id=1, H=20, W=20):
        self.player_id = player_id
        self.H = H
        self.W = W
        
        # MCTS instance with strict 35ms target time limit
        self.mcts = MCTS(time_limit=0.035, max_depth=5)
        self.turn_count = 0
        self.jax_ready = False
        self.known_enemy_gen = None

    def _synchronous_warmup(self, obs):
        """Runs synchronously on Turn 1 during the 10-second grace period."""
        sys.stderr.write("\n[!] Turn 1 Grace Period: Warming up JAX JIT compilation...\n")
        current_state = SimState.from_obs(obs)
        
        # 1. Warm up 2-Player Physics & Step Sim
        dummy_a0 = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
        dummy_a1 = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
        _ = step_sim(current_state, dummy_a0, dummy_a1)
        
        # 2. Warm up Value Function
        from heuristics import evaluate_state
        _ = evaluate_state(current_state, 0)
        
        # 3. Warm up Macro Actions & Terminal checks
        from macro_actions import get_legal_macro_actions
        from mcts import _is_terminal
        _ = get_legal_macro_actions(current_state, 0)
        _ = get_legal_macro_actions(current_state, 1)
        _ = _is_terminal(current_state)
        
        # 4. Warm up Full MCTS Tree (triggers trace for expansion & simulation loops)
        dummy_mcts = MCTS(time_limit=0.01, max_depth=2)
        _ = dummy_mcts.search(obs)
        
        self.jax_ready = True
        sys.stderr.write("[!] JAX Compilation complete! MCTS fully activated.\n")

    def act(self, obs):
        self.turn_count += 1
        current_state = SimState.from_obs(obs)
        
        # BLOCK Turn 1 to compile everything during the 10s grace period
        if self.turn_count == 1:
            self._synchronous_warmup(obs)

        # --- Instant Snipe Check (Pure NumPy, 0 latency) ---
        type_grid = np.asarray(current_state.type_grid)
        opp_mask = np.asarray(current_state.ownership[1])
        my_mask = np.asarray(current_state.ownership[0])
        armies = np.asarray(current_state.armies)
        
        gen_locs = list(zip(*np.where((type_grid == 4) & opp_mask)))
        if gen_locs:
            self.known_enemy_gen = gen_locs[0]

        # --- Instant Snipe Check ---
        # (Update this to use self.known_enemy_gen if you want it to snipe into fog)
        if self.known_enemy_gen:
            gen_r, gen_c = self.known_enemy_gen
            # Ensure we only snipe if we actually know the army size (or guess it's low)
            if opp_mask[gen_r, gen_c]: 
                gen_army = armies[gen_r, gen_c]
                lethal_tiles = list(zip(*np.where(my_mask & (armies > gen_army + 1))))
                
                for r, c in lethal_tiles:
                    for d_idx, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                        nr, nc = r + dr, c + dc
                        if nr == gen_r and nc == gen_c:
                            sys.stderr.write(f"\n[!] INSTANT SNIPE!\n")
                            return [0, int(r), int(c), int(d_idx), 0]
                            
        # Pass the known general location into MCTS and routing
        best_macro_action = self.mcts.search(obs, self.known_enemy_gen)
        final_move = get_fast_rollout_move(current_state, best_macro_action, player_idx=0, known_gen=self.known_enemy_gen)
        
        if final_move is None:
            return [1, 0, 0, 0, 0]
            
        return final_move