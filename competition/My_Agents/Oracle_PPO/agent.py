# belief_agent.py
"""
Phase 3 -- integrated inference agent: foggy live observation -> BeliefNet
reconstruction -> frozen Oracle -> action.

No training happens in this file. It only makes sense once you have:
  - an Oracle checkpoint from oracle_train.py (Phase 1, perfect_info=True)
  - a BeliefNet checkpoint from belief_train.py (Phase 2, supervised
    reconstruction against fog)
Both are loaded frozen (eval mode -- no gradients, no dropout to worry
about since neither network uses any) and never updated here.

Mirrors agent.py's structure (same protocol-obs -> 14-channel frame
conversion, same live action-mask computation) with one addition: before
the stacked frame reaches the policy network, the most recent frame's 4
fog-hidden channels (armies, generals, castles, opponent_cells -- see
belief_model.py) are overwritten with BeliefNet's reconstruction. Older
frames already in the buffer were themselves spliced at their own turn,
so history stays internally consistent as the buffer rolls forward --
only ever the newest slot needs splicing per step.

The live action-legality masks are deliberately computed from the RAW
foggy obs, not the belief-reconstructed one: legality (which cells you
can actually move from) is ground truth given by the engine/protocol
regardless of what the belief model predicts, and running masks off a
reconstruction could mask in genuinely illegal moves or mask out legal
ones. Only the network's *input* gets the belief treatment, never the
mask computation.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from pathlib import Path
import sys

from network import ActorCriticNet
from belief_model import BeliefNet, splice_belief_reconstruction
from official_wrapper import (
    TOTAL_IN_CHANNELS,
    STACK_SIZE,
    NUM_BASE_CHANNELS,
    format_network_action,
)

OFFICIAL_DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)
BELIEF_STACK_SIZE = 8
ORACLE_STACK_SIZE = 4
NUM_BASE_CHANNELS = 14
BELIEF_IN_CHANNELS = NUM_BASE_CHANNELS * BELIEF_STACK_SIZE
AGENT_DIR = Path(__file__).resolve().parent
class Agent:
    """Phase-3 evaluation agent: BeliefNet-augmented Oracle policy."""

    def __init__(
        self,
        player_id: int,
        H: int,
        W: int,
        oracle_path: str = None,
        belief_path: str = None,
    ):
        self.player_id = player_id
        self.H = H
        self.W = W

        # Fallback to relative paths if no explicit path is passed
        if oracle_path is None:
            oracle_path = AGENT_DIR / "oracle_model_iter_4350.eqx"
        if belief_path is None:
            belief_path = AGENT_DIR / "belief_model_iter_1000.eqx"

        key = jax.random.PRNGKey(0)
        k_oracle, k_belief = jax.random.split(key)

        # 1. Load Oracle
        self.oracle = ActorCriticNet(key=k_oracle)
        oracle_file = Path(oracle_path)
        if oracle_file.exists():
            self.oracle = eqx.tree_deserialise_leaves(str(oracle_file), self.oracle)
            # CRITICAL: Print to sys.stderr so it doesn't break stdout protocol
            sys.stderr.write(f"[BeliefAgent] Loaded Oracle weights from {oracle_file}\n")
        else:
            sys.stderr.write(f"[Warning] Oracle checkpoint {oracle_path} not found!\n")

        # 2. Load BeliefNet
        self.belief_net = BeliefNet(in_channels=BELIEF_IN_CHANNELS, key=k_belief)
        belief_file = Path(belief_path)
        if belief_file.exists():
            self.belief_net = eqx.tree_deserialise_leaves(str(belief_file), self.belief_net)
            sys.stderr.write(f"[BeliefAgent] Loaded BeliefNet weights from {belief_file}\n")
        else:
            sys.stderr.write(f"[Warning] BeliefNet checkpoint {belief_path} not found!\n")

        # 3. Initialize Frame Buffer (8 frames for 112-channel BeliefNet)
        self.frame_buffer = jnp.zeros((BELIEF_STACK_SIZE, NUM_BASE_CHANNELS, H, W), dtype=jnp.float32)
        self._jit_infer = jax.jit(self._infer)

        # --- 4. CRITICAL FIX: JAX JIT WARMUP ---
        # Triggers compilation during initialization before turn timer starts
        dummy_frame = jnp.zeros((NUM_BASE_CHANNELS, H, W), dtype=jnp.float32)
        dummy_s_mask = jnp.zeros((H, W), dtype=jnp.bool_)
        dummy_t_mask = jnp.zeros((H, W, 5), dtype=jnp.bool_)
        
        _, self.frame_buffer = self._jit_infer(
            self.oracle, self.belief_net, self.frame_buffer, dummy_frame, dummy_s_mask, dummy_t_mask
        )
        # Reset buffer back to zeros after warmup
        self.frame_buffer = jnp.zeros((BELIEF_STACK_SIZE, NUM_BASE_CHANNELS, H, W), dtype=jnp.float32)
        sys.stderr.write("[BeliefAgent] JAX JIT warmup complete.\n")

    def _convert_obs_to_frame(self, obs) -> np.ndarray:
        """Identical to agent.py's -- builds the raw (foggy) 14-channel
        frame from a live protocol observation, same channel order as
        build_frame_tensor / belief_model.py's ARMY_CH/GENERALS_CH/etc."""
        armies = np.array(obs.army_grid, dtype=np.float32) / 200.0
        types = np.array(obs.type_grid, dtype=np.int32)
        owners = np.array(obs.owner_grid, dtype=np.int32)

        owned_mask = (owners == 1).astype(np.float32)
        opp_mask = (owners == 2).astype(np.float32)
        neutral_mask = (owners == 0).astype(np.float32)

        fog_mask = (types == 0).astype(np.float32)
        mountain_mask = (types == 2).astype(np.float32)
        castle_mask = (types == 3).astype(np.float32)
        general_mask = (types == 4).astype(np.float32)
        struct_in_fog = (types == 5).astype(np.float32)

        H, W = self.H, self.W
        land_me = np.full((H, W), obs.my_land / 400.0, dtype=np.float32)
        army_me = np.full((H, W), obs.my_army / 1000.0, dtype=np.float32)
        land_opp = np.full((H, W), obs.opp_land / 400.0, dtype=np.float32)
        army_opp = np.full((H, W), obs.opp_army / 1000.0, dtype=np.float32)
        step_time = np.full((H, W), obs.turn / 1200.0, dtype=np.float32)

        channels = [
            armies, general_mask, castle_mask, mountain_mask,
            neutral_mask, owned_mask, opp_mask, fog_mask,
            struct_in_fog, land_me, army_me, land_opp, army_opp, step_time
        ]
        return np.stack(channels, axis=0)

    def _get_live_action_masks(self, obs):
        """Identical to agent.py's -- computed from RAW foggy obs, never
        from the belief reconstruction (see module docstring)."""
        armies = np.array(obs.army_grid, dtype=np.int32)
        owners = np.array(obs.owner_grid, dtype=np.int32)
        types = np.array(obs.type_grid, dtype=np.int32)

        owned_cells = (owners == 1)
        source_mask = owned_cells & (armies > 1)

        impassable = (types == 2) | (types == 5)

        i_idx, j_idx = np.ogrid[:self.H, :self.W]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move_masks = []

        for dr, dc in dirs:
            ni, nj = i_idx + dr, j_idx + dc
            in_bounds = (ni >= 0) & (ni < self.H) & (nj >= 0) & (nj < self.W)
            safe_i = np.clip(ni, 0, self.H - 1)
            safe_j = np.clip(nj, 0, self.W - 1)
            is_passable = ~impassable[safe_i, safe_j]
            move_masks.append(in_bounds & is_passable)

        move_mask = np.stack(move_masks, axis=-1)

        is_plain = owned_cells & (types != 3) & (types != 4)
        build_mask = (is_plain & (armies >= 35))[..., None]

        target_mask = np.concatenate([move_mask, build_mask], axis=-1)
        return jnp.array(source_mask), jnp.array(target_mask)

    def _select_action(self, oracle, stacked_obs, source_mask, target_mask):
        """Same greedy (argmax) autoregressive decode as agent.py --
        deterministic at eval time, no sampling."""
        features = oracle.extract_features(stacked_obs)
        raw_s_logits = oracle.get_source_logits(features)
        masked_s_logits = jnp.where(source_mask, raw_s_logits, -1e9)

        source_idx = jnp.argmax(masked_s_logits.reshape(-1))
        selected_r = source_idx // self.W
        selected_c = source_idx % self.W

        raw_d_logits = oracle.get_direction_logits(features, selected_r, selected_c)
        tile_t_mask = target_mask[selected_r, selected_c]
        masked_d_logits = jnp.where(tile_t_mask, raw_d_logits, -1e9)
        dir_idx = jnp.argmax(masked_d_logits)

        no_valid_source = ~jnp.any(source_mask)
        official_action = format_network_action(dir_idx, selected_r, selected_c)
        pass_action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        return jnp.where(no_valid_source, pass_action, official_action)

    def _infer(self, oracle, belief_net, old_buffer, new_raw_frame, source_mask, target_mask):
        # 1. Update 8-frame buffer
        temp_buffer = jnp.concatenate([old_buffer[1:], new_raw_frame[None, ...]], axis=0)
        stacked_112_ch = temp_buffer.reshape(BELIEF_STACK_SIZE * NUM_BASE_CHANNELS, self.H, self.W)

        # 2. Run BeliefNet on all 8 frames
        belief_probs = belief_net.reconstruct_probs(stacked_112_ch)
        
        # 3. Splice predictions into the current (newest) frame
        spliced_last = splice_belief_reconstruction(temp_buffer[-1], belief_probs)
        new_buffer = temp_buffer.at[-1].set(spliced_last)

        # 4. Extract only the last 4 spliced frames (56 channels) for Oracle
        stacked_for_oracle = new_buffer[-ORACLE_STACK_SIZE:].reshape(ORACLE_STACK_SIZE * NUM_BASE_CHANNELS, self.H, self.W)
        action = self._select_action(oracle, stacked_for_oracle, source_mask, target_mask)

        return action, new_buffer

    def act(self, obs) -> tuple:
        new_frame = jnp.array(self._convert_obs_to_frame(obs))
        s_mask, t_mask = self._get_live_action_masks(obs)

        action_array, self.frame_buffer = self._jit_infer(
            self.oracle, self.belief_net, self.frame_buffer, new_frame, s_mask, t_mask
        )
        action_tuple = tuple(int(x) for x in np.array(action_array))

        return action_tuple