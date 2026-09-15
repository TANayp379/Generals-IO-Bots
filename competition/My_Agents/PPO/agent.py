# agent.py
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from pathlib import Path

from competition.agents.PPO.network import ActorCriticNet
from competition.agents.PPO.official_wrapper import (
    TOTAL_IN_CHANNELS,
    STACK_SIZE,
    NUM_BASE_CHANNELS,
    format_network_action
)

OFFICIAL_DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


class Agent:
    """Evaluation agent that loads trained Equinox weights and executes live turns."""

    def __init__(self, player_id: int, H: int, W: int, model_path: str = "checkpoints/official_model_iter_100.eqx"):
        self.player_id = player_id
        self.H = H
        self.W = W

        key = jax.random.PRNGKey(0)
        self.model = ActorCriticNet(key=key)

        ckpt_file = Path(model_path)
        if ckpt_file.exists():
            self.model = eqx.tree_deserialise_leaves(str(ckpt_file), self.model)
            print(f"[Agent] Loaded model weights from {ckpt_file}")
        else:
            print(f"[Warning] Checkpoint {model_path} not found! Agent will play with initial weights.")

        self.frame_buffer = np.zeros((STACK_SIZE, NUM_BASE_CHANNELS, H, W), dtype=np.float32)
        self._jit_act = jax.jit(self._select_action)

    def _convert_obs_to_frame(self, obs) -> np.ndarray:
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

    def _select_action(self, model, stacked_obs, source_mask, target_mask):
        features = model.extract_features(stacked_obs)
        raw_s_logits = model.get_source_logits(features)
        masked_s_logits = jnp.where(source_mask, raw_s_logits, -1e9)

        source_idx = jnp.argmax(masked_s_logits.reshape(-1))
        selected_r = source_idx // self.W
        selected_c = source_idx % self.W

        raw_d_logits = model.get_direction_logits(features, selected_r, selected_c)
        tile_t_mask = target_mask[selected_r, selected_c]
        masked_d_logits = jnp.where(tile_t_mask, raw_d_logits, -1e9)
        dir_idx = jnp.argmax(masked_d_logits)

        no_valid_source = ~jnp.any(source_mask)
        official_action = format_network_action(dir_idx, selected_r, selected_c)
        pass_action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        return jnp.where(no_valid_source, pass_action, official_action)

    def act(self, obs) -> tuple:
        new_frame = self._convert_obs_to_frame(obs)
        self.frame_buffer = np.concatenate([self.frame_buffer[1:], new_frame[None, ...]], axis=0)

        stacked_obs = jnp.array(self.frame_buffer.reshape(TOTAL_IN_CHANNELS, self.H, self.W))
        s_mask, t_mask = self._get_live_action_masks(obs)

        action_array = self._jit_act(self.model, stacked_obs, s_mask, t_mask)
        action_tuple = tuple(int(x) for x in np.array(action_array))

        return action_tuple