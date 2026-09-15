# belief_train.py
import os
import sys
import glob
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from generals import GeneralsEnv
from generals.core import game
from official_wrapper import build_frame_tensor, get_official_action_masks, NUM_BASE_CHANNELS
from belief_model import BeliefNet, belief_loss_fn

NUM_ENVS = 128
ROLLOUT_STEPS = 64
MINI_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
SAVE_DIR = "/content/drive/MyDrive/generals_belief_checkpoints"

# Extended temporal stack specifically for BeliefNet (8 past frames = 112 channels)
BELIEF_STACK_SIZE = 8
BELIEF_IN_CHANNELS = NUM_BASE_CHANNELS * BELIEF_STACK_SIZE  # 14 * 8 = 112


class FrameBufferManager:
    """Rolling-stack logic maintaining K=8 frame history for BeliefNet."""

    @staticmethod
    def init_buffer(frame_batch):
        return jnp.repeat(frame_batch[:, None, ...], BELIEF_STACK_SIZE, axis=1)

    @staticmethod
    def update_buffer(buffer, new_frame_batch):
        return jnp.concatenate([buffer[:, 1:], new_frame_batch[:, None, ...]], axis=1)

    @staticmethod
    def get_stacked_tensor(buffer):
        B, K, C, H, W = buffer.shape
        return buffer.reshape(B, K * C, H, W)


def sample_uniform_action(source_mask, target_mask, key):
    """Uniformly random legal action in official 5-int wire format."""
    k1, k2 = jax.random.split(key)
    H, W = source_mask.shape
    flat_mask = source_mask.reshape(-1)
    no_valid = ~jnp.any(flat_mask)

    src_logits = jnp.where(flat_mask, 0.0, -1e9)
    idx = jax.random.categorical(k1, src_logits)
    r, c = idx // W, idx % W

    tile_mask = target_mask[r, c]
    dir_logits = jnp.where(tile_mask, 0.0, -1e9)
    dir_idx = jax.random.categorical(k2, dir_logits)

    is_build = dir_idx == 4
    kind = jnp.where(is_build, 2, 0)
    direction = jnp.where(is_build, 0, dir_idx)
    action = jnp.array([kind, r, c, direction, 0], dtype=jnp.int32)
    pass_action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    return jnp.where(no_valid, pass_action, action)


def train_loop():
    print(f"[!] Target Device: {jax.devices()[0]}")
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    env = GeneralsEnv(
        min_grid_size=18, max_grid_size=21, pad_to=22, truncation=1200,
        mountain_density_range=(0.24, 0.26), num_castles_range=(9, 11),
        min_generals_distance=14, castle_val_range=(20, 26),
        build_castles=True, deathtouch_turn=800,
        pool_size=NUM_ENVS * 10,
    )

    # Instantiate BeliefNet with 112 input channels (8 stacked frames)
    belief_net = BeliefNet(in_channels=BELIEF_IN_CHANNELS, key=subkey)

    # --- AUTO-RESUME LOGIC ---
    start_iter = 1
    if os.path.exists(SAVE_DIR):
        checkpoints = glob.glob(f"{SAVE_DIR}/belief_model_iter_*.eqx")
        if checkpoints:
            iters = [
                int(re.search(r"iter_(\d+)\.eqx", cp).group(1))
                for cp in checkpoints
                if re.search(r"iter_(\d+)\.eqx", cp)
            ]
            if iters:
                latest_iter = max(iters)
                latest_cp = f"{SAVE_DIR}/belief_model_iter_{latest_iter}.eqx"
                belief_net = eqx.tree_deserialise_leaves(latest_cp, belief_net)
                start_iter = latest_iter + 1
                print(f"[!] Resuming BeliefNet training from iteration {latest_iter}...")

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(belief_net, eqx.is_array))

    key, k_reset = jax.random.split(key)
    pool, _ = env.reset(k_reset)
    states = jax.tree.map(lambda x: x[:NUM_ENVS], pool)

    def get_both_obs(states_in, player):
        foggy = jax.vmap(game.get_observation, in_axes=(0, None))(states_in, player)
        full = jax.vmap(game.get_full_observation, in_axes=(0, None))(states_in, player)
        return foggy, full

    obs_p0_foggy, _ = get_both_obs(states, 0)
    obs_p1_foggy, _ = get_both_obs(states, 1)

    buf_p0 = FrameBufferManager.init_buffer(jax.vmap(build_frame_tensor)(obs_p0_foggy))
    buf_p1 = FrameBufferManager.init_buffer(jax.vmap(build_frame_tensor)(obs_p1_foggy))

    print(f"[+] Initialized {NUM_ENVS} parallel environments for 8-frame BeliefNet pretraining.")

    @eqx.filter_jit
    def collect_and_update(belief_net_in, opt_state_in, key_in, states_in, buf_p0_in, buf_p1_in, obs_p0_foggy_in, obs_p1_foggy_in):
        stacked_list, target_list, fogmask_list = [], [], []

        curr_key = key_in
        curr_states = states_in
        curr_buf_p0, curr_buf_p1 = buf_p0_in, buf_p1_in
        curr_obs_p0_foggy, curr_obs_p1_foggy = obs_p0_foggy_in, obs_p1_foggy_in

        for step in range(ROLLOUT_STEPS):
            curr_key, k_a0, k_a1 = jax.random.split(curr_key, 3)

            sm_p0, tm_p0 = jax.vmap(get_official_action_masks)(curr_obs_p0_foggy)
            sm_p1, tm_p1 = jax.vmap(get_official_action_masks)(curr_obs_p1_foggy)
            actions_p0 = jax.vmap(sample_uniform_action)(sm_p0, tm_p0, jax.random.split(k_a0, NUM_ENVS))
            actions_p1 = jax.vmap(sample_uniform_action)(sm_p1, tm_p1, jax.random.split(k_a1, NUM_ENVS))

            joint_actions = jnp.stack([actions_p0, actions_p1], axis=1)
            timestep, curr_states = jax.vmap(env.step, in_axes=(0, 0, None))(curr_states, joint_actions, pool)

            obs_p0_foggy_new, obs_p0_full_new = get_both_obs(timestep.last_state, 0)
            obs_p1_foggy_new, obs_p1_full_new = get_both_obs(timestep.last_state, 1)

            frame_p0_foggy_new = jax.vmap(build_frame_tensor)(obs_p0_foggy_new)
            frame_p1_foggy_new = jax.vmap(build_frame_tensor)(obs_p1_foggy_new)
            frame_p0_full_new = jax.vmap(build_frame_tensor)(obs_p0_full_new)
            frame_p1_full_new = jax.vmap(build_frame_tensor)(obs_p1_full_new)

            curr_buf_p0 = FrameBufferManager.update_buffer(curr_buf_p0, frame_p0_foggy_new)
            curr_buf_p1 = FrameBufferManager.update_buffer(curr_buf_p1, frame_p1_foggy_new)

            fog_mask_p0 = obs_p0_foggy_new.fog_cells | obs_p0_foggy_new.structures_in_fog
            fog_mask_p1 = obs_p1_foggy_new.fog_cells | obs_p1_foggy_new.structures_in_fog

            stacked_list.append(jnp.concatenate([
                FrameBufferManager.get_stacked_tensor(curr_buf_p0),
                FrameBufferManager.get_stacked_tensor(curr_buf_p1),
            ], axis=0))
            target_list.append(jnp.concatenate([frame_p0_full_new, frame_p1_full_new], axis=0))
            fogmask_list.append(jnp.concatenate([fog_mask_p0, fog_mask_p1], axis=0))

            curr_obs_p0_foggy, curr_obs_p1_foggy = obs_p0_foggy_new, obs_p1_foggy_new

        stacked_all = jnp.concatenate(stacked_list, axis=0)
        target_all = jnp.concatenate(target_list, axis=0)
        fogmask_all = jnp.concatenate(fogmask_list, axis=0)

        dataset_size = stacked_all.shape[0]
        curr_key, perm_key = jax.random.split(curr_key)
        perm = jax.random.permutation(perm_key, dataset_size)
        num_minibatches = dataset_size // MINI_BATCH_SIZE

        def scan_fn(carry, i):
            curr_model, curr_opt_state = carry
            idx = jax.lax.dynamic_slice(perm, (i * MINI_BATCH_SIZE,), (MINI_BATCH_SIZE,))
            s_mb, t_mb, f_mb = stacked_all[idx], target_all[idx], fogmask_all[idx]

            def loss_wrapper(m):
                return belief_loss_fn(m, s_mb, t_mb, f_mb)

            loss, grads = eqx.filter_value_and_grad(loss_wrapper)(curr_model)
            updates, next_opt_state = optimizer.update(grads, curr_opt_state, eqx.filter(curr_model, eqx.is_array))
            next_model = eqx.apply_updates(curr_model, updates)
            return (next_model, next_opt_state), loss

        (belief_net_out, opt_state_out), losses = jax.lax.scan(
            scan_fn, (belief_net_in, opt_state_in), jnp.arange(num_minibatches)
        )

        return (belief_net_out, opt_state_out, jnp.mean(losses), curr_key, curr_states,
                curr_buf_p0, curr_buf_p1, curr_obs_p0_foggy, curr_obs_p1_foggy)

    for iteration in range(start_iter, 10000):
        key, iter_key = jax.random.split(key)
        (belief_net, opt_state, loss, key, states,
         buf_p0, buf_p1, obs_p0_foggy, obs_p1_foggy) = collect_and_update(
            belief_net, opt_state, iter_key, states, buf_p0, buf_p1, obs_p0_foggy, obs_p1_foggy
        )

        if iteration % 10 == 0:
            print(f"Iter {iteration:04d} | Belief Reconstruction MSE: {loss:.5f}")

        if iteration % 50 == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            checkpoint_path = f"{SAVE_DIR}/belief_model_iter_{iteration}.eqx"
            eqx.tree_serialise_leaves(checkpoint_path, belief_net)
            print(f"[+] Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train_loop()