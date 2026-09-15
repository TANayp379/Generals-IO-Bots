# train.py
import os
import sys
from pathlib import Path

# Add repository root to sys.path for dynamic imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import glob  # NEW
import re


from generals import GeneralsEnv
from generals.core import game
from official_wrapper import build_frame_tensor, get_official_action_masks, STACK_SIZE, NUM_BASE_CHANNELS
from network import ActorCriticNet, sample_autoregressive_action
from ppo import BatchTrajectory, update_ppo_step

# Import official composite reward function[cite: 8]
from generals.core.rewards import composite_reward_fn 

NUM_ENVS = 128
ROLLOUT_STEPS = 64
MINI_BATCH_SIZE = 512
LEARNING_RATE = 5e-4
SAVE_DIR = "/content/drive/MyDrive/generals_checkpoints"


class FrameBufferManager:
    @staticmethod
    def init_buffer(obs_batch):
        frame = jax.vmap(build_frame_tensor)(obs_batch)
        return jnp.repeat(frame[:, None, ...], STACK_SIZE, axis=1)

    @staticmethod
    def update_buffer(buffer, new_obs_batch):
        new_frame = jax.vmap(build_frame_tensor)(new_obs_batch)
        return jnp.concatenate([buffer[:, 1:], new_frame[:, None, ...]], axis=1)

    @staticmethod
    def get_stacked_tensor(buffer):
        B, K, C, H, W = buffer.shape
        return buffer.reshape(B, K * C, H, W)


def train_loop():
    print(f"[!] Target Device: {jax.devices()[0]}")
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    env = GeneralsEnv(mode="competition", pool_size=NUM_ENVS * 10)
    
    model = ActorCriticNet(key=subkey)
    start_iter = 1
    if os.path.exists(SAVE_DIR):
        # Scans for the original fog-trained PPO filenames
        checkpoints = glob.glob(f"{SAVE_DIR}/official_model_iter_*.eqx") 
        if checkpoints:
            iters = [int(re.search(r"iter_(\d+)\.eqx", cp).group(1)) for cp in checkpoints if re.search(r"iter_(\d+)\.eqx", cp)]
            if iters:
                latest_iter = max(iters)
                latest_cp = f"{SAVE_DIR}/official_model_iter_{latest_iter}.eqx"
                
                # Deserializes the weights from the file into the model skeleton[cite: 1]
                model = eqx.tree_deserialise_leaves(latest_cp, model)
                start_iter = latest_iter + 1
                print(f"[!] Resuming standard PPO training from iteration {latest_iter}...")
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    key, k_reset = jax.random.split(key)
    pool, _ = env.reset(k_reset)
    states = jax.tree.map(lambda x: x[:NUM_ENVS], pool)
    
    obs_p0 = jax.vmap(game.get_observation, in_axes=(0, None))(states, 0)
    obs_p1 = jax.vmap(game.get_observation, in_axes=(0, None))(states, 1)

    buf_p0 = FrameBufferManager.init_buffer(obs_p0)
    buf_p1 = FrameBufferManager.init_buffer(obs_p1)

    print(f"[+] Initialized {NUM_ENVS} parallel environments with JIT-compiled dense reward rollouts.")

    @eqx.filter_jit
    def collect_rollout(model_in, key_in, states_in, buf_p0_in, buf_p1_in, obs_p0_in, obs_p1_in):
        obs_list, act_list, logprob_list = [], [], []
        rew_list, done_list, val_list = [], [], []
        smask_list, tmask_list = [], []

        curr_key = key_in
        curr_states = states_in
        curr_buf_p0, curr_buf_p1 = buf_p0_in, buf_p1_in
        curr_obs_p0, curr_obs_p1 = obs_p0_in, obs_p1_in

        # XLA naturally unrolls this fixed python loop
        for step in range(ROLLOUT_STEPS):
            curr_key, k_p0, k_p1 = jax.random.split(curr_key, 3)

            stacked_obs_p0 = FrameBufferManager.get_stacked_tensor(curr_buf_p0)
            stacked_obs_p1 = FrameBufferManager.get_stacked_tensor(curr_buf_p1)

            sm_p0, tm_p0 = jax.vmap(get_official_action_masks)(curr_obs_p0)
            sm_p1, tm_p1 = jax.vmap(get_official_action_masks)(curr_obs_p1)

            keys_p0 = jax.random.split(k_p0, NUM_ENVS)
            actions_p0, s_lp0, d_lp0, vals_p0 = jax.vmap(
                sample_autoregressive_action, in_axes=(None, 0, 0, 0, 0)
            )(model_in, stacked_obs_p0, sm_p0, tm_p0, keys_p0)

            keys_p1 = jax.random.split(k_p1, NUM_ENVS)
            actions_p1, s_lp1, d_lp1, vals_p1 = jax.vmap(
                sample_autoregressive_action, in_axes=(None, 0, 0, 0, 0)
            )(model_in, stacked_obs_p1, sm_p1, tm_p1, keys_p1)

            joint_actions = jnp.stack([actions_p0, actions_p1], axis=1)

            # Keep references to calculate reward
            prior_obs_p0, prior_obs_p1 = curr_obs_p0, curr_obs_p1

            timestep, curr_states = jax.vmap(env.step, in_axes=(0, 0, None))(curr_states, joint_actions, pool)

            curr_obs_p0 = jax.tree.map(lambda x: x[:, 0], timestep.observation)
            curr_obs_p1 = jax.tree.map(lambda x: x[:, 1], timestep.observation)

            curr_buf_p0 = FrameBufferManager.update_buffer(curr_buf_p0, curr_obs_p0)
            curr_buf_p1 = FrameBufferManager.update_buffer(curr_buf_p1, curr_obs_p1)

            # Utilize the dense composite reward function[cite: 8]
            rewards_p0 = jax.vmap(composite_reward_fn)(prior_obs_p0, actions_p0, curr_obs_p0)
            rewards_p1 = jax.vmap(composite_reward_fn)(prior_obs_p1, actions_p1, curr_obs_p1)

            dones = timestep.terminated | timestep.truncated

            # Pool both players' data to double the batch size for free
            obs_list.append(jnp.concatenate([stacked_obs_p0, stacked_obs_p1], axis=0))
            act_list.append(jnp.concatenate([actions_p0, actions_p1], axis=0))
            logprob_list.append(jnp.concatenate([s_lp0 + d_lp0, s_lp1 + d_lp1], axis=0))
            rew_list.append(jnp.concatenate([rewards_p0, rewards_p1], axis=0))
            done_list.append(jnp.concatenate([dones, dones], axis=0))
            val_list.append(jnp.concatenate([vals_p0, vals_p1], axis=0))
            smask_list.append(jnp.concatenate([sm_p0, sm_p1], axis=0))
            tmask_list.append(jnp.concatenate([tm_p0, tm_p1], axis=0))

        # Bootstrap Final Value
        final_stacked_p0 = FrameBufferManager.get_stacked_tensor(curr_buf_p0)
        final_stacked_p1 = FrameBufferManager.get_stacked_tensor(curr_buf_p1)
        
        last_val_p0 = jax.vmap(lambda o: model_in.get_value(model_in.extract_features(o)))(final_stacked_p0)
        last_val_p1 = jax.vmap(lambda o: model_in.get_value(model_in.extract_features(o)))(final_stacked_p1)
        last_val_combined = jnp.concatenate([last_val_p0, last_val_p1], axis=0)

        batch = BatchTrajectory(
            obs=jnp.stack(obs_list),
            actions=jnp.stack(act_list),
            old_log_probs=jnp.stack(logprob_list),
            rewards=jnp.stack(rew_list),
            dones=jnp.stack(done_list),
            values=jnp.stack(val_list),
            source_masks=jnp.stack(smask_list),
            target_masks=jnp.stack(tmask_list),
        )

        return batch, last_val_combined, curr_key, curr_states, curr_buf_p0, curr_buf_p1, curr_obs_p0, curr_obs_p1


    for iteration in range(start_iter, 10000):
        key, rollout_key = jax.random.split(key)
        
        # Fast, compiled execution of the rollout loop
        batch, last_value, key, states, buf_p0, buf_p1, obs_p0, obs_p1 = collect_rollout(
            model, rollout_key, states, buf_p0, buf_p1, obs_p0, obs_p1
        )

        key, ppo_key = jax.random.split(key)
        model, opt_state, metrics = update_ppo_step(
            model, opt_state, optimizer, batch, last_value, ppo_key, mini_batch_size=MINI_BATCH_SIZE
        )

        if iteration % 10 == 0:
            print(
                f"Iter {iteration:04d} | Loss: {metrics['total_loss']:.4f} | "
                f"Actor: {metrics['actor_loss']:.4f} | Value: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f}"
            )

        if iteration % 50 == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            checkpoint_path = f"{SAVE_DIR}/official_model_iter_{iteration}.eqx"
            eqx.tree_serialise_leaves(checkpoint_path, model)
            print(f"[+] Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train_loop()