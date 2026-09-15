# oracle_train.py
"""
Phase 1 -- Oracle training: identical PPO setup to train.py, but under
perfect_info=True so the policy learns from the TRUE board state instead
of the foggy one.

This is a separate file rather than a flag on train.py because of the
env.py mode= bug flagged earlier in this project: GeneralsEnv(mode=...)
is authoritative and silently overwrites perfect_info even if you pass
perfect_info=True alongside it (see env.py's __init__ -- every preset
field, including perfect_info, overwrites the matching constructor arg
when mode is set). So this file does NOT pass mode="competition" -- it
reconstructs the competition preset's fields by hand, with perfect_info
flipped to True. If generals/env.py's "competition" preset ever changes,
this dict silently drifts out of sync -- worth checking both places if
you change one.

Everything else (rollout collection, GAE, PPO update, reward shaping) is
untouched from train.py: ActorCriticNet, build_frame_tensor and
get_official_action_masks are already written generically against
Observation fields, so they work whether the observation came from
game.get_observation (foggy) or game.get_full_observation (perfect) --
no network or wrapper code needs to change for this phase.

Important: the resulting Oracle is NOT itself a deployable competition
bot. Real matches never give perfect info, so an Oracle submitted as-is
would receive inputs wildly outside its training distribution. It's only
useful frozen, as the policy Phase 3's belief_agent.py feeds reconstructed
(belief-model) input into.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import glob  # NEW
import re    # NEW

from generals import GeneralsEnv
from generals.core import game
from official_wrapper import build_frame_tensor, get_official_action_masks, STACK_SIZE
from network import ActorCriticNet, sample_autoregressive_action
from ppo import BatchTrajectory, update_ppo_step
from generals.core.rewards import composite_reward_fn

NUM_ENVS = 128
ROLLOUT_STEPS = 64
MINI_BATCH_SIZE = 512
LEARNING_RATE = 5e-4
SAVE_DIR = "checkpoints_oracle"  # separate from checkpoints/ -- do not mix with the fog-trained policy


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


def make_oracle_env(pool_size: int) -> GeneralsEnv:
    """Replicates the 'competition' preset from generals/env.py by hand,
    with perfect_info flipped to True. Do NOT pass mode='competition' here
    -- see module docstring for why that would silently undo this."""
    return GeneralsEnv(
        min_grid_size=18,
        max_grid_size=21,
        pad_to=22,
        truncation=1200,
        perfect_info=True,                # the entire point of this file
        mountain_density_range=(0.24, 0.26),
        num_castles_range=(9, 11),
        min_generals_distance=14,
        castle_val_range=(20, 26),
        build_castles=True,
        deathtouch_turn=800,
        pool_size=pool_size,
    )


def train_loop():
    print(f"[!] Target Device: {jax.devices()[0]}")
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    env = make_oracle_env(pool_size=NUM_ENVS * 10)

    model = ActorCriticNet(key=subkey)
    start_iter = 1
    if os.path.exists(SAVE_DIR):
        checkpoints = glob.glob(f"{SAVE_DIR}/oracle_model_iter_*.eqx")
        if checkpoints:
            # Extract the iteration numbers from the filenames
            iters = [int(re.search(r"iter_(\d+)\.eqx", cp).group(1)) for cp in checkpoints if re.search(r"iter_(\d+)\.eqx", cp)]
            if iters:
                latest_iter = max(iters)
                latest_cp = f"{SAVE_DIR}/oracle_model_iter_{latest_iter}.eqx"
                model = eqx.tree_deserialise_leaves(latest_cp, model)
                start_iter = latest_iter + 1
                print(f"[!] Resuming Oracle training from iteration {latest_iter}...")
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    key, k_reset = jax.random.split(key)
    pool, _ = env.reset(k_reset)
    states = jax.tree.map(lambda x: x[:NUM_ENVS], pool)

    # get_full_observation, not get_observation -- perfect info for both
    # players. env.perfect_info is already True (see make_oracle_env), and
    # env.step() itself picks the right getter internally via that flag,
    # but the very first observation before any step() call has to be
    # built explicitly here, same as train.py does for the foggy case.
    obs_p0 = jax.vmap(game.get_full_observation, in_axes=(0, None))(states, 0)
    obs_p1 = jax.vmap(game.get_full_observation, in_axes=(0, None))(states, 1)

    buf_p0 = FrameBufferManager.init_buffer(obs_p0)
    buf_p1 = FrameBufferManager.init_buffer(obs_p1)

    print(f"[+] Initialized {NUM_ENVS} parallel environments (perfect_info=True) for Oracle pretraining.")

    @eqx.filter_jit
    def collect_rollout(model_in, key_in, states_in, buf_p0_in, buf_p1_in, obs_p0_in, obs_p1_in):
        obs_list, act_list, logprob_list = [], [], []
        rew_list, done_list, val_list = [], [], []
        smask_list, tmask_list = [], []

        curr_key = key_in
        curr_states = states_in
        curr_buf_p0, curr_buf_p1 = buf_p0_in, buf_p1_in
        curr_obs_p0, curr_obs_p1 = obs_p0_in, obs_p1_in

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

            prior_obs_p0, prior_obs_p1 = curr_obs_p0, curr_obs_p1

            # env.step internally uses game.get_full_observation for both
            # players since env.perfect_info=True -- no extra wiring needed.
            timestep, curr_states = jax.vmap(env.step, in_axes=(0, 0, None))(curr_states, joint_actions, pool)

            curr_obs_p0 = jax.tree.map(lambda x: x[:, 0], timestep.observation)
            curr_obs_p1 = jax.tree.map(lambda x: x[:, 1], timestep.observation)

            curr_buf_p0 = FrameBufferManager.update_buffer(curr_buf_p0, curr_obs_p0)
            curr_buf_p1 = FrameBufferManager.update_buffer(curr_buf_p1, curr_obs_p1)

            rewards_p0 = jax.vmap(composite_reward_fn)(prior_obs_p0, actions_p0, curr_obs_p0)
            rewards_p1 = jax.vmap(composite_reward_fn)(prior_obs_p1, actions_p1, curr_obs_p1)

            dones = timestep.terminated | timestep.truncated

            obs_list.append(jnp.concatenate([stacked_obs_p0, stacked_obs_p1], axis=0))
            act_list.append(jnp.concatenate([actions_p0, actions_p1], axis=0))
            logprob_list.append(jnp.concatenate([s_lp0 + d_lp0, s_lp1 + d_lp1], axis=0))
            rew_list.append(jnp.concatenate([rewards_p0, rewards_p1], axis=0))
            done_list.append(jnp.concatenate([dones, dones], axis=0))
            val_list.append(jnp.concatenate([vals_p0, vals_p1], axis=0))
            smask_list.append(jnp.concatenate([sm_p0, sm_p1], axis=0))
            tmask_list.append(jnp.concatenate([tm_p0, tm_p1], axis=0))

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

        batch, last_value, key, states, buf_p0, buf_p1, obs_p0, obs_p1 = collect_rollout(
            model, rollout_key, states, buf_p0, buf_p1, obs_p0, obs_p1
        )

        key, ppo_key = jax.random.split(key)
        model, opt_state, metrics = update_ppo_step(
            model, opt_state, optimizer, batch, last_value, ppo_key, mini_batch_size=MINI_BATCH_SIZE
        )

        if iteration % 10 == 0:
            print(
                f"[Oracle] Iter {iteration:04d} | Loss: {metrics['total_loss']:.4f} | "
                f"Actor: {metrics['actor_loss']:.4f} | Value: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f}"
            )

        if iteration % 50 == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            checkpoint_path = f"{SAVE_DIR}/oracle_model_iter_{iteration}.eqx"
            eqx.tree_serialise_leaves(checkpoint_path, model)
            print(f"[+] Saved Oracle checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train_loop()