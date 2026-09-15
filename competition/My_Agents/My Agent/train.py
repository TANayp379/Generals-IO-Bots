"""
Self-play PPO training for the shared ActorCriticNet policy.

Both players in every parallel game are controlled by the *same* set of
parameters (symmetric self-play). Their experience is pooled into one big
batch for the PPO update — this is the standard, simplest self-play setup
and works well as long as the win/lose signal is symmetric, which it is
here (rewards.py's reward functions are computed independently from each
player's own Observation).

This script intentionally does the rollout loop in plain Python (each step
still calls jitted/vmapped JAX functions under the hood) rather than fusing
everything into one lax.scan. That costs some throughput but keeps the code
readable and easy to debug/modify — worth converting to a fused scan once
the setup is validated and you're ready to scale up num_envs / rollout_len.

Run: python train.py
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from generals.core import action as action_mod
from generals.core import game
from generals.core import rewards as rewards_mod
from generals.core.env import GeneralsEnv

import policy
from model import ActorCriticNet


@dataclass
class Config:
    # Board / env
    grid_dims: tuple[int, int] = (15, 15)
    truncation: int = 500
    build_castles: bool = False
    deathtouch_turn: int | None = None

    # Rollout
    num_envs: int = 64
    rollout_len: int = 128

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    epochs_per_update: int = 4
    num_minibatches: int = 8
    max_grad_norm: float = 0.5

    # Network
    features: int = 64
    num_blocks: int = 6

    # Loop
    total_updates: int = 2000
    seed: int = 0
    checkpoint_every: int = 50
    checkpoint_path: str = "checkpoints/weights.pkl"
    log_every: int = 1

    reward_fn: str = "win_lose"  # "win_lose" | "castle" | "ratio" | "composite"


REWARD_FNS = {
    "win_lose": rewards_mod.win_lose_reward_fn,
    "castle": rewards_mod.castle_reward_fn,
    "ratio": rewards_mod.ratio_reward_fn,
    "composite": rewards_mod.composite_reward_fn,
}


def obs_to_tensor(obs_single) -> jnp.ndarray:
    """Single (unbatched) Observation -> (H, W, 14) channels-last tensor."""
    return jnp.moveaxis(obs_single.as_tensor(), 0, -1)


def build_batched_fns(model: ActorCriticNet, cfg: Config):
    """vmap wrappers over the num_envs axis. model/deterministic are closed
    over (not traced), only params/obs/mask/key/action vary per env."""

    def _act(params, obs_tensor, mask, key):
        return policy.act(model, params, obs_tensor, mask, key, deterministic=False)

    def _evaluate(params, obs_tensor, mask, action):
        return policy.evaluate(model, params, obs_tensor, mask, action)

    batched_act = jax.vmap(_act, in_axes=(None, 0, 0, 0))
    batched_evaluate = jax.vmap(_evaluate, in_axes=(None, 0, 0, 0))
    batched_obs_to_tensor = jax.vmap(obs_to_tensor)
    batched_mask = jax.vmap(action_mod.compute_valid_move_mask_obs)
    batched_reward = jax.vmap(REWARD_FNS[cfg.reward_fn])
    batched_get_obs = jax.vmap(game.get_observation, in_axes=(0, None))

    return {
        "act": batched_act,
        "evaluate": batched_evaluate,
        "obs_to_tensor": batched_obs_to_tensor,
        "mask": batched_mask,
        "reward": batched_reward,
        "get_obs": batched_get_obs,
    }


def compute_gae(rewards, values, dones, bootstrap_value, gamma, lam):
    """
    rewards, values: (T, num_envs, 2)
    dones:           (T, num_envs)          -- shared across players, game ends for both at once
    bootstrap_value: (num_envs, 2)          -- value estimate one step past the last collected step
    Returns advantages, returns: both (T, num_envs, 2)
    """
    T = rewards.shape[0]
    advantages = [None] * T
    last_adv = jnp.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    for t in reversed(range(T)):
        not_done = (1.0 - dones[t].astype(jnp.float32))[:, None]  # (num_envs, 1) broadcast over player axis
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_adv = delta + gamma * lam * not_done * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    advantages = jnp.stack(advantages, axis=0)
    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, cfg: Config, params, opt_state, batch, key):
    """One epoch's worth of minibatch PPO updates over `batch` (already flattened to N samples)."""
    obs, mask, action, old_logprob, advantage, ret = batch
    n = obs.shape[0]
    minibatch_size = n // cfg.num_minibatches

    def loss_fn(p, mb):
        mb_obs, mb_mask, mb_action, mb_old_logprob, mb_adv, mb_ret = mb

        def per_sample(o, m, a, old_lp, adv, r):
            lp, ent, val = policy.evaluate(model, p, o, m, a)
            ratio = jnp.exp(lp - old_lp)
            unclipped = ratio * adv
            clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            pg_loss = -jnp.minimum(unclipped, clipped)
            v_loss = (val - r) ** 2
            return pg_loss, v_loss, ent

        pg_losses, v_losses, entropies = jax.vmap(per_sample)(
            mb_obs, mb_mask, mb_action, mb_old_logprob, mb_adv, mb_ret
        )
        pg_loss = jnp.mean(pg_losses)
        v_loss = jnp.mean(v_losses)
        entropy = jnp.mean(entropies)
        total = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
        return total, (pg_loss, v_loss, entropy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    perm = jrandom.permutation(key, n)
    metrics = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
    for i in range(cfg.num_minibatches):
        idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
        mb = tuple(x[idx] for x in (obs, mask, action, old_logprob, advantage, ret))
        (loss, (pg_loss, v_loss, entropy)), grads = grad_fn(params, mb)
        # Gradient clipping already lives in `optimizer` (see optax.chain in train()).
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics["pg_loss"] += float(pg_loss) / cfg.num_minibatches
        metrics["v_loss"] += float(v_loss) / cfg.num_minibatches
        metrics["entropy"] += float(entropy) / cfg.num_minibatches

    return params, opt_state, metrics


def save_checkpoint(path: str, params, cfg: Config):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "wb") as f:
        pickle.dump({"params": params, "features": cfg.features, "num_blocks": cfg.num_blocks}, f)


def train(cfg: Config):
    key = jrandom.PRNGKey(cfg.seed)
    key, model_key, pool_key, env_key = jrandom.split(key, 4)

    env = GeneralsEnv(
        grid_dims=cfg.grid_dims,
        truncation=cfg.truncation,
        build_castles=cfg.build_castles,
        deathtouch_turn=cfg.deathtouch_turn,
    )
    H, W = cfg.grid_dims

    model = ActorCriticNet(features=cfg.features, num_blocks=cfg.num_blocks)
    dummy_obs = jnp.zeros((H, W, 14), dtype=jnp.float32)
    params = model.init(model_key, dummy_obs)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr),
    )
    opt_state = optimizer.init(params)

    fns = build_batched_fns(model, cfg)

    pool, _ = env.reset(pool_key)
    env_keys = jrandom.split(env_key, cfg.num_envs)
    states = jax.vmap(env.init_state)(env_keys)

    step_fn = jax.jit(jax.vmap(lambda s, a, p: env.step(s, a, p), in_axes=(0, 0, None)))

    obs_p0 = fns["get_obs"](states, 0)
    obs_p1 = fns["get_obs"](states, 1)

    for update in range(cfg.total_updates):
        t0 = time.time()

        buf_obs, buf_mask, buf_action, buf_logprob, buf_value, buf_reward, buf_done = (
            [], [], [], [], [], [], []
        )

        for t in range(cfg.rollout_len):
            key, k0, k1 = jrandom.split(key, 3)
            keys0 = jrandom.split(k0, cfg.num_envs)
            keys1 = jrandom.split(k1, cfg.num_envs)

            tensor_p0 = fns["obs_to_tensor"](obs_p0)
            tensor_p1 = fns["obs_to_tensor"](obs_p1)
            mask_p0 = fns["mask"](obs_p0)
            mask_p1 = fns["mask"](obs_p1)

            action_p0, logprob_p0, value_p0 = fns["act"](params, tensor_p0, mask_p0, keys0)
            action_p1, logprob_p1, value_p1 = fns["act"](params, tensor_p1, mask_p1, keys1)

            actions = jnp.stack([action_p0, action_p1], axis=1)  # (num_envs, 2, 5)
            timestep, states = step_fn(states, actions, pool)

            new_obs_p0 = jax.tree.map(lambda x: x[:, 0], timestep.observation)
            new_obs_p1 = jax.tree.map(lambda x: x[:, 1], timestep.observation)

            reward_p0 = fns["reward"](obs_p0, action_p0, new_obs_p0)
            reward_p1 = fns["reward"](obs_p1, action_p1, new_obs_p1)
            done = timestep.terminated | timestep.truncated

            buf_obs.append(jnp.stack([tensor_p0, tensor_p1], axis=1))
            buf_mask.append(jnp.stack([mask_p0, mask_p1], axis=1))
            buf_action.append(jnp.stack([action_p0, action_p1], axis=1))
            buf_logprob.append(jnp.stack([logprob_p0, logprob_p1], axis=1))
            buf_value.append(jnp.stack([value_p0, value_p1], axis=1))
            buf_reward.append(jnp.stack([reward_p0, reward_p1], axis=1))
            buf_done.append(done)

            obs_p0, obs_p1 = new_obs_p0, new_obs_p1

        # Bootstrap value for the state one step past the rollout.
        tensor_p0 = fns["obs_to_tensor"](obs_p0)
        tensor_p1 = fns["obs_to_tensor"](obs_p1)
        mask_p0 = fns["mask"](obs_p0)
        mask_p1 = fns["mask"](obs_p1)
        key, kb0, kb1 = jrandom.split(key, 3)
        _, _, bootstrap_v0 = fns["act"](params, tensor_p0, mask_p0, jrandom.split(kb0, cfg.num_envs))
        _, _, bootstrap_v1 = fns["act"](params, tensor_p1, mask_p1, jrandom.split(kb1, cfg.num_envs))
        bootstrap_value = jnp.stack([bootstrap_v0, bootstrap_v1], axis=1)  # (num_envs, 2)

        rewards_arr = jnp.stack(buf_reward, axis=0)   # (T, num_envs, 2)
        values_arr = jnp.stack(buf_value, axis=0)     # (T, num_envs, 2)
        dones_arr = jnp.stack(buf_done, axis=0)        # (T, num_envs)

        advantages, returns = compute_gae(
            rewards_arr, values_arr, dones_arr, bootstrap_value, cfg.gamma, cfg.gae_lambda
        )

        # Flatten (T, num_envs, 2, ...) -> (T*num_envs*2, ...) for the PPO update.
        obs_batch = jnp.stack(buf_obs, axis=0).reshape((-1, H, W, 14))
        mask_batch = jnp.stack(buf_mask, axis=0).reshape((-1, H, W, 4))
        action_batch = jnp.stack(buf_action, axis=0).reshape((-1, 5))
        logprob_batch = jnp.stack(buf_logprob, axis=0).reshape((-1,))
        adv_batch = advantages.reshape((-1,))
        ret_batch = returns.reshape((-1,))

        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        batch = (obs_batch, mask_batch, action_batch, logprob_batch, adv_batch, ret_batch)

        for epoch in range(cfg.epochs_per_update):
            key, epoch_key = jrandom.split(key)
            params, opt_state, metrics = ppo_update(model, optimizer, cfg, params, opt_state, batch, epoch_key)

        if update % cfg.log_every == 0:
            win_rate_p0 = float(jnp.mean(jnp.where(dones_arr, rewards_arr[..., 0] > 0, jnp.nan)))
            dt = time.time() - t0
            print(
                f"update {update:5d} | pg {metrics['pg_loss']:.4f} | v {metrics['v_loss']:.4f} "
                f"| ent {metrics['entropy']:.4f} | p0_winrate {win_rate_p0:.3f} | {dt:.1f}s"
            )

        if update % cfg.checkpoint_every == 0 and update > 0:
            save_checkpoint(cfg.checkpoint_path, params, cfg)
            print(f"  saved checkpoint -> {cfg.checkpoint_path}")

    save_checkpoint(cfg.checkpoint_path, params, cfg)
    print(f"training done, final checkpoint -> {cfg.checkpoint_path}")


if __name__ == "__main__":
    train(Config())