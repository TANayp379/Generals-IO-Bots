"""
Action distribution built on top of ActorCriticNet's raw logits.

Single-observation (unbatched) functions on purpose — train.py vmaps over
(env, player) axes explicitly, and agent.py only ever needs one observation
per call anyway. Keeping this file "batch-naive" makes both call sites easier
to get right.

Action encoding matches action.py / agent.py's contract exactly:
    [pass, row, col, direction, split]   direction: 0=UP 1=DOWN 2=LEFT 3=RIGHT
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom


def _bernoulli_logprob(bit: jnp.ndarray, prob: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(bit, jnp.log(prob + 1e-8), jnp.log(1.0 - prob + 1e-8))


def _bernoulli_entropy(prob: jnp.ndarray) -> jnp.ndarray:
    return -(prob * jnp.log(prob + 1e-8) + (1.0 - prob) * jnp.log(1.0 - prob + 1e-8))


def act(model, params, obs_tensor: jnp.ndarray, valid_mask: jnp.ndarray, key: jnp.ndarray, deterministic: bool = False):
    """
    Sample (or, if deterministic, greedily pick) an action.

    Args:
        obs_tensor: (H, W, 14) float32, see model.py for channel layout.
        valid_mask: (H, W, 4) bool, from obs_adapter's mask builder (or
            action.compute_valid_move_mask_obs during training).
        deterministic: True for inference (competition play), False for
            training rollouts (stochastic exploration).

    Returns:
        action: int32 array of shape (5,) — [pass, row, col, direction, split]
        logprob: scalar, log-probability of the returned action under the
            current policy (needed for the PPO ratio; unused at inference).
        value: scalar, critic's value estimate for this state.
    """
    source_logits, dir_logits, pass_logit, split_logit, value = model.apply(params, obs_tensor)
    H, W = source_logits.shape
    valid_any = jnp.any(valid_mask, axis=-1)  # (H, W) — cell has >=1 legal direction
    no_moves = ~jnp.any(valid_any)

    k_pass, k_src, k_dir, k_split = jrandom.split(key, 4)

    pass_prob = jax.nn.sigmoid(pass_logit)
    if deterministic:
        do_pass = (pass_prob > 0.5) | no_moves
    else:
        do_pass = jrandom.bernoulli(k_pass, pass_prob) | no_moves
    log_p_pass = _bernoulli_logprob(do_pass, pass_prob)

    flat_src_logits = jnp.where(valid_any.reshape(-1), source_logits.reshape(-1), -1e9)
    src_idx = jnp.argmax(flat_src_logits) if deterministic else jrandom.categorical(k_src, flat_src_logits)
    log_p_src = jax.nn.log_softmax(flat_src_logits)[src_idx]
    row, col = src_idx // W, src_idx % W

    cell_dir_logits = dir_logits[row, col]
    masked_dir_logits = jnp.where(valid_mask[row, col], cell_dir_logits, -1e9)
    direction = jnp.argmax(masked_dir_logits) if deterministic else jrandom.categorical(k_dir, masked_dir_logits)
    log_p_dir = jax.nn.log_softmax(masked_dir_logits)[direction]

    split_prob = jax.nn.sigmoid(split_logit)
    split = (split_prob > 0.5) if deterministic else jrandom.bernoulli(k_split, split_prob)
    log_p_split = _bernoulli_logprob(split, split_prob)

    move_logprob = log_p_src + log_p_dir + log_p_split
    total_logprob = log_p_pass + jnp.where(do_pass, 0.0, move_logprob)

    action = jnp.array(
        [do_pass.astype(jnp.int32), row.astype(jnp.int32), col.astype(jnp.int32),
         direction.astype(jnp.int32), split.astype(jnp.int32)],
        dtype=jnp.int32,
    )
    return action, total_logprob, value


def evaluate(model, params, obs_tensor: jnp.ndarray, valid_mask: jnp.ndarray, action: jnp.ndarray):
    """
    Log-prob, entropy, and value of a *given* action under the current
    params. Used during PPO update epochs, where actions were sampled by an
    older copy of the policy and we need the current policy's density at
    those same actions (the importance-sampling ratio).
    """
    source_logits, dir_logits, pass_logit, split_logit, value = model.apply(params, obs_tensor)
    H, W = source_logits.shape
    valid_any = jnp.any(valid_mask, axis=-1)

    do_pass = action[0] == 1
    row, col, direction = action[1], action[2], action[3]
    split = action[4] == 1

    pass_prob = jax.nn.sigmoid(pass_logit)
    log_p_pass = _bernoulli_logprob(do_pass, pass_prob)
    pass_entropy = _bernoulli_entropy(pass_prob)

    flat_src_logits = jnp.where(valid_any.reshape(-1), source_logits.reshape(-1), -1e9)
    log_softmax_src = jax.nn.log_softmax(flat_src_logits)
    src_idx = row * W + col
    log_p_src = log_softmax_src[src_idx]
    src_probs = jax.nn.softmax(flat_src_logits)
    src_entropy = -jnp.sum(src_probs * log_softmax_src)

    masked_dir_logits = jnp.where(valid_mask[row, col], dir_logits[row, col], -1e9)
    log_softmax_dir = jax.nn.log_softmax(masked_dir_logits)
    log_p_dir = log_softmax_dir[direction]
    dir_probs = jax.nn.softmax(masked_dir_logits)
    dir_entropy = -jnp.sum(dir_probs * log_softmax_dir)

    split_prob = jax.nn.sigmoid(split_logit)
    log_p_split = _bernoulli_logprob(split, split_prob)
    split_entropy = _bernoulli_entropy(split_prob)

    move_logprob = log_p_src + log_p_dir + log_p_split
    total_logprob = log_p_pass + jnp.where(do_pass, 0.0, move_logprob)

    move_entropy = src_entropy + dir_entropy + split_entropy
    total_entropy = pass_entropy + jnp.where(do_pass, 0.0, move_entropy)

    return total_logprob, total_entropy, value


def action_to_tuple(action: jnp.ndarray) -> tuple[int, int, int, int, int]:
    """int32[5] -> plain python tuple, in the (pass,row,col,dir,split) format main.py writes to stdout."""
    a = [int(x) for x in action]
    return a[0], a[1], a[2], a[3], a[4]