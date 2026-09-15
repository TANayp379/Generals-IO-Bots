# ppo.py
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Tuple, NamedTuple
from network import ActorCriticNet

class BatchTrajectory(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    old_log_probs: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    source_masks: jnp.ndarray
    target_masks: jnp.ndarray

@jax.jit
def compute_gae(
    rewards: jnp.ndarray, 
    values: jnp.ndarray, 
    dones: jnp.ndarray, 
    last_value: jnp.ndarray,  # NEW: Pass in the actual bootstrap value
    gamma: float = 0.99, 
    lam: float = 0.95
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    last_gae = 0.0
    
    # Bootstrap the final step with the critic's actual value instead of 0
    values_next = jnp.concatenate([values[1:], last_value[None]], axis=0)
    
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_next[t] * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages = advantages.at[t].set(last_gae)
        
    returns = advantages + values
    return advantages, returns


def ppo_loss_fn(
    model: ActorCriticNet,
    obs_mb: jnp.ndarray,
    act_mb: jnp.ndarray,
    old_log_probs_mb: jnp.ndarray,
    advs_mb: jnp.ndarray,
    rets_mb: jnp.ndarray,
    s_masks_mb: jnp.ndarray,
    t_masks_mb: jnp.ndarray,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01
) -> Tuple[jnp.ndarray, dict]:
    advs_mb = (advs_mb - jnp.mean(advs_mb)) / (jnp.std(advs_mb) + 1e-8)

    def single_eval(o, act, sm, tm):
        H, W = sm.shape
        r, c, d_idx = act[1], act[2], act[3]
        
        features = model.extract_features(o)
        
        raw_s_logits = model.get_source_logits(features)
        masked_s_logits = jnp.where(sm, raw_s_logits, -1e9)
        flat_s_logits = masked_s_logits.reshape(-1)
        s_log_probs = jax.nn.log_softmax(flat_s_logits)
        
        flat_idx = r * W + c
        s_log_prob = s_log_probs[flat_idx]
        s_entropy = -jnp.sum(jnp.exp(s_log_probs) * s_log_probs)

        raw_d_logits = model.get_direction_logits(features, r, c)
        tile_t_mask = tm[r, c]
        masked_d_logits = jnp.where(tile_t_mask, raw_d_logits, -1e9)
        d_log_probs = jax.nn.log_softmax(masked_d_logits)
        
        d_log_prob = d_log_probs[d_idx]
        d_entropy = -jnp.sum(jnp.exp(d_log_probs) * d_log_probs)
        
        total_log_prob = s_log_prob + d_log_prob
        total_entropy = s_entropy + d_entropy
        
        val = model.get_value(features)
        
        return total_log_prob, val, total_entropy

    new_log_probs, new_values, entropy = jax.vmap(single_eval)(obs_mb, act_mb, s_masks_mb, t_masks_mb)
    
    ratios = jnp.exp(new_log_probs - old_log_probs_mb)
    surr1 = ratios * advs_mb
    surr2 = jnp.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advs_mb
    actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    value_loss = 0.5 * jnp.mean(jnp.square(new_values - rets_mb))
    entropy_loss = -jnp.mean(entropy)
    
    total_loss = actor_loss + vf_coef * value_loss + ent_coef * entropy_loss
    
    metrics = {
        "total_loss": total_loss,
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "entropy": -entropy_loss
    }
    return total_loss, metrics


@eqx.filter_jit  # NEW: Compile the entire PPO update loop
def update_ppo_step(
    model: ActorCriticNet,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: BatchTrajectory,
    last_value: jnp.ndarray,  # NEW: Accept the bootstrap value
    key: jax.random.PRNGKey,
    mini_batch_size: int = 128,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[ActorCriticNet, optax.OptState, dict]:
    advantages, returns = compute_gae(batch.rewards, batch.values, batch.dones, last_value, gamma, lam)
    
    obs = batch.obs.reshape(-1, *batch.obs.shape[2:])
    actions = batch.actions.reshape(-1, 5)
    old_log_probs = batch.old_log_probs.reshape(-1)
    advs = advantages.reshape(-1)
    rets = returns.reshape(-1)
    s_masks = batch.source_masks.reshape(-1, *batch.source_masks.shape[2:])
    t_masks = batch.target_masks.reshape(-1, *batch.target_masks.shape[2:])
    
    dataset_size = obs.shape[0]
    perm = jax.random.permutation(key, dataset_size)
    
    accumulated_metrics = {"total_loss": 0.0, "actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    num_minibatches = dataset_size // mini_batch_size
    
    # We use jax.lax.scan to iterate over minibatches efficiently under JIT
    def scan_fn(carry, i):
        curr_model, curr_opt_state = carry
        idx = jax.lax.dynamic_slice(perm, (i * mini_batch_size,), (mini_batch_size,))
        
        obs_mb, act_mb = obs[idx], actions[idx]
        old_lp_mb, advs_mb, rets_mb = old_log_probs[idx], advs[idx], rets[idx]
        sm_mb, tm_mb = s_masks[idx], t_masks[idx]

        def loss_wrapper(m):
            return ppo_loss_fn(m, obs_mb, act_mb, old_lp_mb, advs_mb, rets_mb, sm_mb, tm_mb)

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_wrapper, has_aux=True)(curr_model)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, eqx.filter(curr_model, eqx.is_array))
        next_model = eqx.apply_updates(curr_model, updates)
        
        return (next_model, next_opt_state), metrics

    (model, opt_state), all_metrics = jax.lax.scan(scan_fn, (model, opt_state), jnp.arange(num_minibatches))
    
    # Average the metrics over all minibatches
    for k in accumulated_metrics:
        accumulated_metrics[k] = jnp.mean(all_metrics[k])
            
    return model, opt_state, accumulated_metrics