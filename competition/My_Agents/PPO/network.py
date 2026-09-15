# network.py
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from official_wrapper import TOTAL_IN_CHANNELS, format_network_action

class ResBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm

    def __init__(self, channels: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=k2)
        self.norm1 = eqx.nn.GroupNorm(groups=4, channels=channels)
        self.norm2 = eqx.nn.GroupNorm(groups=4, channels=channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = jax.nn.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return jax.nn.relu(x + residual)

class ActorCriticNet(eqx.Module):
    conv_in: eqx.nn.Conv2d
    res_blocks: list
    source_conv: eqx.nn.Conv2d
    dir_fc1: eqx.nn.Linear
    dir_fc2: eqx.nn.Linear
    value_fc1: eqx.nn.Linear
    value_fc2: eqx.nn.Linear

    def __init__(
        self, 
        in_channels: int = TOTAL_IN_CHANNELS,
        hidden_dim: int = 128, 
        num_blocks: int = 4, 
        *, 
        key: jax.random.PRNGKey
    ):
        # FIX: Ensure 6 unique keys are generated for the architecture instead of 5
        keys = jax.random.split(key, 6 + num_blocks)
        
        self.conv_in = eqx.nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, key=keys[0])
        self.res_blocks = [ResBlock(hidden_dim, k) for k in keys[1:1 + num_blocks]]
        
        self.source_conv = eqx.nn.Conv2d(hidden_dim, 1, kernel_size=1, key=keys[-5])
        self.dir_fc1 = eqx.nn.Linear(hidden_dim * 2, 64, key=keys[-4])
        self.dir_fc2 = eqx.nn.Linear(64, 5, key=keys[-3])
        self.value_fc1 = eqx.nn.Linear(hidden_dim, 64, key=keys[-2])
        self.value_fc2 = eqx.nn.Linear(64, 1, key=keys[-1])

    def extract_features(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.conv_in(x))
        for block in self.res_blocks:
            x = block(x)
        return x

    def get_source_logits(self, features: jnp.ndarray) -> jnp.ndarray:
        return jnp.squeeze(self.source_conv(features), axis=0)

    def get_direction_logits(self, features: jnp.ndarray, selected_r: int, selected_c: int) -> jnp.ndarray:
        local_feat = features[:, selected_r, selected_c]
        global_feat = jnp.mean(features, axis=(-2, -1))
        combined = jnp.concatenate([local_feat, global_feat], axis=-1)
        x = jax.nn.relu(self.dir_fc1(combined))
        return self.dir_fc2(x)

    def get_value(self, features: jnp.ndarray) -> jnp.ndarray:
        global_feat = jnp.mean(features, axis=(-2, -1))
        x = jax.nn.relu(self.value_fc1(global_feat))
        return jnp.squeeze(self.value_fc2(x), axis=-1)

def sample_autoregressive_action(
    net: ActorCriticNet, 
    obs_tensor: jnp.ndarray, 
    source_mask: jnp.ndarray, 
    target_mask: jnp.ndarray, 
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k1, k2 = jax.random.split(key)
    H, W = source_mask.shape
    
    features = net.extract_features(obs_tensor)
    
    raw_s_logits = net.get_source_logits(features)
    masked_s_logits = jnp.where(source_mask, raw_s_logits, -1e9)
    flat_s_logits = masked_s_logits.reshape(-1)
    
    no_valid_source = ~jnp.any(source_mask)
    
    source_idx = jax.random.categorical(k1, flat_s_logits)
    selected_r = source_idx // W
    selected_c = source_idx % W
    
    s_log_probs = jax.nn.log_softmax(flat_s_logits)
    s_log_prob = s_log_probs[source_idx]
    
    raw_d_logits = net.get_direction_logits(features, selected_r, selected_c)
    tile_t_mask = target_mask[selected_r, selected_c]
    masked_d_logits = jnp.where(tile_t_mask, raw_d_logits, -1e9)
    
    dir_idx = jax.random.categorical(k2, masked_d_logits)
    d_log_probs = jax.nn.log_softmax(masked_d_logits)
    d_log_prob = d_log_probs[dir_idx]
    
    value = net.get_value(features)
    
    official_action = format_network_action(dir_idx, selected_r, selected_c)
    pass_action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    
    final_action = jnp.where(no_valid_source, pass_action, official_action)
    
    return final_action, s_log_prob, d_log_prob, value