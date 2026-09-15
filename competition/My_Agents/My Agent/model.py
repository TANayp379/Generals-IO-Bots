"""
Shared actor-critic network.

Imported by both train.py (training, self-play PPO) and agent.py (inference
inside the competition harness). Keeping it in one file guarantees the
architecture used at inference time exactly matches what was trained.

Input tensor layout: (H, W, C) or (N, H, W, C), channels-last, float32.
The 14 channels match generals.core.observation.Observation.as_tensor()
(with axes moved so channel is last instead of first):

    0: armies              5: owned_cells         10: owned_army_count
    1: generals             6: opponent_cells      11: opponent_land_count
    2: castles               7: fog_cells          12: opponent_army_count
    3: mountains             8: structures_in_fog  13: timestep
    4: neutral_cells         9: owned_land_count

Fully convolutional -> works for any board size without retraining/reshaping,
which matters since the training env supports variable grid sizes and the
competition board size is only known at the player/H/W handshake.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

NUM_CHANNELS_IN = 14
NUM_DIRECTIONS = 4


class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        h = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        h = nn.relu(h)
        h = nn.Conv(self.features, (3, 3), padding="SAME")(h)
        return nn.relu(h + residual)


class ActorCriticNet(nn.Module):
    """
    Fully-convolutional actor-critic.

    Outputs (all before masking/softmax, i.e. raw logits):
        source_logits: (..., H, W)     which owned cell to move from
        dir_logits:    (..., H, W, 4)  which direction, conditioned on source
        pass_logit:    (..., )         scalar, pass-vs-move
        split_logit:   (..., )         scalar, split-vs-move-most
        value:         (..., )         scalar state value estimate

    `dir_logits` is produced for every cell (not just the chosen source) so
    that, at the source cell finally selected, the 4 direction logits can be
    masked and sampled independently. This keeps the whole head convolutional
    with no dynamic gather-then-dense step.
    """

    features: int = 64
    num_blocks: int = 6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        h = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        h = nn.relu(h)
        for _ in range(self.num_blocks):
            h = ResidualBlock(self.features)(h)

        source_logits = nn.Conv(1, (1, 1))(h)[..., 0]        # (..., H, W)
        dir_logits = nn.Conv(NUM_DIRECTIONS, (1, 1))(h)       # (..., H, W, 4)

        pooled = jnp.mean(h, axis=(-3, -2))                   # (..., features)
        pass_logit = nn.Dense(1)(pooled)[..., 0]
        split_logit = nn.Dense(1)(pooled)[..., 0]
        value = nn.Dense(1)(pooled)[..., 0]

        return source_logits, dir_logits, pass_logit, split_logit, value