# belief_model.py
"""
Phase 2 -- BeliefNet: pure supervised reconstruction network (Option A).

Takes the SAME stacked input tensor the policy network consumes (a
STACK_SIZE x NUM_BASE_CHANNELS frame stack built from the foggy
Observation, i.e. shape (TOTAL_IN_CHANNELS, H, W)) and predicts only the
channels that fog actually hides:

    0: armies          (true army count, same normalization as build_frame_tensor)
    1: generals         (true general mask)
    2: castles          (true castle mask)
    3: opponent_cells   (true opponent-ownership mask)

Why only these four (matching build_frame_tensor's channel indices 0, 1,
2, 6): under fog, owned_cells/neutral_cells/mountains-once-revealed and
all positional/scalar channels (land/army counts, timestep) are either
already exactly known to the agent or purely derivable -- BeliefNet gains
nothing by re-predicting them. What's actually hidden:
  - armies: game.get_observation zeroes army counts on every invisible
    cell (armies * visible), so the true count under both fog_cells AND
    structures_in_fog is unknown.
  - generals: masked the same way (generals * visible); note a hidden
    general does NOT show up in structures_in_fog (that mask is only
    mountains | castles) -- it just looks like ordinary fog_cells, which
    is exactly the case this channel needs to cover.
  - castles: masked the same way; a hidden castle shows only as
    structures_in_fog (ambiguous with mountain) or as plain fog_cells,
    never as `castles=True`.
  - opponent_cells: masked the same way -- an invisible enemy-owned tile
    reads as unowned/neutral until reconstructed.
Since all four go to zero/false under the exact same condition (any
invisible cell, i.e. fog_cells | structures_in_fog), one shared loss mask
covers all four -- see belief_loss_fn.

No PPO, no Oracle, no RL loss anywhere in this file -- belief_loss_fn is
plain supervised regression/classification against the true full-info
frame (built via generals.core.game.get_full_observation +
build_frame_tensor), masked to only the tiles that were actually hidden
at that step. Training BeliefNet has no ordering dependency on the Oracle
(train.py) -- they can run in either order or in parallel, as
belief_train.py's docstring already notes.

At inference (Phase 3, not implemented in this file -- see
splice_belief_reconstruction below for the integration point): foggy
stacked obs -> BeliefNet.reconstruct_probs -> 4 channels -> splice into a
copy of the foggy 14-channel frame (replacing indices 0, 1, 2, 6) -> feed
into the frozen Oracle exactly as if it were real full_observation input.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from official_wrapper import TOTAL_IN_CHANNELS

# Indices into the 14-channel build_frame_tensor layout that BeliefNet
# reconstructs. Must match build_frame_tensor's channel order exactly --
# if that function's channel order ever changes, update these too.
ARMY_CH = 0
GENERALS_CH = 1
CASTLES_CH = 2
OPPONENT_CH = 6
NUM_BELIEF_OUTPUTS = 4  # army, generals, castles, opponent_cells


class BeliefResBlock(eqx.Module):
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


class BeliefNet(eqx.Module):
    """Predicts the 4 fog-hidden channels from a stacked foggy frame.

    Deliberately a *smaller* network than ActorCriticNet (this is a
    reconstruction task on local+neighborhood texture, not a policy) --
    hidden_dim/num_blocks are separate knobs, tune independently.
    """
    conv_in: eqx.nn.Conv2d
    res_blocks: list
    army_head: eqx.nn.Conv2d   # (1, H, W), linear regression output
    mask_head: eqx.nn.Conv2d   # (3, H, W), logits: generals, castles, opponent_cells

    def __init__(
        self,
        in_channels: int = TOTAL_IN_CHANNELS,
        hidden_dim: int = 96,
        num_blocks: int = 4,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 3 + num_blocks)
        self.conv_in = eqx.nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, key=keys[0])
        self.res_blocks = [BeliefResBlock(hidden_dim, k) for k in keys[1:1 + num_blocks]]
        self.army_head = eqx.nn.Conv2d(hidden_dim, 1, kernel_size=1, key=keys[-2])
        self.mask_head = eqx.nn.Conv2d(hidden_dim, 3, kernel_size=1, key=keys[-1])

    def _features(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.conv_in(x))
        for block in self.res_blocks:
            x = block(x)
        return x

    def __call__(self, stacked_obs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            stacked_obs: (TOTAL_IN_CHANNELS, H, W) foggy frame stack --
                same tensor the policy network consumes.
        Returns:
            (4, H, W): [army (raw regression scale), generals_logit,
            castles_logit, opponent_logit]. Logits, not probabilities --
            kept raw so belief_loss_fn can use a numerically-stable
            BCE-with-logits. Use reconstruct_probs() for sigmoided output.
        """
        feats = self._features(stacked_obs)
        army = self.army_head(feats)
        mask_logits = self.mask_head(feats)
        return jnp.concatenate([army, mask_logits], axis=0)

    def reconstruct_probs(self, stacked_obs: jnp.ndarray) -> jnp.ndarray:
        """Inference convenience: (4, H, W) with mask channels sigmoided into [0, 1]."""
        raw = self(stacked_obs)
        army = raw[0:1]
        probs = jax.nn.sigmoid(raw[1:4])
        return jnp.concatenate([army, probs], axis=0)


def belief_loss_fn(
    model: BeliefNet,
    stacked_obs_batch: jnp.ndarray,   # (B, TOTAL_IN_CHANNELS, H, W)
    target_frame_batch: jnp.ndarray,  # (B, 14, H, W) -- full build_frame_tensor output
    fog_mask_batch: jnp.ndarray,      # (B, H, W) bool -- fog_cells | structures_in_fog
    army_weight: float = 1.0,
    mask_weight: float = 1.0,
) -> jnp.ndarray:
    """
    Masked reconstruction loss, hidden tiles only (visible tiles are
    already exactly known, so they'd contribute pure noise to the
    gradient if included).

    - army: MSE. It's the one genuinely continuous channel here
      (normalized count), so squared error is the right loss.
    - generals / castles / opponent_cells: BCE-from-logits. These are
      0.0/1.0 boolean channels in build_frame_tensor's own encoding, and
      cross-entropy is the correct loss for a boolean target -- MSE would
      still train but gives worse-calibrated, vanishing gradients near
      the extremes.

    Both terms are normalized by the hidden-tile count (not H*W*B), so
    the loss doesn't shrink just because most of a large board is
    visible early in an episode.
    """
    preds = jax.vmap(model)(stacked_obs_batch)  # (B, 4, H, W)

    target_army = target_frame_batch[:, ARMY_CH]
    target_generals = target_frame_batch[:, GENERALS_CH]
    target_castles = target_frame_batch[:, CASTLES_CH]
    target_opponent = target_frame_batch[:, OPPONENT_CH]

    fog = fog_mask_batch.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(fog), 1.0)

    army_sq_err = jnp.square(preds[:, 0] - target_army) * fog
    army_loss = jnp.sum(army_sq_err) / denom

    def masked_bce(logits, target):
        bce = jnp.maximum(logits, 0) - logits * target + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.sum(bce * fog) / denom

    generals_loss = masked_bce(preds[:, 1], target_generals)
    castles_loss = masked_bce(preds[:, 2], target_castles)
    opponent_loss = masked_bce(preds[:, 3], target_opponent)
    mask_loss = (generals_loss + castles_loss + opponent_loss) / 3.0

    return army_weight * army_loss + mask_weight * mask_loss


def splice_belief_reconstruction(
    foggy_frame: jnp.ndarray,       # (NUM_BASE_CHANNELS, H, W) -- ONE unstacked frame
    belief_probs: jnp.ndarray,      # (4, H, W) from BeliefNet.reconstruct_probs
) -> jnp.ndarray:
    """
    Phase-3 integration point: overwrite the 4 hidden channels of a single
    foggy frame with BeliefNet's reconstruction, leaving the other 10
    channels (owned_cells, neutral_cells, mountains, fog_cells,
    structures_in_fog, land/army scalars, timestep) untouched since those
    are already exactly correct.

    Intended use in a Phase-3 agent: build the foggy frame as usual (same
    as agent.py today), run BeliefNet on the *stacked* version to get
    belief_probs, splice it into the *current* (most recent, unstacked)
    frame before it goes into the frame buffer that feeds the frozen
    Oracle. Only the most recent frame needs splicing per step -- once
    spliced, it becomes part of history for subsequent stacked frames
    naturally as the buffer rolls forward.
    """
    out = foggy_frame
    out = out.at[ARMY_CH].set(belief_probs[0])
    out = out.at[GENERALS_CH].set(belief_probs[1])
    out = out.at[CASTLES_CH].set(belief_probs[2])
    out = out.at[OPPONENT_CH].set(belief_probs[3])
    return out