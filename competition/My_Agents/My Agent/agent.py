"""
Edit this file to implement your agent.

`Agent.act(obs)` is called once per turn. See main.py for the Observation
field docs. `act` must return a 5-tuple `(pass, row, col, direction, split)`.

This version loads a checkpoint trained by train.py and runs the policy
network deterministically (greedy) for inference. It requires:
    - model.py, policy.py, obs_adapter.py   (bundled alongside this file)
    - checkpoints/weights.pkl               (produced by train.py)

If for some reason the checkpoint or JAX/Flax aren't available at runtime,
`Agent` falls back to the simple expander heuristic so the bot still plays
a legal game instead of crashing out of the competition.
"""
import os

import jax
import jax.numpy as jnp
import jax.random as jrandom

from model import ActorCriticNet
from obs_adapter import wire_obs_to_tensor, compute_valid_move_mask
import policy

# A no-op action — used as an ultimate fallback.
PASS = (1, 0, 0, 0, 0)

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "weights.pkl")


class _ExpanderFallback:
    """Same heuristic as the original starter agent — used only if the
    trained checkpoint can't be loaded, so the bot never forfeits by crashing."""

    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    @staticmethod
    def _is_passable(t):
        return t != 2 and t != 5

    def act(self, obs):
        best_score, best_move, first_valid = -1.0, None, None
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue
                for d, (dr, dc) in enumerate(self.DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                        continue
                    if not self._is_passable(obs.type_grid[nr][nc]):
                        continue
                    move = (0, r, c, d, 0)
                    if first_valid is None:
                        first_valid = move
                    dest_owner = obs.owner_grid[nr][nc]
                    dest_army = obs.army_grid[nr][nc]
                    if src_army <= dest_army + 1:
                        continue
                    is_opp = dest_owner == 2
                    dest_type = obs.type_grid[nr][nc]
                    is_visible_neutral = (dest_owner == 0) and dest_type not in (0, 5)
                    is_expansion = is_opp or is_visible_neutral
                    score = float(src_army)
                    if is_expansion:
                        score *= 10.0
                    if is_opp:
                        score *= 2.0
                    if score > best_score:
                        best_score, best_move = score, move
        return best_move or first_valid or PASS


class Agent:
    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.fallback = _ExpanderFallback()
        self._key = jrandom.PRNGKey(0)  # unused (deterministic inference), kept for API symmetry with policy.act

        self.model = None
        self.params = None
        try:
            self._load_checkpoint()
            # Compile once now (H, W known), rather than on the first act() call,
            # so we don't eat JIT latency against the game clock on turn 1.
            dummy_obs = jnp.zeros((H, W, 14), dtype=jnp.float32)
            dummy_mask = jnp.zeros((H, W, 4), dtype=bool)
            self._infer = jax.jit(
                lambda p, o, m, k: policy.act(self.model, p, o, m, k, deterministic=True)
            )
            self._infer(self.params, dummy_obs, dummy_mask, self._key)
        except Exception as e:  # noqa: BLE001 — never let a load/compile error crash the match
            print(f"[agent] falling back to heuristic ({e})", flush=True)
            self.model = None

    def _load_checkpoint(self):
        import pickle
        with open(CHECKPOINT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        self.model = ActorCriticNet(features=ckpt["features"], num_blocks=ckpt["num_blocks"])
        self.params = ckpt["params"]

    def act(self, obs):
        if self.model is None:
            return self.fallback.act(obs)

        try:
            obs_tensor = jnp.asarray(wire_obs_to_tensor(obs))
            mask = jnp.asarray(compute_valid_move_mask(obs))
            action, _, _ = self._infer(self.params, obs_tensor, mask, self._key)
            return policy.action_to_tuple(action)
        except Exception as e:  # noqa: BLE001 — a bad turn shouldn't end the match
            print(f"[agent] inference error, passing this turn ({e})", flush=True)
            return self.fallback.act(obs)