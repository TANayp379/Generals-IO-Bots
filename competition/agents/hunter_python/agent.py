from functools import partial
import jax
import jax.numpy as jnp


GARRISON = 4  # army kept on the general; only its surplus (above 2x) is sent out

def _bfs(passable, sources):
    """Steps from `sources` to every cell over passable terrain (large = unreachable)."""
    H, W = passable.shape
    INF = jnp.int32(H * W + 5)

    def relax(_, d):
        nb = jnp.minimum(
            jnp.minimum(jnp.roll(d, 1, 0).at[0].set(INF), jnp.roll(d, -1, 0).at[-1].set(INF)),
            jnp.minimum(jnp.roll(d, 1, 1).at[:, 0].set(INF), jnp.roll(d, -1, 1).at[:, -1].set(INF)),
        )
        return jnp.where(sources, jnp.int32(0), jnp.where(passable, jnp.minimum(d, nb + 1), INF))

    return jax.lax.fori_loop(0, H * W, relax, jnp.where(sources, jnp.int32(0), INF))

def _toward(field, passable):
    """Per cell: (direction of its lowest-`field` passable neighbour, that neighbour's value)."""
    INF = jnp.int32(field.size + 7)

    def shift(arr, fill, s, ax):
        arr = jnp.roll(arr, s, ax)
        e = 0 if s == 1 else -1
        return arr.at[e, :].set(fill) if ax == 0 else arr.at[:, e].set(fill)

    vals = jnp.stack([
        jnp.where(shift(passable, False, s, ax), shift(field, INF, s, ax), INF)
        for s, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1))
    ])
    return jnp.argmin(vals, 0).astype(jnp.int32), jnp.min(vals, 0)


@jax.jit
def _hunter_logic(a, mine, generals, mountains, structures_in_fog, castles, opponent_cells, fog_cells):
    """Pure JAX function containing the core Hunter logic."""
    H, W = a.shape
    reach = jnp.int32(H * W)
    passable = ~(mountains | structures_in_fog | (castles & ~mine))
    mine_army = jnp.where(mine, a, 0)
    movable = mine & (a > 1)

    gen = mine & generals
    gen_army = jnp.sum(jnp.where(gen, a, 0))
    g = jnp.argmax(gen.reshape(-1).astype(jnp.int32))
    from_gen = _bfs(passable, gen)

    # Goal: the enemy general, else nearest enemy land, else the farthest cell to scout.
    egen = opponent_cells & generals
    enemy = opponent_cells & ~castles
    fog = fog_cells & passable & (from_gen < reach)
    open_ = passable & ~mine & (from_gen < reach)
    farthest = lambda m: m & (from_gen == jnp.max(jnp.where(m, from_gen, -1)))
    
    goal = jnp.where(jnp.any(egen), egen,
           jnp.where(jnp.any(enemy), enemy,
           jnp.where(jnp.any(fog), farthest(fog), farthest(open_))))

    to_goal = _bfs(passable, goal)
    direction, nbr = _toward(to_goal, passable)
    advances = nbr < to_goal
    dirn = direction.reshape(-1)

    egen_army = jnp.sum(jnp.where(egen, a, 0))
    kill = jnp.any(egen) & movable & (to_goal == 1) & advances & (a - 1 > egen_army)
    ki = jnp.argmax(jnp.where(kill, mine_army, -1).reshape(-1))
    feed = (gen_army >= 2 * GARRISON) & advances.reshape(-1)[g]
    fwd = movable & ~gen & advances
    ci = jnp.argmax(jnp.where(fwd, mine_army, -1).reshape(-1))

    do_kill = jnp.any(kill)
    do_feed = ~do_kill & feed
    do_conv = ~do_kill & ~do_feed & jnp.any(fwd)
    i = jnp.where(do_kill, ki, jnp.where(do_feed, g, ci))
    
    return jnp.array([~(do_kill | do_feed | do_conv), i // W, i % W, dirn[i], do_feed], dtype=jnp.int32)


class Agent:
    """Agent that garrisons its general and hunts down the enemy general to win."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W

    def act(self, obs):
        """Adapter to translate between stdio Python dataclass and JAX arrays."""
        
        # 1. Convert Python lists to JAX arrays
        type_grid = jnp.array(obs.type_grid)
        owner_grid = jnp.array(obs.owner_grid)
        a = jnp.array(obs.army_grid)

        # 2. Reconstruct the boolean masks expected by the Hunter logic based on the wire protocol
        mine = (owner_grid == 1)
        opponent_cells = (owner_grid == 2)
        generals = (type_grid == 4)
        mountains = (type_grid == 2)
        structures_in_fog = (type_grid == 5)
        castles = (type_grid == 3)
        fog_cells = (type_grid == 0)

        # 3. Execute the JIT-compiled logic
        action_array = _hunter_logic(
            a, mine, generals, mountains, structures_in_fog, 
            castles, opponent_cells, fog_cells
        )

        # 4. Convert the JAX array back to a standard Python tuple so main.py can unpack it
        return tuple(int(x) for x in action_array)