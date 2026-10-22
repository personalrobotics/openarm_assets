# openarm_assets/src/openarm_assets/collision_cover.py
"""A ball cover of the robot's collision meshes, and the pairs of balls that ever collide.

gafro keeps one part of the robot clear of another by balls rather than meshes (its SelfCollisionObjective), so the
robot has to say what its balls are and which pairs of them can touch. Both are derived from the collision geometry
offline and written into the model beside the chains and task spaces (see `add_collision_cover`), so they live with the
robot and not in each controller's configuration.

The cover is built to be conservative: every triangle of a link's collision mesh lies inside one of its balls, so a
mesh contact is a ball overlap. MuJoCo collides the convex hull of a mesh, which can stick out of the mesh by a little;
`inflate` is the allowance for that, and `check_conservative` is the sweep that shows whether it is enough.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

# The geom group the vendored model puts collision meshes in (visual meshes are in group 2).
COLLISION_GROUP = 3


@dataclass(frozen=True)
class Ball:
    """A ball that covers `body`'s collision geometry and is carried by `carrier`, its centre in the carrier's frame.

    The carrier is the body itself unless the body is rigid with the root (no moving joint between them): such a body has no
    chain of its own that moves it, so the root carries its balls, at the same place in the world."""

    name: str
    body: str
    center: tuple[float, float, float]
    radius: float
    carrier: str = ""

    def __post_init__(self):
        if not self.carrier:
            object.__setattr__(self, "carrier", self.body)


@dataclass
class Cover:
    """The balls, the pairs of them that can collide, and what the sweep found out along the way."""

    balls: list[Ball]
    pairs: list[tuple[str, str]]
    # Body pairs whose collision geometry overlaps in (nearly) every sampled configuration: the vendor meshes of
    # link5 and link7, the two fingers of a hand. They are not collisions to avoid, so they are left out.
    permanent: list[tuple[str, str]] = field(default_factory=list)
    # Body pairs that collided in the sweep, each with the number of samples it did so in.
    colliding: dict[tuple[str, str], int] = field(default_factory=dict)
    # Body pairs whose meshes came within the probe margin (a collision is within it), likewise counted.
    near: dict[tuple[str, str], int] = field(default_factory=dict)


def _rotation(quaternion) -> np.ndarray:
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def _body_name(model: mujoco.MjModel, body: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)


def collision_triangles(model: mujoco.MjModel) -> dict[int, np.ndarray]:
    """Each body's collision mesh triangles in its own frame, as an (n, 3, 3) array of vertex positions."""
    triangles: dict[int, list[np.ndarray]] = {}
    for geom in range(model.ngeom):
        if model.geom_group[geom] != COLLISION_GROUP or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        faces = model.mesh_face[model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
        in_body = vertices @ _rotation(model.geom_quat[geom]).T + model.geom_pos[geom]
        triangles.setdefault(int(model.geom_bodyid[geom]), []).append(in_body[faces])
    return {body: np.concatenate(parts) for body, parts in triangles.items()}


def _enclose(points: np.ndarray, iterations: int = 200) -> tuple[np.ndarray, float]:
    """A ball around `points` that is close to the smallest one (Badoiu-Clarkson); the radius is exact for the centre."""
    center = points.mean(axis=0)
    for step in range(1, iterations + 1):
        farthest = points[np.argmax(np.linalg.norm(points - center, axis=1))]
        center = center + (farthest - center) / (step + 1)
    return center, float(np.linalg.norm(points - center, axis=1).max())


def fit_balls(triangles: np.ndarray, count: int, seed: int = 0) -> list[tuple[np.ndarray, float]]:
    """`count` balls whose union contains every triangle: triangles are clustered by their centroids, and a cluster's ball
    encloses all the vertices of its triangles, so no triangle straddles two balls' gap."""
    centroids = triangles.mean(axis=1)
    count = max(1, min(count, len(triangles)))
    rng = np.random.default_rng(seed)
    # k-means++ seeding, then Lloyd's iterations; deterministic for a seed.
    centers = [centroids[rng.integers(len(centroids))]]
    for _ in range(count - 1):
        nearest = np.min([np.sum((centroids - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(centroids[np.argmax(nearest)])
    centers = np.array(centers)
    for _ in range(30):
        owner = np.argmin(((centroids[:, None, :] - centers[None]) ** 2).sum(axis=2), axis=1)
        moved = np.array([centroids[owner == k].mean(axis=0) if np.any(owner == k) else centers[k] for k in range(count)])
        if np.allclose(moved, centers):
            break
        centers = moved
    owner = np.argmin(((centroids[:, None, :] - centers[None]) ** 2).sum(axis=2), axis=1)
    return [_enclose(triangles[owner == k].reshape(-1, 3)) for k in range(count) if np.any(owner == k)]


def _root_body(model: mujoco.MjModel) -> int:
    """The body gafro takes as the base of the mechanism: the one the world holds."""
    roots = [body for body in range(1, model.nbody) if model.body_parentid[body] == 0]
    if len(roots) != 1:
        raise ValueError(f"expected one body under the world, found {len(roots)}")
    return roots[0]


def _is_static(model: mujoco.MjModel, body: int, root: int) -> bool:
    """Whether no joint lies between `body` and the root, so it never moves relative to it."""
    while body != root:
        if model.body_jntnum[body]:
            return False
        body = int(model.body_parentid[body])
    return True


def balls_for(model: mujoco.MjModel, target_length: float = 0.1, max_balls: int = 5, inflate: float = 0.003, seed: int = 0) -> list[Ball]:
    """The cover of every body that has collision geometry: about one ball per `target_length` metres of the body's
    longest extent, at most `max_balls`, each fattened by `inflate`. A body rigid with the root has its balls carried by
    the root (see `Ball`)."""
    root = _root_body(model)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    root_rotation, root_position = data.xmat[root].reshape(3, 3), data.xpos[root]
    balls = []
    for body, triangles in sorted(collision_triangles(model).items()):
        carrier = root if _is_static(model, body, root) else body
        extent = float(np.ptp(triangles.reshape(-1, 3), axis=0).max())
        if carrier != body:
            rotation, position = data.xmat[body].reshape(3, 3), data.xpos[body]
            triangles = ((triangles @ rotation.T + position) - root_position) @ root_rotation
        count = int(np.clip(np.ceil(extent / target_length), 1, max_balls))
        for index, (center, radius) in enumerate(fit_balls(triangles, count, seed)):
            balls.append(
                Ball(
                    f"{_body_name(model, body)}_b{index}",
                    _body_name(model, body),
                    tuple(float(x) for x in center),
                    radius + inflate,
                    _body_name(model, carrier),
                )
            )
    return balls


def _may_collide(model: mujoco.MjModel, first: int, second: int) -> bool:
    """Whether MuJoCo would ever test the two bodies: not a body with itself, not parent and child, not excluded."""
    if first == second or model.body_parentid[first] == second or model.body_parentid[second] == first:
        return False
    for index in range(model.nexclude):
        signature = int(model.exclude_signature[index])
        if {signature >> 16, signature & 0xFFFF} == {first + 1, second + 1} or {signature >> 16, signature & 0xFFFF} == {first, second}:
            return False
    return True


def _sample(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, span: float) -> None:
    """Uniform joint angles over `span` of each limited joint's range, centred on the ready pose; the fingers stay put."""
    ready = model.key_qpos[0] if model.nkey else model.qpos0
    for joint in range(model.njnt):
        address = model.jnt_qposadr[joint]
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[joint]:
            data.qpos[address] = ready[address]
            continue
        low, high = model.jnt_range[joint]
        half = 0.5 * span * (high - low)
        centre = np.clip(ready[address], low + half, high - half)
        data.qpos[address] = rng.uniform(centre - half, centre + half)


def _center(model: mujoco.MjModel, data: mujoco.MjData, ball: Ball) -> np.ndarray:
    """Where the ball is in the world, with `data` already through kinematics."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ball.carrier)
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ np.array(ball.center)


def _smallest_cover(events: dict[frozenset, int], redundancy: int) -> list[tuple[str, str]]:
    """Few ball pairs such that every sampled event has `redundancy` of them (or all it has, if fewer): greedily, the pair
    seen in the most events that still want one. Every pair that was ever within the margin would also do, at the price of
    a row each; a single pair per event covers the sample and nothing past it, which is what the slack is for."""
    wanted = {seen: min(redundancy, len(seen)) for seen in events}
    chosen: set[tuple[str, str]] = set()
    while any(wanted.values()):
        votes: dict[tuple[str, str], int] = {}
        for seen, count in events.items():
            if wanted[seen] > 0:
                for ball_pair in seen - chosen:
                    votes[ball_pair] = votes.get(ball_pair, 0) + count
        best = max(sorted(votes), key=votes.__getitem__)
        chosen.add(best)
        for seen in events:
            if best in seen and wanted[seen] > 0:
                wanted[seen] -= 1
    return sorted(chosen)


def sweep(
    model: mujoco.MjModel,
    balls: list[Ball],
    samples: int = 30000,
    span: float = 0.5,
    probe_margin: float = 0.02,
    permanent_fraction: float = 0.99,
    redundancy: int = 2,
    seed: int = 0,
) -> Cover:
    """Sample configurations and keep the pairs of balls that overlap whenever their bodies' collision meshes do.

    A pair of bodies is worth avoiding if their meshes come within `probe_margin` of each other in some sample but overlap in
    not (nearly) all of them; a pair of balls goes on the list if, in a sample where its bodies are that close, the two balls
    are within `probe_margin` of each other too. Counting near misses as well as contacts matters because contacts can be
    rare -- a hand folding back onto the base is a few in a thousand samples -- and a pair never seen colliding is a pair
    the list would not hold. Every mesh collision of the sweep has a listed pair of balls that sees it, which
    `check_conservative` re-measures on fresh samples.

    `span` is the fraction of each joint's range sampled, centred on the ready pose, and it is the one choice that sets how
    long the list is: half the range gives a few dozen pairs, the whole range about three times as many, and a controller pays
    for every row (a stage of 80 rows costs about 1.3 ms, of 280 about 8 ms). The cover sees the collisions the sweep
    does, so a collision outside the sampled region, or a pair of bodies that never came near in it, is not listed."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    by_body: dict[int, list[Ball]] = {}
    for ball in balls:
        by_body.setdefault(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ball.body), []).append(ball)
    geoms: dict[int, list[int]] = {}
    for geom in range(model.ngeom):
        if model.geom_group[geom] == COLLISION_GROUP and model.geom_bodyid[geom] in by_body:
            geoms.setdefault(int(model.geom_bodyid[geom]), []).append(geom)

    candidates = [(a, b) for a in sorted(by_body) for b in sorted(by_body) if a < b and _may_collide(model, a, b)]
    colliding = {pair: 0 for pair in candidates}
    near = {pair: 0 for pair in candidates}
    # For each body pair, each distinct set of ball pairs that were within the probe margin at once, and how often.
    events: dict[tuple[int, int], dict[frozenset, int]] = {pair: {} for pair in candidates}
    fromto = np.zeros(6)

    for _ in range(samples):
        _sample(model, data, rng, span)
        mujoco.mj_kinematics(model, data)
        centres = None
        for pair in candidates:
            distance = min(mujoco.mj_geomDistance(model, data, g1, g2, probe_margin, fromto) for g1 in geoms[pair[0]] for g2 in geoms[pair[1]])
            if distance >= probe_margin:
                continue
            near[pair] += 1
            colliding[pair] += distance < 0.0
            if centres is None:
                centres = {ball.name: _center(model, data, ball) for group in by_body.values() for ball in group}
            seen = frozenset(
                (first.name, second.name)
                for first in by_body[pair[0]]
                for second in by_body[pair[1]]
                if np.linalg.norm(centres[first.name] - centres[second.name]) - first.radius - second.radius < probe_margin
            )
            events[pair][seen] = events[pair].get(seen, 0) + 1

    names = {body: _body_name(model, body) for body in by_body}
    permanent = [(names[a], names[b]) for (a, b), count in colliding.items() if count >= permanent_fraction * samples]
    live = [pair for pair in candidates if near[pair] > 0 and colliding[pair] < permanent_fraction * samples]
    pairs = [first_second for pair in live for first_second in _smallest_cover(events[pair], redundancy)]
    return Cover(
        balls=balls,
        pairs=pairs,
        permanent=permanent,
        colliding={(names[a], names[b]): count for (a, b), count in colliding.items() if count},
        near={(names[a], names[b]): count for (a, b), count in near.items() if count},
    )


def check_conservative(model: mujoco.MjModel, cover: Cover, samples: int = 1000, span: float = 0.5, seed: int = 1) -> list[tuple[int, str, str]]:
    """Fresh samples (another seed than the sweep's) in which two bodies' meshes collide and no listed pair of balls of
    those bodies overlaps -- the cover missing a collision. Empty when the cover is conservative over these samples."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    ball = {b.name: b for b in cover.balls}
    # Body pairs are keyed by their sorted names: the cover lists them in body order, this walks them alphabetically.
    permanent = {tuple(sorted(pair)) for pair in cover.permanent}
    listed: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for first, second in cover.pairs:
        listed.setdefault(tuple(sorted((ball[first].body, ball[second].body))), []).append((first, second))
    geoms: dict[str, list[int]] = {}
    for geom in range(model.ngeom):
        if model.geom_group[geom] == COLLISION_GROUP:
            geoms.setdefault(_body_name(model, int(model.geom_bodyid[geom])), []).append(geom)
    bodies = sorted({b.body for b in cover.balls})
    fromto = np.zeros(6)
    missed = []
    for index in range(samples):
        _sample(model, data, rng, span)
        mujoco.mj_kinematics(model, data)

        def centre(name):
            return _center(model, data, ball[name])

        for i, first in enumerate(bodies):
            for second in bodies[i + 1 :]:
                a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, first)
                b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, second)
                if (first, second) in permanent or not _may_collide(model, a_id, b_id):
                    continue
                if not any(mujoco.mj_geomDistance(model, data, g1, g2, 0.0, fromto) < 0.0 for g1 in geoms[first] for g2 in geoms[second]):
                    continue
                seen = any(
                    np.linalg.norm(centre(x) - centre(y)) - ball[x].radius - ball[y].radius < 0.0 for x, y in listed.get((first, second), [])
                )
                if not seen:
                    missed.append((index, first, second))
    return missed


def generate(model: mujoco.MjModel, **options) -> Cover:
    """The cover of `model` and its pair list: `balls_for` then `sweep`."""
    sweep_keys = ("samples", "span", "probe_margin", "permanent_fraction", "redundancy")
    ball_options = {k: v for k, v in options.items() if k not in sweep_keys}
    sweep_options = {k: v for k, v in options.items() if k in sweep_keys}
    return sweep(model, balls_for(model, **ball_options), seed=options.get("seed", 0), **sweep_options)
