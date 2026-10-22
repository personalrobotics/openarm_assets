"""The ball cover of the collision meshes, and the pairs of balls that collide: derived, conservative, and written into the model."""

import mujoco
import numpy as np
import pytest

from openarm_assets import collision_cover as covers
from openarm_assets.assembly import build_openarm


@pytest.fixture(scope="module")
def model():
    return build_openarm(collision_cover=False).compile()


@pytest.fixture(scope="module")
def cover(model):
    return covers.generate(model)


def test_every_point_of_every_collision_surface_lies_inside_a_ball_of_its_body(model, cover):
    """The union of a body's balls contains its collision surface: its triangles, split finely, have every vertex and centre inside one."""
    triangles = covers.collision_triangles(model)
    for body, mesh in triangles.items():
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        balls = [ball for ball in cover.balls if ball.body == name]
        assert balls, f"{name} has collision geometry and no ball"
        centres = np.array([ball.center for ball in balls])
        radii = np.array([ball.radius for ball in balls])
        # A ball's centre is in its carrier's frame; a body rigid with the root is carried by it.
        carrier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, balls[0].carrier)
        assert {ball.carrier for ball in balls} == {balls[0].carrier}
        if carrier != body:
            data = mujoco.MjData(model)
            mujoco.mj_kinematics(model, data)
            world = mesh @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            mesh = (world - data.xpos[carrier]) @ data.xmat[carrier].reshape(3, 3)
        fine = covers.subdivide(mesh, 0.02)
        points = np.concatenate([mesh.reshape(-1, 3), mesh.mean(axis=1), fine.reshape(-1, 3), fine.mean(axis=1)])
        inside = (np.linalg.norm(points[:, None, :] - centres[None], axis=2) <= radii[None]).any(axis=1)
        assert inside.all(), f"{name}: {np.count_nonzero(~inside)} of {len(points)} surface points outside every ball"


def test_balls_are_named_once_and_few(model, cover):
    names = [ball.name for ball in cover.balls]
    assert len(set(names)) == len(names)
    assert all(ball.radius > 0 and " " not in ball.name for ball in cover.balls)
    bodies = {ball.body for ball in cover.balls}
    assert len(cover.balls) <= 8 * len(bodies)


def test_the_pairs_are_the_ones_that_collide_and_none_that_cannot(model, cover):
    ball = {b.name: b for b in cover.balls}
    assert cover.pairs and len(set(cover.pairs)) == len(cover.pairs)
    for first, second in cover.pairs:
        a, b = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ball[n].body) for n in (first, second))
        assert covers._may_collide(model, a, b), f"{first}, {second}: MuJoCo never tests these bodies"
        assert (ball[first].body, ball[second].body) in cover.near, f"{first}, {second}: their bodies never came near each other"
    # A short list, not every pair of balls.
    assert len(cover.pairs) < len(cover.balls) * (len(cover.balls) - 1) / 4


def test_the_cover_sees_every_mesh_collision_between_bodies_the_sweep_saw_near_each_other(model, cover):
    """A listed body pair has no collision of fresh samples that its balls miss. A collision between a pair of bodies the sweep
    never saw near each other cannot be listed, and is the sweep's limit rather than the cover's: it has to be rare."""
    samples = 3000
    missed = covers.check_conservative(model, cover, samples=samples, seed=7)
    known = {tuple(sorted(pair)) for pair in cover.near}
    assert [m for m in missed if tuple(sorted(m[1:])) in known] == []
    assert len({index for index, _, _ in missed}) < 0.005 * samples


def test_the_check_can_fail(model, cover):
    """A cover with no pairs misses every collision: the check is not a rubber stamp."""
    blind = covers.Cover(balls=cover.balls, pairs=[], permanent=cover.permanent, colliding=cover.colliding, near=cover.near)
    missed = covers.check_conservative(model, blind, samples=2000, seed=7)
    assert missed, "no collision in 1000 samples, so the check above proved nothing"


def test_the_sweep_finds_the_collisions_mujoco_does(model):
    """The same samples through MuJoCo's own contact detection: both say a sample collides, or neither does."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(3)
    bodies = sorted({int(model.geom_bodyid[g]) for g in range(model.ngeom) if model.geom_group[g] == covers.COLLISION_GROUP})
    geoms = {b: [g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_group[g] == covers.COLLISION_GROUP] for b in bodies}
    fromto = np.zeros(6)
    colliding = 0
    for _ in range(400):
        covers._sample(model, data, rng, 1.0)
        mujoco.mj_forward(model, data)
        contact = any(
            model.geom_group[c.geom1] == covers.COLLISION_GROUP and model.geom_group[c.geom2] == covers.COLLISION_GROUP and c.dist < 0
            for c in data.contact[: data.ncon]
        )
        distance = any(
            mujoco.mj_geomDistance(model, data, g1, g2, 0.0, fromto) < 0.0
            for i, a in enumerate(bodies)
            for b in bodies[i + 1 :]
            if covers._may_collide(model, a, b)
            for g1 in geoms[a]
            for g2 in geoms[b]
        )
        assert contact == distance
        colliding += contact
    assert colliding > 0


def test_the_model_carries_the_cover_it_was_built_with():
    spec = build_openarm(collision_cover=True, collision_options={"samples": 3000})
    texts = {t.name: t.data for t in spec.texts}
    spheres = {name.removeprefix("gafro_collision_sphere:"): data.split() for name, data in texts.items() if name.startswith("gafro_collision_sphere:")}
    pairs = [data.split() for name, data in texts.items() if name.startswith("gafro_collision_pair:")]
    assert spheres and pairs
    for body, *numbers in spheres.values():
        assert f"gafro_kinematic_chain:{body}" in texts, "a ball's body is a chain the model declares"
        assert len(numbers) == 4 and float(numbers[3]) > 0
    for first, second in pairs:
        assert first in spheres and second in spheres


def test_it_can_be_left_out():
    spec = build_openarm(collision_cover=False)
    assert not [t for t in spec.texts if t.name.startswith("gafro_collision")]


def test_subdividing_keeps_the_surface_and_shortens_every_edge():
    rng = np.random.default_rng(0)
    triangles = rng.uniform(-1.0, 1.0, size=(20, 3, 3))
    triangles[0] = [[0, 0, 0], [0.01, 0, 0], [0.005, 0.7, 0]]  # a sliver down the side of a pole

    def area(t):
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()

    pieces = covers.subdivide(triangles, 0.1)
    edges = np.linalg.norm(pieces[:, (1, 2, 0)] - pieces, axis=2)
    assert edges.max() <= 0.1 + 1e-12
    assert area(pieces) == pytest.approx(area(triangles))


def test_no_ball_is_large_because_one_triangle_is_long(model, cover):
    """A thin pole is covered by balls the size of the pole's width, not of its length (a strip of triangles down its side)."""
    assert max(ball.radius for ball in cover.balls) < 0.12


def test_nothing_in_the_list_overlaps_at_the_ready_pose(model, cover):
    """A guard on a pair that overlaps at rest fires at every pose and drives the arm away from whatever it was asked to do."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    ball = {b.name: b for b in cover.balls}
    clearance = [
        np.linalg.norm(covers._center(model, data, ball[first]) - covers._center(model, data, ball[second])) - ball[first].radius - ball[second].radius
        for first, second in cover.pairs
    ]
    assert min(clearance) > 0.0


def test_rigidly_adjacent_bodies_are_left_out(model, cover):
    """A link and the hand and fingers beyond it overlap at every pose; that is the model's construction, not a collision."""
    assert ("openarm_left_link7", "openarm_left_hand") in cover.permanent
    listed = {tuple(sorted((ball[first].body, ball[second].body))) for ball in [{b.name: b for b in cover.balls}] for first, second in cover.pairs}
    assert not (listed & {tuple(sorted(pair)) for pair in cover.permanent})
