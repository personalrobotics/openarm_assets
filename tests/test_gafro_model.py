"""The decorated OpenArm model agrees with MuJoCo on what gafro computes from it.

Kinematics: each chain's tip pose against MuJoCo's tool body. Dynamics: the gravity torque and the
mass matrix against MuJoCo's. Gravcomp is off by default, so MuJoCo's own bias force is the
gravity torque a torque controller has to cancel.
"""

import mujoco
import numpy as np
import pytest

gafro = pytest.importorskip("gafro")

from openarm_assets import MODELS_DIR  # noqa: E402

PATH = MODELS_DIR / "openarm" / "generated" / "openarm_bimanual_decorated.xml"
SIDES = ("left", "right")
N_CONFIGS = 30


@pytest.fixture(scope="module")
def pair():
    if not PATH.exists():
        pytest.skip("run `python -m openarm_assets.assembly --sides bimanual` first")

    system = gafro.SystemSerialization.load(str(PATH))
    model = mujoco.MjModel.from_xml_path(str(PATH))

    # gafro orders its positions by actuator, MuJoCo by joint address: map one onto the other by name.
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, a.joint_name) for a in system.get_actuators()]

    return {
        "system": system,
        "model": model,
        "qadr": np.array([model.jnt_qposadr[j] for j in joints]),
        "dadr": np.array([model.jnt_dofadr[j] for j in joints]),
        "range": np.array([model.jnt_range[j] for j in joints]),
    }


def _configurations(pair):
    rng = np.random.default_rng(0)

    for _ in range(N_CONFIGS):
        q = rng.uniform(pair["range"][:, 0], pair["range"][:, 1])
        data = mujoco.MjData(pair["model"])
        data.qpos[pair["qadr"]] = q
        mujoco.mj_forward(pair["model"], data)
        yield q, data


def test_groups_and_task_spaces(pair):
    system = pair["system"]

    assert [g.get_name() for g in system.get_control_groups()] == ["left_arm", "right_arm"]
    assert system.get_task_space_names() == [*SIDES, "bimanual"]
    assert len(system.get_task_space("bimanual").get_joint_indices()) == 14


@pytest.mark.parametrize("side", SIDES)
def test_chain_tip_is_the_mujoco_tool_body(pair, side):
    model, system = pair["model"], pair["system"]
    tip = f"openarm_{side}_hand_tcp"
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tip)

    for q, data in _configurations(pair):
        T = np.array(system.compute_kinematic_chain_motor(tip, q).to_transformation_matrix())

        np.testing.assert_allclose(T[:3, 3], data.xpos[body], atol=1e-6)
        np.testing.assert_allclose(T[:3, :3], data.xmat[body].reshape(3, 3), atol=1e-6)


def test_gravity_torque_in_the_world_frame(pair):
    """Gravity is along the system root's -z: the arms are mounted rotated about x, so a gravity along each arm's own z would not match."""
    system, dadr = pair["system"], pair["dadr"]
    zero = np.zeros(system.get_dof())

    for q, data in _configurations(pair):
        np.testing.assert_allclose(system.compute_inverse_dynamics(q, zero, zero), data.qfrc_bias[dadr], atol=1e-6)


def test_mass_matrix(pair):
    model, system, dadr = pair["model"], pair["system"], pair["dadr"]

    for q, data in _configurations(pair):
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, full)

        np.testing.assert_allclose(system.compute_mass_matrix(q), full[np.ix_(dadr, dadr)], atol=1e-6)


def test_actuator_limits_are_the_urdfs_rated_ones(pair):
    """Torque and velocity limits of every arm joint are enactic's, from the vendored URDF."""
    from openarm_assets.assembly import urdf_limits

    model = pair["model"]
    rated = urdf_limits()

    for name, (effort, velocity) in rated.items():
        if "finger" in name or "joint" not in name:
            continue
        side, joint = name.split("_")[1], name.split("joint")[-1]
        torque = model.actuator(f"{side}_joint{joint}_torque")
        speed = model.actuator(f"{side}_joint{joint}_velocity")
        position = model.actuator(f"{side}_joint{joint}_position")

        assert torque.ctrlrange[1] == pytest.approx(effort)
        assert torque.forcerange[1] == pytest.approx(effort)
        assert speed.ctrlrange[1] == pytest.approx(velocity)
        assert position.forcerange[1] == pytest.approx(effort)


@pytest.mark.parametrize("side", SIDES)
def test_joint_force_range_is_the_rating_and_bounds_the_sum_of_the_actuators(pair, side):
    """The three actuators on a joint act together in plain MuJoCo; the joint's own range keeps the total at the rating."""
    from openarm_assets.assembly import urdf_limits

    model = pair["model"]
    rated = urdf_limits()

    for joint_index in range(1, 8):
        name = f"openarm_{side}_joint{joint_index}"
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        effort = rated[name][0]

        assert model.jnt_actfrclimited[joint]
        assert model.jnt_actfrcrange[joint][0] == pytest.approx(-effort)
        assert model.jnt_actfrcrange[joint][1] == pytest.approx(effort)

    # All three actuators of one joint at full command, in plain MuJoCo (no group disabled): the force stays at the rating.
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint3")
    effort = rated[f"openarm_{side}_joint3"][0]
    for kind, command in (("torque", 1e3), ("velocity", 1e3), ("position", 1e3)):
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_joint3_{kind}")
        data.ctrl[actuator] = np.clip(command, *model.actuator_ctrlrange[actuator])
    mujoco.mj_forward(model, data)

    assert abs(data.qfrc_actuator[model.jnt_dofadr[joint]]) <= effort + 1e-9


def test_a_robot_can_narrow_a_joint_and_never_widen_it():
    """A robot narrows the arm's range for what is around it; the range reaches the model, the position actuator and gafro's joint."""
    from openarm_assets.assembly import build_openarm

    narrowed = build_openarm(joint_ranges={"openarm_right_joint2": (-0.1745, 2.7402), "openarm_left_joint2": (-2.7402, 0.1745)}).compile()

    for name, (lower, upper) in (("openarm_right_joint2", (-0.1745, 2.7402)), ("openarm_left_joint2", (-2.7402, 0.1745))):
        joint = mujoco.mj_name2id(narrowed, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert narrowed.jnt_range[joint] == pytest.approx([lower, upper])

        side = name.split("_")[1]
        actuator = mujoco.mj_name2id(narrowed, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_joint2_position")
        assert narrowed.actuator_ctrlrange[actuator] == pytest.approx([lower, upper])

    # Other joints are the arm's.
    right1 = mujoco.mj_name2id(narrowed, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_joint1")
    assert narrowed.jnt_range[right1][1] == pytest.approx(3.490659, abs=1e-5)

    with pytest.raises(ValueError, match="wider than the arm's"):
        build_openarm(joint_ranges={"openarm_right_joint2": (-0.5, 2.7402)})
    with pytest.raises(ValueError, match="wider than the arm's"):
        build_openarm(joint_ranges={"openarm_right_joint2": (-0.1745, 3.5)})
    with pytest.raises(ValueError, match="no joint named"):
        build_openarm(joint_ranges={"openarm_right_joint9": (0.0, 1.0)})
    with pytest.raises(ValueError, match="empty"):
        build_openarm(joint_ranges={"openarm_right_joint2": (1.0, 1.0)})


def test_a_robot_can_give_its_own_ready_pose():
    """The ready keyframe is the robot's, in its joint names; the arm's own is nearly singular, and out-of-range or unknown joints are refused."""
    from openarm_assets.assembly import build_openarm

    pose = {"openarm_right_joint2": 0.7, "openarm_right_joint4": 1.8, "openarm_left_joint2": -0.7, "openarm_left_joint4": 1.8}
    model = build_openarm(ready=pose).compile()
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")

    for name, value in pose.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert model.key_qpos[key][model.jnt_qposadr[joint]] == pytest.approx(value)
    other = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_joint3")
    assert model.key_qpos[key][model.jnt_qposadr[other]] == 0.0

    # Without it the keyframe is READY_POSE, and the near-stow pose is still there as `rest`.
    from openarm_assets.assembly import READY_POSE

    default = build_openarm().compile()
    ready, rest = (mujoco.mj_name2id(default, mujoco.mjtObj.mjOBJ_KEY, name) for name in ("ready", "rest"))
    assert default.key_qpos[ready][default.jnt_qposadr[mujoco.mj_name2id(default, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_joint4")]] == pytest.approx(1.8)
    for name, value in READY_POSE.items():
        assert default.key_qpos[ready][default.jnt_qposadr[mujoco.mj_name2id(default, mujoco.mjtObj.mjOBJ_JOINT, name)]] == pytest.approx(value)
    for arm in ("left", "right"):
        for n in range(1, 8):
            address = default.jnt_qposadr[mujoco.mj_name2id(default, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{arm}_joint{n}")]
            assert default.key_qpos[rest][address] == pytest.approx(0.3 if n == 4 else 0.0)

    with pytest.raises(ValueError, match="outside the joint's range"):
        build_openarm(ready={"openarm_right_joint6": 2.0})
    with pytest.raises(ValueError, match="no joint named"):
        build_openarm(ready={"openarm_right_joint9": 0.1})


def test_the_ready_pose_is_the_right_arms_mirror_in_the_left():
    """Joints 2 and 3 flip sign between the arms (their axes and ranges are mirrored); the rest are equal."""
    from openarm_assets.assembly import READY_POSE

    for n in range(1, 8):
        left, right = READY_POSE.get(f"openarm_left_joint{n}", 0.0), READY_POSE.get(f"openarm_right_joint{n}", 0.0)
        assert left == pytest.approx(-right if n in (2, 3) else right), n


def test_the_ready_pose_is_what_a_hand_tracking_controller_can_start_from():
    """Not the straight arm: every arm joint of `ready` is inside its range with room to move, and joint 4 is clear of its lower limit."""
    from openarm_assets.assembly import build_openarm

    model = build_openarm().compile()
    ready = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    for arm in ("left", "right"):
        for n in range(1, 8):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{arm}_joint{n}")
            low, high = model.jnt_range[joint]
            q = model.key_qpos[ready][model.jnt_qposadr[joint]]
            assert low + 0.05 < q < high - 0.05 or q == 0.0, f"{arm} joint {n}: {q} is at its limit [{low}, {high}]"
    joint4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_joint4")
    assert model.key_qpos[ready][model.jnt_qposadr[joint4]] > 1.0


# -- the collision cover, as gafro reads it -------------------------------------------------------------------------------------

needs_cover = pytest.mark.skipif(not hasattr(gafro, "CollisionSphere"), reason="this gafro predates the model's collision cover")


@needs_cover
def test_the_model_carries_the_cover_gafro_reads(pair):
    system = pair["system"]

    spheres, pairs = system.get_collision_spheres(), system.get_collision_pairs()
    assert spheres and pairs
    assert len({s.name for s in spheres}) == len(spheres)
    for sphere in spheres:
        assert sphere.radius > 0
        # Every ball rides on a chain the model declares, so an objective can resolve it.
        assert system.has_kinematic_chain(sphere.chain)
    names = {s.name for s in spheres}
    assert all(p.first in names and p.second in names for p in pairs)


def _ball_centre(pair, data, sphere):
    model = pair["model"]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sphere.chain)
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ np.asarray(sphere.offset)


def _state(system, q):
    state = gafro.control.SystemState(system.get_dof())
    state.set_position(q), state.set_velocity(np.zeros(system.get_dof())), state.set_torque(np.zeros(system.get_dof()))
    return state


@needs_cover
def test_gafro_places_every_ball_where_mujoco_does(pair):
    """The cover's centres are in a carrier body's frame, and gafro's chain tip frame is the MuJoCo body frame, so the clearance of
    a pair is the same number from either side."""
    system = pair["system"]
    spheres = {s.name: s for s in system.get_collision_spheres()}
    pairs = system.get_collision_pairs()
    objective = gafro.objective.SelfCollisionObjective(system, system.get_task_space("bimanual"), 0.02)

    for q, data in _configurations(pair):
        expected = [
            np.linalg.norm(_ball_centre(pair, data, spheres[p.first]) - _ball_centre(pair, data, spheres[p.second]))
            - spheres[p.first].radius
            - spheres[p.second].radius
            for p in pairs
        ]
        np.testing.assert_allclose(objective.get_clearances(_state(system, q)), expected, atol=1e-9)


@needs_cover
def test_a_mesh_collision_is_a_ball_overlap_in_gafro(pair):
    """Configurations MuJoCo says collide, sampled where the cover was built: gafro sees a listed pair of balls overlap in each,
    except where the bodies in contact are a pair the sweep never saw (those are the sweep's limit; they have to be rare)."""
    from openarm_assets import collision_cover as covers

    model, system = pair["model"], pair["system"]
    spheres = {s.name: s for s in system.get_collision_spheres()}
    covered = {tuple(sorted((spheres[p.first].name.rsplit("_b", 1)[0], spheres[p.second].name.rsplit("_b", 1)[0]))) for p in system.get_collision_pairs()}
    objective = gafro.objective.SelfCollisionObjective(system, system.get_task_space("bimanual"), 0.0)

    rng = np.random.default_rng(21)
    data = mujoco.MjData(model)
    colliding = unseen = 0
    for _ in range(6000):
        covers._sample(model, data, rng, 0.5)
        mujoco.mj_forward(model, data)
        touching = {
            tuple(sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) for g in (c.geom1, c.geom2)))
            for c in data.contact[: data.ncon]
            if model.geom_group[c.geom1] == covers.COLLISION_GROUP and model.geom_group[c.geom2] == covers.COLLISION_GROUP and c.dist < 0
        }
        if not touching:
            continue
        colliding += 1
        clearances = objective.get_clearances(_state(system, data.qpos[pair["qadr"]]))
        if touching <= covered:
            assert clearances.min() < 0.0, f"{sorted(touching)} touch and no listed pair of balls overlaps"
        else:
            unseen += 1
    assert colliding > 20, "the sample found too few collisions to say anything"
    assert unseen <= 0.05 * colliding
