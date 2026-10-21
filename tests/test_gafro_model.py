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
    assert system.get_task_space_names() == list(SIDES)


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
