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


@pytest.mark.xfail(
    strict=True,
    reason="gafro applies gravity along -z of the chain's base link, not of the system root; the OpenArm "
    "arms are mounted rotated about x, so its gravity torque is for a differently oriented robot. "
    "Passes once gafro takes gravity in the root frame.",
)
def test_gravity_torque_in_the_world_frame(pair):
    system, dadr = pair["system"], pair["dadr"]
    zero = np.zeros(system.get_dof())

    for q, data in _configurations(pair):
        np.testing.assert_allclose(system.compute_inverse_dynamics(q, zero, zero), data.qfrc_bias[dadr], atol=1e-6)


@pytest.mark.parametrize("side,joints", [("left", slice(0, 7)), ("right", slice(9, 16))])
def test_gravity_torque_along_the_arm_base(pair, side, joints):
    """What gafro computes today: MuJoCo's gravity torque with gravity along -z of the arm's link0."""
    model, system, dadr = pair["model"], pair["system"], pair["dadr"]
    link0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_link0")
    zero = np.zeros(system.get_dof())

    for q, data in _configurations(pair):
        model.opt.gravity[:] = data.xmat[link0].reshape(3, 3) @ [0.0, 0.0, -9.81]
        mujoco.mj_forward(model, data)

        np.testing.assert_allclose(system.compute_inverse_dynamics(q, zero, zero)[joints], data.qfrc_bias[dadr][joints], atol=1e-6)

    model.opt.gravity[:] = [0.0, 0.0, -9.81]


def test_mass_matrix(pair):
    model, system, dadr = pair["model"], pair["system"], pair["dadr"]

    for q, data in _configurations(pair):
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, full)

        np.testing.assert_allclose(system.compute_mass_matrix(q), full[np.ix_(dadr, dadr)], atol=1e-6)
