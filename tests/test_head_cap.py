# SPDX-FileCopyrightText: 2026 Tobias Loew <tobias.loew@gafro.ch>
# SPDX-License-Identifier: MPL-2.0
"""The head-cap variant: both arms, joint 2 capped on the outward side by a head beside them, every joint at the driver's velocity limit."""
import mujoco
import numpy as np

from openarm_assets.assembly import HEAD_CAP, _OUT_PATHS

HEAD_CAP_MODEL = _OUT_PATHS["bimanual_head_cap"]
DECORATED = _OUT_PATHS["bimanual"]


def joint_ranges(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): tuple(model.jnt_range[j]) for j in range(model.njnt)}


def test_joint2_is_capped_outward_on_each_arm_and_the_other_joints_are_the_arms():
    capped = joint_ranges(mujoco.MjModel.from_xml_path(str(HEAD_CAP_MODEL)))
    arm = joint_ranges(mujoco.MjModel.from_xml_path(str(DECORATED)))

    # The XML holds six significant digits.
    assert np.allclose(capped["openarm_right_joint2"], (arm["openarm_right_joint2"][0], min(arm["openarm_right_joint2"][1], HEAD_CAP)), atol=1e-5)
    assert np.allclose(capped["openarm_left_joint2"], (max(arm["openarm_left_joint2"][0], -HEAD_CAP), arm["openarm_left_joint2"][1]), atol=1e-5)
    assert np.isclose(capped["openarm_right_joint2"][1], np.radians(155.0), atol=1e-5)
    for name, limits in arm.items():
        if name not in ("openarm_right_joint2", "openarm_left_joint2"):
            assert np.allclose(capped[name], limits), name


def test_the_committed_head_cap_model_is_what_the_generator_builds():
    from openarm_assets.assembly import DRIVER_VELOCITY_LIMIT, build_openarm, head_cap_ranges

    # The velocity actuators' command ranges are the joints' velocity limits: capped at what the driver runs.
    spec = build_openarm(joint_ranges=head_cap_ranges(), velocity_limit=DRIVER_VELOCITY_LIMIT)
    built = spec.compile()
    committed = mujoco.MjModel.from_xml_path(str(HEAD_CAP_MODEL))
    assert np.allclose(built.jnt_range, committed.jnt_range, atol=1e-5)
    assert np.allclose(built.actuator_ctrlrange, committed.actuator_ctrlrange, atol=1e-5)
    assert built.nkey == committed.nkey and np.allclose(built.key_qpos, committed.key_qpos)


def test_the_head_cap_model_names_the_hands():
    """The arms run the head-cap model on hardware, the hands too: each a control group openarm-driver's `openarm_hand` drives."""
    committed = mujoco.MjModel.from_xml_path(str(HEAD_CAP_MODEL))
    texts = {mujoco.mj_id2name(committed, mujoco.mjtObj.mjOBJ_TEXT, i) for i in range(committed.ntext)}

    assert {"gafro_control_group:left_hand", "gafro_control_group:right_hand", "gafro_task_space:arms_and_hands"} <= texts
