# SPDX-FileCopyrightText: 2026 Tobias Loew <tobias.loew@gafro.ch>
# SPDX-License-Identifier: MPL-2.0
"""The node configs do their job in MuJoCo, driven as a client drives them (gafro_robot_node.testing):
each arm and both follow a joint path while the other holds, the hands open and shut, and reach plans
an arm to a pose. What a robot's config extends these with (the hardware backend, a guard, a
transport) is the robot package's to test."""

import os
import pathlib

import numpy as np
import pytest

testing = pytest.importorskip("gafro_robot_node.testing", reason="gafro_robot_node with its testing module")
pytest.importorskip("gafro.control")

from openarm_assets import MODELS_DIR  # noqa: E402

# The package's configs, where they are installed (or src/ in a checkout): a release check tests the files that shipped.
CONFIGS = pathlib.Path(MODELS_DIR).parent / "configs"
MODEL = MODELS_DIR / "openarm" / "generated" / "openarm_bimanual_decorated.xml"
STEP = np.array(
    [0.3, 0.2, -0.2, 0.3, 0.2, 0.2, 0.3]
)  # rad per arm joint: a move of every joint, inside the limits from the start
# How close an arm settles: the position controller's lead (node.yaml's maximum_lead, 0.02 rad) is
# the most it commands past the measurement, and in MuJoCo joint 4, holding the forearm up, ends at
# about that (18 mrad on the left arm, 8 on the right) with its lead saturated. The other joints
# settle within 7 mrad.
ARM_TOLERANCE = 0.025


@pytest.mark.parametrize(
    "behaviour,moving", [("track_right", ["right"]), ("track_left", ["left"]), ("track_both", ["right", "left"])]
)
def test_the_arms_follow_their_paths_and_an_arm_sent_nothing_holds(behaviour, moving):
    with testing.NodeHarness(CONFIGS / "node.yaml", model=MODEL) as node:
        node.run_for(0.5)
        start = node.q
        goals = {}
        for side in moving:
            arm = node.controlled(side)
            goals[side] = start[arm] + STEP
            node.send_joint_path(f"{side}/path", testing.smooth_path(start[arm], goals[side]))
        assert node.run(behaviour) == "success", node.lifecycle.reason
        node.run_for(0.3)
        q = node.q
        for side, goal in goals.items():
            assert np.abs(q[node.controlled(side)] - goal).max() < ARM_TOLERANCE, side
        for side in {"right", "left"} - set(moving):
            assert np.abs(q[node.controlled(side)] - start[node.controlled(side)]).max() < 0.01, f"{side} moved"


def test_the_hands_open_and_shut():
    with testing.NodeHarness(CONFIGS / "node_hands.yaml", model=MODEL) as node:
        node.run_for(0.5)
        fingers = {side: node.controlled(f"{side}_hand") for side in ("right", "left")}
        opening = float(
            np.asarray(node.system.get_joint_limits_max().coefficients())[fingers["right"][0]]
        )  # the finger joints' upper limit
        for target in (opening, 0.0):
            for side, joints in fingers.items():
                node.send_joint_path(f"{side}/hand", [np.full(len(joints), target)] * 21)
                # track_all follows all four inputs: the arms get a path that holds them where they are
                arm = node.controlled(side)
                node.send_joint_path(f"{side}/path", [node.q[arm]] * 21)
            assert node.run("track_all") == "success", node.lifecycle.reason
            node.run_for(0.3)
            for side, joints in fingers.items():
                assert np.abs(node.q[joints] - target).max() < 0.002, f"{side} hand at {target}"


def _reach_plugin() -> bool:
    return any(
        (pathlib.Path(d) / "libgafro_self_motion_charts_reach_task_nodes.so").is_file()
        for d in os.environ.get("GAFRO_PLUGIN_PATH", "").split(os.pathsep)
        if d
    )


@pytest.mark.skipif(
    not _reach_plugin(), reason="the reach plugin (self-motion-charts) on GAFRO_PLUGIN_PATH, built against this gafro"
)
def test_reach_plans_the_hand_to_a_pose():
    import gafro

    with testing.NodeHarness(CONFIGS / "node_reach.yaml", model=MODEL) as node:
        node.pose("right/pose")  # follow the hand's pose from here
        node.run_for(0.5)
        start = node.q
        hand = np.asarray(node.pose("right/pose").to_transformation_matrix(), dtype=float)
        goal = hand.copy()
        goal[0, 3] += 0.05
        node.send_pose_path("right/reach", [gafro.Motor.from_matrix(goal)])
        assert node.run("reach_right") == "success", node.lifecycle.reason
        node.run_for(0.3)
        reached = np.asarray(node.pose("right/pose").to_transformation_matrix(), dtype=float)
        assert np.linalg.norm(reached[:3, 3] - goal[:3, 3]) < 0.01
        assert np.abs(node.q[node.controlled("left")] - start[node.controlled("left")]).max() < 0.01
