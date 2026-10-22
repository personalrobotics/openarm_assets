# openarm_assets/src/openarm_assets/assembly.py
"""Robot assembly utilities for building OpenArm configurations."""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from openarm_assets import MODELS_DIR
from openarm_assets import collision_cover as covers

Side = Literal["left", "right"]

# joint4's real resting position sits right at its raw lower limit (0) -- see
# add_ready_keyframe's docstring below. The onboard P+D controller's
# steady-state settling error (no integral term) puts the live joint a hair
# negative often enough that planning's start-configuration validity check
# was rejecting the arm's own current pose during hardware soak testing.
# This comfortably covers the observed drift (~-0.001 to -0.0013 rad) with
# room to spare -- it's not meant to allow deliberately commanding the joint
# negative, just to accept an already-settled pose that's a hair past zero.
JOINT4_LOWER_LIMIT_MARGIN_RAD = -0.05

# The upstream `motor_finger` default class intends every finger actuator to be
# a position servo with kp=100 (ctrl = target finger position in metres), and the
# right arm's fingers are declared that way. The left arm's fingers are declared
# as <motor> instead, which resets the gain to 1 and drops the position feedback:
# they take a raw force, so commanding 0.044 only applies 0.044 N. This restores
# the intended behaviour without editing the vendored XML.
FINGER_KP = 100.0

# Default source model and the decorated output this module produces.
_MODEL_DIR = MODELS_DIR / "openarm"
_DEFAULT_SRC = _MODEL_DIR / "vendor" / "openarm_bimanual.xml"
_GENERATED_DIR = _MODEL_DIR  / "generated"
_OUT_PATHS: dict[str, Path] = {
    "bimanual": _GENERATED_DIR / "openarm_bimanual_decorated.xml",
    "left": _GENERATED_DIR / "openarm_left_decorated.xml",
    "right": _GENERATED_DIR / "openarm_right_decorated.xml",
}

_SIDES = {
    "left": {
        "ee_site_name": "openarm_left_ee_site",
        "tcp_body": "openarm_left_hand_tcp",
        "base_body": "openarm_left_link0",
        "left_finger_body": "openarm_left_left_finger",
        "right_finger_body": "openarm_left_right_finger",
        "joint4": "openarm_left_joint4",
        "finger_actuators": ["left_finger1_ctrl", "left_finger2_ctrl"],
    },
    "right": {
        "ee_site_name": "openarm_right_ee_site",
        "tcp_body": "openarm_right_hand_tcp",
        "base_body": "openarm_right_link0",
        "left_finger_body": "openarm_right_left_finger",
        "right_finger_body": "openarm_right_right_finger",
        "joint4": "openarm_right_joint4",
        "finger_actuators": ["right_finger1_ctrl", "right_finger2_ctrl"],
    }
}


"""Scene decoration"""

def add_ee_site(spec: mujoco.MjSpec, tcp_body: str, site_name: str) -> None:
    """Add an end-effector site to the OpenArm left hand TCP."""
    tcp = spec.body(tcp_body)
    if tcp is None:
        raise RuntimeError(f"No body named {tcp_body!r} in this spec")
    site = tcp.add_site()
    site.name = site_name
    site.pos = [0.0, 0.0, 0.0]

def add_gravcomp(spec: mujoco.MjSpec, root_body: str) -> int:
    """Enable gravity compensation on `root_body` and all of its descendants.

    Must run before `spec.compile()`: MuJoCo optimizes gravcomp away at compile
    time if every body has gravcomp=0, and runtime writes to
    `model.body_gravcomp` are silently ignored.

    Adapted from `mj_manipulator.arm.add_subtree_gravcomp` (MIT,
    github.com/personalrobotics/mj_manipulator), inlined here so this package
    doesn't need mj_manipulator and its planning dependencies just for this.

    Returns the number of bodies that had gravcomp set.
    """
    root = spec.body(root_body)
    if root is None:
        available = [b.name for b in spec.worldbody.bodies if b.name]
        raise ValueError(f"No body named {root_body!r} in this spec. Top-level worldbody children: {available}")

    count = 0
    stack = [root]
    while stack:
        body = stack.pop()
        body.gravcomp = 1.0
        count += 1
        stack.extend(body.bodies)
    return count

def add_gafro_annotations(spec: mujoco.MjSpec, sides: tuple[Side, ...]) -> None:
    """Name each arm's kinematic chain and task space for gafro's MJCF parser.

    gafro reads these from `<custom><text>` entries (see gafro's MJCFParser): a chain is named by its
    tip body (gafro looks the tip up by that name), a task space by the chains it spans. The tip is the hand TCP body, the same frame
    `add_ee_site` puts `openarm_<side>_ee_site` on, so the task space is the MuJoCo tool frame and not
    the URDFs' `hand_tcp` (see `opendubs/notes/design/arm-kinematic-models.md`). Each arm's seven joints are one
    control group, which the task space names. With both arms there is also a `bimanual` task space spanning both.
    """
    for side_name in sides:
        side = _SIDES[side_name]
        # gafro looks the chain's tip up by the chain's name, so the two must be the same string.
        chain = side["tcp_body"]
        spec.add_text(name=f"gafro_kinematic_chain:{chain}", data=f"{chain};")
        # One control group per arm: the seven arm joints, which is what one CAN bus and one driver
        # command. The gripper is not in it; it is commanded as a gripper, not as a joint group.
        group = f"{side_name}_arm"
        joints = " ".join(f"openarm_{side_name}_joint{j}" for j in range(1, 8))
        spec.add_text(name=f"gafro_control_group:{group}", data=f"joint: {joints}")
        spec.add_text(name=f"gafro_task_space:{side_name}", data=f"{chain};{group}")

    # Both arms as one task space, for a controller over both at once: a RobotInterface runs one controller, and a stage binds to one task
    # space, so moving both arms together (engaging them from stow) needs a task space that is both.
    if len(sides) == 2:
        chains = " ".join(_SIDES[side_name]["tcp_body"] for side_name in sides)
        groups = " ".join(f"{side_name}_arm" for side_name in sides)
        spec.add_text(name="gafro_task_space:bimanual", data=f"{chains};{groups}")


def add_collision_cover(spec: mujoco.MjSpec, **options) -> covers.Cover:
    """Write the robot's ball cover and the pairs of balls that can collide, for gafro's self-collision objective.

    The cover is derived from the collision meshes (see `covers`) of the spec as it stands, so call this after
    the joint ranges and the ready pose are final: the pairs are the ones that collide within those ranges. gafro reads
    each ball as `gafro_collision_sphere:<name>` with data `<carrier> <x> <y> <z> <radius>` (the centre in the carrier body's frame),
    and each pair as `gafro_collision_pair:<first>|<second>` with data `<first> <second>`. A ball's carrier is the tip of the
    kinematic chain that carries it, so every carrier is declared as a chain, as the arm's tool body is.
    `options` are those of `covers.generate`.
    """
    cover = covers.generate(spec.copy().compile(), **options)
    for carrier in dict.fromkeys(ball.carrier for ball in cover.balls):
        spec.add_text(name=f"gafro_kinematic_chain:{carrier}", data=f"{carrier};")
    for ball in cover.balls:
        x, y, z = ball.center
        spec.add_text(name=f"gafro_collision_sphere:{ball.name}", data=f"{ball.carrier} {x!r} {y!r} {z!r} {ball.radius!r}")
    for first, second in cover.pairs:
        spec.add_text(name=f"gafro_collision_pair:{first}|{second}", data=f"{first} {second}")
    return cover


# Each arm joint has three actuators, one per command mode a gafro controller can run it in, in the actuator groups the gafro
# robot interface switches between (it disables the position and velocity ones while a joint is torque-controlled, and
# the position one while it is velocity-controlled; see MujocoBackend::applyCommands):
#
#   torque    group 1   <motor>     force = command                              (MIT: kp = kd = 0, torque feed-forward)
#   velocity  group 2   <velocity>  force = kd (command - dq)                    (MIT: kp = 0, velocity target)
#   position  group 3   <position>  force = kp (command - q) - kd dq             (MIT: position target)
#
# The gains are the ones the openarm-driver holds and tracks a position with (its `kp` and `kd`), so a position command means the
# same thing on the arm as in the sim. They are starting values for bring-up, not tuned on an arm. Per joint 1..7; the left and
# right arms are alike.
ARM_KP = (120.0, 120.0, 60.0, 60.0, 20.0, 20.0, 20.0)
ARM_KD = (2.0, 2.0, 1.5, 1.5, 0.5, 0.5, 0.5)

# The arm's rated joint velocity and torque are enactic's, in the vendored URDF (<limit effort velocity>): rad/s and N m. They are lower
# than what a motor frame can encode (the Damiao ranges), and they are the limits of the arm, so they are the velocity command range.
_URDF = _MODEL_DIR / "urdf" / "example" / "v1.urdf"


def urdf_limits(path: Path = _URDF) -> dict[str, tuple[float, float]]:
    """Each joint's (effort, velocity) limit from the URDF."""
    limits = {}
    for joint in ET.parse(path).getroot().iter("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("effort") and limit.get("velocity"):
            limits[joint.get("name")] = (float(limit.get("effort")), float(limit.get("velocity")))
    return limits

TORQUE_GROUP, VELOCITY_GROUP, POSITION_GROUP = 1, 2, 3


def add_arm_actuators(spec: mujoco.MjSpec, side_name: Side) -> None:
    """Give each arm joint a torque, a velocity and a position actuator.

    The joint's existing actuator (gain 1, no bias, limited by its motor class's torque) becomes the torque one; the other two are
    added with the same torque limit. Call after the joint limits are final: the position actuator's command range is the joint's.
    """
    for joint_index in range(1, 8):
        joint = f"openarm_{side_name}_joint{joint_index}"
        kp, kd = ARM_KP[joint_index - 1], ARM_KD[joint_index - 1]
        rated_torque, speed = urdf_limits()[joint]

        torque = spec.actuator(f"{side_name}_joint{joint_index}_ctrl")
        limit = torque.forcerange[1]

        # The motor classes' torque limit and the URDF's rating are the same number; if they ever differ, one of them is wrong.
        if abs(limit - rated_torque) > 1e-6:
            raise RuntimeError(f"{joint}: the model's torque limit {limit} differs from the URDF's effort {rated_torque}")

        # The three actuators act on the joint together in plain MuJoCo, each up to the rating, so without a joint-level limit the joint
        # can deliver three times it. The joint's own actuator force range is the total the motor can give; the same number as each
        # actuator's, so that whichever is read agrees.
        joint_spec = spec.joint(joint)
        joint_spec.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint_spec.actfrcrange = [-limit, limit]

        torque.name = f"{side_name}_joint{joint_index}_torque"
        torque.group = TORQUE_GROUP
        torque.ctrllimited = True
        torque.ctrlrange = [-limit, limit]

        velocity = spec.add_actuator(name=f"{side_name}_joint{joint_index}_velocity", target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        velocity.group = VELOCITY_GROUP
        velocity.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        velocity.gainprm[0] = kd
        velocity.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        velocity.biasprm[:3] = [0.0, 0.0, -kd]
        velocity.ctrllimited = True
        velocity.ctrlrange = [-speed, speed]
        velocity.forcelimited = True
        velocity.forcerange = [-limit, limit]

        lower, upper = spec.joint(joint).range
        position = spec.add_actuator(name=f"{side_name}_joint{joint_index}_position", target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        position.group = POSITION_GROUP
        position.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        position.gainprm[0] = kp
        position.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        position.biasprm[:3] = [0.0, -kp, -kd]
        position.ctrllimited = True
        position.ctrlrange = [lower, upper]
        position.forcelimited = True
        position.forcerange = [-limit, limit]


_ARM_ACTUATOR = re.compile(r'<general name="(\w+_joint\d_(torque|velocity|position))" [^>]*/>')


def to_xml(spec: mujoco.MjSpec) -> str:
    """The spec as MJCF text, written the way gafro reads it.

    One thing differs from `spec.to_xml()`, for gafro and not a change to the model MuJoCo compiles:

    - The arm joints' three actuators are written as `<motor>`, `<velocity>` and `<position>`, with the parameters the spec
      gives them. gafro registers a command mode from the element kind: a plain `<general>` (gain 1, no bias, which is what a
      torque motor is) gets no mode at all, so nothing could command the joint and RI's moves would wait forever.
    """
    xml = spec.to_xml()

    def shortcut(match: re.Match) -> str:
        name, kind = match.groups()
        actuator = spec.actuator(name)
        joint = actuator.target
        lo, hi = actuator.ctrlrange
        torque = actuator.forcerange[1]
        limits = f'ctrllimited="true" ctrlrange="{lo:.10g} {hi:.10g}" forcelimited="true" forcerange="{-torque:.10g} {torque:.10g}"'
        if kind == "torque":
            return f'<motor name="{name}" joint="{joint}" gear="1" group="{actuator.group}" {limits}/>'
        if kind == "velocity":
            return f'<velocity name="{name}" joint="{joint}" kv="{actuator.gainprm[0]:.10g}" group="{actuator.group}" {limits}/>'
        return f'<position name="{name}" joint="{joint}" kp="{actuator.gainprm[0]:.10g}" kv="{-actuator.biasprm[2]:.10g}" group="{actuator.group}" {limits}/>'

    xml, replaced = _ARM_ACTUATOR.subn(shortcut, xml)
    if replaced != 42:
        raise RuntimeError(f"expected 42 arm joint actuators (3 for each of 14 joints), found {replaced}; has the vendored model's actuator layout changed?")

    return xml


def narrow_joint_ranges(spec: mujoco.MjSpec, ranges: dict[str, tuple[float, float]]) -> None:
    """Narrow joints' ranges to a robot's own: where this arm may go on the robot it is mounted on.

    The arm's range is the arm's; a robot narrows it for what is around the arm (the opendubs head), and never widens it, so a range wider than
    the joint's is refused. Call before `add_arm_actuators`: the position actuator's command range is the joint's.
    """
    for name, (lower, upper) in ranges.items():
        joint = spec.joint(name)
        if joint is None:
            raise ValueError(f"no joint named {name!r}")
        if not lower < upper:
            raise ValueError(f"{name}: the range [{lower}, {upper}] is empty")
        arm_lower, arm_upper = joint.range
        if lower < arm_lower - 1e-9 or upper > arm_upper + 1e-9:
            raise ValueError(f"{name}: the range [{lower}, {upper}] is wider than the arm's [{arm_lower}, {arm_upper}]")
        joint.range = [lower, upper]


def add_finger_exclude(spec: mujoco.MjSpec, left_finger: str, right_finger: str) -> None:
    """Exclude finger-finger self-collision at the closed position."""
    exclude = spec.add_exclude()
    exclude.bodyname1 = left_finger
    exclude.bodyname2 = right_finger

def add_ready_keyframe(spec: mujoco.MjSpec, joint4_names: list[str], joint4_pos: float = 0.3, pose: dict[str, float] | None = None) -> None:
    """Add the "ready" keyframe: all joints at 0 except `joint4` on each
    arm. joint4's range is [0, 2.44346] (see hw_validation/joint_limits.yaml),
    so 0 sits exactly on its lower limit -- a singular pose for planning to
    start from. `joint4_pos` keeps it clear of that limit instead.

    That is only clear of the limit, and still nearly singular (the elbow is almost straight). A robot that works from a
    ready pose gives its own as `pose`: joint name -> angle, the others at 0.
    """
    model = spec.compile()
    qpos = np.zeros(model.nq)
    for name in joint4_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"No joint named {name!r} in this spec")
        qpos[model.jnt_qposadr[jid]] = joint4_pos

    if pose is not None:
        qpos[:] = 0.0
        for name, value in pose.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"no joint named {name!r}")
            low, high = model.jnt_range[jid]
            if not low - 1e-9 <= value <= high + 1e-9:
                raise ValueError(f"{name}: {value} is outside the joint's range [{low}, {high}]")
            qpos[model.jnt_qposadr[jid]] = value

    key = spec.add_key()
    key.name = "ready"
    key.qpos = qpos

def widen_joint4_lower_limit(spec: mujoco.MjSpec, joint4_name: str) -> None:
    """Widen `joint4_name`'s lower range bound by JOINT4_LOWER_LIMIT_MARGIN_RAD.

    See that constant's comment and add_ready_keyframe's docstring above for
    why joint4's raw lower limit sitting exactly at its resting position is
    fragile against real controller settling noise.
    """
    joint = spec.joint(joint4_name)
    joint.range[0] = JOINT4_LOWER_LIMIT_MARGIN_RAD

def make_finger_actuators_position(spec: mujoco.MjSpec, actuator_names: list[str], kp: float = FINGER_KP) -> None:
    """Make the named finger actuators position servos (ctrl = target position, m).

    Sets a fixed gain of `kp` and an affine bias of `-kp * q`, which is exactly
    what MuJoCo's <position> shortcut expands to. Already-position actuators
    (the right arm's) are left effectively unchanged, so this is idempotent.
    """
    for name in actuator_names:
        actuator = spec.actuator(name)
        if actuator is None:
            raise RuntimeError(f"No actuator named {name!r} in this spec")
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = kp
        actuator.biasprm[0] = 0.0
        actuator.biasprm[1] = -kp
        actuator.biasprm[2] = 0.0

""" Build """

def build_openarm(
        src: Path | str = _DEFAULT_SRC, 
        sides: tuple[Side, ...] = ("left", "right"),
        gravcomp: bool = False,
        joint_ranges: dict[str, tuple[float, float]] | None = None,
        ready: dict[str, float] | None = None,
        collision_cover: bool = True,
        collision_options: dict | None = None,
) -> mujoco.MjSpec:
    """Load the OpenArm scene and apply all decorations.

    The chains, task spaces and control groups are named for gafro's parser. MuJoCo's gravity
    compensation is off unless asked for: it makes the simulated arm weightless, so a torque
    controller (or a driver test) would never see the gravity the real arm has, which the
    controller has to cancel. Pass `gravcomp` for a weightless sim, as the impedance controller's
    onboard compensation made the real arm.

    `joint_ranges` narrows named joints' ranges for a robot (see `narrow_joint_ranges`); `ready` is the robot's own ready pose
    (joint name -> angle, the others at 0), in place of the arm's nearly straight one (see `add_ready_keyframe`).

    `collision_cover` writes the ball cover and its pairs for gafro's self-collision objective (see `add_collision_cover`);
    it is derived from the model as decorated here, so it follows `joint_ranges` and `ready`. `collision_options` are
    `covers.generate`'s (`span`, `samples`, `redundancy`, ...).
    """

    spec = mujoco.MjSpec.from_file(str(src))
    spec.compiler.meshdir = "../meshes"

    for side_name in sides:
        side = _SIDES[side_name]
        add_ee_site(spec, side["tcp_body"], side["ee_site_name"])
        if gravcomp:
            add_gravcomp(spec, side["base_body"])
        add_finger_exclude(spec, side["left_finger_body"], side["right_finger_body"])

    # Both arms' joints and actuators are always present in the loaded scene
    # regardless of which side(s) get decorated above, so the limit margin,
    # finger actuator fix and keyframe cover both arms -- not just `sides` --
    # to keep every output out of that singular pose and consistent across arms.
    for s in _SIDES.values():
        widen_joint4_lower_limit(spec, s["joint4"])
        make_finger_actuators_position(spec, s["finger_actuators"])

    # After the arm's own widening, and before the actuators, whose command range is the joint's.
    narrow_joint_ranges(spec, joint_ranges or {})

    for side_name in _SIDES:
        add_arm_actuators(spec, side_name)
    add_ready_keyframe(spec, [s["joint4"] for s in _SIDES.values()], pose=ready)
    add_gafro_annotations(spec, sides)
    if collision_cover:
        add_collision_cover(spec, **(collision_options or {}))

    return spec

def build_openarm_bimanual(src: Path | str = _DEFAULT_SRC, gravcomp: bool = False) -> mujoco.MjSpec:
    """Decorate both arms."""
    return build_openarm(src, sides=("left", "right"), gravcomp=gravcomp)

def build_openarm_left(src: Path | str = _DEFAULT_SRC, gravcomp: bool = False) -> mujoco.MjSpec:
    """Decorate only the left arm."""
    return build_openarm(src, sides=("left",), gravcomp=gravcomp)

def build_openarm_right(src: Path | str = _DEFAULT_SRC, gravcomp: bool = False) -> mujoco.MjSpec:
    """Decorate only the right arm."""
    return build_openarm(src, sides=("right",), gravcomp=gravcomp)

""" CLI """

def main() -> None:
   parser = argparse.ArgumentParser(
       description="Decorate the OpenArm bimanual scene (EE site, finger exclude, gafro "
       "chains and task spaces) for one or both arms and write the result to disk."
   )
   parser.add_argument(
       "-s", "--src", type=Path, default=_DEFAULT_SRC,
       help=f"Source scene XML (default: {_DEFAULT_SRC}).",
   )
   parser.add_argument(
       "--sides", choices=["bimanual", "left", "right"], default="bimanual",
       help="Which arm(s) to decorate (default: bimanual, both arms).",
   )
   parser.add_argument(
       "-o", "--out", type=Path, default=None,
       help="Output XML path (default: generated/openarm_<sides>_decorated.xml).",
   )
   parser.add_argument(
       "--joint-range", action="append", default=[], metavar="NAME=LOW,HIGH",
       help="Narrow a joint's range, radians (repeatable): a robot's own limit, for what is around the arm. Never wider than the arm's.",
   )
   parser.add_argument(
       "--ready", action="append", default=[], metavar="NAME=VALUE",
       help="The robot's own ready pose, radians, one joint per flag (repeatable); the others are at 0. Inside the joint's range.",
   )
   parser.add_argument(
       "--collision-span", type=float, default=0.5,
       help="Fraction of each joint's range the collision sweep samples, centred on ready (default 0.5; 1.0 is the whole range and "
       "about three times as many pairs, which a controller pays for in every row; see collision_cover.sweep).",
   )
   parser.add_argument(
       "--no-collision-cover", action="store_true",
       help="Leave out the ball cover and collision pairs gafro's self-collision objective reads (see add_collision_cover).",
   )
   parser.add_argument(
       "--gravcomp", action="store_true",
       help="Turn on MuJoCo gravity compensation on the decorated arms (off by default; see build_openarm).",
   )
   args = parser.parse_args()
 
   sides = ("left", "right") if args.sides == "bimanual" else (args.sides,)
   ranges = {}
   for item in args.joint_range:
       name, _, bounds = item.partition("=")
       lower, _, upper = bounds.partition(",")
       ranges[name] = (float(lower), float(upper))
   ready = {item.partition("=")[0]: float(item.partition("=")[2]) for item in args.ready} or None
   spec = build_openarm(args.src, sides=sides, gravcomp=args.gravcomp, joint_ranges=ranges, ready=ready, collision_cover=not args.no_collision_cover,
                         collision_options={"span": args.collision_span})
 
   # Sanity check: does it actually compile?
   model = spec.compile()
   print(f"Compiled OK — {model.nq} qpos, {model.nu} actuators")
 
   out = args.out or _OUT_PATHS[args.sides]
   out.parent.mkdir(parents=True, exist_ok=True)
   out.write_text(to_xml(spec))
   print(f"Wrote {out}")
 
 
if __name__ == "__main__":
   main()
