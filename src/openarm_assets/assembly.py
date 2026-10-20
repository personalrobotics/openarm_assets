# openarm_assets/src/openarm_assets/assembly.py
"""Robot assembly utilities for building OpenArm configurations."""

import argparse
import re
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from openarm_assets import MODELS_DIR

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
    control group, which the task space names.
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


_ARM_ACTUATOR = re.compile(r'<general name="(\w+_joint\d_ctrl)" class="\w+" joint="(\w+)"[^>]*/>')


def add_command_range(spec: mujoco.MjSpec, side_name: Side) -> None:
    """Give each arm joint's motor a command range equal to its torque limit.

    The motors have gain 1 and no gear, so a command is a torque and the two ranges are the same limit. gafro reads
    the command range (`control_input_range`) and not the force range, so both are set, and `to_xml` writes both.
    """
    for joint in range(1, 8):
        actuator = spec.actuator(f"{side_name}_joint{joint}_ctrl")
        limit = actuator.forcerange[1]
        actuator.ctrllimited = True
        actuator.ctrlrange = [-limit, limit]


def to_xml(spec: mujoco.MjSpec) -> str:
    """The spec as MJCF text, written the way gafro reads it.

    Two things differ from `spec.to_xml()`, both for gafro and neither a change to the model MuJoCo compiles:

    - The seven arm joints per side are written as `<motor>`. gafro registers a torque mode only for a motor element, and a
      plain `<general>` (which is what these are: gain 1, no bias) gets no mode at all, so nothing can command the joint
      and RI's position moves wait forever. The motor keeps the joint's torque limit as its force range and, from
      `add_command_range`, its command range.
    - `meshdir` is `assets`, which is where gafro will look: it drops an MJCF `meshdir` on load and writes the default back out,
      so the meshes have to be at `assets` beside the model file (see `link_meshes`).
    """
    xml = spec.to_xml()

    def motor(match: re.Match) -> str:
        name, joint = match.groups()
        limit = spec.actuator(name).forcerange[1]
        return (
            f'<motor name="{name}" joint="{joint}" gear="1" ctrllimited="true" ctrlrange="{-limit:g} {limit:g}" '
            f'forcelimited="true" forcerange="{-limit:g} {limit:g}"/>'
        )

    xml, replaced = _ARM_ACTUATOR.subn(motor, xml)
    if replaced == 0:
        raise RuntimeError("no arm joint actuators found to write as motors; has the vendored model's actuator layout changed?")

    return xml.replace('meshdir="../meshes/"', 'meshdir="assets"')


def link_meshes(out_dir: Path) -> None:
    """Make `assets` in `out_dir` a link to the vendored meshes, where gafro looks for them."""
    link = out_dir / "assets"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path("..") / "meshes")


def add_finger_exclude(spec: mujoco.MjSpec, left_finger: str, right_finger: str) -> None:
    """Exclude finger-finger self-collision at the closed position."""
    exclude = spec.add_exclude()
    exclude.bodyname1 = left_finger
    exclude.bodyname2 = right_finger

def add_ready_keyframe(spec: mujoco.MjSpec, joint4_names: list[str], joint4_pos: float = 0.3) -> None:
    """Add the "ready" home pose: all joints at 0 except `joint4` on each
    arm. joint4's range is [0, 2.44346] (see hw_validation/joint_limits.yaml),
    so 0 sits exactly on its lower limit -- a singular pose for planning to
    start from. `joint4_pos` keeps it clear of that limit instead.
    """
    model = spec.compile()
    qpos = np.zeros(model.nq)
    for name in joint4_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"No joint named {name!r} in this spec")
        qpos[model.jnt_qposadr[jid]] = joint4_pos

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
) -> mujoco.MjSpec:
    """Load the OpenArm scene and apply all decorations.

    The chains, task spaces and control groups are named for gafro's parser. MuJoCo's gravity
    compensation is off unless asked for: it makes the simulated arm weightless, so a torque
    controller (or a driver test) would never see the gravity the real arm has, which the
    controller has to cancel. Pass `gravcomp` for a weightless sim, as the impedance controller's
    onboard compensation made the real arm.
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
    for side_name, s in _SIDES.items():
        add_command_range(spec, side_name)
        widen_joint4_lower_limit(spec, s["joint4"])
        make_finger_actuators_position(spec, s["finger_actuators"])
    add_ready_keyframe(spec, [s["joint4"] for s in _SIDES.values()])
    add_gafro_annotations(spec, sides)

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
       "--gravcomp", action="store_true",
       help="Turn on MuJoCo gravity compensation on the decorated arms (off by default; see build_openarm).",
   )
   args = parser.parse_args()
 
   sides = ("left", "right") if args.sides == "bimanual" else (args.sides,)
   spec = build_openarm(args.src, sides=sides, gravcomp=args.gravcomp)
 
   # Sanity check: does it actually compile?
   model = spec.compile()
   print(f"Compiled OK — {model.nq} qpos, {model.nu} actuators")
 
   out = args.out or _OUT_PATHS[args.sides]
   out.parent.mkdir(parents=True, exist_ok=True)
   out.write_text(to_xml(spec))
   link_meshes(out.parent)
   print(f"Wrote {out}")
 
 
if __name__ == "__main__":
   main()
