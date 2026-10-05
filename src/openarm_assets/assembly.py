# openarm_assets/src/openarm_assets/assembly.py
"""Robot assembly utilities for building OpenArm configurations."""

import argparse
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
    },
    "right": {
        "ee_site_name": "openarm_right_ee_site",
        "tcp_body": "openarm_right_hand_tcp",
        "base_body": "openarm_right_link0",
        "left_finger_body": "openarm_right_left_finger",
        "right_finger_body": "openarm_right_right_finger",
        "joint4": "openarm_right_joint4",
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

""" Build """

def build_openarm(
        src: Path | str = _DEFAULT_SRC, 
        sides: tuple[Side, ...] = ("left", "right"),
) -> mujoco.MjSpec:
    """Load the OpenArm scene and apply all decorations"""

    spec = mujoco.MjSpec.from_file(str(src))
    spec.compiler.meshdir = "../meshes"

    for side_name in sides:
        side = _SIDES[side_name]
        add_ee_site(spec, side["tcp_body"], side["ee_site_name"])
        add_gravcomp(spec, side["base_body"])
        add_finger_exclude(spec, side["left_finger_body"], side["right_finger_body"])

    # Both arms' joints are always present in the loaded scene regardless of
    # which side(s) get decorated above, so the limit margin and keyframe
    # cover joint4 on both -- not just `sides` -- to keep every output out
    # of that singular pose.
    for s in _SIDES.values():
        widen_joint4_lower_limit(spec, s["joint4"])
    add_ready_keyframe(spec, [s["joint4"] for s in _SIDES.values()])

    return spec

def build_openarm_bimanual(src: Path | str = _DEFAULT_SRC) -> mujoco.MjSpec:
    """Decorate both arms."""
    return build_openarm(src, sides=("left", "right"))

def build_openarm_left(src: Path | str = _DEFAULT_SRC) -> mujoco.MjSpec:
    """Decorate only the left arm."""
    return build_openarm(src, sides=("left",))

def build_openarm_right(src: Path | str = _DEFAULT_SRC) -> mujoco.MjSpec:
    """Decorate only the right arm."""
    return build_openarm(src, sides=("right",))

""" CLI """

def main() -> None:
   parser = argparse.ArgumentParser(
       description="Decorate the OpenArm bimanual scene (EE site, gravcomp, "
       "finger exclude) for one or both arms and write the result to disk."
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
   args = parser.parse_args()
 
   sides = ("left", "right") if args.sides == "bimanual" else (args.sides,)
   spec = build_openarm(args.src, sides=sides)
 
   # Sanity check: does it actually compile?
   model = spec.compile()
   print(f"Compiled OK — {model.nq} qpos, {model.nu} actuators")
 
   out = args.out or _OUT_PATHS[args.sides]
   out.parent.mkdir(parents=True, exist_ok=True)
   out.write_text(spec.to_xml())
   print(f"Wrote {out}")
 
 
if __name__ == "__main__":
   main()
