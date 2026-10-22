# openarm_assets

MuJoCo models for the bimanual [OpenArm](https://github.com/enactic/openarm) robot: two 7-DOF humanoid arms with parallel-jaw grippers on a shared body.

The base models are vendored from [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco) (`v1/` at commit [`5c6e1b4`](https://github.com/enactic/openarm_mujoco/commit/5c6e1b4c71cbe27c6a4b58c7f198f080449e1b9c)). This package adds Python helpers for locating them, plus an assembly step that decorates them for use with planners and controllers (end-effector sites, a "ready" keyframe, and the names gafro reads: kinematic chains, task spaces, control groups).

## Installation

From source:
```bash
git clone https://github.com/personalrobotics/openarm_assets.git
cd openarm_assets
uv sync
```

Or with pip:
```bash
pip install -e .
```

Requires Python 3.10+. The only runtime dependencies are `mujoco` and `numpy-stl`.

## Quick Start

```python
import mujoco
from openarm_assets import get_generated_model_path

# Load the decorated bimanual OpenArm
model = mujoco.MjModel.from_xml_path(str(get_generated_model_path()))
data = mujoco.MjData(model)

# Reset to ready pose
ready_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
mujoco.mj_resetDataKeyframe(model, data, ready_key)

# Step simulation
mujoco.mj_step(model, data)
```

View the robot in MuJoCo's interactive viewer:
```bash
python -m mujoco.viewer --mjcf="$(python -c 'from openarm_assets import get_generated_model_path; print(get_generated_model_path())')"
```

> **Note on macOS:** `mujoco.viewer` needs `mjpython` for the non-blocking `launch_passive` API, which requires a "framework" Python build (e.g. from Homebrew or python.org).

### Collision Visualization

Visual meshes are in geom group 2 and collision meshes are in geom group 3. In the viewer's left sidebar:

- Enable **Group Enable > Geom groups > Geom 3** (and **Rendering > Model Elements > Convex Hull**) to show collision geometry
- Deselect **Geom 2** to hide the visual meshes

### Collision Viewer

An interactive collision viewer is available to visualize collision geometry and contact detection:
```bash
uv run python -u examples/view_collisions.py                # bimanual (default)
uv run python -u examples/view_collisions.py --sides left   # left/right/bimanual generated model
```

- Use joint sliders to move the robot (Tab to show/hide control panel)
- Colliding bodies turn **red**, and each new set of contacts is printed to the terminal
- Press `3` to toggle collision geometry and `2` to toggle visual geometry
- Press `C` to toggle contact point visualization
- Press `Backspace` to reset (or `R` to return to the `ready` keyframe if using the non-blocking viewer)
- Press `P` to list every collision-enabled geom, and `V` to print the camera parameters (non-blocking viewer only)

The viewer turns off gravity and steps the physics so the actuator sliders work. Contact forces are simulated too, so a colliding arm gets pushed back rather than held in the penetrating pose.

> **Note on macOS:** The viewer automatically falls back to blocking mode when using uv's Python. For non-blocking mode with `launch_passive`, you need a "framework" Python build (e.g., from Homebrew or python.org) and can run: `mjpython examples/view_collisions.py`

## Robot Configuration

The OpenArm model consists of:
- **Body** (`openarm_body_link0`) fixed to the world at the origin
- **Two 7-DOF arms**, left and right (`openarm_left_*`, `openarm_right_*`)
- **Parallel-jaw gripper** on each arm: two sliding fingers, mirrored by an equality constraint
- **Gravity compensation** on both arms' kinematic chains (decorated models only)

Physics timestep is 2 ms. There are 18 DOFs and 18 actuators in total (7 arm joints + 2 finger joints, per arm).

### Joints and Actuators

Arm actuators are torque motors: `ctrl` is torque in N·m, clamped to the force limit below. Joint limits are in radians. Left and right arms are mirrored on joints 1 and 2.

| Joint | Motor | Torque limit | Left range | Right range |
|-------|-------|--------------|------------|-------------|
| `joint1` | DM8009 | ±40 N·m | [-3.491, 1.396] | [-1.396, 3.491] |
| `joint2` | DM8009 | ±40 N·m | [-3.316, 0.175] | [-0.175, 3.316] |
| `joint3` | DM4340 | ±27 N·m | [-1.571, 1.571] | [-1.571, 1.571] |
| `joint4` | DM4340 | ±27 N·m | [-0.05, 2.443] | [-0.05, 2.443] |
| `joint5` | DM4310 | ±7 N·m | [-1.571, 1.571] | [-1.571, 1.571] |
| `joint6` | DM4310 | ±7 N·m | [-0.785, 0.785] | [-0.785, 0.785] |
| `joint7` | DM4310 | ±7 N·m | [-1.571, 1.571] | [-1.571, 1.571] |

> **Note on `joint4`:** the raw lower limit is 0, which is exactly the arm's resting position. The assembly step widens it to -0.05 rad so that a pose the real controller has settled a hair past zero is still considered valid.

| Actuator | Range | Description |
|----------|-------|-------------|
| `left_joint1_ctrl` ... `left_joint7_ctrl` | torque, see above | Left arm joints 1-7 |
| `right_joint1_ctrl` ... `right_joint7_ctrl` | torque, see above | Right arm joints 1-7 |
| `left_finger1_ctrl`, `left_finger2_ctrl` | 0-0.044 | Left gripper fingers: target position in m (0 = closed, 0.044 = open) |
| `right_finger1_ctrl`, `right_finger2_ctrl` | 0-0.044 | Right gripper fingers: target position in m (0 = closed, 0.044 = open) |

Finger actuators are position servos (`kp = 100`, force limit ±333 N), and a positive finger value opens the gripper: with both fingers at 0 the collision meshes just touch (about 3.5 mm of overlap), and at 0.044 they are about 85 mm apart.

> **Note on the vendored finger actuators:** in the upstream v1 XML the left fingers are declared as `<motor>` while the right fingers are `<position>`, so the left ones would take a raw force instead of a target position. The assembly step converts them (see [Assembly](#assembly)), so the **generated** models behave identically on both arms. The undecorated models in `vendor/` are unmodified upstream and still have the mismatch.

You can list every actuator and its force limits from the loaded model:

```python
import mujoco
from openarm_assets import get_generated_model_path

model = mujoco.MjModel.from_xml_path(str(get_generated_model_path()))

for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    lo, hi = model.actuator_forcerange[i]
    print(f"{name}: [{lo:.1f}, {hi:.1f}]")
```

### Named Sites

| Site | Description |
|------|-------------|
| `openarm_left_ee_site` | End effector at the left hand TCP (`openarm_left_hand_tcp`) |
| `openarm_right_ee_site` | End effector at the right hand TCP (`openarm_right_hand_tcp`) |
| `world_site` | Origin of the world frame |

An arm's EE site exists only if that arm was decorated, so the `left` and `right` generated models each contain just their own EE site.

### Keyframes

| Keyframe | Description |
|----------|-------------|
| `ready` | All joints at 0 except `joint4` on each arm at 0.3 rad, which keeps the arms off the singular zero pose |

## Available Models

```python
from openarm_assets import get_model_path, get_generated_model_path, AVAILABLE_MODELS

AVAILABLE_MODELS                       # ["openarm"]
get_generated_model_path()             # decorated bimanual (recommended)
get_generated_model_path("left")       # left arm decorated
get_generated_model_path("right")      # right arm decorated
get_model_path()                       # vendored single-arm openarm.xml (undecorated)
```

**Generated (decorated) models** in `models/openarm/generated/`:

| Model | Description |
|-------|-------------|
| `openarm_bimanual_decorated.xml` | Both arms decorated: EE sites, finger-finger collision exclusion, matching finger actuators, `ready` keyframe, gafro chains / task spaces / control groups |
| `openarm_left_decorated.xml` | Same, applied to the left arm only |
| `openarm_right_decorated.xml` | Same, applied to the right arm only |

All three contain both arms (18 DOF). The `left` and `right` variants differ only in which arm receives the decorations. The generated files are committed, so you don't need to run the assembly step to use them.

**Vendored (undecorated) models** in `models/openarm/vendor/`, from upstream with one change (see [Provenance](#provenance)):

| Model | Description |
|-------|-------------|
| `openarm.xml` | Single OpenArm arm with gripper |
| `openarm_bimanual.xml` | Both arms on the body |
| `scene.xml` | Bimanual model with ground plane, skybox, and lighting |

The meshes live in `models/openarm/meshes/` (`visual/` as `.obj`, `collision/` as `.stl`), shared by the vendored and generated models. All of these load directly:

```python
import mujoco
from openarm_assets import get_model_path

model = mujoco.MjModel.from_xml_path(str(get_model_path()))               # single arm
scene = mujoco.MjModel.from_xml_path(str(get_model_path().parent / "scene.xml"))  # bimanual + floor, sky, light
```

`models/openarm/urdf/` also contains the upstream URDF and xacro sources (including `example/v1.urdf`).

## Assembly

`openarm_assets.assembly` takes a vendored model and applies the decorations that planners and controllers typically need:

- an end-effector site at each hand TCP
- three actuators per arm joint, one per command mode: a `<motor>` (torque, group 1), a `<velocity>` servo (group 2) and a `<position>` servo (group 3), the groups gafro's robot interface switches between. The velocity command range and the torque limit are the arm's rated ones, from enactic's URDF (`<limit effort velocity>`); the assembly step checks that the URDF's effort matches the model's torque limit. The position and velocity gains are the ones the [openarm-driver](https://github.com/personalrobotics/openarm-driver) uses for the same modes on the arm (`ARM_KP`, `ARM_KD` in `assembly.py`); they are starting values, not tuned on an arm. Each actuator is limited to the joint's motor torque, and so is the joint's total (`actuatorfrcrange`): in plain MuJoCo the three act together, and without that limit a joint would deliver three times its rating
- the names [gafro](https://github.com/personalrobotics/gafro) reads from an MJCF: a kinematic chain, a task space and a control group (`left_arm`, `right_arm`) per decorated arm; and the arm joints written as `<motor>` torque actuators, which is what makes gafro register a torque mode on them
- *optionally* (`--gravcomp`) gravity compensation on each arm's kinematic subtree; **off by default**, since a weightless simulated arm hides the gravity a torque controller has to cancel
- a collision cover for gafro's self-collision objective: a few balls per link that contain the link's collision mesh, and the pairs of balls that can collide, written as `gafro_collision_sphere:` and `gafro_collision_pair:` entries beside the chains (see [Collision Cover](#collision-cover))
- a collision exclusion between each gripper's two fingers
- a wider `joint4` lower limit
- position-servo finger actuators on both arms (fixes the left arm's `<motor>` fingers)
- a `ready` keyframe

### Command Line

```bash
# Regenerate the decorated bimanual model
python -m openarm_assets.assembly --sides bimanual

# Decorate only one arm, writing to a custom path
python -m openarm_assets.assembly --sides left -o ./openarm_left.xml
```

| Option | Description |
|--------|-------------|
| `-s`, `--src` | Source XML (default: `vendor/openarm_bimanual.xml`) |
| `--sides` | `bimanual` (default), `left`, or `right` |
| `-o`, `--out` | Output path (default: `generated/openarm_<sides>_decorated.xml`) |

The CLI compiles the model as a sanity check before writing it. If you write the output somewhere other than `generated/`, the mesh path (`../meshes`) will no longer resolve, so keep the output next to a `meshes/` directory or adjust `meshdir`.

### Collision Cover

gafro keeps one part of the robot clear of another with balls rather than meshes, so the model says what its balls are and which pairs of them can touch, and no controller has to. `openarm_assets.collision_cover` derives both from the collision meshes (geom group 3):

- **Balls.** The triangles of each link are clustered and every cluster's ball contains all of its triangles, so a mesh contact is a ball overlap. A link gets about one ball per 10 cm, at most five, each fattened by 3 mm (MuJoCo collides the convex hull of a mesh, which sticks out of the mesh a little). A body that is rigid with the root (the arm bases) has its balls carried by the root, because gafro has no moving chain to carry them.
- **Pairs.** Configurations are sampled uniformly over a fraction of each joint's range centred on the `ready` pose (`--collision-span`, default 0.5). A pair of bodies is listed when their meshes come within 2 cm of each other in some sample, and then the fewest pairs of their balls that overlap in every such sample, each sample covered twice. MuJoCo's exclusions (adjacent links, the vendor's overlapping link5 to link7) are not tested.

The span is the one choice that matters, because a controller pays for every pair: a guard stage of about 80 pairs, the default, costs about 1.3 ms per control step; the whole range gives about 280 pairs and about eight. The cover sees what the sweep sees: a collision outside the sampled region, or between bodies that never came near in it (the two arms reaching each other, at the default span), is not listed.

```bash
python -m openarm_assets.assembly --sides bimanual --collision-span 1.0   # the whole range
python -m openarm_assets.assembly --no-collision-cover                     # leave it out
```

`collision_cover.check_conservative` re-measures a cover on fresh samples: every configuration in which two meshes collide and no listed pair of their balls overlaps.

### Python API

```python
from openarm_assets.assembly import (
    build_openarm,
    build_openarm_bimanual,
    build_openarm_left,
    build_openarm_right,
)

spec = build_openarm_bimanual()             # decorate both arms
spec = build_openarm_left()                 # decorate only the left arm
spec = build_openarm(sides=("left",))       # equivalent, with an explicit tuple

model = spec.compile()
```

These functions return a `mujoco.MjSpec`, so you can keep editing the model (add objects, cameras, sensors) before compiling.

## Development

```bash
uv sync --dev

# Test
uv run pytest tests/ -v

# Lint
uv run ruff check .
uv run ruff format .
```

## Provenance

The MJCF files, meshes, and URDF/xacro sources under `models/openarm/` are vendored from [Enactic's OpenArm projects](https://github.com/enactic/openarm_mujoco) (v1 models at commit `5c6e1b4`). The only local change to the vendored XMLs is `meshdir="meshes"` -> `meshdir="../meshes"` in `vendor/openarm.xml` and `vendor/openarm_bimanual.xml`, because the meshes are shared with the generated models one level up. To pick up a newer upstream release, re-vendor the files, reapply that change, and update the commit reference above.

OpenArm is developed by [Enactic, Inc.](https://github.com/enactic) See the [OpenArm project](https://github.com/enactic/openarm) for hardware designs, ROS 2 drivers, and documentation.

## License

- Vendored OpenArm assets (`src/openarm_assets/models/openarm/`): Apache License 2.0, Copyright 2025 Enactic, Inc. See [`src/openarm_assets/models/openarm/LICENSE`](src/openarm_assets/models/openarm/LICENSE).
- Everything else, including the packaging and assembly code (`src/openarm_assets/*.py`): MIT. See [`LICENSE`](LICENSE).

`add_gravcomp` in `assembly.py` is adapted from `mj_manipulator.arm.add_subtree_gravcomp` (MIT). `examples/view_collisions.py` is adapted from [geodude_assets](https://github.com/personalrobotics/geodude_assets) (MIT).
