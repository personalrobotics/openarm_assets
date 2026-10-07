# SPDX-License-Identifier: MIT

"""Tests for the openarm_assets package."""

import runpy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pytest

from openarm_assets import (
    AVAILABLE_MODELS,
    MODELS_DIR,
    get_generated_model_path,
    get_model_path,
)
from openarm_assets.assembly import (
    FINGER_KP,
    JOINT4_LOWER_LIMIT_MARGIN_RAD,
    build_openarm,
)

ALL_SIDES = ["bimanual", "left", "right"]
ARMS = ["left", "right"]
VENDOR_DIR = MODELS_DIR / "openarm" / "vendor"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def load(path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))


def decorated_arms(sides: str) -> list[str]:
    return ARMS if sides == "bimanual" else [sides]


def name2id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    return mujoco.mj_name2id(model, obj, name)


@pytest.fixture(params=ALL_SIDES)
def generated(request):
    """(sides, model) for each generated model."""
    return request.param, load(get_generated_model_path(request.param))


class TestModelPaths:
    """Tests for model path functions."""

    def test_models_dir_exists(self):
        assert MODELS_DIR.is_dir()

    def test_get_model_path_default(self):
        """get_model_path defaults to the vendored single-arm model."""
        path = get_model_path()
        assert path == VENDOR_DIR / "openarm.xml"
        assert path.exists()

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_get_model_path_all_models(self, model_name):
        path = get_model_path(model_name)
        assert path.exists()
        assert path.suffix == ".xml"

    def test_get_model_path_invalid(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            get_model_path("nonexistent_model")

    @pytest.mark.parametrize("sides", ALL_SIDES)
    def test_get_generated_model_path(self, sides):
        path = get_generated_model_path(sides)
        assert path.exists()
        assert path.name == f"openarm_{sides}_decorated.xml"

    def test_get_generated_model_path_missing_has_build_hint(self):
        with pytest.raises(FileNotFoundError, match="python -m openarm_assets.assembly"):
            get_generated_model_path("bimanual", name="nonexistent_model")


class TestModelLoading:
    """Every shipped model loads and steps in MuJoCo."""

    @pytest.mark.parametrize("filename", ["openarm.xml", "openarm_bimanual.xml", "scene.xml"])
    def test_vendored_models_load_and_step(self, filename):
        model = load(VENDOR_DIR / filename)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        assert model.nq > 0
        assert data.time > 0

    def test_generated_models_load_and_step(self, generated):
        _, model = generated
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        assert data.time > 0

    def test_every_model_file_is_well_formed_xml(self):
        """Every model parses with a strict XML parser, not only MuJoCo's lenient one."""
        malformed = []
        for path in sorted(MODELS_DIR.rglob("*.xml")):
            try:
                ET.parse(path)
            except ET.ParseError as e:
                malformed.append(f"{path.relative_to(MODELS_DIR)}: {e}")
        assert malformed == []


class TestAssembly:
    """The generated models are what assembly.py produces, with every decoration applied."""

    @pytest.mark.parametrize("sides", ALL_SIDES)
    def test_generated_model_matches_its_sources(self, sides):
        """Each generated XML compiles to exactly the model the assembly builds from the
        vendored source, so no change lives only in the generated file.

        Compiled models are compared rather than XML text because MjSpec.to_xml()
        formatting (e.g. content_type, identity quats) varies between MuJoCo versions,
        and to a relative tolerance because the XML rounds values to 6 significant digits.
        """
        path = get_generated_model_path(sides)
        committed = load(path)
        spec = build_openarm(sides=tuple(decorated_arms(sides)))
        spec.modelfiledir = str(path.parent)
        regenerated = spec.compile()

        differing = []
        for field in dir(committed):
            value = getattr(committed, field, None)
            if field.startswith("_") or not isinstance(value, np.ndarray) or value.dtype == object:
                continue
            other = getattr(regenerated, field)
            if value.shape != other.shape or not np.allclose(value, other, rtol=1e-5, atol=1e-6, equal_nan=True):
                differing.append(field)
        assert differing == [], (
            f"openarm_{sides}_decorated.xml differs from its sources in {differing}; regenerate it with "
            f"`uv run python -m openarm_assets.assembly --sides {sides}`"
        )

    def test_ee_sites_only_on_decorated_arms(self, generated):
        sides, model = generated
        for arm in ARMS:
            site = name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"openarm_{arm}_ee_site")
            assert (site != -1) == (arm in decorated_arms(sides)), f"{arm} ee_site"

    def test_gravcomp_only_on_decorated_arms(self, generated):
        sides, model = generated
        for arm in ARMS:
            link1 = name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{arm}_link1")
            expected = 1.0 if arm in decorated_arms(sides) else 0.0
            assert model.body_gravcomp[link1] == expected, f"{arm} gravcomp"

    def test_finger_exclusion_only_on_decorated_arms(self, generated):
        sides, model = generated
        excluded = set()
        for sig in model.exclude_signature[: model.nexclude]:
            b1, b2 = sig >> 16, sig & 0xFFFF
            excluded.add((min(b1, b2), max(b1, b2)))
        for arm in ARMS:
            b1 = name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{arm}_left_finger")
            b2 = name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{arm}_right_finger")
            assert b1 != -1 and b2 != -1
            is_excluded = (min(b1, b2), max(b1, b2)) in excluded
            assert is_excluded == (arm in decorated_arms(sides)), f"{arm} finger exclusion"

    def test_joint4_lower_limit_widened_on_both_arms(self, generated):
        _, model = generated
        for arm in ARMS:
            joint = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{arm}_joint4")
            assert model.jnt_range[joint][0] == pytest.approx(JOINT4_LOWER_LIMIT_MARGIN_RAD)

    def test_finger_actuators_are_position_servos_on_both_arms(self, generated):
        """Every finger actuator (one per finger, four in all) is a position servo with gain FINGER_KP."""
        _, model = generated
        finger_actuators = 0
        for a in range(model.nu):
            joint = model.actuator_trnid[a][0]
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE:
                continue
            finger_actuators += 1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            assert model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE, name
            assert model.actuator_gainprm[a][0] == pytest.approx(FINGER_KP), name
            assert model.actuator_biasprm[a][1] == pytest.approx(-FINGER_KP), name
        assert finger_actuators == 4

    def test_has_ready_keyframe(self, generated):
        _, model = generated
        assert name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready") != -1


class TestKeyframes:
    """Each keyframe is a valid, collision-free starting pose."""

    def test_every_keyframe_is_within_joint_limits_and_free_of_contact(self, generated, request):
        sides, model = generated
        if sides != "bimanual":
            request.applymarker(
                pytest.mark.xfail(
                    strict=True,
                    reason="Single-arm models only add the finger exclusion to the decorated arm, so "
                    "the other arm's fingers interpenetrate (~3.5 mm) at the ready pose",
                )
            )
        data = mujoco.MjData(model)
        problems = []
        for key in range(model.nkey):
            key_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, key)
            for j in range(model.njnt):
                if not model.jnt_limited[j]:
                    continue
                q = model.key_qpos[key][model.jnt_qposadr[j]]
                low, high = model.jnt_range[j]
                if not low <= q <= high:
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    problems.append(f"{key_name}: {name}={q:.4f} not in [{low:.4f}, {high:.4f}]")
            mujoco.mj_resetDataKeyframe(model, data, key)
            mujoco.mj_forward(model, data)
            if data.ncon:
                problems.append(f"{key_name}: {data.ncon} contacts")
        assert model.nkey > 0 and problems == []

    def test_closing_grippers_at_ready_makes_no_contacts(self):
        """The fingers can close on empty space without touching each other or the hand."""
        model = load(get_generated_model_path("bimanual"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
        for a in range(model.nu):
            if model.jnt_type[model.actuator_trnid[a][0]] == mujoco.mjtJoint.mjJNT_SLIDE:
                data.ctrl[a] = model.actuator_ctrlrange[a][0]
        for _ in range(500):
            mujoco.mj_step(model, data)
        assert data.ncon == 0


class _FakeViewer:
    """Stands in for mujoco.viewer.launch_passive: drives one arm into the body,
    then records which geoms the collision viewer colored red."""

    red_geoms = None
    printed_geoms = False

    def __init__(self, model, data, key_callback):
        self.model, self.data, self.key_callback = model, data, key_callback
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.frames = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def is_running(self):
        self.frames += 1
        if self.frames == 1:
            # Move to a random (seeded) pose that collides.
            rng = np.random.default_rng(0)
            low, high = self.model.jnt_range.T
            while True:
                self.data.qpos[:] = rng.uniform(low, high)
                mujoco.mj_forward(self.model, self.data)
                if self.data.ncon:
                    break
        elif self.frames == 2:
            red = np.all(self.model.geom_rgba == [1.0, 0.0, 0.0, 1.0], axis=1)
            _FakeViewer.red_geoms = set(np.flatnonzero(red))
            self.key_callback(ord("R"))
        return self.frames < 3

    def sync(self):
        pass


class TestCollisionViewer:
    """examples/view_collisions.py runs headless with a stand-in viewer."""

    def test_colliding_bodies_turn_red(self, monkeypatch, capsys):
        monkeypatch.setattr(mujoco.viewer, "launch_passive", _FakeViewer)
        monkeypatch.setattr(sys, "platform", "linux")  # take the launch_passive path
        monkeypatch.setattr(sys, "argv", ["view_collisions.py"])
        _FakeViewer.red_geoms = None

        runpy.run_path(str(EXAMPLES_DIR / "view_collisions.py"), run_name="__main__")

        out = capsys.readouterr().out
        assert "Collisions (" in out
        assert _FakeViewer.red_geoms, "no geoms were colored red"
        assert "Done." in out
