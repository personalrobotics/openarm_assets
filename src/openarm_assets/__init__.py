"""MuJoCo models for the OpenArm robot."""

from pathlib import Path
from typing import Literal

__version__ = "0.1.0"
MODELS_DIR = Path(__file__).parent / "models"

# Just OpenArm for now
AVAILABLE_MODELS = ["openarm"]

GeneratedSide = Literal["bimanual", "left", "right"]

def get_model_path(name: str = "openarm") -> Path:
    """Get the path to a vendored (undecorated) MuJoCo model XML file.

    This looks in vendor/ only -- for a decorated model with an
    ee_site/gravcomp/finger-exclude already applied, use
    get_generated_model_path() instead.
    """
    model_dir = MODELS_DIR / name

    if not model_dir.exists():
        raise FileNotFoundError(f"Model '{name}' not found. Available models: {AVAILABLE_MODELS}")

    xml_candidates = [
        model_dir / "vendor" / f"{name}.xml",
        model_dir / "vendor" / "scene.xml",
    ]

    for xml_path in xml_candidates:
        if xml_path.exists():
            return xml_path

    raise FileNotFoundError(f"No XML file found in {model_dir}/vendor. Tried {[p.name for p in xml_candidates]}")

def get_generated_model_path(sides: GeneratedSide = "bimanual", name: str = "openarm") -> Path:
    """Get the path to a decorated (generated) MuJoCo model XML file.

    Raises FileNotFoundError with a build hint if the file hasn't been
    generated yet -- these are build output (from assembly.py's
    build_openarm* functions), not committed source, so a missing file
    here means "run the build," not "something's broken."
    """
    xml_path = MODELS_DIR / name / "generated" / f"{name}_{sides}_decorated.xml"

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Generated model not found at {xml_path}. "
            f"Run: python -m openarm_assets.assembly --sides {sides}"
        )

    return xml_path

