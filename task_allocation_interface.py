"""
task_allocation_interface.py

What this is
------------
A small PySide6 interface for manually testing a trained RLlib PPO/MAPPO task-allocation checkpoint.

The interface lets you:
  - load a trained checkpoint or a trial folder containing checkpoint_* subfolders;
  - type a natural-language task instruction;
  - set requirement classes manually, or auto-infer them from the instruction;
  - run deterministic multi-agent inference and inspect each agent's bid, eligibility, and winner status.

The MAPPO inference path intentionally stays aligned with the offline evaluator:
  - the checkpoint env_config and evaluation env_config are merged;
  - tasks are injected directly into env.current_task with normalized requirements;
  - observations are built by env._get_obs();
  - actions are computed with Algorithm.compute_single_action(..., policy_id=..., explore=False);
  - winner/eligibility/payment/profit logic comes from env._calculate_rewards(...).

Requirement auto-inference
--------------------------
The auto-inference logic is heuristic and is copied from the requirement-inference part of
marl_bridge_gui.py. It extracts masses in kg, reaches in cm, and phrase hints such as
"no mobility required", "narrow spaces", or "uneven terrain".

Important: the checkpoint was trained with explicit requirement classes. For exact parity with
dataset/evaluation results, verify the inferred Mobility / Manipulation / Payload classes before
running bids. Auto-inference is optional and is disabled by default.

Requirements
------------
You need:
  - ray[rllib]
  - PySide6
  - project modules importable from the current working directory:
      heterogeneous_team_env.py, train.py, task_generator.py

How to run
----------
From the project root:
  python task_allocation_interface.py
"""

import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import ray

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from heterogeneous_team_env import HeterogeneousTeamEnv
from train import MAPPOCentralCriticModel


# Global evaluator thresholds.
MAX_SUPPORTED_MASS_KG = 50.0
MAX_SUPPORTED_REACH_CM = 130.0
HUMAN_ONLY_REACH_CM = 5.0

# Requirement tuples with no available agent; the interface warns instead of blocking.
UNSUPPORTED_REQUIREMENT_TUPLES = {
    (1, 2, 4),
    (1, 3, 4),
    (1, 4, 4),
    (2, 2, 4),
    (2, 3, 4),
    (2, 4, 4),
    (3, 1, 4),
    (3, 2, 4),
    (3, 3, 4),
    (4, 2, 4),
    (4, 3, 4),
    (4, 4, 4),
    (3, 4, 4),
    (4, 1, 4),
    (1, 1, 1),
    (1, 1, 2),
    (1, 1, 3),
    (1, 1, 4),
}

# Payload classes from your definition:
# class 1: 0..3 kg
# class 2: >3..5 kg
# class 3: >5..10 kg
# class 4: >10..50 kg
PAYLOAD_CLASS_THRESHOLDS_KG = (3.0, 5.0, 10.0)

# Places mentioned in your templates.
KNOWN_PLACES = [
    "warehouse",
    "assembly line",
    "station a",
    "station b",
    "station c",
    "workshop",
    "quality control room",
    "workstation",
    "maintenance zone",
    "logistics area",
    "assembly zone",
    "inspection station",
    "storage",
    "office",
    "production floor",
]


# -----------------------------------------------------------------------------
# INSTRUCTION CLASS HINTS
# Phrase-first dictionaries extracted from your templates.
# -----------------------------------------------------------------------------
INSTRUCTION_CLASS_HINTS: Dict[str, Dict[int, Dict[str, List[str]]]] = {
    "manipulation": {
        1: {
            "phrases": [
                "no manipulation required",
                "zero manipulation expected",
                "no object handling required",
                "with no manipulation expected",
                "zero manipulation required",
                "handling actions not required",
                "without performing any manipulation",
                "no manipulation needed",
                "no picking or placing",
                "without any picking or placing",
                "without any pick and place actions",
                "requiring no object handling",
                "no need for manipulation",
                "with zero manipulation expected",
                "no object handling involved",
                "no object handling needed",
            ],
            "keywords": [
                "no manipulation",
                "zero manipulation",
                "no object handling",
                "handling actions not required",
                "no picking or placing",
                "no pick and place",
                "without performing any manipulation",
                "without any pick and place actions",
                "no need for manipulation",
            ],
        },
    },
    "mobility": {
        1: {
            "phrases": [
                "no mobility required",
                "no repositioning needed",
                "without relocating",
                "no change in location",
                "done in place",
                "from a fixed position",
                "requiring no relocation",
                "with no need for mobility",
                "no mobility expected",
                "without needing to move",
                "without needing to move to other place",
                "without needing to move to other location",
                "without moving to other location",
                "workspace next to you",
                "no movement",
                "requiring no movement",
                "no movement to other place",
                "no movement to other location",
                "close requiring no movement",
                "in place",
                "fixed position",
                "at your workstation",
                "your workstation",
            ],
            "keywords": [
                "no mobility",
                "no repositioning",
                "without relocating",
                "fixed position",
                "done in place",
                "no relocation",
                "no need for mobility",
                "workspace next to you",
                "no movement",
                "in place",
            ],
        },
        2: {
            "phrases": [
                "navigate to",
                "navigate from coordinates",
                "transport",
                "deliver",
                "move from",
                "move to",
                "move to coordinates",
                "go to",
                "proceed to",
                "carry",
                "from warehouse to",
                "to station",
                "to workstation",
                "to assembly line",
                "to inspection station",
            ],
            "keywords": [
                "navigate",
                "transport",
                "deliver",
                "move",
                "go",
                "proceed",
                "carry",
            ],
        },
        3: {
            "phrases": [
                "through narrow spaces",
                "via tight access routes",
                "through confined paths",
                "through narrow corridors",
                "through compact areas",
                "across limited passageways",
                "through tight access points",
                "within constrained space",
                "through restricted paths",
                "via narrow aisles",
                "through tight passages",
                "across confined areas",
                "through compact access routes",
                "through limited space",
                "requiring high maneuverability",
                "through compact entryways",
                "within narrow workspace conditions",
                "demanding high mobility",
                "high planar maneuverability",
                "confined spaces",
                "constrained space",
                "restricted paths",
                "confined areas",
                "limited space",
                "tight access points",
                "tight passages",
                "narrow spaces",
                "narrow aisles",
                "compact entryways",
                "compact access routes",
                "narrow passages",
            ],
            "keywords": [
                "narrow",
                "tight",
                "confined",
                "constrained",
                "restricted",
                "compact",
                "maneuverability",
                "high mobility",
                "high planar maneuverability",
            ],
        },
        4: {
            "phrases": [
                "stepped and uneven ground",
                "stepped and uneven terrain",
                "stairways and irregular ground",
                "up stairs",
                "over level changes",
                "uneven ground and obstacles",
                "climb a ramp and stepped surface",
                "across uneven floors",
                "up ramps and stairs",
                "restricted access points and stairways",
                "cluttered, unstructured environment",
                "through uneven terrain",
                "over irregular floor with elevation changes",
                "across rough, discontinuous terrain",
                "over steps and narrow passages",
                "through stepped walkways and uneven surfaces",
                "across irregular terrain with multiple elevation changes",
                "through stairways and narrow, uneven corridors",
                "through irregular floor and stepped surfaces",
                "through stairs and uneven surfaces",
                "reach upper office",
                "reach upper floor",
                "reach upper room",
                "reach upper level",
                "upper platform",
                "upper office",
                "upper floor",
                "upper room",
                "upper level",
                "stairways",
                "uneven terrain",
                "uneven ground",
                "irregular ground",
                "irregular floor",
                "elevation changes",
                "stepped surfaces",
                "stepped walkways",
                "ramp",
                "ramps",
                "stairs",
                "obstacles",
                "rough terrain",
                "discontinuous terrain",
            ],
            "keywords": [
                "stairs",
                "stairways",
                "ramp",
                "ramps",
                "uneven",
                "irregular",
                "stepped",
                "elevation changes",
                "obstacles",
                "rough terrain",
                "upper level",
                "upper floor",
                "upper office",
                "upper room",
                "upper platform",
            ],
        },
    },
    "payload": {
        1: {
            "phrases": ["small light", "small light parts", "lightweight"],
            "keywords": ["small", "tiny", "lightweight", "delicate", "light"],
        },
        2: {
            "phrases": ["small heavy", "medium-weight", "compact heavy"],
            "keywords": ["medium", "heavy"],
        },
        3: {
            "phrases": ["small very heavy", "very heavy", "big size", "big size heavy", "large heavy"],
            "keywords": ["big", "large"],
        },
        4: {
            "phrases": ["super heavy", "big size super heavy", "large super heavy"],
            "keywords": ["super heavy"],
        },
    },
}

PICK_PLACE_MANIPULATION_PHRASES = [
    "pick",
    "pick up",
    "grasp",
    "handle",
    "place",
    "position",
    "remove",
    "feed",
    "load",
    "unload",
    "manipulate",
    "assembly",
    "assemble",
    "lift",
    "sort",
    "transfer",
]

TRANSPORT_VERBS = [
    "deliver",
    "transport",
    "carry",
    "navigate",
    "move",
    "go to",
    "proceed to",
]

UNIT_COUNT_NOUNS = [
    "unit",
    "units",
    "part",
    "parts",
    "component",
    "components",
    "product",
    "products",
    "finished product",
    "finished products",
    "bolt",
    "bolts",
    "box",
    "boxes",
    "item",
    "items",
    "tool",
    "tools",
    "instrument",
    "instruments",
    "material",
    "materials",
    "package",
    "packages",
    "load",
    "loads",
]

MOBILITY_PRIORITY = [4, 3, 1, 2]


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class ValidationResult:
    warnings: List[str] = field(default_factory=list)
    hard_stop: bool = False
    force_human: bool = False
    masses_kg: List[float] = field(default_factory=list)
    reaches_cm: List[float] = field(default_factory=list)


@dataclass
class RequirementGuess:
    mobility: int
    manipulation: int
    payload: int
    reasons: List[str] = field(default_factory=list)


@dataclass
class ProgramMatch:
    program: Optional[str]
    score: int = -1
    matched_phrases: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Text utilities
# -----------------------------------------------------------------------------
def normalize_instruction_for_marl(text: str) -> str:
    """
    Canonical normalization before MARL parsing / embedding.

    - converts decimal commas to decimal dots: 3,5 -> 3.5
    - collapses repeated whitespace
    - removes trailing dots at the very end of the instruction
      so 'Pick up box.' and 'Pick up box' embed identically
    """
    if not text:
        return text

    text = str(text)

    # 1) decimal comma -> decimal dot
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    # 2) normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 3) remove one or more trailing dots only at the end
    text = re.sub(r"\.+$", "", text).strip()

    return text


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9\s:.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_number(raw: str) -> float:
    return float(raw.replace(",", "."))


def phrase_in_text(text_norm: str, phrase: str) -> bool:
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return False

    if " " in phrase_norm or ":" in phrase_norm:
        haystack = f" {text_norm} "
        needle = f" {phrase_norm} "
        return needle in haystack

    pattern = rf"\b{re.escape(phrase_norm)}\b"
    return re.search(pattern, text_norm) is not None


def first_matching_phrase(text_norm: str, phrases: Sequence[str]) -> Optional[str]:
    for phrase in phrases:
        if phrase_in_text(text_norm, phrase):
            return phrase
    return None


def match_hint(text_norm: str, metric: str, class_id: int) -> Optional[str]:
    bucket = INSTRUCTION_CLASS_HINTS.get(metric, {}).get(class_id, {})
    hit = first_matching_phrase(text_norm, bucket.get("phrases", []))
    if hit is not None:
        return hit
    return first_matching_phrase(text_norm, bucket.get("keywords", []))


def extract_masses_kg(text: str) -> List[float]:
    pattern = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(?:kg|kgs|kilogram|kilograms)\b", re.IGNORECASE)
    return [parse_number(m.group(1)) for m in pattern.finditer(text)]


def extract_reaches_cm(text: str) -> List[float]:
    pattern = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(?:cm|centimeter|centimeters)\b", re.IGNORECASE)
    return [parse_number(m.group(1)) for m in pattern.finditer(text)]


def extract_unit_counts(text_norm: str) -> List[int]:
    noun_pattern = "|".join(sorted({normalize_text(x) for x in UNIT_COUNT_NOUNS}, key=len, reverse=True))
    pattern = re.compile(
        rf"(?<!\w)(\d+(?:\.\d+)?)\s+(?:[a-z]+\s+){{0,2}}(?:{noun_pattern})\b",
        re.IGNORECASE,
    )
    counts: List[int] = []
    for m in pattern.finditer(text_norm):
        try:
            counts.append(int(round(float(m.group(1)))))
        except Exception:
            pass
    return counts


# -----------------------------------------------------------------------------
# Global evaluator
# -----------------------------------------------------------------------------
def evaluate_instruction_constraints(instruction: str) -> ValidationResult:
    result = ValidationResult()

    masses = extract_masses_kg(instruction)
    reaches = extract_reaches_cm(instruction)

    result.masses_kg = masses
    result.reaches_cm = reaches

    for mass in masses:
        if mass > MAX_SUPPORTED_MASS_KG:
            result.hard_stop = True
            result.warnings.append(
                f"Unsupported mass detected: {mass:.3f} kg is above the supported limit of {MAX_SUPPORTED_MASS_KG:.1f} kg."
            )

    for reach in reaches:
        if reach > MAX_SUPPORTED_REACH_CM:
            result.hard_stop = True
            result.warnings.append(
                f"Unsupported reach detected: {reach:.1f} cm is above the supported limit of {MAX_SUPPORTED_REACH_CM:.1f} cm."
            )
        elif reach < HUMAN_ONLY_REACH_CM:
            result.force_human = True
            result.warnings.append(
                f"Human-only task: detected reach {reach:.1f} cm, which is below {HUMAN_ONLY_REACH_CM:.1f} cm."
            )

    return result


def requirement_tuple_warning(req_1to4: Sequence[int]) -> Optional[str]:
    req_tuple = tuple(int(x) for x in req_1to4)
    if req_tuple in UNSUPPORTED_REQUIREMENT_TUPLES:
        return (
            "None of the agents considered can perform that task."
            f"Requirement tuple detected: [{req_tuple[0]}, {req_tuple[1]}, {req_tuple[2]}]."
        )
    return None


# -----------------------------------------------------------------------------
# Requirement inference
# -----------------------------------------------------------------------------
def infer_payload_class(text_norm: str, masses_kg: Sequence[float]) -> Tuple[int, str]:
    if masses_kg:
        mass = max(float(x) for x in masses_kg)
        t1, t2, t3 = PAYLOAD_CLASS_THRESHOLDS_KG
        if mass <= t1:
            return 1, f"payload class 1 from mass {mass:.3f} kg"
        if mass <= t2:
            return 2, f"payload class 2 from mass {mass:.3f} kg"
        if mass <= t3:
            return 3, f"payload class 3 from mass {mass:.3f} kg"
        return 4, f"payload class 4 from mass {mass:.3f} kg"

    for class_id in [4, 3, 2, 1]:
        hit = match_hint(text_norm, "payload", class_id)
        if hit is not None:
            return class_id, f"payload class {class_id} from '{hit}' wording"

    counts = extract_unit_counts(text_norm)
    if counts:
        count = max(counts)
        if count <= 3:
            return 1, f"payload class 1 from unit count {count}"
        if count <= 6:
            return 2, f"payload class 2 from unit count {count}"
        if count <= 9:
            return 3, f"payload class 3 from unit count {count}"
        return 4, f"payload class 4 from unit count {count}"

    return 1, "payload defaulted to class 1"


def infer_mobility_class(text_norm: str) -> Tuple[int, str]:
    for class_id in MOBILITY_PRIORITY:
        hit = match_hint(text_norm, "mobility", class_id)
        if hit is not None:
            return class_id, f"mobility class {class_id} from '{hit}'"

    place_hits = [place for place in KNOWN_PLACES if phrase_in_text(text_norm, place)]
    has_coordinate_motion = phrase_in_text(text_norm, "coordinates") or phrase_in_text(text_norm, "x:") or phrase_in_text(text_norm, "y:")
    has_transport = first_matching_phrase(text_norm, TRANSPORT_VERBS) is not None

    if has_transport or has_coordinate_motion or place_hits:
        why_bits: List[str] = []
        if has_transport:
            why_bits.append("transport/navigation wording")
        if has_coordinate_motion:
            why_bits.append("coordinate wording")
        if place_hits:
            why_bits.append(f"place wording ({', '.join(place_hits[:2])})")
        reason = " + ".join(why_bits) if why_bits else "default travel wording"
        return 2, f"mobility class 2 from {reason}"

    return 2, "mobility defaulted to class 2"


def infer_manipulation_class(text_norm: str, reaches_cm: Sequence[float]) -> Tuple[int, str]:
    hit = match_hint(text_norm, "manipulation", 1)
    if hit is not None:
        return 1, f"manipulation class 1 from '{hit}'"

    if reaches_cm:
        reach = max(float(x) for x in reaches_cm)
        if reach <= 50.0:
            return 2, f"manipulation class 2 from reach {reach:.1f} cm"
        if reach <= 90.0:
            return 3, f"manipulation class 3 from reach {reach:.1f} cm"
        return 4, f"manipulation class 4 from reach {reach:.1f} cm"

    has_pick_place = first_matching_phrase(text_norm, PICK_PLACE_MANIPULATION_PHRASES)
    has_transport = first_matching_phrase(text_norm, TRANSPORT_VERBS)
    if has_pick_place and has_transport:
        return 3, f"manipulation class 3 from '{has_pick_place}' + transport wording without explicit reach"
    if has_pick_place:
        return 2, f"manipulation class 2 from '{has_pick_place}' wording without explicit reach"

    return 1, "manipulation defaulted to class 1"


def infer_requirement_classes(instruction: str) -> RequirementGuess:
    instruction_norm = normalize_instruction_for_marl(instruction)
    text_norm = normalize_text(instruction_norm)
    masses = extract_masses_kg(instruction_norm)
    reaches = extract_reaches_cm(instruction_norm)

    mobility, reason_m = infer_mobility_class(text_norm)
    manipulation, reason_man = infer_manipulation_class(text_norm, reaches)
    payload, reason_p = infer_payload_class(text_norm, masses)

    return RequirementGuess(
        mobility=int(np.clip(mobility, 1, 4)),
        manipulation=int(np.clip(manipulation, 1, 4)),
        payload=int(np.clip(payload, 1, 4)),
        reasons=[reason_m, reason_man, reason_p],
    )


# -----------------------------------------------------------------------------
# RLlib registries
# -----------------------------------------------------------------------------
def _ensure_rllib_registries():
    # Env registration can be repeated; Ray may warn but it's OK.
    register_env("heterogeneous_team_env", lambda cfg: HeterogeneousTeamEnv(cfg))

    # Guard against double-registration
    if not getattr(ModelCatalog, "_mappo_cc_model_registered", False):
        ModelCatalog.register_custom_model("mappo_cc_model", MAPPOCentralCriticModel)
        ModelCatalog._mappo_cc_model_registered = True


# -----------------------------------------------------------------------------
# Checkpoint resolution (same idea as your evaluator)
# -----------------------------------------------------------------------------
def _looks_like_rllib_checkpoint_dir(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    files = set(os.listdir(p))
    expected_any = {
        "rllib_checkpoint.json",
        "algorithm_state.pkl",
        "checkpoint.json",
        "policies",
    }
    return len(files.intersection(expected_any)) > 0


def _checkpoint_sort_key(path: str) -> int:
    base = os.path.basename(path)
    if not base.startswith("checkpoint_"):
        return -1
    suffix = base.split("checkpoint_", 1)[-1]
    try:
        return int(suffix)
    except Exception:
        digits = "".join([c for c in suffix if c.isdigit()])
        return int(digits) if digits else -1


def resolve_checkpoint_dir(checkpoint_path: str) -> str:
    ckpt = os.path.abspath(os.path.expanduser(checkpoint_path))

    # Case 1: passed a checkpoint dir directly
    if os.path.isdir(ckpt) and _looks_like_rllib_checkpoint_dir(ckpt):
        return ckpt

    # Case 2: passed a folder containing checkpoint_* subdirs
    if os.path.isdir(ckpt):
        subs = []
        for name in os.listdir(ckpt):
            if name.startswith("checkpoint_"):
                sub = os.path.join(ckpt, name)
                if os.path.isdir(sub):
                    subs.append(sub)
        if subs:
            subs_sorted = sorted(subs, key=_checkpoint_sort_key)
            pick = subs_sorted[-1]
            if _looks_like_rllib_checkpoint_dir(pick):
                return pick

    raise FileNotFoundError(
        f"Could not resolve a valid RLlib checkpoint directory from:\n{checkpoint_path}\n"
        f"Tip: pass a checkpoint_*/ directory, or a folder containing checkpoint_*/ subfolders."
    )


def _algo_config_to_dict(algo) -> Dict[str, Any]:
    cfg = getattr(algo, "config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    # AlgorithmConfig object
    if hasattr(cfg, "to_dict"):
        try:
            return cfg.to_dict()
        except Exception:
            pass
    return dict(cfg) if hasattr(cfg, "items") else {}


def _build_env_config_from_checkpoint(algo, *, max_steps: int = 1) -> Dict[str, Any]:
    """
    Merge base env_config + evaluation_config.env_config like evaluator does.
    Then force dataset_mode=False (GUI injects tasks), train_mode=False, etc.
    """
    cfg = _algo_config_to_dict(algo)
    base_env_cfg = dict(cfg.get("env_config", {}) or {})
    eval_cfg = dict(cfg.get("evaluation_config", {}) or {})
    eval_env_cfg = dict((eval_cfg.get("env_config", {}) or {}))

    merged = dict(base_env_cfg)
    merged.update(eval_env_cfg)

    merged["dataset_mode"] = False  # GUI injects tasks
    merged["train_mode"] = False
    merged["show_detailed_episodes"] = False
    merged["max_steps"] = int(max_steps)

    return merged


def _load_algorithm_from_checkpoint(checkpoint_path: str):
    _ensure_rllib_registries()
    ckpt_dir = resolve_checkpoint_dir(checkpoint_path)

    from ray.rllib.algorithms.algorithm import Algorithm
    return Algorithm.from_checkpoint(ckpt_dir), ckpt_dir


# -----------------------------------------------------------------------------
# Task injection using env conventions
# -----------------------------------------------------------------------------
def _inject_task_into_env(env: HeterogeneousTeamEnv, instruction: str, req_1to4: np.ndarray, task_id: int = 0) -> None:
    """
    IMPORTANT: env expects current_task["requirements"] to be NORMALIZED ([-1..1] scale),
    because env later does _unnormalize(current_task["requirements"]) to compute eligibility/masks.
    """
    req_1to4 = np.asarray(req_1to4, dtype=np.float32).reshape(3,)
    req_norm = env._normalize(req_1to4).astype(np.float32, copy=False)

    env.current_task = {
        "id": int(task_id),
        "instruction": str(instruction),
        "category": "gui",
        "requirements": req_norm,
        # optional extras (harmless)
        "requirements_raw": req_1to4,
        "requirements_norm": req_norm,
    }

    # If your env uses per-task mask decision, resample it to match injected task.
    if hasattr(env, "_sample_mask_decision_for_task"):
        try:
            env._sample_mask_decision_for_task()
        except Exception:
            pass


def _action_from_compute(ret) -> int:
    # RLlib version differences: action OR (action, state, info)
    if isinstance(ret, tuple):
        ret = ret[0]
    return int(np.asarray(ret).reshape(-1)[0])



# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
class TaskAllocationInterface(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Task Allocation Interface (MAPPO/RLlib)")
        self.resize(1080, 760)

        self._algo = None
        self._env: Optional[HeterogeneousTeamEnv] = None
        self._checkpoint_path = ""

        self._build_ui()
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # -------------------------- UI build --------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # 1) Checkpoint controls
        ckpt_group = QGroupBox("1) RLlib checkpoint")
        ckpt_layout = QHBoxLayout(ckpt_group)

        self.ckpt_line = QLineEdit()
        self.ckpt_line.setPlaceholderText(
            "Select checkpoint dir (checkpoint_0000xx, exported_checkpoints/best_checkpoint, or a trial folder)"
        )
        ckpt_layout.addWidget(self.ckpt_line)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.on_browse_checkpoint)
        ckpt_layout.addWidget(self.btn_browse)

        self.btn_load = QPushButton("Load checkpoint")
        self.btn_load.clicked.connect(self.on_load_checkpoint)
        ckpt_layout.addWidget(self.btn_load)

        layout.addWidget(ckpt_group)

        # 2) Instruction
        task_group = QGroupBox("2) Task instruction")
        task_layout = QVBoxLayout(task_group)

        self.prompt_text = QTextEdit()
        self.prompt_text.setPlaceholderText(
            "Example: Pick up a 3.5 kg component within a 65 cm reach and deliver it to the assembly line"
        )
        self.prompt_text.setMinimumHeight(120)
        self.prompt_text.textChanged.connect(self.refresh_instruction_analysis)
        task_layout.addWidget(self.prompt_text)

        layout.addWidget(task_group)

        # 3) Requirement inference + manual controls
        req_group = QGroupBox("3) Requirement classes used by the MAPPO checkpoint")
        req_layout = QGridLayout(req_group)

        self.chk_auto_requirements = QCheckBox(
            "Auto-infer requirement classes from instruction (heuristic; verify for exact parity)"
        )
        self.chk_auto_requirements.setChecked(False)
        self.chk_auto_requirements.stateChanged.connect(self.refresh_instruction_analysis)
        req_layout.addWidget(self.chk_auto_requirements, 0, 0, 1, 5)

        self.btn_autofill = QPushButton("Auto-fill now")
        self.btn_autofill.clicked.connect(self.on_autofill_requirements)
        req_layout.addWidget(self.btn_autofill, 0, 5, 1, 1)

        req_layout.addWidget(QLabel("Mobility"), 1, 0)
        self.spin_mob = QSpinBox()
        self.spin_mob.setRange(1, 4)
        self.spin_mob.setValue(2)
        self.spin_mob.valueChanged.connect(self.refresh_instruction_analysis)
        req_layout.addWidget(self.spin_mob, 1, 1)

        req_layout.addWidget(QLabel("Manipulation"), 1, 2)
        self.spin_man = QSpinBox()
        self.spin_man.setRange(1, 4)
        self.spin_man.setValue(2)
        self.spin_man.valueChanged.connect(self.refresh_instruction_analysis)
        req_layout.addWidget(self.spin_man, 1, 3)

        req_layout.addWidget(QLabel("Payload"), 1, 4)
        self.spin_pay = QSpinBox()
        self.spin_pay.setRange(1, 4)
        self.spin_pay.setValue(2)
        self.spin_pay.valueChanged.connect(self.refresh_instruction_analysis)
        req_layout.addWidget(self.spin_pay, 1, 5)

        self.req_summary = QLabel("Requirements: Mobility=2, Manipulation=2, Payload=2")
        self.req_summary.setWordWrap(True)
        req_layout.addWidget(self.req_summary, 2, 0, 1, 6)

        self.parsed_info_lbl = QLabel("Detected parameters: none")
        self.parsed_info_lbl.setWordWrap(True)
        self.parsed_info_lbl.setStyleSheet("color: #555;")
        req_layout.addWidget(self.parsed_info_lbl, 3, 0, 1, 6)

        self.req_reason_lbl = QLabel("Requirement inference notes: none")
        self.req_reason_lbl.setWordWrap(True)
        self.req_reason_lbl.setStyleSheet("color: #555;")
        req_layout.addWidget(self.req_reason_lbl, 4, 0, 1, 6)

        self.warning_box = QPlainTextEdit()
        self.warning_box.setReadOnly(True)
        self.warning_box.setMaximumHeight(90)
        self.warning_box.setPlaceholderText("Warnings will appear here.")
        req_layout.addWidget(self.warning_box, 5, 0, 1, 6)

        layout.addWidget(req_group)

        # Run/Clear buttons
        run_row = QHBoxLayout()
        layout.addLayout(run_row)

        self.btn_run = QPushButton("Run bids")
        self.btn_run.clicked.connect(self.on_run_bids)
        self.btn_run.setEnabled(False)
        run_row.addWidget(self.btn_run)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.on_clear)
        run_row.addWidget(self.btn_clear)

        run_row.addStretch(1)

        # Output table
        bids_group = QGroupBox("4) Predicted bids and environment outcome")
        bids_layout = QVBoxLayout(bids_group)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Agent ID", "Agent Type", "Eligible", "Bid", "Won", "Notes"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bids_layout.addWidget(self.table)

        layout.addWidget(bids_group)

        self.status_lbl = QLabel("Status: no checkpoint loaded")
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

        self.refresh_instruction_analysis()

    # -------------------------- lifecycle --------------------------
    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        super().closeEvent(event)

    # -------------------------- helper UI methods --------------------------
    def set_status(self, text: str) -> None:
        self.status_lbl.setText(f"Status: {text}")

    def current_requirement_vector(self) -> np.ndarray:
        return np.array(
            [self.spin_mob.value(), self.spin_man.value(), self.spin_pay.value()],
            dtype=np.float32,
        )

    def set_warnings(self, warnings: Sequence[str]) -> None:
        if warnings:
            self.warning_box.setPlainText("\n".join(str(w) for w in warnings))
        else:
            self.warning_box.setPlainText("No warnings.")

    def _update_requirement_summary(self, req: Sequence[int]) -> None:
        req = tuple(int(x) for x in req)
        self.req_summary.setText(
            f"Requirements: Mobility={req[0]}, Manipulation={req[1]}, Payload={req[2]}"
        )

    # -------------------------- requirement analysis --------------------------
    def refresh_instruction_analysis(self) -> None:
        instruction_raw = self.prompt_text.toPlainText().strip()
        current_req = tuple(int(x) for x in self.current_requirement_vector())
        self._update_requirement_summary(current_req)

        if not instruction_raw:
            self.parsed_info_lbl.setText("Detected parameters: none")
            self.req_reason_lbl.setText("Requirement inference notes: none")
            self.set_warnings([])
            return

        instruction = normalize_instruction_for_marl(instruction_raw)
        validation = evaluate_instruction_constraints(instruction)
        guess = infer_requirement_classes(instruction)

        if self.chk_auto_requirements.isChecked():
            self.spin_mob.setValue(guess.mobility)
            self.spin_man.setValue(guess.manipulation)
            self.spin_pay.setValue(guess.payload)
            req_preview = (guess.mobility, guess.manipulation, guess.payload)
        else:
            req_preview = tuple(int(x) for x in self.current_requirement_vector())

        self._update_requirement_summary(req_preview)

        warnings = list(validation.warnings)
        req_tuple_msg = requirement_tuple_warning(req_preview)
        if req_tuple_msg is not None:
            warnings.append(req_tuple_msg)

        masses_txt = ", ".join(f"{m:.3f} kg" for m in validation.masses_kg) if validation.masses_kg else "none"
        reaches_txt = ", ".join(f"{d:.1f} cm" for d in validation.reaches_cm) if validation.reaches_cm else "none"
        self.parsed_info_lbl.setText(
            "Detected parameters: "
            f"masses=[{masses_txt}] | reaches=[{reaches_txt}] | "
            f"inferred requirements(Mob,Man,Pay)=({guess.mobility}, {guess.manipulation}, {guess.payload})"
        )
        self.req_reason_lbl.setText("Requirement inference notes: " + " | ".join(guess.reasons))
        self.set_warnings(warnings)

    def on_autofill_requirements(self) -> None:
        instruction_raw = self.prompt_text.toPlainText().strip()
        if not instruction_raw:
            QMessageBox.warning(self, "Missing instruction", "Type an instruction first.")
            return

        guess = infer_requirement_classes(normalize_instruction_for_marl(instruction_raw))
        self.spin_mob.setValue(guess.mobility)
        self.spin_man.setValue(guess.manipulation)
        self.spin_pay.setValue(guess.payload)
        self._update_requirement_summary((guess.mobility, guess.manipulation, guess.payload))
        self.req_reason_lbl.setText("Requirement inference notes: " + " | ".join(guess.reasons))
        self.refresh_instruction_analysis()

    # -------------------------- checkpoint --------------------------
    def on_browse_checkpoint(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select RLlib checkpoint directory", os.getcwd())
        if path:
            self.ckpt_line.setText(path)

    def on_load_checkpoint(self) -> None:
        ckpt_in = self.ckpt_line.text().strip()
        if not ckpt_in:
            QMessageBox.warning(self, "Missing path", "Please select a checkpoint directory first.")
            return
        if not os.path.isdir(ckpt_in):
            QMessageBox.critical(self, "Invalid path", f"Not a directory:\n{ckpt_in}")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.set_status(f"loading checkpoint from {ckpt_in}")

            self._algo, resolved = _load_algorithm_from_checkpoint(ckpt_in)
            self._checkpoint_path = resolved

            env_cfg = _build_env_config_from_checkpoint(self._algo, max_steps=1)
            self._env = HeterogeneousTeamEnv(env_cfg)

            try:
                self._env.reset(seed=0)
            except Exception:
                pass

            self.btn_run.setEnabled(True)
            self.set_status(f"checkpoint loaded OK from {resolved}")

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load failed", f"{e}\n\n{tb}")
            self.set_status("checkpoint load failed")
            self._algo = None
            self._env = None
            self.btn_run.setEnabled(False)
        finally:
            QApplication.restoreOverrideCursor()

    # -------------------------- inference --------------------------
    def on_run_bids(self) -> None:
        if self._algo is None or self._env is None:
            QMessageBox.warning(self, "Not ready", "Load a checkpoint first.")
            return

        instruction_raw = self.prompt_text.toPlainText().strip()
        if not instruction_raw:
            QMessageBox.warning(self, "Missing instruction", "Please type an instruction first.")
            return

        instruction = normalize_instruction_for_marl(instruction_raw)
        validation = evaluate_instruction_constraints(instruction)
        guess = infer_requirement_classes(instruction)

        if self.chk_auto_requirements.isChecked():
            self.spin_mob.setValue(guess.mobility)
            self.spin_man.setValue(guess.manipulation)
            self.spin_pay.setValue(guess.payload)

        req = self.current_requirement_vector()
        self._update_requirement_summary(req)

        warnings = list(validation.warnings)
        req_tuple_msg = requirement_tuple_warning(req)
        if req_tuple_msg is not None:
            warnings.append(req_tuple_msg)
        self.set_warnings(warnings)
        self.req_reason_lbl.setText("Requirement inference notes: " + " | ".join(guess.reasons))

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Inject task into env and use env._get_obs() for inference parity.
            _inject_task_into_env(self._env, instruction, req, task_id=0)
            obs = self._env._get_obs()

            # Compute bids through the Algorithm API, one policy per agent.
            bids: Dict[int, int] = {}
            for aid in self._env._agent_ids:
                pid = f"policy_{int(aid)}"
                act = self._algo.compute_single_action(obs[int(aid)], policy_id=pid, explore=False)
                bids[int(aid)] = int(np.clip(_action_from_compute(act), 0, 10))

            # Ask the environment to compute outcomes so GUI tie-breaking cannot drift from env logic.
            _rewards, outcomes = self._env._calculate_rewards(bids, return_outcomes=True, raw_actions=bids)

            rows = []
            for aid in self._env._agent_ids:
                atype = self._env.agent_type_names.get(self._env.agent_types[aid], f"Type{self._env.agent_types[aid]}")
                od = outcomes.get(int(aid), {})
                eligible = bool(od.get("eligible", False))
                won = bool(od.get("won", False))
                bid = int(bids[int(aid)])

                notes = []
                if bid == 0:
                    notes.append("abstain")
                if (not eligible) and bid > 0:
                    notes.append("INELIGIBLE BID")
                if won:
                    notes.append("WINNER")

                rows.append((aid, atype, "yes" if eligible else "no", bid, "yes" if won else "", "; ".join(notes)))

            self.table.setRowCount(len(rows))
            for r, (aid, atype, elig_str, bid, won_str, note) in enumerate(rows):
                self.table.setItem(r, 0, QTableWidgetItem(str(aid)))
                self.table.setItem(r, 1, QTableWidgetItem(str(atype)))
                self.table.setItem(r, 2, QTableWidgetItem(str(elig_str)))
                self.table.setItem(r, 3, QTableWidgetItem(str(bid)))
                self.table.setItem(r, 4, QTableWidgetItem(str(won_str)))
                self.table.setItem(r, 5, QTableWidgetItem(str(note)))

            winner = None
            for aid in self._env._agent_ids:
                if bool(outcomes.get(int(aid), {}).get("won", False)):
                    winner = int(aid)
                    break

            if winner is None:
                self.set_status("inference finished with no winner")
            else:
                winner_name = self._env.agent_type_names.get(self._env.agent_types[winner], f"Agent {winner}")
                self.set_status(f"inference finished: winner={winner} ({winner_name}), bid={bids[winner]}")

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Inference failed", f"{e}\n\n{tb}")
            self.set_status("inference failed")
        finally:
            QApplication.restoreOverrideCursor()

    # -------------------------- clear --------------------------
    def on_clear(self) -> None:
        self.prompt_text.clear()
        self.table.setRowCount(0)
        self.req_summary.setText("Requirements: Mobility=2, Manipulation=2, Payload=2")
        self.parsed_info_lbl.setText("Detected parameters: none")
        self.req_reason_lbl.setText("Requirement inference notes: none")
        self.set_warnings([])
        self.set_status("ready")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    app = QApplication(sys.argv)
    window = TaskAllocationInterface()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
