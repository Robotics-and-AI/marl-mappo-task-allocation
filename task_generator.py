"""
task_generator.py

This module provides three main components used by the RL environment:

1) TaskInstructionProcessor
   - Encodes a natural-language instruction string into a fixed-size vector embedding
     using SentenceTransformer (required dependency).
   - Appends lightweight "hybrid" features derived from regex-based parsing of the text
     (units + coarse class one-hots).

2) NLTaskGenerator
   - Generates natural-language tasks from parameterized templates.
   - Each "scenarioX" category has:
       - parameter ranges (weight/distance/etc.)
       - train templates
       - eval templates
   - The generator returns:
       {"instruction": str, "category": str, "requirements": np.ndarray(shape=(3,), dtype=float32)}
     where requirements are RAW in [1..4] using the ordering:
       [mobility_class, manipulation_class, payload_class]

3) BiddingTracker
   - Optional debugging helper to log bids/rewards per step and print summaries.

Also included:
- build_dataset(): a simple offline dataset generator for train/eval prompt generation.
"""

import os
from pathlib import Path
import numpy as np
import random
from tabulate import tabulate  # tables for debugging output
import re
from typing import Dict, Any, List, Optional, Tuple

# Optional at module import time. TaskInstructionProcessor still requires this dependency when used.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # allows using NLTaskGenerator without installing sentence-transformers
    SentenceTransformer = None  # type: ignore[assignment]


# ============================================================================
# 1) Instruction -> Embedding (SentenceTransformer required)
# ============================================================================
class TaskInstructionProcessor:
    """
    Encodes a natural language instruction into a fixed-dimensional vector.

    Why do we need this?
    --------------------
    RL policies typically consume numeric vectors, not text.
    Instruction text is converted into a dense embedding (e.g., 384 dims).

    Dependency policy:
    ------------------
    Sentence-transformers MUST be installed. If loading fails, an error is raised.

    Performance note (Ray workers):
    -------------------------------
    This class keeps a process-local shared model instance so that:
      - multiple TaskInstructionProcessor instances in ONE Python process reuse the same model
      - each Ray worker process will still load its own model once
    """

    # Shared cache across instances in the same process
    _shared_model: Optional[SentenceTransformer] = None
    _shared_model_name: Optional[str] = None
    _shared_device: Optional[str] = None
    _printed_once: bool = False

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        normalize_embeddings: bool = True,
        device: str = "cpu",
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "TaskInstructionProcessor requires sentence-transformers. "
                "Install it with `pip install sentence-transformers`."
            )

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.device = device

        # Load or reuse the shared model
        reload_needed = (
            TaskInstructionProcessor._shared_model is None
            or TaskInstructionProcessor._shared_model_name != model_name
            or TaskInstructionProcessor._shared_device != device
        )

        if reload_needed:
            # Fail if loading fails; no fallback
            TaskInstructionProcessor._shared_model = SentenceTransformer(model_name, device=device)
            TaskInstructionProcessor._shared_model_name = model_name
            TaskInstructionProcessor._shared_device = device

        self.model = TaskInstructionProcessor._shared_model
        assert self.model is not None  # for type checkers

        # Semantic embedding dim from model (e.g., 384)
        self.semantic_dim = int(self.model.get_sentence_embedding_dimension())

        # Hybrid features count:
        # 3 (unit flags) + 5 (payload bins incl unknown) + 5 (manip bins incl unknown) + 5 (mobility bins incl unknown) = 18
        self.hybrid_features_count = 18

        # Total embedding dimension returned by encode_instruction()
        self.embedding_dim = int(self.semantic_dim + self.hybrid_features_count)

        # Regex to extract values tied to specific units (avoids coordinate/max-number pollution)
        self._kg_re = re.compile(r"(\d+\.?\d*)\s*(kg|kilograms?|kilo(?:gram)?s?)\b", re.IGNORECASE)
        self._cm_re = re.compile(r"(\d+\.?\d*)\s*(cm|centimeters?|centimetres?)\b", re.IGNORECASE)

        # Detect meters as a unit without matching "cm" or "mm":
        # - matches "meter"/"meters"
        # - matches standalone "m" only when it is not adjacent to letters
        #   (so "cm" or "mm" will NOT trigger)
        self._meter_re = re.compile(r"\bmeters?\b|(?<![a-zA-Z])m(?![a-zA-Z])", re.IGNORECASE)

        # Print only once per process (Ray creates many envs)
        if not TaskInstructionProcessor._printed_once:
            print(
                f"Using SentenceTransformer for embeddings "
                f"(model={model_name}, device={device}, dim={self.semantic_dim} + {self.hybrid_features_count} = {self.embedding_dim})"
            )
            TaskInstructionProcessor._printed_once = True

    def _extract_max_value(self, pattern: "re.Pattern", text: str) -> Optional[float]:
        """Extract the maximum numeric value matched by a regex (used for kg/cm parsing)."""
        matches = pattern.findall(text)
        if not matches:
            return None

        vals: List[float] = []
        for m in matches:
            # m is tuple like ("3.50", "kg")
            try:
                vals.append(float(m[0]))
            except Exception:
                pass

        return float(np.max(vals)) if vals else None

    def _extract_hybrid_features(self, text: str) -> np.ndarray:
        """
        Build small, human-interpretable features from the text.

        Output layout (18 dims total):
          - unit flags (3): [has_kg, has_cm, has_meter]
          - payload one-hot (5): [unknown, class1, class2, class3, class4]
          - manipulation one-hot (5): [unknown, class1(no manipulation), class2, class3, class4]
          - mobility one-hot (5): [unknown, class1, class2, class3, class4]

        Notes:
        - "unknown" is used when the relevant cues are missing from the instruction text.
        - These are heuristic features; the *ground truth* task requirement classes come from NLTaskGenerator.
        """
        lower = text.lower()

        kg_val = self._extract_max_value(self._kg_re, text)
        cm_val = self._extract_max_value(self._cm_re, text)

        # 1) Unit flags (3)
        has_kg = 1.0 if kg_val is not None else 0.0
        has_cm = 1.0 if cm_val is not None else 0.0
        has_meter = 1.0 if self._meter_re.search(text) else 0.0
        unit_vec = np.array([has_kg, has_cm, has_meter], dtype=np.float32)

        # Helper: 5-way one-hot with unknown at index 0
        def onehot5_unknown() -> np.ndarray:
            v = np.zeros(5, dtype=np.float32)
            v[0] = 1.0  # unknown by default
            return v

        # 2) Payload one-hot (5)
        # [unknown, class1, class2, class3, class4]
        payload = onehot5_unknown()
        if kg_val is not None:
            payload[:] = 0.0
            if kg_val <= 3.0:
                payload[1] = 1.0
            elif kg_val <= 5.0:
                payload[2] = 1.0
            elif kg_val <= 10.0:
                payload[3] = 1.0
            else:
                payload[4] = 1.0

        # 3) Manipulation one-hot (5)
        # [unknown, class1(no manipulation), class2, class3, class4]
        manip = onehot5_unknown()

        # Optional: detect explicit "no manipulation" phrasing
        no_manip_phrases = [
            "no manipulation", "no reach", "no arm", "fixed gripper not available", "cannot manipulate", "zero manipulation",
            "no object handling required", "handling actions not required", "without performing any manipulation",
            "without any pick and place actions", "no need for manipulation",
        ]
        if any(p in lower for p in no_manip_phrases):
            manip[:] = 0.0
            manip[1] = 1.0
        elif cm_val is not None:
            manip[:] = 0.0
            # Your original classes:
            # class2 <=50cm, class3 <=90cm, class4 >90cm
            # (class1 is explicitly "no manipulation", handled above)
            if cm_val <= 50.0:
                manip[2] = 1.0
            elif cm_val <= 90.0:
                manip[3] = 1.0
            else:
                manip[4] = 1.0

        # 4) Mobility one-hot (5): [unknown, class1, class2, class3, class4]
        #
        # Mobility class semantics (as used by your templates):
        # 1 = stationary/no mobility
        # 2 = planar/open-area mobility
        # 3 = high maneuverability in constrained spaces (tight corridors, narrow aisles)
        # 4 = uneven terrain / stairs / obstacles
        mobility = onehot5_unknown()

        class1_keys = [
            "no mobility", "fixed position", "stationary", "no repositioning needed",
            "without relocating", "no change in location", "in place", "no relocation",
            "without moving", "no need for mobility", "without needing to move",
            "requiring no movement", "next to you"
        ]
        class2_keys = ["planar", "flat", "smooth"]
        class3_keys = [
            "high mobility", "narrow workspace", "high planar", "compact entryways",
            "limited space", "narrow spaces", "tight access", "confined paths",
            "narrow corridors", "compact areas", "limited passageways",
            "constrained space", "restricted paths", "narrow aisles",
            "tight passages", "compact access"
        ]
        class4_keys = [
            "uneven ground", "uneven terrain", "stairways", "stairs", "uneven floors",
            "irregular ground", "stepped surface", "ramps and stairs",
            "unstructured environment", "irregular floor", "discontinuous terrain",
            "steps", "uneven surfaces", "uneven corridors"
        ]

        if any(k in lower for k in class4_keys):
            mobility[:] = 0.0
            mobility[4] = 1.0
        elif any(k in lower for k in class3_keys):
            mobility[:] = 0.0
            mobility[3] = 1.0
        elif any(k in lower for k in class2_keys):
            mobility[:] = 0.0
            mobility[2] = 1.0
        elif any(k in lower for k in class1_keys):
            mobility[:] = 0.0
            mobility[1] = 1.0
        # else remains unknown

        # Concatenate: 3 + 5 + 5 + 5 = 18
        return np.concatenate([unit_vec, payload, manip, mobility], axis=0).astype(np.float32)

    def encode_instruction(self, instruction: str) -> np.ndarray:
        """
        Encode a single instruction into a fixed-length embedding vector.

        Returns: np.ndarray shape (embedding_dim,), dtype float32

        Raises: RuntimeError if the returned embedding dimension is unexpected.
        """
        # Semantic embedding from SentenceTransformer
        semantic = self.model.encode(
            instruction,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        semantic = np.asarray(semantic, dtype=np.float32).reshape(-1)

        if semantic.shape[0] != self.semantic_dim:
            raise RuntimeError(
                f"SentenceTransformer returned embedding dim={semantic.shape[0]}, "
                f"but expected dim={self.semantic_dim} (model={self.model_name})."
            )

        # Hybrid features (18 dims)
        hybrid = self._extract_hybrid_features(instruction)

        # Final embedding: [semantic, hybrid]
        embedding = np.concatenate([semantic, hybrid], axis=0).astype(np.float32, copy=False)

        if embedding.shape[0] != self.embedding_dim:
            raise RuntimeError(
                f"Hybrid embedding dim={embedding.shape[0]}, but expected dim={self.embedding_dim} "
                f"(semantic={self.semantic_dim}, hybrid={self.hybrid_features_count})."
            )

        return embedding

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension (int)."""
        return int(self.embedding_dim)


# ============================================================================
# 2) Debug Tracker (Optional)
# ============================================================================
class BiddingTracker:
    """
    Tracks and summarizes bidding behavior across agents during an episode.

    Intended for human-readable debugging:
      - record bids and rewards per step
      - print tables
      - summarize per-agent statistics
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all recorded data (start of new episode)."""
        self.current_step_data: List[Dict[str, Any]] = []
        # Kept for backward compatibility (some older code may read episode_data)
        self.episode_data: List[Dict[str, Any]] = []

    def record_bidding_round(
        self,
        step: int,
        task: Dict[str, Any],
        bids_per_task: Dict[int, float],
        rewards: Dict[int, float],
        agent_types: Dict[int, int],
        eligibility: Optional[Dict[int, bool]] = None,
        computed_costs: Optional[Dict[int, float]] = None,
        effective_costs: Optional[Dict[int, float]] = None,
        actions: Optional[Dict[int, Any]] = None,
        mask_sums: Optional[Dict[int, float]] = None,
        winner_id: Optional[int] = None,
        # Optional extra debug fields:
        raw_actions: Optional[Dict[int, int]] = None,
        slack_mins: Optional[Dict[int, float]] = None,
        payments: Optional[Dict[int, float]] = None,
        profits: Optional[Dict[int, float]] = None,
    ):
        """
        Record bids and rewards for one step.

        IMPORTANT:
        - If optional dicts are NOT passed, we store None (unknown) instead of
          defaulting to False/0.0.
        """
        instr = task.get("instruction", "")
        short_instr = (instr[:60] + "...") if len(instr) > 60 else instr

        task_data = {
            "task_id": task.get("id", -1),
            "instruction": short_instr,
            "category": task.get("category", "unknown"),
            "requirements": task.get("requirements"),
            "agent_info": {},
            "winner_id": winner_id,
        }

        # Normalize optional maps to dicts (so membership checks are safe)
        raw_actions = raw_actions if isinstance(raw_actions, dict) else {}
        slack_mins = slack_mins if isinstance(slack_mins, dict) else {}
        payments = payments if isinstance(payments, dict) else {}
        profits = profits if isinstance(profits, dict) else {}

        for raw_agent_id in bids_per_task:
            agent_id = int(raw_agent_id)
            agent_name = self._get_agent_name(agent_id, agent_types)

            # None means "unknown / not provided"
            if eligibility is None:
                eligible_val = None
            else:
                eligible_val = None if agent_id not in eligibility else bool(eligibility.get(agent_id))

            if winner_id is None:
                won_val = None
            else:
                won_val = (agent_id == int(winner_id))

            cc_val = None if (computed_costs is None or agent_id not in computed_costs) else float(computed_costs[agent_id])
            ec_val = None if (effective_costs is None or agent_id not in effective_costs) else float(effective_costs[agent_id])
            act_val = None if (actions is None or agent_id not in actions) else actions.get(agent_id, None)
            ms_val = None if (mask_sums is None or agent_id not in mask_sums) else float(mask_sums[agent_id])

            # Slightly more robust bid lookup (handles np.int keys too)
            bid_val = bids_per_task.get(agent_id, bids_per_task.get(raw_agent_id, 0.0))

            task_data["agent_info"][agent_id] = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "bid": float(bid_val),
                "reward": float(rewards.get(agent_id, 0.0)),

                "eligible": eligible_val,
                "won": won_val,
                "computed_cost": cc_val,
                "effective_cost": ec_val,
                "action": act_val,
                "mask_sum": ms_val,

                # Safe access + store None when not available
                "raw_action": int(raw_actions[agent_id]) if agent_id in raw_actions else None,
                "slack_min": float(slack_mins[agent_id]) if agent_id in slack_mins else None,
                "payment": float(payments[agent_id]) if agent_id in payments else None,
                "profit": float(profits[agent_id]) if agent_id in profits else None,
            }

        step_data = {"step": int(step), "task": task_data}
        self.current_step_data.append(step_data)
        self.episode_data.append(step_data)

    @staticmethod
    def _get_agent_name(agent_id: int, agent_types: Dict[int, int]) -> str:
        """Convert agent_id + type_id into a readable label."""
        type_names = {
            0: "Mobile Robot",
            1: "Mobile Manipulator",
            2: "Legged Robot",
            3: "Robotic Arm1",
            4: "Robotic Arm2",
            5: "Human",
        }
        agent_type = agent_types.get(agent_id, -1)
        return f"{type_names.get(agent_type, 'Agent')}-{agent_id}"

    def print_step_summary(self, step_data: Dict[str, Any]):
        """Print a formatted summary for a single bidding round."""
        print(f"\n{'='*100}")
        print(f"STEP {step_data['step']} - TASK SUMMARY")
        print(f"{'='*100}")

        task = step_data["task"]
        print(f"\nTASK {task['task_id']}: {task['instruction']}")
        print(f"   Category: {task['category']}")

        req = task.get("requirements")

        # req may be a numpy array
        if req is not None and len(req) >= 3:
            req = np.asarray(req, dtype=np.float32).reshape(-1)

            # If these look normalized [-1..1], convert back to RAW [1..4] for printing
            # (raw = ((norm + 1) / 2) * 3 + 1)
            if float(np.min(req)) >= -1.01 and float(np.max(req)) <= 1.01:
                req_raw = (((req + np.float32(1.0)) / np.float32(2.0)) * np.float32(3.0)) + np.float32(1.0)
            else:
                req_raw = req

            req_raw = np.round(req_raw).astype(int)
            print(f"   Requirements: Mobility={req_raw[0]}, Manipulation={req_raw[1]}, Payload={req_raw[2]}")
        else:
            print("   Requirements: (not available)")

        if task["agent_info"]:
            # Stable ordering
            infos = sorted(task["agent_info"].values(), key=lambda x: x.get("agent_id", 0))

            table = [
                [
                    info["agent_name"],
                    "X" if info.get("eligible") is True else "",
                    "🏆" if info.get("won") is True else "",
                    # Note: bid is cast to int for readability; change to {info['bid']:.2f} if bids are continuous.
                    f"{int(info['bid'])}/10",
                    f"{info['computed_cost']:.3f}" if info.get("computed_cost") is not None else "",
                    f"{info['effective_cost']:.3f}" if info.get("effective_cost") is not None else "",

                    f"{int(info['raw_action'])}" if info.get("raw_action") is not None else "",
                    f"{info['slack_min']:.3f}" if info.get("slack_min") is not None else "",
                    f"{info['payment']:.3f}" if info.get("payment") is not None else "",
                    f"{info['profit']:.3f}" if info.get("profit") is not None else "",

                    "" if info.get("action") is None else str(info.get("action")),
                    f"{info['mask_sum']:.1f}" if info.get("mask_sum") is not None else "",
                    f"{info['reward']:.3f}",
                ]
                for info in infos
            ]

            headers = [
                "Agent",
                "Eligible",
                "Won",
                "Bid",
                "Cost",
                "EffCost",
                "RawAct",
                "SlackMin",
                "Payment",
                "Profit",
                "Action",
                "MaskSum",
                "Reward",
            ]
            print("\n   BIDDING RESULTS:")
            print(tabulate(table, headers=headers, tablefmt="grid"))

    def print_episode_summary(self):
        """Print aggregated statistics for the full episode."""
        if not self.current_step_data:
            print("No bidding data recorded for this episode.")
            return

        total_steps = len(self.current_step_data)
        print(f"\n{'='*100}")
        print("EPISODE SUMMARY - BIDDING OVERVIEW")
        print(f"{'='*100}")
        print(f"Total steps/tasks: {total_steps}")

        agent_stats: Dict[int, Dict[str, Any]] = {}
        category_counts: Dict[str, int] = {}

        # Aggregate over steps
        for step_data in self.current_step_data:
            task = step_data["task"]
            cat = task.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

            for agent_id, info in task["agent_info"].items():
                if agent_id not in agent_stats:
                    agent_stats[agent_id] = {
                        "agent_name": info["agent_name"],
                        "sum_bid": 0.0,
                        "sum_reward": 0.0,
                        "count": 0,
                        "bid_nonzero": 0,
                    }
                s = agent_stats[agent_id]
                s["sum_bid"] += float(info["bid"])
                s["sum_reward"] += float(info["reward"])
                s["count"] += 1
                if info["bid"] > 0:
                    s["bid_nonzero"] += 1

        # Print agent stats table
        agent_table = []
        for agent_id in sorted(agent_stats):
            s = agent_stats[agent_id]
            count = max(int(s["count"]), 1)
            agent_table.append(
                [
                    s["agent_name"],
                    f"{s['sum_bid'] / count:.2f}",
                    f"{s['sum_reward'] / count:.3f}",
                    f"{100.0 * s['bid_nonzero'] / count:.1f}%",
                ]
            )

        print("\nPER-AGENT STATS:")
        print(tabulate(agent_table, headers=["Agent", "Mean Bid", "Mean Reward", "Non-zero Bid Rate"], tablefmt="grid"))

        # Print category distribution
        cat_table = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        print("\nTASK CATEGORY COUNTS:")
        print(tabulate(cat_table, headers=["Category", "Count"], tablefmt="grid"))


# ============================================================================
# 3) Natural Language Task Generator
# ============================================================================
class NLTaskGenerator:
    """
    Generates task instructions + requirements using parameterized natural language templates.

    Key design:
    -----------
    - Each category ("scenario1", "scenario2", ...) has:
        - ranges for numeric placeholders (weight/distance, etc.)
        - lists for categorical placeholders (adj/place/units)
        - train templates list
        - eval templates list

    The environment controls whether to use train or eval templates by passing: train_mode=True/False

    Output contract:
    ----------------
    generate_task(...) returns:
      {
        "instruction": str,
        "category": str,
        "requirements": np.ndarray shape(3,), dtype float32 in RAW [1..4]
      }
    """

    def __init__(self, templates_path: Optional[str] = None):
        """
        Load task templates and requirement labels from an external YAML dataset.

        Args:
          templates_path:
            Optional path to a YAML file with the schema used by
            ``task_templates/task_templates.yaml``. When omitted, the bundled dataset next
            to this module is used.
        """
        dataset = self._load_template_dataset(templates_path)
        self.task_templates: Dict[str, Dict[str, Any]] = dataset["scenarios"]
        self.task_requirements_map: Dict[str, List[int]] = dataset["requirements"]

        # Ensure the scenario keys match between templates and the requirements map.
        template_keys = set(self.task_templates.keys())
        req_keys = set(self.task_requirements_map.keys())

        missing = template_keys - req_keys
        extra = req_keys - template_keys

        if missing:
            raise KeyError(f"Missing requirements for scenarios: {sorted(missing)}")
        if extra:
            raise KeyError(f"Requirements map has unknown scenarios: {sorted(extra)}")

        # Validate template placeholders and scenario range structure.
        self._validate_templates()

    @staticmethod
    def _load_template_dataset(templates_path: Optional[str] = None) -> Dict[str, Any]:
        """Load the task-template dataset from YAML."""
        if templates_path is None:
            path = Path(__file__).resolve().parent / "task_templates" / "task_templates.yaml"
        else:
            path = Path(templates_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Task template dataset not found: {path}")

        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "Loading task templates from YAML requires PyYAML. "
                "Install it with `pip install pyyaml`, or add it to requirements.txt."
            ) from exc

        with path.open("r", encoding="utf-8") as f:
            dataset = yaml.safe_load(f)

        if not isinstance(dataset, dict):
            raise ValueError(f"Template dataset must be a mapping, got {type(dataset).__name__}")
        if "scenarios" not in dataset:
            raise ValueError("Template dataset missing required top-level key: scenarios")
        if "requirements" not in dataset:
            raise ValueError("Template dataset missing required top-level key: requirements")
        if not isinstance(dataset["scenarios"], dict):
            raise ValueError("Template dataset key 'scenarios' must be a mapping")
        if not isinstance(dataset["requirements"], dict):
            raise ValueError("Template dataset key 'requirements' must be a mapping")

        return dataset

    # ----------------------------------------------------------------------
    # Placeholder sampling + template filling
    # ----------------------------------------------------------------------
    @staticmethod
    def _sample_placeholders(scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a sampler for placeholders used inside templates.

        Supported placeholders (base keys):
          {w}     -> weight float
          {d}     -> distance int
          {adj}   -> adjective from size_adjectives
          {place} -> location string
          {units} -> integer from "units" list
          {place_src}, {place_dst} -> paired locations (stable within an instruction)
          {x1},{y1},{x2},{y2} -> coordinates if X_range/Y_range exist

        Also supports indexed variants for diversity:
          {adj1}, {adj2}, ...  (each index sampled independently, but stable within the instruction)
          {w1}, {w2}, ...
          {d1}, {d2}, ...
          {place1}, {place2}, ...
          {units1}, {units2}, ...

        Notes:
        - Caching is per-key, so repeated use of {adj2} in a template stays consistent.
        - {adj1} and {adj2} will usually differ (unless RNG picks the same adjective).
        """
        w_min, w_max = scenario["weight_range"]
        d_min, d_max = scenario["distance_range"]

        adj_list = scenario.get("size_adjectives", ["medium"])
        place_list = scenario.get("place", ["station"])
        units_list = scenario.get("units", [1])

        # Pre-sample paired places so {place_src}/{place_dst} are consistent
        if len(place_list) >= 2:
            place_src_val, place_dst_val = random.sample(place_list, 2)
        else:
            place_src_val = place_dst_val = place_list[0]

        # Optional coordinates
        x1_val = y1_val = x2_val = y2_val = None
        if "X_range" in scenario and "Y_range" in scenario:
            x1_val = float(round(random.uniform(*scenario["X_range"]), 2))
            y1_val = float(round(random.uniform(*scenario["Y_range"]), 2))
            x2_val = float(round(random.uniform(*scenario["X_range"]), 2))
            y2_val = float(round(random.uniform(*scenario["Y_range"]), 2))

        cache: Dict[str, Any] = {}

        # Base generators
        def gen_w() -> float: return float(round(random.uniform(w_min, w_max), 2))
        def gen_d() -> int: return int(random.randint(d_min, d_max))
        def gen_adj() -> str: return str(random.choice(adj_list))
        def gen_place() -> str: return str(random.choice(place_list))
        def gen_units() -> int: return int(random.choice(units_list))

        # Fixed/paired generators
        def gen_place_src() -> str: return str(place_src_val)
        def gen_place_dst() -> str: return str(place_dst_val)
        def gen_x1(): return x1_val
        def gen_y1(): return y1_val
        def gen_x2(): return x2_val
        def gen_y2(): return y2_val

        base_generators = {
            "w": gen_w,
            "d": gen_d,
            "adj": gen_adj,
            "place": gen_place,
            "units": gen_units,
            "place_src": gen_place_src,
            "place_dst": gen_place_dst,
            "x1": gen_x1, "y1": gen_y1, "x2": gen_x2, "y2": gen_y2,
        }

        def get(key: str):
            # Cache per key so placeholders are stable inside one instruction
            if key in cache:
                return cache[key]

            fn = base_generators.get(key)
            if fn is None:
                # Try indexed placeholders (adj1, adj2, w2, d3, etc.)
                m = NLTaskGenerator._INDEXED_KEY_RE.match(key)
                if m:
                    base = m.group(1)
                    fn = base_generators.get(base)

            val = fn() if fn else None
            cache[key] = val
            return val

        return {"_get": get}

    @staticmethod
    def _fill_template(template: str, sampler: Dict[str, Any]) -> str:
        """
        Replace placeholders like:
          {w:.2f}, {d}, {adj1}, {place_dst}, {x1:.2f}, {units}, {place_src}
        """
        get = sampler["_get"]

        def replace(match):
            key = match.group(1)
            fmt = match.group(2) or ""
            value = get(key)

            if value is None:
                return match.group(0)  # leave unchanged

            try:
                return f"{value:{fmt}}" if fmt else str(value)
            except Exception:
                return str(value)

        return re.sub(r"\{([^}:]+)(?::([^}]*))?\}", replace, template)

    # Compiled once per class (avoids recompilation on every call)
    _PLACEHOLDER_RE = re.compile(r"\{([^}:]+)(?::[^}]*)?\}")
    _INDEXED_KEY_RE = re.compile(r"^(w|d|adj|place|units)(\d+)$")

    def _validate_templates(self) -> None:
        """
        Validate scenario structure and ensure every placeholder used in templates is supported.
        Also ensures x/y placeholders are only used when X_range/Y_range exist.
        Raises ValueError with a helpful message if something is wrong.
        """
        allowed_base = {
            "w", "d", "adj", "place", "units",
            "place_src", "place_dst",
            "x1", "y1", "x2", "y2",
        }
        coord_keys = {"x1", "y1", "x2", "y2"}

        errors: List[str] = []

        def _is_num(x) -> bool:
            return isinstance(x, (int, float, np.integer, np.floating))

        def _check_range(name: str, rng, scenario_name: str):
            """Validate (min,max) ranges: type, numeric, and ordering."""
            if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
                errors.append(f"{scenario_name}: {name} must be a (min,max) tuple/list, got {rng}")
                return
            lo, hi = rng[0], rng[1]
            if not (_is_num(lo) and _is_num(hi)):
                errors.append(f"{scenario_name}: {name} values must be numeric, got {rng}")
                return
            if float(lo) > float(hi):
                errors.append(f"{scenario_name}: {name} must satisfy min <= max, got {rng}")

        for scenario_name, scenario in self.task_templates.items():
            # Required field existence
            for required in ("weight_range", "distance_range", "train", "eval"):
                if required not in scenario:
                    errors.append(f"{scenario_name}: missing required field '{required}'")

            # Range checks (only if present, to avoid KeyError cascades)
            if "weight_range" in scenario:
                _check_range("weight_range", scenario.get("weight_range"), scenario_name)
            if "distance_range" in scenario:
                _check_range("distance_range", scenario.get("distance_range"), scenario_name)

            # Train/eval type checks
            for split in ("train", "eval"):
                templates = scenario.get(split)
                if not isinstance(templates, list) or len(templates) == 0:
                    errors.append(f"{scenario_name}.{split}: must be a non-empty list of strings")
                elif not all(isinstance(t, str) for t in templates):
                    errors.append(f"{scenario_name}.{split}: all templates must be strings")

            # Coordinate range consistency + validation
            has_x = "X_range" in scenario
            has_y = "Y_range" in scenario
            if has_x != has_y:
                errors.append(f"{scenario_name}: X_range and Y_range must be provided together")

            has_coords = has_x and has_y
            if has_coords:
                _check_range("X_range", scenario.get("X_range"), scenario_name)
                _check_range("Y_range", scenario.get("Y_range"), scenario_name)

            # Placeholder validation
            for split in ("train", "eval"):
                templates = scenario.get(split, [])
                if not isinstance(templates, list):
                    continue

                for idx, tmpl in enumerate(templates):
                    if not isinstance(tmpl, str):
                        continue

                    keys = self._PLACEHOLDER_RE.findall(tmpl)
                    for key in keys:
                        # Allowed base placeholders
                        if key in allowed_base:
                            if key in coord_keys and not has_coords:
                                errors.append(
                                    f"{scenario_name}.{split}[{idx}]: uses {{{key}}} but scenario has no X_range/Y_range\n"
                                    f"  Template: {tmpl}"
                                )
                            continue

                        # Allowed indexed placeholders: adj7, w2, d3, place10, units4...
                        if self._INDEXED_KEY_RE.match(key):
                            continue

                        errors.append(
                            f"{scenario_name}.{split}[{idx}]: unknown placeholder {{{key}}}\n"
                            f"  Template: {tmpl}"
                        )

        if errors:
            msg = "Template/scenario validation failed:\n" + "\n".join(f"- {e}" for e in errors[:80])
            if len(errors) > 80:
                msg += f"\n...and {len(errors) - 80} more."
            raise ValueError(msg)

    # ----------------------------------------------------------------------
    # Main generation functions
    # ----------------------------------------------------------------------
    def generate_task(self, category: str, train_mode: bool = True) -> Dict[str, Any]:
        """
        Generate one task for the given category.

        Args:
          category: scenario key (e.g., "scenario1")
          train_mode:
            True  -> sample scenario["train"] templates
            False -> sample scenario["eval"] templates
        """
        if category not in self.task_templates:
            raise ValueError(f"Unknown category: {category}")

        scenario = self.task_templates[category]
        if "train" not in scenario or "eval" not in scenario:
            raise ValueError(f"Scenario '{category}' must define both 'train' and 'eval' template lists")

        templates = scenario["train"] if train_mode else scenario["eval"]
        if not templates:
            raise ValueError(
                f"No templates for category '{category}' in {'train' if train_mode else 'eval'} mode"
            )

        template = random.choice(templates)
        instruction = self._fill_template(template, self._sample_placeholders(scenario))

        # Requirements are ground-truth classes (RAW [1..4]) in the order:
        # [mobility_class, manipulation_class, payload_class]
        requirements = np.asarray(self.task_requirements_map[category], dtype=np.float32)

        return {
            "instruction": instruction,
            "category": category,
            "requirements": requirements,
        }

    def generate_task_random(self, train_mode: bool = True) -> Dict[str, Any]:
        """Generate a task by sampling a random category."""
        category = random.choice(list(self.task_templates.keys()))
        return self.generate_task(category, train_mode=train_mode)


# ============================================================================
# 4) Optional offline dataset generator
# ============================================================================
TRAIN_PER_CLASS = 500
EVAL_PER_CLASS = 125
SEED = 42


def build_dataset(seed: int = SEED) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a list of generated tasks for offline inspection or dataset export.

    Returns:
      (train_tasks, eval_tasks)
    """
    # Save RNG states so calling build_dataset() doesn't affect training randomness
    py_state = random.getstate()
    np_state = np.random.get_state()

    random.seed(seed)
    np.random.seed(seed)

    try:
        generator = NLTaskGenerator()
        cats = list(generator.task_templates.keys())
        n_classes = len(cats)

        print(
            f"Generating dataset: {n_classes} classes × "
            f"{TRAIN_PER_CLASS} train + {EVAL_PER_CLASS} eval = "
            f"{n_classes * (TRAIN_PER_CLASS + EVAL_PER_CLASS)} samples"
        )

        train_tasks = [generator.generate_task(c, train_mode=True) for c in cats for _ in range(TRAIN_PER_CLASS)]
        eval_tasks = [generator.generate_task(c, train_mode=False) for c in cats for _ in range(EVAL_PER_CLASS)]

        return train_tasks, eval_tasks

    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def save_dataset_npz(path: str, train_tasks: List[Dict[str, Any]], eval_tasks: List[Dict[str, Any]]) -> None:
    """
    Save a generated dataset to disk for repeatable runs.

    The environment can later load this file and cycle tasks deterministically.
    """
    def pack(tasks: List[Dict[str, Any]]):
        instr = np.array([t["instruction"] for t in tasks], dtype=object)
        cat = np.array([t["category"] for t in tasks], dtype=object)
        req = np.stack([np.asarray(t["requirements"], dtype=np.float32) for t in tasks], axis=0)
        return instr, cat, req

    train_instr, train_cat, train_req = pack(train_tasks)
    eval_instr, eval_cat, eval_req = pack(eval_tasks)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        train_instructions=train_instr,
        train_categories=train_cat,
        train_requirements=train_req,
        eval_instructions=eval_instr,
        eval_categories=eval_cat,
        eval_requirements=eval_req,
    )


def load_dataset_npz(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load a dataset saved by save_dataset_npz().

    Returns:
      (train_tasks, eval_tasks)
    """
    data = np.load(path, allow_pickle=True)

    def unpack(prefix: str) -> List[Dict[str, Any]]:
        instr = data[f"{prefix}_instructions"]
        cat = data[f"{prefix}_categories"]
        req = data[f"{prefix}_requirements"].astype(np.float32, copy=False)

        tasks: List[Dict[str, Any]] = []
        for i in range(len(instr)):
            tasks.append(
                {
                    "instruction": str(instr[i]),
                    "category": str(cat[i]),
                    "requirements": np.asarray(req[i], dtype=np.float32),
                }
            )
        return tasks

    return unpack("train"), unpack("eval")
