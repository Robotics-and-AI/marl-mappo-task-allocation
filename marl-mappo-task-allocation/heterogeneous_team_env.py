"""
heterogeneous_team_env.py

Multi-agent task-bidding environment for RLlib (MAPPO/CTDE style).

High-level idea
---------------
At every step:
  1) The environment samples ONE natural-language task instruction (train or eval templates)
  2) All agents observe:
       - their own normalized capabilities (local part)
       - the task embedding (local part)
       - a global state for centralized critic: (all agents' capabilities + task embedding + eligibility flags)
     NOTE: The structured requirement vector is used internally to compute eligibility and action masks,
           but it is NOT included in the observation vectors emitted by this environment.
  3) Each agent outputs an integer bid in {0..10}
       - 0 means "I don't bid" / abstain
  4) Rewards are computed using a procurement-style auction:
       - only eligible agents can win
       - winner = lowest bid among eligible bidders
       - payment = 2nd-lowest bid if ≥2 bidders, else first-price fallback (payment = winning bid)
       - winner reward is based on profit = payment - (estimated true cost), plus additional shaping

CTDE / MAPPO structure
----------------------
- Actor policy uses local observation: (agent_caps + task_embedding)
- Critic can use global state: (all_agent_caps + task_embedding + elig_flags)

Train vs Eval templates
-----------------------
Task generator has scenario templates separated into:
  - scenario["train"] templates
  - scenario["eval"] templates

This env uses env_config["train_mode"] to choose which to sample:
  - train_mode=True  -> uses train templates
  - train_mode=False -> uses eval templates

Per-agent policies
------------------
If you want ONE policy per agent (no parameter sharing), RLlib must be configured with:
  - policies = {"policy_0": ..., "policy_1": ..., ...}
  - policy_mapping_fn(agent_id, ...) -> f"policy_{agent_id}"

This file includes:
  - HeterogeneousTeamEnv.get_policy_ids() helper

The policy-mapping helper lives in train.py because it is part of the RLlib training configuration.

Action masking
--------------
- Always includes "action_mask" in observations (and observation_space).
- Boolean switch:
    env_config["use_action_mask"] = False -> mask is all ones (does NOT constrain bidding)
    env_config["use_action_mask"] = True  -> mask blocks bids>0 for ineligible agents (only 0 allowed)
- Optional constraint:
    env_config["force_eligible_to_bid"] = True -> mask also blocks action 0 for eligible agents.
- IMPORTANT (Option A): This env does NOT modify/correct actions in step().
  Masking must be enforced in the RLlib model/policy (masked logits) so PPO remains consistent.

Per-step info fields
--------------------
Adds per-step outcome info fields in step() info[agent_id]:
    eligible / won / action / raw_action / mask_sum
  plus:
    slack_min / payment / profit

Dataset sampling (optional)
---------------------------
When dataset_mode=True, tasks are loaded from a .npz file and sampled via:
  env_config["dataset_sampling"] in {"permute","random","sequential","category_weighted"}

The "category_weighted" mode (train-only) biases sampling toward categories where specialist agents
are eligible, then samples a random task from that category.

Config knobs:
  - category_mix_uniform (default 0.30): mix weighted sampling with uniform to avoid overfitting.
  - category_weight_nonspecialist (default 0.20): weight for categories with no specialist eligibility.
  - category_weight_floor (default 0.05): very small weight for "human-only" categories.
  - print_category_sampling_cache (default False): prints weights once per worker.
"""

import os
import numpy as np
import random
from collections import defaultdict
from typing import Dict, Any, List, Optional
from gymnasium.spaces import Dict as SpaceDict, Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

# Task custom generator module
from task_generator import BiddingTracker, NLTaskGenerator, TaskInstructionProcessor, load_dataset_npz


class HeterogeneousTeamEnv(MultiAgentEnv):
    """
    Environment:
    - 6 agents with different "types" and capability vectors
    Each step:
      - sample a new natural-language instruction
      - agents bid [0..10]
      - compute reward per agent
    """

    # Agent type IDs (used for labeling only)
    MOBILE_ROBOT = 0
    MOBILE_MANIPULATOR = 1
    LEGGED_ROBOT = 2
    ROBOTIC_ARM1 = 3
    ROBOTIC_ARM2 = 4
    HUMAN = 5

    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}

        # ----------------------------
        # Dataset vs Online mode + sampling mode
        # ----------------------------
        self.dataset_mode = bool(env_config.get("dataset_mode", False))  # True -> use .npz, False -> online generator
        # "permute"|"random"|"sequential"|"category_weighted"
        self.dataset_sampling = str(env_config.get("dataset_sampling", "permute"))

        # RNGs (do NOT seed global np.random/random)
        self._rng = np.random.default_rng()
        self._py_rng = random.Random()

        self._dataset_order: Optional[np.ndarray] = None

        # ----------------------------
        # category-weighted sampling knobs (train-only)
        # ----------------------------
        self.category_mix_uniform = float(env_config.get("category_mix_uniform", 0.30))
        self.category_weight_nonspecialist = float(env_config.get("category_weight_nonspecialist", 0.20))
        self.category_weight_floor = float(env_config.get("category_weight_floor", 0.05))
        self.print_category_sampling_cache = bool(env_config.get("print_category_sampling_cache", False))

        # Caches for category-weighted sampling (built lazily)
        self._cat_to_indices: Optional[Dict[str, np.ndarray]] = None
        self._cat_sampling_categories: Optional[List[str]] = None
        self._cat_sampling_probs: Optional[np.ndarray] = None
        self._cat_cache_ready: bool = False

        # ----------------------------
        # Team definition (6 agents)
        # ----------------------------
        self._num_agents = 6
        self.agent_types = [
            self.MOBILE_ROBOT,         # Agent 0
            self.MOBILE_MANIPULATOR,   # Agent 1
            self.LEGGED_ROBOT,         # Agent 2
            self.ROBOTIC_ARM1,         # Agent 3
            self.ROBOTIC_ARM2,         # Agent 4
            self.HUMAN,                # Agent 5
        ]

        # Capability vectors per agent: [mobility, manipulation, payload_class] each in {1..4}
        raw_caps = np.array(
            [
                [2, 1, 4],  # Mobile robot
                [3, 4, 3],  # Mobile manipulator
                [4, 1, 1],  # Legged robot
                [1, 3, 2],  # Robotic arm 1
                [1, 2, 1],  # Robotic arm 2
                [4, 4, 3],  # Human
            ],
            dtype=np.float32,
        )

        # Normalize capabilities into [-1, 1] so models train stably.
        self.agent_capabilities = self._normalize(raw_caps)

        self.agent_type_names = {
            0: "Mobile Robot",
            1: "Mobile Manipulator",
            2: "Legged Robot",
            3: "Robotic Arm1",
            4: "Robotic Arm2",
            5: "Human",
        }

        # Episode config
        self.max_steps = int(env_config.get("max_steps", 50))
        self.current_step = 0

        # Counts tasks generated within the current episode.
        # NOTE: This is reset in reset(). If you intended a global ramp across training,
        # you would not reset this counter per episode.
        self._total_generated_tasks = 0
        # Lifetime task counter for curriculum scheduling (NOT reset each episode)
        self._mask_curriculum_tasks_total = 0

        # If True, store detailed bidding logs and print them if requested
        self.show_detailed_episodes = bool(env_config.get("show_detailed_episodes", False))

        # TRAIN/EVAL SWITCH:
        self.train_mode = bool(env_config.get("train_mode", True))

        # Reserved debug knob. Currently not used by the environment logic.
        self.eval_mask_print_every_n = int(env_config.get("eval_mask_print_every_n", 0))

        # Resume (debug metrics) print frequency (0 disables)
        self.resume_print_every_n = int(env_config.get("resume_print_every_n", 0))
        self.eval_resume_print_every_n = int(env_config.get("eval_resume_print_every_n", 0))

        # Reward shaping knobs (configurable from train script via env_config)
        self.eligible_bid_bonus = float(env_config.get("eligible_bid_bonus", 0.0))
        self.bid_shaping_coeff = float(env_config.get("bid_shaping_coeff", 0.50))
        self.missed_assignment_penalty = float(env_config.get("missed_assignment_penalty", -0.50))

        # Penalize eligible agents that abstain (bid=0) EVEN if others bid.
        self.eligible_abstain_penalty = float(env_config.get("eligible_abstain_penalty", -0.10))

        # Optional fixed bonus awarded to the winning eligible agent
        self.winner_reward = float(env_config.get("winner_reward", 0.10))

        # Human winner bonus (separate from winner_reward)
        self.human_winner_reward = float(env_config.get("human_winner_reward", 0.5))

        # Make these configurable
        self.unqualified_bid_penalty = float(env_config.get("unqualified_bid_penalty", -1.00))
        self.lost_auction_penalty = float(env_config.get("lost_auction_penalty", -0.05))

        # Human cost offset (only affects true_cost used in shaping/profit, not eligibility.)
        self.human_cost_offset = int(env_config.get("human_cost_offset", 2))

        # ----------------------------
        # Specialist preference shaping (NO change to winner selection logic)
        # Specialists default: MobileRobot(0), Legged(2), Arm1(3), Arm2(4)
        # ----------------------------
        self.specialist_agent_ids = set(env_config.get("specialist_agent_ids", [0, 2, 3, 4]))

        # --- Specialist "true cost" discount (drives specialists to bid lower via shaping) ---
        self.specialist_cost_multiplier = float(env_config.get("specialist_cost_multiplier", 1.0))  # e.g. 0.90
        self.specialist_cost_offset = int(env_config.get("specialist_cost_offset", 0))              # e.g. -1

        # --- Arm2-only bid/cost shaping knobs (optional) ---
        self.arm2_cost_multiplier = float(env_config.get("arm2_cost_multiplier", 1.0))  # e.g. 0.65
        self.arm2_cost_offset = int(env_config.get("arm2_cost_offset", 0))              # e.g. -2

        # Kept for visibility/debugging, but the current discount logic does not depend on this threshold.
        self.specialist_discount_min_bn_slack = float(env_config.get("specialist_discount_min_bn_slack", 0.0))

        # Debug flag for specialist cost discount printouts (off by default)
        self.debug_specialist_cost = bool(env_config.get("debug_specialist_cost", False))

        # Entry discouragement for non-specialists when any specialist is eligible
        self.ns_bid_penalty_when_spec_eligible = float(env_config.get("ns_bid_penalty_when_spec_eligible", 0.05))

        # "Steal" penalty: non-specialist wins while some specialist was eligible
        self.ns_win_penalty_when_spec_eligible = float(env_config.get("ns_win_penalty_when_spec_eligible", 0.40))

        # Bonus: specialist wins in a setting where a specialist was eligible
        self.spec_win_bonus_when_spec_eligible = float(env_config.get("spec_win_bonus_when_spec_eligible", 0.15))

        # Optional ramp for the above shaping (based on number of tasks generated).
        self.specialist_shaping_warmup_tasks = int(env_config.get("specialist_shaping_warmup_tasks", 0))
        self.specialist_shaping_ramp_tasks = int(env_config.get("specialist_shaping_ramp_tasks", 0))

        # Natural language task components
        self.bidding_tracker = BiddingTracker()
        self.task_generator = NLTaskGenerator()
        self.instruction_processor = TaskInstructionProcessor()

        # Bidding action space definition: 0..10 (0 = no bid)
        self.bids_range = 10
        self.action_space = Discrete(self.bids_range + 1)

        # ------------------------------------------------------------------
        # Action masking support
        # ------------------------------------------------------------------
        # If False: still emit action_mask, but it's all-ones (no constraint).
        # If True : action_mask blocks bids>0 for ineligible agents.
        self.use_action_mask = bool(env_config.get("use_action_mask", False))
        self.debug_action_mask = bool(env_config.get("debug_action_mask", False))

        # Probabilistic mask curriculum (only relevant when use_action_mask=True)
        self.mask_curriculum = bool(env_config.get("mask_curriculum", False))
        self._specialist_shaping_tasks_total = 0

        # Helps tie breaking Arm2 vs Arm1
        pref = env_config.get("tie_break_preference", [4, 3])  # default: Arm2 > Arm1
        # Higher number = higher priority
        self.tie_priority = {int(a): len(pref) - i for i, a in enumerate(pref)}

        # Cached per-task decision: should we apply the real eligibility mask on this task?
        self._apply_mask_this_task = True

        # Task-based schedule parameters (in number of tasks/steps)
        self.mask_warmup_tasks = int(env_config.get("mask_warmup_tasks", 0))          # e.g., 100000
        self.mask_decay_tasks = int(env_config.get("mask_decay_tasks", 0))            # e.g., 500000
        self.mask_min_prob = float(env_config.get("mask_min_prob", 0.0))              # usually 0.0
        
        # Optional: per-step independent randomness for curriculum decision
        # Use env's local RNG (self._rng) already exists.

        # When True (and use_action_mask=True), eligible agents are forced to bid (no abstain).
        # NOTE: Env does not enforce this in step(); policy/model must respect the mask.
        self.force_eligible_to_bid = bool(env_config.get("force_eligible_to_bid", True))

        # Observation space (CTDE):
        instruction_dim = int(self.instruction_processor.get_embedding_dim())
        capabilities_dim = int(raw_caps.shape[1])  # 3

        # Eligibility flag dimensions (GLOBAL ONLY; not in local obs)
        self._elig_flag_dim_global = self._num_agents   # appended to global state (one per agent)

        # Local: caps + task_embedding
        self.local_obs_dim = capabilities_dim + instruction_dim

        # Global: all_caps + task_embedding + elig_flags
        self.global_state_dim = (self._num_agents * capabilities_dim) + instruction_dim + self._elig_flag_dim_global

        local_low = np.full((self.local_obs_dim,), -10.0, dtype=np.float32)
        local_high = np.full((self.local_obs_dim,), 10.0, dtype=np.float32)
        global_low = np.full((self.global_state_dim,), -10.0, dtype=np.float32)
        global_high = np.full((self.global_state_dim,), 10.0, dtype=np.float32)

        local_obs_space = Box(low=local_low, high=local_high, dtype=np.float32)
        global_state_space = Box(low=global_low, high=global_high, dtype=np.float32)

        action_mask_space = Box(
            low=0.0,
            high=1.0,
            shape=(self.bids_range + 1,),
            dtype=np.float32,
        )

        # Always include action_mask in the observation space
        obs_space_dict = {
            "obs": local_obs_space,
            ENV_STATE: global_state_space,  # "state"
            "action_mask": action_mask_space,
        }
        self.observation_space = SpaceDict(obs_space_dict)

        # RLlib agent bookkeeping
        self._agent_ids = list(range(self._num_agents))  # stable IDs: 0..5
        self.possible_agents = self._agent_ids[:]
        self.agents = self.possible_agents[:]  # static team

        # Embedding cache: instruction string -> embedding vector
        self.task_embeddings: Dict[str, np.ndarray] = {}

        # Simple history log of previous tasks/rewards
        self.task_history: List[Dict[str, Any]] = []

        # Current task dict is created in reset()
        self.current_task: Optional[Dict[str, Any]] = None

        # ----------------------------
        # Optional: fixed pre-generated dataset (ONLY if dataset_mode=True)
        # ----------------------------
        self.dataset_path: Optional[str] = env_config.get("dataset_path", None)
        self._dataset_train_tasks: Optional[List[Dict[str, Any]]] = None
        self._dataset_eval_tasks: Optional[List[Dict[str, Any]]] = None
        self._dataset_tasks: Optional[List[Dict[str, Any]]] = None
        self._dataset_cursor: int = 0

        if self.dataset_mode:
            if not self.dataset_path:
                raise ValueError("dataset_mode=True but dataset_path is not set.")
            if not os.path.exists(self.dataset_path):
                raise ValueError(f"dataset_path does not exist: {self.dataset_path}")

            train_tasks, eval_tasks = load_dataset_npz(self.dataset_path)

            if not isinstance(train_tasks, list) or not isinstance(eval_tasks, list):
                raise ValueError("Loaded dataset format invalid (expected two lists).")
            if len(train_tasks) == 0 or len(eval_tasks) == 0:
                raise ValueError("Loaded dataset is empty (train or eval list has 0 tasks).")

            self._dataset_train_tasks = train_tasks
            self._dataset_eval_tasks = eval_tasks
            self._dataset_tasks = self._dataset_train_tasks if self.train_mode else self._dataset_eval_tasks

        # Resume counters (train/eval buckets)
        self._init_bidding_resume()

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def get_policy_ids(self) -> List[str]:
        """Convenience helper for RLlib config when using one-policy-per-agent."""
        return [f"policy_{aid}" for aid in self._agent_ids]

    # ------------------------------------------------------------------
    # Resume (debug metrics)
    # ------------------------------------------------------------------
    def _init_bidding_resume(self) -> None:
        def _blank():
            return {
                "tasks": 0,
                "tasks_no_eligible": 0,
                "tasks_no_bid": 0,
                "eligible": {aid: 0 for aid in self._agent_ids},
                "wins": {aid: 0 for aid in self._agent_ids},
            }

        self._bidding_resume = {
            "train": _blank(),
            "eval": _blank(),
        }

    def _update_bidding_resume(self, outcomes: Dict[int, Dict[str, Any]], actions: Dict[int, int]) -> None:
        phase = "train" if bool(self.train_mode) else "eval"
        s = self._bidding_resume[phase]

        s["tasks"] += 1

        any_eligible = False
        for aid in self._agent_ids:
            if bool(outcomes.get(aid, {}).get("eligible", False)):
                any_eligible = True
                s["eligible"][aid] += 1
            if bool(outcomes.get(aid, {}).get("won", False)):
                s["wins"][aid] += 1

        if not any_eligible:
            s["tasks_no_eligible"] += 1

        any_bid = any(int(actions.get(aid, 0)) > 0 for aid in self._agent_ids)
        if not any_bid:
            s["tasks_no_bid"] += 1

    def _format_bidding_resume(self, phase: str) -> str:
        s = self._bidding_resume[phase]
        tasks = int(s["tasks"])
        no_elig = int(s["tasks_no_eligible"])
        no_bid = int(s["tasks_no_bid"])

        lines = []
        lines.append(f"[BIDDING RESUME] phase={phase} | tasks={tasks} | no_eligible={no_elig} | no_bid={no_bid}")
        lines.append("  Agent  Name               Eligible   Wins   Win% (wins/eligible)")
        lines.append("  -----  -----------------  --------  -----  ---------------------")

        for aid in self._agent_ids:
            name = self.agent_type_names[self.agent_types[aid]]
            elig = int(s["eligible"][aid])
            wins = int(s["wins"][aid])
            pct = (100.0 * wins / elig) if elig > 0 else 0.0
            lines.append(f"  {aid:<5d}  {name:<17s}  {elig:>8d}  {wins:>5d}  {pct:>8.2f}%")

        return "\n".join(lines)


    def _print_bidding_resume(self) -> None:
        sep = "=" * 90
        print(sep)
        print(self._format_bidding_resume("train"))
        print("-" * 90)
        print(self._format_bidding_resume("eval"))
        print(sep)

    # ------------------------------------------------------------------
    # RL API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """
        Start a new episode:
          - reset counters
          - generate first task
          - return initial observations
        """
        # Do NOT seed global np.random/random.
        # Keep seeding local RNGs only.
        if seed is not None:
            s = int(seed)
            self._rng = np.random.default_rng(s)
            self._py_rng.seed(s)

        self.current_step = 0
        self.task_history = []

        # Keep embedding cache across episodes when using dataset (speeds up repeats).
        # In online mode, clear it to avoid unbounded growth.
        if self._dataset_tasks is None:
            self.task_embeddings = {}

        self._total_generated_tasks = 0

        if hasattr(self, "bidding_tracker"):
            self.bidding_tracker.reset()

        # Dataset per-episode sampling order
        if self._dataset_tasks is not None:
            n = len(self._dataset_tasks)
            if n <= 0:
                raise ValueError("Dataset task list is empty.")

            if seed is not None:
                self._rng = np.random.default_rng(int(seed))
            else:
                self._rng = np.random.default_rng()

            self._dataset_cursor = 0
            if self.dataset_sampling == "permute":
                self._dataset_order = self._rng.permutation(n)
            else:
                self._dataset_order = None

            # build category-weighted cache when requested (train-only)
            if self.train_mode and self.dataset_sampling == "category_weighted":
                self._maybe_build_category_sampling_cache()

        self.current_task = self._generate_nl_task()
        self._sample_mask_decision_for_task()
        obs = self._get_obs()

        info = {aid: {} for aid in self._agent_ids}
        return obs, info

    def step(self, actions):
        """
        One environment step:
          - collect bids from all agents
          - compute rewards for the current task
          - sample next task
          - return new obs + rewards + termination flags + info

        Emits per-agent step info for metrics/debug.

        NOTE (Option A): This env does NOT correct invalid actions.
        Action masks must be enforced by the RLlib model/policy.
        """
        self.current_step += 1

        # Ensure every agent has an action; missing -> 0 (abstain)
        raw_actions = {aid: int(actions.get(aid, 0)) for aid in self._agent_ids}
        safe_actions = {
            aid: int(np.clip(int(actions.get(aid, 0)), 0, self.bids_range))
            for aid in self._agent_ids
        }

        prev_task = self.current_task
        prev_instruction = prev_task["instruction"]

        # Build masks for THIS task (the one actions apply to)
        task_req_1to4 = self._unnormalize(prev_task["requirements"]).astype(np.float32, copy=False)

        masks: Dict[int, np.ndarray] = {}
        mask_sums: Dict[int, float] = {}
        for aid in self._agent_ids:
            m = self._build_action_mask(aid, task_req_1to4)
            masks[aid] = m
            mask_sums[aid] = float(np.sum(m))

        # Compute rewards + outcomes for bids on the CURRENT task
        # NOTE: pass raw_actions so BiddingTracker can log policy output vs executed action.
        task_rewards, outcomes = self._calculate_rewards(
            safe_actions,
            return_outcomes=True,
            raw_actions=raw_actions,
        )

        # Update counters before any optional progress printing.
        self._update_bidding_resume(outcomes, safe_actions)

        # Now use the updated count
        phase = "train" if self.train_mode else "eval"

        n = int(self.resume_print_every_n)
        if not self.train_mode:
            n = int(self.eval_resume_print_every_n) or int(self.resume_print_every_n)

        tasks = int(self._bidding_resume[phase]["tasks"])

        if n > 0 and tasks > 0 and (tasks % n) == 0:
            self._print_bidding_resume()

        terminated = {aid: False for aid in self._agent_ids}
        truncated = {aid: False for aid in self._agent_ids}

        if self.current_step >= self.max_steps:
            truncated["__all__"] = True
            terminated["__all__"] = False
        else:
            truncated["__all__"] = False
            terminated["__all__"] = False

        # Optional logging for debugging
        self.task_history.append(
            {
                "instruction": prev_instruction,
                "category": prev_task.get("category", "unknown"),
                "agent_types": [self.agent_type_names[self.agent_types[aid]] for aid in self._agent_ids],
                "rewards": task_rewards.copy(),
            }
        )

        # Advance to the next task
        self.current_task = self._generate_nl_task()
        self._sample_mask_decision_for_task()
        obs = self._get_obs()

        # Winner summary for __common__
        winner_aid = None
        for aid in self._agent_ids:
            if bool(outcomes.get(aid, {}).get("won", False)):
                winner_aid = aid
                break

        # Specialist metrics (based on previous task outcomes / actions)
        spec_ids = self.specialist_agent_ids
        spec_eligible = any(bool(outcomes.get(aid, {}).get("eligible", False)) for aid in spec_ids)
        spec_bid_any = any(int(safe_actions.get(aid, 0)) > 0 for aid in spec_ids)
        spec_won = (winner_aid in spec_ids) if winner_aid is not None else False

        info = {
            "__common__": {
                "current_step": self.current_step,
                "previous_task_instruction": prev_instruction,
                "previous_task_category": prev_task.get("category", "unknown"),
                "bidding_results": safe_actions.copy(),
                "winner_agent_id": winner_aid,
                "winning_bid": int(safe_actions[winner_aid]) if winner_aid is not None else None,
                # convenience (winner-only)
                "winning_payment": outcomes.get(winner_aid, {}).get("payment", None) if winner_aid is not None else None,
                "winning_profit": outcomes.get(winner_aid, {}).get("profit", None) if winner_aid is not None else None,
                # specialist-focused diagnostics
                "spec_eligible": bool(spec_eligible),
                "spec_bid_any": bool(spec_bid_any),
                "spec_won": bool(spec_won),
            }
        }

        # Per-agent info fields needed for custom metrics
        for aid in self._agent_ids:
            o = outcomes.get(aid, {})
            info[aid] = {
                "agent_type": self.agent_types[aid],
                "agent_type_name": self.agent_type_names[self.agent_types[aid]],
                "eligible": bool(o.get("eligible", False)),
                "won": bool(o.get("won", False)),
                "raw_action": int(raw_actions.get(aid, 0)),          # what policy output
                "action": int(safe_actions.get(aid, 0)),             # executed action
                "mask_sum": float(mask_sums.get(aid, float(self.action_space.n))),
                "slack_min": o.get("slack_min", None),
                "payment": o.get("payment", None),
                "profit": o.get("profit", None),
            }

        return obs, task_rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers: embeddings and observations
    # ------------------------------------------------------------------
    def _hashable_key(self, instruction: str) -> str:
        return str(instruction)

    def _get_task_embedding(self, instruction: str) -> np.ndarray:
        key = self._hashable_key(instruction)
        if key not in self.task_embeddings:
            emb = self.instruction_processor.encode_instruction(instruction)
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            self.task_embeddings[key] = emb
        return self.task_embeddings[key]
    
    def _mask_probability(self) -> float:
        """
        Returns p_mask in [mask_min_prob, 1.0] based on how many tasks have been generated so far.
        Uses a lifetime counter (self._mask_curriculum_tasks_total) so the schedule works across episodes.

        IMPORTANT:
        - Curriculum is applied only during training (train_mode=True).
        - During eval (train_mode=False), we return 1.0 so the mask behavior is deterministic.
        """
        # Disable curriculum in eval to keep evaluation stable/deterministic
        if (not self.mask_curriculum) or (not self.train_mode):
            return 1.0

        t = int(getattr(self, "_mask_curriculum_tasks_total", 0))
        warm = max(0, int(self.mask_warmup_tasks))
        decay = max(0, int(self.mask_decay_tasks))

        if decay <= 0:
            return 1.0 if t < warm else float(self.mask_min_prob)

        if t < warm:
            return 1.0

        x = min(1.0, max(0.0, (t - warm) / float(decay)))
        p = (1.0 - x) * 1.0 + x * float(self.mask_min_prob)
        return float(np.clip(p, float(self.mask_min_prob), 1.0))
    
    def _sample_mask_decision_for_task(self) -> None:
        """
        Sample ONCE per task whether masking is applied (curriculum).
        Ensures consistent masks across obs/step/debug logging.
        """
        # Default: apply mask deterministically.
        self._apply_mask_this_task = True

        if not self.use_action_mask:
            return

        # If curriculum is off, always apply real mask.
        if not self.mask_curriculum:
            return

        # If you want curriculum only in training, keep this:
        if not self.train_mode:
            return

        p = float(self._mask_probability())
        u = float(self._rng.random())
        self._apply_mask_this_task = (u < p)


    def _build_action_mask(self, agent_id: int, task_req_1to4: np.ndarray) -> np.ndarray:
        """
        When use_action_mask=False: allow all actions (no constraint).
        When use_action_mask=True :
          - ineligible agents: only action 0 allowed
          - eligible agents  : if force_eligible_to_bid=True, action 0 disallowed
        """
        mask = np.ones((self.bids_range + 1,), dtype=np.float32)

        if not self.use_action_mask:
            return mask

        # If curriculum decided "no mask" for this task, return all-ones.
        if self.mask_curriculum and (not self._apply_mask_this_task):
            return mask

        EPS = 1e-5
        cap_1to4 = self._unnormalize(self.agent_capabilities[agent_id]).astype(np.float32, copy=False)
        eligible = bool(np.all(cap_1to4 + EPS >= task_req_1to4))

        if not eligible:
            mask[1:] = 0.0
        else:
            if self.force_eligible_to_bid:
                mask[0] = 0.0

        return mask

    def _get_obs(self):
        """
        obs[agent_id] = {
            "obs": local_obs,
            "state": global_state (ENV_STATE),
            "action_mask": mask
        }

        Local obs:  [agent_caps, task_embedding]
        Global state: [all_caps, task_embedding, elig_flags]

        Requirements are still used internally to compute:
          - elig_flags
          - action_mask
        but are NOT included in the observation vectors anymore.
        """
        obs: Dict[int, Dict[str, Any]] = {}

        task_emb = self._get_task_embedding(self.current_task["instruction"]).astype(np.float32, copy=False)

        # requirements remain internal (used for eligibility + mask), but not emitted in obs/state vectors
        req_norm = np.asarray(self.current_task["requirements"], dtype=np.float32).reshape(-1)
        task_req_1to4 = self._unnormalize(req_norm).astype(np.float32, copy=False)

        EPS = 1e-5
        elig_flags = np.zeros((self._num_agents,), dtype=np.float32)
        for aid in self._agent_ids:
            cap_1to4 = self._unnormalize(self.agent_capabilities[aid]).astype(np.float32, copy=False)
            elig_flags[aid] = 1.0 if bool(np.all(cap_1to4 + EPS >= task_req_1to4)) else 0.0

        global_state = np.concatenate(
            [
                self.agent_capabilities.reshape(-1).astype(np.float32, copy=False),
                task_emb,
                elig_flags,
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        for aid in self._agent_ids:
            local_obs = np.concatenate(
                [
                    self.agent_capabilities[aid].astype(np.float32, copy=False),
                    task_emb,
                ],
                axis=0,
            ).astype(np.float32, copy=False)

            agent_obs = {
                "obs": local_obs,
                ENV_STATE: global_state,
            }

            # Always emit action_mask (still depends on requirements internally)
            mask = self._build_action_mask(aid, task_req_1to4)
            agent_obs["action_mask"] = mask

            if self.debug_action_mask and (self.current_step <= 3):
                msum = float(np.sum(mask))
                print(f"[MASK DEBUG] step={self.current_step} agent={aid} sum={msum} mask={mask.tolist()}")

            obs[aid] = agent_obs

        return obs

    # ------------------------------------------------------------------
    # Helpers: normalize / unnormalize capability scale
    # ------------------------------------------------------------------
    def _normalize(self, unnormalized):
        """Map values from [1..4] -> [-1..1] linearly. 1 -> -1, 4 -> +1."""
        unnormalized = np.asarray(unnormalized, dtype=np.float32)
        return (((unnormalized - np.float32(1.0)) / np.float32(3.0)) * np.float32(2.0) - np.float32(1.0)).astype(
            np.float32, copy=False
        )

    def _unnormalize(self, normalized):
        """Inverse of _normalize: map [-1..1] -> [1..4]."""
        normalized = np.asarray(normalized, dtype=np.float32)
        return ((((normalized + np.float32(1.0)) / np.float32(2.0)) * np.float32(3.0)) + np.float32(1.0)).astype(
            np.float32, copy=False
        )

    # ------------------------------------------------------------------
    # category-weighted sampling cache (train-only)
    # ------------------------------------------------------------------
    def _maybe_build_category_sampling_cache(self) -> None:
        if self._cat_cache_ready:
            return
        if self._dataset_tasks is None or len(self._dataset_tasks) == 0:
            return

        cat_to_idxs = defaultdict(list)
        for i, t in enumerate(self._dataset_tasks):
            cat_to_idxs[str(t.get("category", "unknown"))].append(int(i))

        self._cat_to_indices = {c: np.asarray(idxs, dtype=np.int32) for c, idxs in cat_to_idxs.items()}
        cats = sorted(self._cat_to_indices.keys())

        SPECIALIST_BOOST = {
            int(self.LEGGED_ROBOT): 2.0,
            int(self.ROBOTIC_ARM1): 2.0,
            int(self.ROBOTIC_ARM2): 4.0,
            int(self.MOBILE_ROBOT): 1.0,
        }

        cap_1to4_all = self._unnormalize(self.agent_capabilities).astype(np.float32, copy=False)

        weights = []
        for cat in cats:
            idxs = self._cat_to_indices[cat]
            if idxs.size == 0:
                weights.append(self.category_weight_floor)
                continue

            # NOTE: This assumes all tasks in a category share the same requirement vector (true for your generator).
            req_raw = np.asarray(self._dataset_tasks[int(idxs[0])]["requirements"], dtype=np.float32).reshape(-1)
            if req_raw.shape[0] != 3:
                weights.append(self.category_weight_floor)
                continue

            eligible_aids = [aid for aid in self._agent_ids if bool(np.all(cap_1to4_all[aid] + 1e-5 >= req_raw))]

            if eligible_aids == [int(self.HUMAN)]:
                w = self.category_weight_floor
            else:
                spec_eligible = [aid for aid in eligible_aids if int(aid) in SPECIALIST_BOOST]
                if len(spec_eligible) == 0:
                    w = self.category_weight_nonspecialist
                else:
                    w = 1.0 + float(sum(SPECIALIST_BOOST[int(aid)] for aid in spec_eligible))

            weights.append(float(w))

        w = np.asarray(weights, dtype=np.float32)
        w = np.clip(w, 1e-8, None)
        w = w / float(w.sum())

        mix = float(np.clip(self.category_mix_uniform, 0.0, 1.0))
        u = np.ones_like(w, dtype=np.float32) / float(len(w))
        p = (1.0 - mix) * w + mix * u
        p = p / float(p.sum())

        self._cat_sampling_categories = cats
        self._cat_sampling_probs = p.astype(np.float32, copy=False)
        self._cat_cache_ready = True

        if self.print_category_sampling_cache:
            top = sorted(zip(cats, weights), key=lambda x: x[1], reverse=True)[:12]
            print("\n[CATEGORY_WEIGHTED] Top category weights (train-only):")
            for c, ww in top:
                print(f"  {c:>10s}  weight={ww:.3f}")
            print(f"[CATEGORY_WEIGHTED] mix_uniform={mix:.2f} | n_categories={len(cats)}\n")

    # ------------------------------------------------------------------
    # Helpers: task sampling
    # ------------------------------------------------------------------
    def _choose_valid_category(self) -> str:
        tmpl_keys = set(self.task_generator.task_templates.keys())
        req_keys = set(self.task_generator.task_requirements_map.keys())
        valid = sorted(tmpl_keys & req_keys)
        if not valid:
            raise ValueError("No valid categories: task_templates keys do not overlap task_requirements_map keys!")
        return self._py_rng.choice(valid)

    def _generate_nl_task(self):
        """
        Generate ONE task dict, from dataset or generator.
        """
        if self._dataset_tasks is not None:
            n = len(self._dataset_tasks)
            if n <= 0:
                raise ValueError("Dataset task list is empty.")

            if self.dataset_sampling == "sequential":
                idx = self._dataset_cursor
                self._dataset_cursor = (self._dataset_cursor + 1) % n

            elif self.dataset_sampling == "random":
                idx = int(self._rng.integers(0, n))

            elif self.dataset_sampling == "category_weighted":
                if self.train_mode:
                    self._maybe_build_category_sampling_cache()
                    if not self._cat_cache_ready or self._cat_to_indices is None:
                        idx = int(self._rng.integers(0, n))
                    else:
                        cat = str(self._rng.choice(self._cat_sampling_categories, p=self._cat_sampling_probs))
                        idx = int(self._rng.choice(self._cat_to_indices[cat]))
                else:
                    if self._dataset_order is None or self._dataset_cursor >= n:
                        self._dataset_order = self._rng.permutation(n)
                        self._dataset_cursor = 0
                    idx = int(self._dataset_order[self._dataset_cursor])
                    self._dataset_cursor += 1

            else:  # "permute" (default)
                if self._dataset_order is None or self._dataset_cursor >= n:
                    self._dataset_order = self._rng.permutation(n)
                    self._dataset_cursor = 0
                idx = int(self._dataset_order[self._dataset_cursor])
                self._dataset_cursor += 1

            nl_task = self._dataset_tasks[idx]

        else:
            category = self._choose_valid_category()
            nl_task = self.task_generator.generate_task(category, train_mode=self.train_mode)

        raw_req = np.asarray(nl_task["requirements"], dtype=np.float32)
        if raw_req.shape != (3,):
            raise ValueError(f"Invalid requirements shape: {raw_req.shape}")
        normalized_req = self._normalize(raw_req).astype(np.float32, copy=False)

        task_id = int(self._total_generated_tasks)
        task = {
            "id": task_id,
            "instruction": str(nl_task["instruction"]),
            "category": str(nl_task["category"]),
            "requirements": normalized_req,
        }

        self._total_generated_tasks += 1
        self._mask_curriculum_tasks_total += 1
        self._specialist_shaping_tasks_total += 1
        return task

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------
    def _calculate_rewards(
        self,
        bidding_results,
        return_outcomes: bool = False,
        raw_actions: Optional[Dict[int, int]] = None,
    ):
        """
        Procurement auction reward (2nd-price with first-price fallback if only one bidder).

        Design:
        - Eligible bidders get a small positive participation bonus.
        - Bid-shaping is applied winner-only.
        - Eligible non-bidders get a small negative penalty (to learn to bid).
        - Eligible bidders are guaranteed not to go negative (floored to their participation bonus).
        """
        UNQUALIFIED_BID_PENALTY = float(self.unqualified_bid_penalty)
        LOST_AUCTION_PENALTY = float(self.lost_auction_penalty)

        EPS = 1e-5

        MISSED_ASSIGNMENT_PENALTY = float(self.missed_assignment_penalty)
        ELIGIBLE_ABSTAIN_PENALTY = float(self.eligible_abstain_penalty)

        BID_SHAPING_COEFF = float(self.bid_shaping_coeff)
        ELIGIBLE_BID_BONUS = float(self.eligible_bid_bonus)

        rewards = {aid: 0.0 for aid in self._agent_ids}
        outcomes = {
            aid: {"eligible": False, "won": False, "slack_min": None, "payment": None, "profit": None}
            for aid in self._agent_ids
        }

        # Winner-only shaping accumulator
        shaping_term = {aid: 0.0 for aid in self._agent_ids}

        # Participation bonus actually granted (used as the positivity floor for eligible bidders)
        participation_bonus = {aid: 0.0 for aid in self._agent_ids}

        # Convert requirements back to [1..4]
        req = self._unnormalize(self.current_task["requirements"]).astype(np.float32, copy=False)

        def is_eligible(cap_1to4: np.ndarray) -> bool:
            return bool(np.all(cap_1to4 + EPS >= req))

        def slack_min_minus1_to1(cap_1to4: np.ndarray) -> float:
            slack = (cap_1to4 - req) / np.float32(3.0)
            return float(np.min(slack))

        def bottleneck_slack_0to1(cap_1to4: np.ndarray) -> float:
            slack = (cap_1to4 - req) / np.float32(3.0)
            slack = np.clip(slack, 0.0, 1.0)
            return float(np.min(slack))

        def base_cost_1to10(cap_1to4: np.ndarray) -> int:
            slack = (cap_1to4 - req) / np.float32(3.0)
            slack = np.clip(slack, 0.0, 1.0)

            bn = float(np.min(slack))
            mn = float(np.mean(slack))
            effective = 0.5 * bn + 0.5 * mn

            cost = int(round(self.bids_range - (self.bids_range - 1) * effective))
            return int(np.clip(cost, 1, self.bids_range))

        # --- OPTION A: specialist discount whenever specialist, plus human offset ---
        def effective_cost_1to10(cap_1to4: np.ndarray, agent_id: int) -> int:
            cost = base_cost_1to10(cap_1to4)

            if int(agent_id) in self.specialist_agent_ids:
                if self.debug_specialist_cost:
                    bn = bottleneck_slack_0to1(cap_1to4)
                    print(
                        "DBG",
                        int(agent_id),
                        "base",
                        base_cost_1to10(cap_1to4),
                        "bn",
                        bn,
                        "th",
                        self.specialist_discount_min_bn_slack,
                        "mult",
                        self.specialist_cost_multiplier,
                        "off",
                        self.specialist_cost_offset,
                    )

                cost = int(round(float(cost) * float(self.specialist_cost_multiplier))) + int(self.specialist_cost_offset)

            # Arm2-only extra discount (lets Arm2 bid more competitively)
            if int(agent_id) == int(self.ROBOTIC_ARM2):
                cost = int(round(float(cost) * float(self.arm2_cost_multiplier))) + int(self.arm2_cost_offset)

            if int(agent_id) == int(self.HUMAN) and self.human_cost_offset != 0:
                cost += int(self.human_cost_offset)

            return int(np.clip(cost, 1, self.bids_range))

        # Debug dicts for BiddingTracker
        actions_dbg: Dict[int, int] = {
            aid: int(np.clip(int(bidding_results.get(aid, 0)), 0, self.bids_range))
            for aid in self._agent_ids
        }
        mask_sums_dbg: Dict[int, float] = {
            aid: float(np.sum(self._build_action_mask(aid, req)))
            for aid in self._agent_ids
        }
        eligibility_dbg: Dict[int, bool] = {}
        computed_costs_dbg: Dict[int, float] = {}
        effective_costs_dbg: Dict[int, float] = {}

        slack_mins_dbg: Dict[int, float] = {}
        payments_dbg: Dict[int, float] = {}
        profits_dbg: Dict[int, float] = {}

        eligible_agents: List[int] = []
        bidders: List[Any] = []  # (bid_value, bottleneck_slack, agent_id, eff_cost)
        caps_1to4: Dict[int, np.ndarray] = {}

        # Pass 1: eligibility + diagnostics
        for aid in self._agent_ids:
            cap = self._unnormalize(self.agent_capabilities[aid]).astype(np.float32, copy=False)
            caps_1to4[aid] = cap

            sm = slack_min_minus1_to1(cap)
            slack_mins_dbg[aid] = float(sm)
            outcomes[aid]["slack_min"] = float(sm)

            elig = is_eligible(cap)
            outcomes[aid]["eligible"] = bool(elig)
            eligibility_dbg[aid] = bool(elig)

            if elig:
                eligible_agents.append(aid)
                bc = base_cost_1to10(cap)
                ec = effective_cost_1to10(cap, aid)
                computed_costs_dbg[aid] = float(bc)
                effective_costs_dbg[aid] = float(ec)

        spec_eligible = any(bool(eligibility_dbg.get(aid, False)) for aid in self.specialist_agent_ids)

        warmup = max(0, int(self.specialist_shaping_warmup_tasks))
        ramp = max(0, int(self.specialist_shaping_ramp_tasks))
        if ramp <= 0:
            alpha = 1.0
        else:
            t = max(0, int(self._specialist_shaping_tasks_total) - warmup)
            alpha = float(min(1.0, t / float(ramp)))

        # Interpreted as a FRACTION of the participation bonus (keeps eligible bidder rewards >= 0)
        entry_tax_frac = alpha * float(self.ns_bid_penalty_when_spec_eligible)

        steal_pen = alpha * float(self.ns_win_penalty_when_spec_eligible)
        spec_bonus = alpha * float(self.spec_win_bonus_when_spec_eligible)

        # Early exit: no eligible agents
        if len(eligible_agents) == 0:
            if self.show_detailed_episodes:
                self.bidding_tracker.record_bidding_round(
                    self.current_step,
                    self.current_task,
                    bidding_results,
                    rewards,
                    {i: self.agent_types[i] for i in self._agent_ids},
                    eligibility=eligibility_dbg,
                    computed_costs=computed_costs_dbg,
                    effective_costs=effective_costs_dbg,
                    actions=actions_dbg,
                    mask_sums=mask_sums_dbg,
                    winner_id=None,
                    raw_actions=raw_actions,
                    slack_mins=slack_mins_dbg,
                    payments=payments_dbg,
                    profits=profits_dbg,
                )
            return (rewards, outcomes) if return_outcomes else rewards

        # Pass 2: participation bonus for eligible bidders + compute shaping (winner-only later) + build bidders list
        for aid in self._agent_ids:
            bid = actions_dbg[aid]
            elig = bool(eligibility_dbg[aid])
            cap = caps_1to4[aid]

            if bid <= 0:
                continue

            if not elig:
                rewards[aid] += UNQUALIFIED_BID_PENALTY
                continue

            # participation bonus
            pb = float(ELIGIBLE_BID_BONUS)

            # reduce participation bonus (non-negative) for non-specialists when any specialist is eligible
            if spec_eligible and (aid not in self.specialist_agent_ids) and entry_tax_frac != 0.0:
                pb *= float(max(0.0, 1.0 - entry_tax_frac))

            rewards[aid] += pb
            participation_bonus[aid] = pb

            # compute shaping term (apply only to winner)
            tc_eff = effective_cost_1to10(cap, aid)
            bn = bottleneck_slack_0to1(cap)

            err = abs(bid - tc_eff) / float(self.bids_range - 1)
            shaping_term[aid] = float(BID_SHAPING_COEFF * (1.0 - 2.0 * err))

            bidders.append((bid, bn, aid, tc_eff))

        # If nobody bid (>0): penalize eligible agents for missing assignment (learn-to-bid signal)
        if len(bidders) == 0:
            if MISSED_ASSIGNMENT_PENALTY != 0.0:
                for aid in eligible_agents:
                    rewards[aid] += float(MISSED_ASSIGNMENT_PENALTY)

            if self.show_detailed_episodes:
                self.bidding_tracker.record_bidding_round(
                    self.current_step,
                    self.current_task,
                    bidding_results,
                    rewards,
                    {i: self.agent_types[i] for i in self._agent_ids},
                    eligibility=eligibility_dbg,
                    computed_costs=computed_costs_dbg,
                    effective_costs=effective_costs_dbg,
                    actions=actions_dbg,
                    mask_sums=mask_sums_dbg,
                    winner_id=None,
                    raw_actions=raw_actions,
                    slack_mins=slack_mins_dbg,
                    payments=payments_dbg,
                    profits=profits_dbg,
                )
            return (rewards, outcomes) if return_outcomes else rewards

        # With at least one bidder: penalize eligible abstainers (small negative)
        if ELIGIBLE_ABSTAIN_PENALTY != 0.0:
            for aid in eligible_agents:
                if actions_dbg.get(aid, 0) == 0:
                    rewards[aid] += float(ELIGIBLE_ABSTAIN_PENALTY)

        # Winner selection:
        #  1) lowest bid
        #  2) tie on bid -> prefer tie_priority (e.g., Arm2 > Arm1)
        #  3) then higher bottleneck slack
        #  4) final fallback: lower agent id
        bidders_sorted = sorted(
            bidders,
            key=lambda x: (x[0], -self.tie_priority.get(int(x[2]), 0), -x[1], x[2])
        )
        winning_bid, winner_bn, winner_aid, winner_true_cost_eff = bidders_sorted[0]
        outcomes[winner_aid]["won"] = True

        # =======================
        # competition + single-bidder payment fallback
        # =======================
        num_bidders = len(bidders_sorted)
        num_eligible = len(eligible_agents)

        # Competition factor in [0, 1]
        # - 0 when only one bidder OR only one eligible agent
        # - increases as more eligible agents actually bid
        competition = 0.0
        if num_eligible > 1:
            competition = (num_bidders - 1) / float(num_eligible - 1)
        competition = float(np.clip(competition, 0.0, 1.0))

        # 2nd price payment (or first-price fallback if only one bidder)
        if num_bidders >= 2:
            payment = float(bidders_sorted[1][0])
        else:
            payment = float(winning_bid)
        # =======================

        profit = payment - float(winner_true_cost_eff)

        payments_dbg[int(winner_aid)] = float(payment)
        profits_dbg[int(winner_aid)] = float(profit)
        outcomes[int(winner_aid)]["payment"] = float(payment)
        outcomes[int(winner_aid)]["profit"] = float(profit)

        # Winner gets profit-based reward + (scaled) fixed winner_reward + (scaled) winner-only shaping
        rewards[winner_aid] += float(profit / float(self.bids_range))

        # Fixed winner bonus (human can differ)
        win_bonus = float(self.winner_reward)
        bonus_scale = float(competition)

        # IMPORTANT: human fixed win reward should not disappear when competition=0
        if int(winner_aid) == int(self.HUMAN):
            win_bonus = float(self.human_winner_reward)
            bonus_scale = 1.0

        rewards[winner_aid] += bonus_scale * win_bonus

        rewards[winner_aid] += competition * float(shaping_term.get(winner_aid, 0.0))

        # Specialist win bonus / non-specialist steal penalty (scaled)
        if spec_eligible:
            if (winner_aid in self.specialist_agent_ids) and spec_bonus != 0.0:
                rewards[winner_aid] += competition * float(spec_bonus)
            elif (winner_aid not in self.specialist_agent_ids) and steal_pen != 0.0:
                rewards[winner_aid] -= competition * float(steal_pen)

        # Situational bonuses for Mobile Robot, Legged Robot and Robotic Arms
        mobile_bonus = 0.0
        legged_bonus = 0.0
        arm_bonus = 0.0

        mob_req = float(req[0])
        man_req = float(req[1])
        pay_req = float(req[2])

        task_desc_lower = str(self.current_task.get("instruction", "")).lower()

        if int(winner_aid) == 0:  # Mobile Robot
            mobile_keywords = [
                "deliver", "delivery", "transport", "carry", "haul", "pickup", "pick up",
                "dropoff", "drop off", "move", "navigate", "patrol", "warehouse",
                "corridor", "hallway", "route", "across", "between",
            ]
            kw = any(k in task_desc_lower for k in mobile_keywords)

            if (mob_req >= 3.0 and man_req <= 1.8) or (pay_req >= 3.2 and man_req <= 2.0):
                mobile_bonus = 0.25
            elif mob_req >= 2.4 and man_req <= 1.6:
                mobile_bonus = 0.18
            elif kw and mob_req >= 2.0 and man_req <= 2.2:
                mobile_bonus = 0.15
            elif mob_req >= 2.0 and man_req <= 1.5:
                mobile_bonus = 0.10

        elif int(winner_aid) == 2:  # Legged Robot
            hard_keywords = ["stairs", "stairway", "step", "stepped", "ramp", "sloped", "slope"]
            medium_keywords = [
                "uneven",
                "irregular terrain",
                "obstacle",
                "obstacles",
                "tight corridors",
                "narrow passages",
                "restricted paths",
                "confined spaces",
            ]

            if any(kw in task_desc_lower for kw in hard_keywords):
                legged_bonus = 0.6
            elif any(kw in task_desc_lower for kw in medium_keywords) or mob_req >= 3.8:
                legged_bonus = 0.24
            else:
                legged_bonus = 0.18

        elif int(winner_aid) in [3, 4]:  # Robotic Arm1 or Arm2
            if mob_req <= 1.5 and man_req >= 2.0:
                arm_bonus = 0.25

        rewards[winner_aid] += competition * (mobile_bonus + legged_bonus + arm_bonus)

        # Penalize losing bidders (still won't go negative for eligible bidders due to floor below)
        if LOST_AUCTION_PENALTY != 0.0:
            for _bid, _bn, aid, _tc_eff in bidders_sorted[1:]:
                rewards[aid] += float(LOST_AUCTION_PENALTY)

        # Guarantee: eligible BIDDERS never end negative (abstainers may be negative by design)
        for aid in eligible_agents:
            if actions_dbg.get(aid, 0) > 0:
                pb = float(participation_bonus.get(aid, 0.0))
                if pb > 0.0:
                    rewards[aid] = max(float(rewards[aid]), pb)
                else:
                    rewards[aid] = max(float(rewards[aid]), 0.0)

        if self.show_detailed_episodes:
            self.bidding_tracker.record_bidding_round(
                self.current_step,
                self.current_task,
                bidding_results,
                rewards,
                {i: self.agent_types[i] for i in self._agent_ids},
                eligibility=eligibility_dbg,
                computed_costs=computed_costs_dbg,
                effective_costs=effective_costs_dbg,
                actions=actions_dbg,
                mask_sums=mask_sums_dbg,
                winner_id=int(winner_aid),
                raw_actions=raw_actions,
                slack_mins=slack_mins_dbg,
                payments=payments_dbg,
                profits=profits_dbg,
            )

        return (rewards, outcomes) if return_outcomes else rewards

    # ------------------------------------------------------------------
    # Rendering / debug
    # ------------------------------------------------------------------
    def render(self):
        if hasattr(self, "bidding_tracker") and getattr(self.bidding_tracker, "current_step_data", None):
            latest_step = self.bidding_tracker.current_step_data[-1]
            self.bidding_tracker.print_step_summary(latest_step)
        else:
            print(f"\n=== Step {self.current_step} ===")
            print(f"Active task: {self.current_task.get('instruction', '')}")

            print("Recent task completions:")
            for history in self.task_history[-3:]:
                rewards = list(history["rewards"].values())
                mean_reward = float(np.mean(rewards)) if rewards else 0.0
                print(f"  '{history['instruction']}' -> mean reward: {mean_reward:.2f}")

    def print_episode_summary(self):
        if hasattr(self, "bidding_tracker"):
            self.bidding_tracker.print_episode_summary()
        # Also print the cumulative resume table (train/eval buckets)
        self._print_bidding_resume()