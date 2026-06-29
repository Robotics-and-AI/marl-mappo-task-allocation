"""
train.py

Goal
----
Train multi-agent PPO with a MAPPO-style CTDE setup:

  - Actor consumes *local* observation:
        agent_capabilities + text_embedding
    (NOTE: requirements are not included in obs anymore; they are used internally
     inside the environment for eligibility, masking, and reward computation.)

  - Critic consumes *global* state:
        all_agents_capabilities + text_embedding + elig_flags
    (elig_flags are computed in the env from capabilities vs requirements.)

Centralized critic via a custom TorchModelV2:
  - forward(): produces policy logits using local obs
  - value_function(): produces V(s) using global state

We also keep a strict TRAIN vs EVAL prompt split:
  - training env_config: {"train_mode": True}   -> use scenario["train"] templates
  - evaluation env_config: {"train_mode": False} -> use scenario["eval"] templates

One policy per agent (no parameter sharing):
  - policies: policy_0 .. policy_5
  - policy_mapping_fn(agent_id) -> "policy_<agent_id>"

Env-config knobs this file passes through (must exist in HeterogeneousTeamEnv):
  - eligible_bid_bonus
  - bid_shaping_coeff
  - missed_assignment_penalty
  - winner_reward
  - unqualified_bid_penalty
  - lost_auction_penalty
  - eligible_abstain_penalty
  - use_action_mask
  - force_eligible_to_bid
  - human_cost_offset
  - debug_action_mask
  - eval_mask_print_every_n

Additional env-config knobs:
  - human_winner_reward
  - resume_print_every_n  (optional debug printing inside env)

Specialist true-cost discount (for shaping/profit only; does NOT change auction winner rule)
  - specialist_agent_ids
  - specialist_cost_multiplier
  - specialist_cost_offset

Notes
-----
- The env is the single source of truth for whether masking constrains actions:
    env_config["use_action_mask"] controls whether the mask is restrictive or all-ones.
- For "always exactly one winner" behavior (assuming each task has at least 1 eligible agent),
  use:
    use_action_mask=True
    force_eligible_to_bid=True
- The model ALWAYS applies action_mask when present.
  (When env masking is "disabled", the env returns an all-ones mask → applying it is a no-op.)
"""

import os
import copy
import logging
from typing import Dict, Any, Optional

import numpy as np
import gymnasium as gym
import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune.registry import register_env

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO  # used in the sanity check helper
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.multi_agent_env import ENV_STATE

import torch
import torch.nn as nn

from heterogeneous_team_env import HeterogeneousTeamEnv


# Try to import CheckpointConfig (Ray AIR). Keep a fallback for older Ray versions.
try:
    from ray.air.config import CheckpointConfig
except Exception:  # pragma: no cover
    CheckpointConfig = None


# Logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("train")


# -----------------------------------------------------------------------------
# Helpers for per-agent TRAIN/EVAL resume tables + metric robustness
# -----------------------------------------------------------------------------
def _get_first(d: dict, keys, default=None):
    """Return the first non-None key present in dict d (or default)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _print_resume_table_from_df(df, prefix: str, agent_names: dict):
    """
    Print a readable per-agent totals table from a Tune Result.metrics_dataframe.

    prefix examples:
      - "bidding_resume/train"
      - "evaluation/bidding_resume/eval"
    """
    if df is None or getattr(df, "empty", True):
        print(f"[{prefix}] No metrics_dataframe to print table from.")
        return

    tasks_col = f"{prefix}/tasks_total"
    if tasks_col not in df.columns:
        print(f"[{prefix}] Table columns not found (did callbacks write totals into result?).")
        return

    dff = df.dropna(subset=[tasks_col])
    if dff.empty:
        print(f"[{prefix}] No rows with {tasks_col}.")
        return

    row = dff.iloc[-1]

    def _ival(col):
        try:
            v = row.get(col, 0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0
            return int(round(float(v)))
        except Exception:
            return 0

    tasks_total = _ival(tasks_col)
    title = "TRAIN" if prefix.endswith("/train") else "EVAL"

    print("\n" + "=" * 90)
    print(f"[BIDDING RESUME TABLE] phase={title} | tasks={tasks_total}")
    print("-" * 90)
    print(f"{'Agent':<5} {'Name':<18} {'Eligible':>10} {'Wins':>8} {'Win% (wins/eligible)':>22}")
    print("-" * 90)

    for aid, name in agent_names.items():
        elig = _ival(f"{prefix}/agent_{aid}/eligible_total")
        wins = _ival(f"{prefix}/agent_{aid}/wins_total")
        winpct = (100.0 * wins / max(1, elig))
        print(f"{aid:<5} {name:<18} {elig:>10} {wins:>8} {winpct:>21.2f}%")

    print("=" * 90 + "\n")


def _safe_get_best_result(results, metric: str, mode: str, fallback_metric: Optional[str] = None):
    """
    Get best result by `metric`. If metric is missing (e.g., eval disabled),
    optionally fall back to `fallback_metric`.

    Returns: (best_result, used_metric)
    """
    try:
        return results.get_best_result(metric=metric, mode=mode), metric
    except Exception as e:
        if fallback_metric is None:
            raise
        logger.warning(
            f"get_best_result(metric={metric}) failed (likely eval disabled/absent). "
            f"Falling back to metric={fallback_metric}. Error: {repr(e)}"
        )
        return results.get_best_result(metric=fallback_metric, mode=mode), fallback_metric


def _merge_custom_metrics(res: dict) -> dict:
    """
    RLlib placement of custom_metrics varies by API stack/version.
    Merge the common locations into one dict.
    """
    out = {}
    if isinstance(res.get("custom_metrics"), dict):
        out.update(res["custom_metrics"])

    envr = res.get("env_runners")
    if isinstance(envr, dict) and isinstance(envr.get("custom_metrics"), dict):
        out.update(envr["custom_metrics"])

    samp = res.get("sampler_results")
    if isinstance(samp, dict) and isinstance(samp.get("custom_metrics"), dict):
        out.update(samp["custom_metrics"])

    return out


def _cm_get_agent_value(cm: dict, aid: int, stem: str):
    """
    Try exact keys first, then a tolerant scan.

    Expected stems:
      - "eligible_steps"
      - "wins"
    """
    exact_keys = [
        f"agent_{aid}/{stem}_mean",
        f"agent_{aid}/{stem}",
        f"agent_{aid}_{stem}_mean",
        f"agent_{aid}_{stem}",
    ]
    for k in exact_keys:
        if k in cm:
            return cm[k]

    # Tolerant scan fallback
    needle_a = f"agent_{aid}"
    for k, v in cm.items():
        ks = str(k)
        if needle_a in ks and stem in ks:
            # prefer *_mean if present
            if ks.endswith("_mean"):
                return v
            # else accept a bare value
            if ks.endswith(stem) or f"{stem}_" in ks:
                return v

    return None


# -----------------------------------------------------------------------------
# Custom Model: MAPPO-style Centralized Critic (TorchModelV2)
# -----------------------------------------------------------------------------
class MAPPOCentralCriticModel(TorchModelV2, nn.Module):
    """
    Centralized-critic model for PPO in multi-agent setting.

    Observation layout (from env):
      obs is a Dict with keys:
        - "obs":         local observation vector
        - "state" or ENV_STATE: global state vector
        - "action_mask": action mask vector

    Current env layout:
      - Local "obs"  = [agent_capabilities, task_embedding]
      - Global state = [all_agent_capabilities, task_embedding, elig_flags]

    IMPORTANT:
      RLlib will NOT automatically enforce action_mask unless you apply it to logits.
      This model ALWAYS applies action_mask when it is present.
      (When env masking is "disabled", your env returns an all-ones mask, so this is a no-op.)
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        ccfg = (model_config.get("custom_model_config") or {}).copy()
        self.local_obs_dim = int(ccfg["local_obs_dim"])
        self.state_dim = int(ccfg["state_dim"])

        self.debug_action_mask = bool(ccfg.get("debug_action_mask", False))
        self._printed_mask_debug_once = False

        actor_hiddens = list(ccfg.get("actor_hiddens", [128, 128]))
        critic_hiddens = list(ccfg.get("critic_hiddens", [256, 256]))

        # Actor network: local_obs -> action logits
        actor_layers = []
        in_dim = self.local_obs_dim
        for h in actor_hiddens:
            actor_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, num_outputs))
        self.actor = nn.Sequential(*actor_layers)

        # Critic network: global_state -> scalar value
        critic_layers = []
        in_dim = self.state_dim
        for h in critic_hiddens:
            critic_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self._last_state_tensor: Optional[torch.Tensor] = None
        self._num_outputs = int(num_outputs)

    def _split_obs(self, obs_any):
        # Case 1: dict-like obs (standard RLlib dict obs)
        if isinstance(obs_any, dict):
            local = obs_any["obs"]
            state = obs_any.get("state", None)
            if state is None:
                state = obs_any.get(ENV_STATE, None)
            if state is None:
                raise KeyError(f"Expected global state under 'state' or '{ENV_STATE}', got keys={list(obs_any.keys())}")
            mask = obs_any.get("action_mask", None)
            return local, state, mask

        # Case 2: flattened tensor [B, local+state] (no mask)
        if isinstance(obs_any, torch.Tensor):
            if obs_any.dim() != 2:
                raise ValueError(f"Expected flattened obs tensor dim=2, got shape {tuple(obs_any.shape)}")
            local = obs_any[:, : self.local_obs_dim]
            state = obs_any[:, self.local_obs_dim : self.local_obs_dim + self.state_dim]
            return local, state, None

        # Case 3: flattened numpy array [B, local+state] (no mask)
        if isinstance(obs_any, np.ndarray):
            if obs_any.ndim != 2:
                raise ValueError(f"Expected flattened obs ndarray ndim=2, got shape {obs_any.shape}")
            local = torch.from_numpy(obs_any[:, : self.local_obs_dim])
            state = torch.from_numpy(obs_any[:, self.local_obs_dim : self.local_obs_dim + self.state_dim])
            return local, state, None

        # Some RLlib dict-like wrappers
        if hasattr(obs_any, "get"):
            local = obs_any.get("obs", None)
            state = obs_any.get("state", None)
            if state is None:
                state = obs_any.get(ENV_STATE, None)
            if local is not None and state is not None:
                return local, state, obs_any.get("action_mask", None)

        raise TypeError(f"Unsupported obs type in model forward: {type(obs_any)}")

    def forward(self, input_dict, state, seq_lens):
        obs_any = input_dict["obs"]
        local_obs, global_state, action_mask = self._split_obs(obs_any)

        # Ensure tensors (RLlib can sometimes hand dict values as numpy arrays)
        local_obs = torch.as_tensor(local_obs, dtype=torch.float32, device=self.actor[0].weight.device)
        global_state = torch.as_tensor(global_state, dtype=torch.float32, device=self.actor[0].weight.device)

        self._last_state_tensor = global_state
        logits = self.actor(local_obs)

        # ---------------------------
        # APPLY ACTION MASK (ALWAYS if present)
        # ---------------------------
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.as_tensor(action_mask, dtype=torch.float32, device=logits.device)
            else:
                action_mask = action_mask.to(device=logits.device, dtype=torch.float32)

            # Ensure shape is [B, A]
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

            if action_mask.shape[-1] != self._num_outputs:
                raise ValueError(
                    f"action_mask last dim {action_mask.shape[-1]} != num_outputs {self._num_outputs}"
                )

            # log(0) = -inf; clamp to a large negative number for numerical safety
            inf_mask = torch.clamp(torch.log(action_mask), min=-1e9)
            logits = logits + inf_mask

            if self.debug_action_mask and not self._printed_mask_debug_once:
                with torch.no_grad():
                    ms = float(action_mask.sum().item())
                    mn = float(action_mask.min().item())
                    mx = float(action_mask.max().item())
                print(
                    f"[MODEL MASK DEBUG] action_mask stats: sum={ms:.1f} min={mn:.1f} max={mx:.1f} "
                    f"(num_outputs={self._num_outputs})"
                )
                self._printed_mask_debug_once = True

        return logits, state

    def value_function(self):
        assert self._last_state_tensor is not None, "value_function called before forward"
        return self.critic(self._last_state_tensor).squeeze(-1)
    
# -----------------------------------------------------------------------------
# Register custom model once per process (avoid duplicate-registration warnings)
# -----------------------------------------------------------------------------
if not getattr(ModelCatalog, "_mappo_cc_model_registered", False):
    ModelCatalog.register_custom_model("mappo_cc_model", MAPPOCentralCriticModel)
    ModelCatalog._mappo_cc_model_registered = True


# -----------------------------------------------------------------------------
# Callback: Eval-only debug printing + metric shim + per-agent custom metrics
# -----------------------------------------------------------------------------
class EvalOnlyDebugCallbacks(DefaultCallbacks):
    """
    Callback that:
      - collects per-episode per-agent stats using info dicts emitted by the env
      - writes custom_metrics (RLlib aggregates these across episodes/workers)
      - accumulates TRAIN/EVAL lifetime totals into result dict keys so they appear in metrics_dataframe
    """

    NUM_BIDS = 11  # bids 0..10

    def __init__(self):
        super().__init__()
        self._agent_ids = [0, 1, 2, 3, 4, 5]

        self._train_tasks_total = 0
        self._train_eligible_total = {aid: 0 for aid in self._agent_ids}
        self._train_wins_total = {aid: 0 for aid in self._agent_ids}

        self._eval_tasks_total = 0
        self._eval_eligible_total = {aid: 0 for aid in self._agent_ids}
        self._eval_wins_total = {aid: 0 for aid in self._agent_ids}

    @staticmethod
    def _agent_ids_from_episode(episode) -> list:
        try:
            ids = list(episode.get_agents())
            if ids:
                return ids
        except Exception:
            pass
        return [0, 1, 2, 3, 4, 5]

    @staticmethod
    def _last_info_for(episode, aid):
        try:
            return episode.last_info_for(aid)
        except Exception:
            pass
        try:
            return episode.last_info_for(str(aid))
        except Exception:
            return None

    @staticmethod
    def _find_key(d: dict, k_int: int):
        if k_int in d:
            return k_int
        ks = str(k_int)
        if ks in d:
            return ks
        return None

    def on_episode_start(self, *, worker, base_env, episode, env_index, **kwargs):
        agent_ids = self._agent_ids_from_episode(episode)
        episode.user_data["mappo_agent_ids"] = agent_ids

        episode.user_data["env_steps"] = 0
        episode.user_data["eligible_steps"] = {aid: 0 for aid in agent_ids}
        episode.user_data["wins"] = {aid: 0 for aid in agent_ids}
        episode.user_data["eligible_bid0"] = {aid: 0 for aid in agent_ids}
        episode.user_data["eligible_loses"] = {aid: 0 for aid in agent_ids}

        # Use RAW action distribution when eligible (policy output, before env enforcement)
        episode.user_data["bid_counts_when_eligible"] = {aid: [0 for _ in range(self.NUM_BIDS)] for aid in agent_ids}

        # SlackMin stats (over eligible steps)
        episode.user_data["slack_sum_eligible"] = {aid: 0.0 for aid in agent_ids}
        episode.user_data["slack_cnt_eligible"] = {aid: 0 for aid in agent_ids}

        # Payment/Profit stats (over wins only)
        episode.user_data["payment_sum_wins"] = {aid: 0.0 for aid in agent_ids}
        episode.user_data["profit_sum_wins"] = {aid: 0.0 for aid in agent_ids}
        episode.user_data["payprof_cnt_wins"] = {aid: 0 for aid in agent_ids}

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        episode.user_data["env_steps"] = int(episode.user_data.get("env_steps", 0)) + 1

        agent_ids = episode.user_data.get("mappo_agent_ids", [0, 1, 2, 3, 4, 5])
        bid_counts = episode.user_data.get("bid_counts_when_eligible", {})

        for aid in agent_ids:
            info = self._last_info_for(episode, aid)
            if not isinstance(info, dict):
                continue

            eligible = bool(info.get("eligible", False))
            won = bool(info.get("won", False))

            # Prefer RAW action (policy output) if present
            action = info.get("raw_action", info.get("action", None))

            slack_min = info.get("slack_min", None)
            payment = info.get("payment", None)
            profit = info.get("profit", None)

            if not eligible:
                continue

            episode.user_data["eligible_steps"][aid] += 1

            if slack_min is not None:
                try:
                    episode.user_data["slack_sum_eligible"][aid] += float(slack_min)
                    episode.user_data["slack_cnt_eligible"][aid] += 1
                except Exception:
                    pass

            if action is not None:
                try:
                    a = int(action)
                except Exception:
                    a = None
                if a is not None and 0 <= a < self.NUM_BIDS:
                    if aid in bid_counts:
                        bid_counts[aid][a] += 1

            if action is not None and int(action) == 0:
                episode.user_data["eligible_bid0"][aid] += 1
                continue

            if won:
                episode.user_data["wins"][aid] += 1
                if payment is not None and profit is not None:
                    try:
                        episode.user_data["payment_sum_wins"][aid] += float(payment)
                        episode.user_data["profit_sum_wins"][aid] += float(profit)
                        episode.user_data["payprof_cnt_wins"][aid] += 1
                    except Exception:
                        pass
            else:
                episode.user_data["eligible_loses"][aid] += 1

    def on_episode_end(self, *, worker, base_env, episode, env_index, **kwargs):
        # -----------------------
        # START original body (kept for minimal diffs)
        # -----------------------
        if env_index == 0 and getattr(worker, "worker_index", 0) == 0:
            envs = base_env.get_sub_environments()
            if envs and env_index < len(envs):
                env = envs[env_index]

                if not getattr(env, "train_mode", True):
                    try:
                        print("\n" + "=" * 120)
                        print(
                            f"[EVAL DEBUG] episode_end | env_step={getattr(env, 'current_step', None)} | "
                            f"max_steps={getattr(env, 'max_steps', None)}"
                        )

                        tracker = getattr(env, "bidding_tracker", None)
                        if tracker is not None and getattr(tracker, "current_step_data", None):
                            last_k = 10
                            steps = tracker.current_step_data[-last_k:]
                            for s in steps:
                                tracker.print_step_summary(s)
                        else:
                            print(
                                "No bidding tracker data recorded yet. "
                                "(Set show_detailed_episodes=True in eval env_config)"
                            )
                        print("=" * 120 + "\n")
                    except Exception as e:
                        print(f"[EVAL DEBUG] Failed to print debug info: {repr(e)}")

        agent_ids = episode.user_data.get("mappo_agent_ids", [0, 1, 2, 3, 4, 5])
        env_steps = int(episode.user_data.get("env_steps", 0))
        denom_tasks = max(1, env_steps)

        eligible_steps = episode.user_data.get("eligible_steps", {})
        wins = episode.user_data.get("wins", {})
        eligible_bid0 = episode.user_data.get("eligible_bid0", {})
        eligible_loses = episode.user_data.get("eligible_loses", {})
        bid_counts = episode.user_data.get("bid_counts_when_eligible", {})

        per_agent_summary = []

        for aid in agent_ids:
            elig = int(eligible_steps.get(aid, 0))
            w = int(wins.get(aid, 0))
            b0 = int(eligible_bid0.get(aid, 0))
            l = int(eligible_loses.get(aid, 0))

            denom_elig = max(1, elig)

            episode.custom_metrics[f"agent_{aid}/win_rate"] = float(w) / float(denom_elig)
            episode.custom_metrics[f"agent_{aid}/eligible_bid0_rate"] = float(b0) / float(denom_elig)
            episode.custom_metrics[f"agent_{aid}/eligible_lose_rate"] = float(l) / float(denom_elig)

            episode.custom_metrics[f"agent_{aid}/eligible_steps"] = float(elig)
            episode.custom_metrics[f"agent_{aid}/wins"] = float(w)
            episode.custom_metrics[f"agent_{aid}/eligible_bid0"] = float(b0)
            episode.custom_metrics[f"agent_{aid}/eligible_loses"] = float(l)

            pct_tasks_won = float(w) / float(denom_tasks)
            episode.custom_metrics[f"agent_{aid}/pct_tasks_won"] = pct_tasks_won
            episode.custom_metrics[f"agent_{aid}/pct_tasks_eligible"] = float(elig) / float(denom_tasks)

            counts = bid_counts.get(aid, [0] * self.NUM_BIDS)
            total_elig_bids = int(sum(counts))

            slack_sum = float(episode.user_data.get("slack_sum_eligible", {}).get(aid, 0.0))
            slack_cnt = int(episode.user_data.get("slack_cnt_eligible", {}).get(aid, 0))
            episode.custom_metrics[f"agent_{aid}/slack_min_mean_eligible"] = (slack_sum / max(1, slack_cnt))

            pp_cnt = int(episode.user_data.get("payprof_cnt_wins", {}).get(aid, 0))
            pay_sum = float(episode.user_data.get("payment_sum_wins", {}).get(aid, 0.0))
            prof_sum = float(episode.user_data.get("profit_sum_wins", {}).get(aid, 0.0))

            episode.custom_metrics[f"agent_{aid}/payment_mean_when_won"] = (pay_sum / max(1, pp_cnt))
            episode.custom_metrics[f"agent_{aid}/profit_mean_when_won"] = (prof_sum / max(1, pp_cnt))
            episode.custom_metrics[f"agent_{aid}/payprof_cnt_wins"] = float(pp_cnt)

            if total_elig_bids <= 0:
                for b in range(self.NUM_BIDS):
                    episode.custom_metrics[f"agent_{aid}/bid_pct_eligible_{b}"] = 0.0
                episode.custom_metrics[f"agent_{aid}/bid_entropy_eligible"] = 0.0
                entropy_bits = 0.0
            else:
                probs = np.asarray(counts, dtype=np.float32) / float(total_elig_bids)
                for b in range(self.NUM_BIDS):
                    episode.custom_metrics[f"agent_{aid}/bid_pct_eligible_{b}"] = float(probs[b])

                eps = 1e-12
                entropy_bits = float(-np.sum(probs * np.log2(probs + eps)))
                episode.custom_metrics[f"agent_{aid}/bid_entropy_eligible"] = entropy_bits

            per_agent_summary.append((aid, pct_tasks_won, entropy_bits))

        human_key_w = self._find_key(wins, 5)
        human_key_b = self._find_key(bid_counts, 5)
        if human_key_w is not None:
            human_w = int(wins.get(human_key_w, 0))
            episode.custom_metrics["assignment_mix/human_pct_tasks_won"] = float(human_w) / float(denom_tasks)
        if human_key_b is not None:
            counts = bid_counts.get(human_key_b, [0] * self.NUM_BIDS)
            tot = int(sum(counts))
            if tot > 0:
                probs = np.asarray(counts, dtype=np.float32) / float(tot)
                eps = 1e-12
                episode.custom_metrics["bid_entropy/human"] = float(-np.sum(probs * np.log2(probs + eps)))
            else:
                episode.custom_metrics["bid_entropy/human"] = 0.0

        try:
            if env_index == 0 and getattr(worker, "worker_index", 0) == 0:
                envs = base_env.get_sub_environments()
                if envs and env_index < len(envs):
                    env = envs[env_index]
                    if not getattr(env, "train_mode", True):
                        print("[EVAL SUMMARY] Assignment mix (wins/tasks) + eligible-bid entropy")
                        for (aid, pct_won, ent) in per_agent_summary:
                            print(f"  agent={aid} pct_tasks_won={pct_won*100:.1f}% bid_entropy_eligible={ent:.2f} bits")
                        print("")
        except Exception:
            pass
        # -----------------------
        # END original body
        # -----------------------

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # -----------------------
        # Metric shim (normalizes key names across RLlib versions)
        # -----------------------
        if "env_runners/episode_return_mean" not in result:
            if "episode_reward_mean" in result:
                result["env_runners/episode_return_mean"] = result["episode_reward_mean"]
            elif "episode_return_mean" in result:
                result["env_runners/episode_return_mean"] = result["episode_return_mean"]

        if "evaluation" in result and isinstance(result["evaluation"], dict):
            ev = result["evaluation"]
            if "env_runners/episode_return_mean" not in ev:
                if "episode_reward_mean" in ev:
                    ev["env_runners/episode_return_mean"] = ev["episode_reward_mean"]
                elif "episode_return_mean" in ev:
                    ev["env_runners/episode_return_mean"] = ev["episode_return_mean"]
            if "evaluation/env_runners/episode_return_mean" not in result and "env_runners/episode_return_mean" in ev:
                result["evaluation/env_runners/episode_return_mean"] = ev["env_runners/episode_return_mean"]

        # -----------------------
        # Minimal init guards (in case callback object is recreated)
        # -----------------------
        if not hasattr(self, "_agent_ids"):
            self._agent_ids = [0, 1, 2, 3, 4, 5]

        if not hasattr(self, "_train_eligible_total"):
            self._train_eligible_total = {aid: 0 for aid in self._agent_ids}
        if not hasattr(self, "_train_wins_total"):
            self._train_wins_total = {aid: 0 for aid in self._agent_ids}
        if not hasattr(self, "_eval_eligible_total"):
            self._eval_eligible_total = {aid: 0 for aid in self._agent_ids}
        if not hasattr(self, "_eval_wins_total"):
            self._eval_wins_total = {aid: 0 for aid in self._agent_ids}

        if not hasattr(self, "_train_tasks_total"):
            self._train_tasks_total = 0
        if not hasattr(self, "_eval_tasks_total"):
            self._eval_tasks_total = 0

        if not hasattr(self, "_last_train_env_steps_lifetime"):
            self._last_train_env_steps_lifetime = 0
        if not hasattr(self, "_last_eval_env_steps_lifetime"):
            self._last_eval_env_steps_lifetime = 0

        # -----------------------
        # TRAIN: use lifetime steps (prevents double counting)
        # In your env, 1 env step == 1 task, so "tasks_total" tracks step totals.
        # -----------------------
        train_life = 0
        try:
            train_life = int(result.get("num_env_steps_sampled_lifetime", 0) or 0)
        except Exception:
            train_life = 0

        train_delta = max(0, train_life - int(self._last_train_env_steps_lifetime))
        self._last_train_env_steps_lifetime = train_life
        self._train_tasks_total = train_life

        ep_len = result.get("env_runners/episode_len_mean", None)
        if ep_len is None:
            envr = result.get("env_runners")
            if isinstance(envr, dict):
                ep_len = envr.get("episode_len_mean", None)
        try:
            ep_len = float(ep_len) if ep_len is not None else 50.0
        except Exception:
            ep_len = 50.0

        episodes_this_iter = 0
        if train_delta > 0:
            episodes_this_iter = max(1, int(train_delta / max(1.0, ep_len)))

        cm = _merge_custom_metrics(result)

        for aid in self._agent_ids:
            elig_v = _cm_get_agent_value(cm, aid, "eligible_steps")
            wins_v = _cm_get_agent_value(cm, aid, "wins")

            if elig_v is not None:
                try:
                    elig_f = float(elig_v)
                    if episodes_this_iter > 0 and elig_f <= (ep_len + 1.0):
                        self._train_eligible_total[aid] += int(round(elig_f * episodes_this_iter))
                    else:
                        self._train_eligible_total[aid] += int(round(elig_f))
                except Exception:
                    pass

            if wins_v is not None:
                try:
                    wins_f = float(wins_v)
                    if episodes_this_iter > 0 and wins_f <= (ep_len + 1.0):
                        self._train_wins_total[aid] += int(round(wins_f * episodes_this_iter))
                    else:
                        self._train_wins_total[aid] += int(round(wins_f))
                except Exception:
                    pass

        # Write TRAIN totals (root keys -> metrics_dataframe columns)
        result["bidding_resume/train/tasks_total"] = int(self._train_tasks_total)
        for aid in self._agent_ids:
            result[f"bidding_resume/train/agent_{aid}/eligible_total"] = int(self._train_eligible_total[aid])
            result[f"bidding_resume/train/agent_{aid}/wins_total"] = int(self._train_wins_total[aid])

        # -----------------------
        # EVAL: accumulate + mirror into root "evaluation/..." keys so metrics_dataframe has columns
        # -----------------------
        ev = result.get("evaluation", None)
        if isinstance(ev, dict):
            ev_life = 0
            try:
                ev_life = int(ev.get("num_env_steps_sampled_lifetime", 0) or 0)
            except Exception:
                ev_life = 0

            ev_delta = 0
            if ev_life > 0:
                ev_delta = max(0, ev_life - int(self._last_eval_env_steps_lifetime))
                self._last_eval_env_steps_lifetime = ev_life
                self._eval_tasks_total = ev_life

            ev_ep_len = ev.get("env_runners/episode_len_mean", None)
            if ev_ep_len is None:
                evr = ev.get("env_runners")
                if isinstance(evr, dict):
                    ev_ep_len = evr.get("episode_len_mean", None)
            try:
                ev_ep_len = float(ev_ep_len) if ev_ep_len is not None else 50.0
            except Exception:
                ev_ep_len = 50.0

            if ev_delta <= 0 and ev_life <= 0:
                eval_duration = int(getattr(algorithm.config, "evaluation_duration", 0) or 0)
                if eval_duration > 0:
                    est_steps = int(eval_duration * max(1.0, ev_ep_len))
                    self._eval_tasks_total += est_steps
                    ev_delta = est_steps

            ev_eps = 0
            if ev_delta > 0:
                ev_eps = max(1, int(ev_delta / max(1.0, ev_ep_len)))

            ev_cm = _merge_custom_metrics(ev)

            for aid in self._agent_ids:
                elig_v = _cm_get_agent_value(ev_cm, aid, "eligible_steps")
                wins_v = _cm_get_agent_value(ev_cm, aid, "wins")

                if elig_v is not None:
                    try:
                        elig_f = float(elig_v)
                        if ev_eps > 0 and elig_f <= (ev_ep_len + 1.0):
                            self._eval_eligible_total[aid] += int(round(elig_f * ev_eps))
                        else:
                            self._eval_eligible_total[aid] += int(round(elig_f))
                    except Exception:
                        pass

                if wins_v is not None:
                    try:
                        wins_f = float(wins_v)
                        if ev_eps > 0 and wins_f <= (ev_ep_len + 1.0):
                            self._eval_wins_total[aid] += int(round(wins_f * ev_eps))
                        else:
                            self._eval_wins_total[aid] += int(round(wins_f))
                    except Exception:
                        pass

            ev["bidding_resume/eval/tasks_total"] = int(self._eval_tasks_total)
            for aid in self._agent_ids:
                ev[f"bidding_resume/eval/agent_{aid}/eligible_total"] = int(self._eval_eligible_total[aid])
                ev[f"bidding_resume/eval/agent_{aid}/wins_total"] = int(self._eval_wins_total[aid])

            result["evaluation/bidding_resume/eval/tasks_total"] = int(self._eval_tasks_total)
            for aid in self._agent_ids:
                result[f"evaluation/bidding_resume/eval/agent_{aid}/eligible_total"] = int(self._eval_eligible_total[aid])
                result[f"evaluation/bidding_resume/eval/agent_{aid}/wins_total"] = int(self._eval_wins_total[aid])


def apply_env_runners_compat(cfg: PPOConfig, num_runners: int, num_envs_per_runner: int) -> PPOConfig:
    """Compatibility shim across RLlib versions for setting rollout workers/env runners."""
    if not hasattr(cfg, "env_runners"):
        raise ValueError("This Ray version does not have AlgorithmConfig.env_runners().")

    try:
        return cfg.env_runners(
            num_env_runners=num_runners,
            num_envs_per_env_runner=num_envs_per_runner,
        )
    except TypeError:
        return cfg.env_runners(
            num_rollout_workers=num_runners,
            num_envs_per_worker=num_envs_per_runner,
        )


def apply_training_compat(cfg: PPOConfig, desired: Dict[str, Any]) -> PPOConfig:
    """Compatibility shim across RLlib versions for training() keyword differences."""
    if not hasattr(cfg, "training"):
        raise ValueError("This Ray version does not have AlgorithmConfig.training().")

    kw = dict(desired)

    while True:
        try:
            return cfg.training(**kw)
        except TypeError as e:
            msg = str(e)
            bad_key = None
            if "unexpected keyword argument" in msg:
                parts = msg.split("'")
                if len(parts) >= 2:
                    bad_key = parts[1]
            if not bad_key:
                raise

            if bad_key == "minibatch_size" and "minibatch_size" in kw:
                kw["sgd_minibatch_size"] = kw.pop("minibatch_size")
                continue
            if bad_key == "sgd_minibatch_size" and "sgd_minibatch_size" in kw:
                kw["minibatch_size"] = kw.pop("sgd_minibatch_size")
                continue
            if bad_key == "num_epochs" and "num_epochs" in kw:
                kw["num_sgd_iter"] = kw.pop("num_epochs")
                continue
            if bad_key == "num_sgd_iter" and "num_sgd_iter" in kw:
                kw["num_epochs"] = kw.pop("num_sgd_iter")
                continue

            if bad_key in kw:
                logger.warning(f"Dropping unsupported training() kwarg for this RLlib version: {bad_key}")
                kw.pop(bad_key)
                if not kw:
                    return cfg
                continue

            raise


def _metric_total_from_df(df, lifetime_col: str, this_iter_col: str) -> int:
    """
    Prefer cumulative '*_lifetime' column; fallback to summing '*_this_iter' if needed.
    Returns 0 if not available.
    """
    if df is None:
        return 0

    if lifetime_col in getattr(df, "columns", []):
        try:
            v = df[lifetime_col].dropna()
            if len(v) > 0:
                return int(v.max())
        except Exception:
            pass

    if this_iter_col in getattr(df, "columns", []):
        try:
            v = df[this_iter_col].dropna()
            if len(v) > 0:
                return int(v.sum())
        except Exception:
            pass

    return 0


def per_agent_policy_mapping_fn(agent_id, *a, **k) -> str:
    """
    Policy mapping: one policy per agent id.
    Agent IDs are expected to be ints (0..5), but this is tolerant to string IDs too.
    """
    try:
        aid = int(agent_id)
    except Exception:
        aid = int(str(agent_id).split("_")[-1])
    return f"policy_{aid}"


def build_single_trial_config(args: Dict[str, Any]) -> Dict[str, Any]:

    env_name = "heterogeneous_team_env"

    env_config_train = {
        "max_steps": int(args.get("max_steps", 50)),
        "train_mode": True,
        "show_detailed_episodes": False,

        "dataset_mode": bool(args.get("dataset_mode", True)),
        "dataset_path": args.get("dataset_path", None),
        "dataset_sampling": str(args.get("dataset_sampling", "permute")),

        "eligible_bid_bonus": float(args.get("eligible_bid_bonus", 0.0)),
        "bid_shaping_coeff": float(args.get("bid_shaping_coeff", 0.50)),
        "missed_assignment_penalty": float(args.get("missed_assignment_penalty", -0.50)),
        "winner_reward": float(args.get("winner_reward", 0.10)),

        "unqualified_bid_penalty": float(args.get("unqualified_bid_penalty", -1.00)),
        "lost_auction_penalty": float(args.get("lost_auction_penalty", -0.05)),

        "eligible_abstain_penalty": float(args.get("eligible_abstain_penalty", -0.10)),

        "human_cost_offset": int(args.get("human_cost_offset", 2)),

        "use_action_mask": bool(args.get("use_action_mask", False)),
        "force_eligible_to_bid": bool(args.get("force_eligible_to_bid", False)),
        "debug_action_mask": bool(args.get("debug_action_mask", False)),

        "eval_mask_print_every_n": 0,

        # Specialist true-cost discount (shaping/profit only)
        "specialist_agent_ids": list(args.get("specialist_agent_ids", [0, 2, 3, 4])),
        "specialist_cost_multiplier": float(args.get("specialist_cost_multiplier", 1.0)),
        "specialist_cost_offset": int(args.get("specialist_cost_offset", 0)),
        "debug_specialist_cost": bool(args.get("debug_specialist_cost", False)),

        # pass-through for env changes
        "resume_print_every_n": int(args.get("resume_print_every_n", 0)),
        "human_winner_reward": float(args.get("human_winner_reward", 0.5)),

        # --- Probabilistic action-mask curriculum ---
        "mask_curriculum": bool(args.get("mask_curriculum", False)),
        "mask_warmup_tasks": int(args.get("mask_warmup_tasks", 0)),
        "mask_decay_tasks": int(args.get("mask_decay_tasks", 0)),
        "mask_min_prob": float(args.get("mask_min_prob", 0.0)),

        # --- Arm2-only shaping ---
        "arm2_cost_multiplier": float(args.get("arm2_cost_multiplier", 1.0)),
        "arm2_cost_offset": int(args.get("arm2_cost_offset", 0)),

        # If bids tie: prefer Arm2 over Arm1 (and optionally others)
        "tie_break_preference": list(args.get("tie_break_preference", [4, 3])),
    }

    logger.info(
        f"ENV CONFIG (train): use_action_mask={env_config_train.get('use_action_mask')} "
        f"mask_curriculum={env_config_train.get('mask_curriculum')} "
        f"warmup={env_config_train.get('mask_warmup_tasks')} "
        f"decay={env_config_train.get('mask_decay_tasks')} "
        f"min_prob={env_config_train.get('mask_min_prob')} "
        f"| arm2_mult={env_config_train.get('arm2_cost_multiplier')} "
        f"arm2_off={env_config_train.get('arm2_cost_offset')}"
    )

    # Only apply category-weighted config if requested.
    if (
        bool(env_config_train.get("dataset_mode", False))
        and bool(env_config_train.get("train_mode", True))
        and str(env_config_train.get("dataset_sampling", "permute")) == "category_weighted"
    ):
        env_config_train.update(
            {
                "category_mix_uniform": float(args.get("category_mix_uniform", 0.30)),
                "category_weight_nonspecialist": float(args.get("category_weight_nonspecialist", 0.20)),
                "category_weight_floor": float(args.get("category_weight_floor", 0.05)),
                "print_category_sampling_cache": bool(args.get("print_category_sampling_cache", False)),
            }
        )

    # Discover dims directly from the env instance (source of truth).
    tmp_env = HeterogeneousTeamEnv(env_config_train)
    local_obs_dim = int(tmp_env.local_obs_dim)
    state_dim = int(tmp_env.global_state_dim)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    num_actions = int(act_space.n)

    try:
        num_agents = int(getattr(tmp_env, "num_agents"))
    except Exception:
        num_agents = int(len(getattr(tmp_env, "possible_agents", [])) or 6)

    logger.info(
        f"Discovered dims from env: local_obs_dim={local_obs_dim}, "
        f"state_dim={state_dim}, num_actions={num_actions}, num_agents={num_agents}"
    )
    logger.info(f"Discovered obs_space from env: {obs_space}")

    policies = {f"policy_{i}": (None, obs_space, act_space, {}) for i in range(num_agents)}
    policies_to_train = list(policies.keys())

    cfg = (
        PPOConfig()
        .framework("torch")
        .environment(env=env_name, env_config=env_config_train, disable_env_checking=True)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=per_agent_policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .resources(num_gpus=float(args.get("num_gpus", 0.0)))
        .callbacks(EvalOnlyDebugCallbacks)
        .debugging(log_level="INFO")
    )

    desired_training = dict(
        gamma=float(args.get("gamma", 0.99)),
        lr=float(args.get("lr", 3e-5)),
        lambda_=float(args.get("lambda_", 0.95)),
        train_batch_size=int(args.get("train_batch_size", 4096)),
        minibatch_size=int(args.get("minibatch_size", 512)),
        num_epochs=int(args.get("num_epochs", 10)),
        grad_clip=float(args.get("grad_clip", 0.5)),
        clip_param=float(args.get("clip_param", 0.1)),
        vf_clip_param=float(args.get("vf_clip_param", 10.0)),
        vf_loss_coeff=float(args.get("vf_loss_coeff", 1.0)),
        entropy_coeff=float(args.get("entropy_coeff", 0.01)),
    )

    if args.get("kl_target", None) is not None:
        desired_training["kl_target"] = float(args["kl_target"])
    if args.get("entropy_coeff_schedule", None) is not None:
        desired_training["entropy_coeff_schedule"] = args["entropy_coeff_schedule"]

    cfg = apply_training_compat(cfg, desired_training)

    cfg = apply_env_runners_compat(
        cfg,
        num_runners=int(args.get("num_env_runners", 4)),
        num_envs_per_runner=int(args.get("num_envs_per_env_runner", 1)),
    )

    eval_env_config = dict(env_config_train)
    eval_env_config["train_mode"] = False
    eval_env_config["mask_curriculum"] = False
    eval_env_config["resume_print_every_n"] = int(args.get("eval_resume_print_every_n", 0))
    eval_env_config["show_detailed_episodes"] = True
    eval_env_config["eval_mask_print_every_n"] = int(args.get("eval_mask_print_every_n", 50))

    eval_env_config["use_action_mask"] = bool(args.get("eval_use_action_mask", False))   # disable masking effect in eval (env returns all-ones mask)
    eval_env_config["force_eligible_to_bid"] = False  # optional: keep behavior consistent with “no mask”
    eval_env_config["debug_action_mask"] = False      # optional: reduce noise
    eval_env_config["eval_mask_print_every_n"] = 0    # optional: don’t print mask debug
    eval_env_config["tie_break_preference"] = list(args.get("tie_break_preference", [4, 3]))  # eval_env_config["tie_break_mode"] = str(args.get("tie_break_mode", "priority_then_slack_then_id"))

    if args.get("eval_resume_print_every_n", None) is not None:
        eval_env_config["resume_print_every_n"] = int(args.get("eval_resume_print_every_n", 0))

    try:
        cfg = cfg.evaluation(
            evaluation_interval=int(args.get("evaluation_interval", 10)),
            evaluation_duration=int(args.get("evaluation_duration", 5)),
            evaluation_duration_unit="episodes",
            evaluation_config={"env_config": eval_env_config, "explore": False},
        )
    except TypeError:
        cfg = cfg.evaluation(
            evaluation_interval=int(args.get("evaluation_interval", 10)),
            evaluation_duration=int(args.get("evaluation_duration", 5)),
            evaluation_config={"env_config": eval_env_config, "explore": False},
        )

    cfg.model = {
        "custom_model": "mappo_cc_model",
        "custom_model_config": {
            "actor_hiddens": [128, 128],
            "critic_hiddens": [256, 256],
            "local_obs_dim": local_obs_dim,
            "state_dim": state_dim,
            "debug_action_mask": bool(args.get("debug_action_mask", False)),
        },
        "_disable_preprocessor_api": True,
    }

    return cfg.to_dict()


def rllib_action_mask_sanity_check():
    logger.info("Running RLlib Action Mask Sanity Check")
    ray.init(ignore_reinit_error=True)

    register_env("heterogeneous_team_env", lambda cfg: HeterogeneousTeamEnv(cfg))

    cfg_dict = build_single_trial_config(
        {
            "max_steps": 5,
            "num_env_runners": 1,
            "num_envs_per_env_runner": 1,
            "num_gpus": 0.0,

            "gamma": 0.99,
            "lambda_": 0.95,
            "lr": 3e-5,
            "entropy_coeff": 0.01,
            "clip_param": 0.1,
            "train_batch_size": 1024,
            "minibatch_size": 256,
            "num_epochs": 1,

            "evaluation_interval": 0,
            "evaluation_duration": 1,

            "dataset_mode": True,
            "dataset_sampling": "category_weighted",
            "dataset_path": os.path.abspath("./datasets/tasks_v1.npz"),

            "eligible_bid_bonus": 0.03,
            "bid_shaping_coeff": 0.10,
            "missed_assignment_penalty": -0.05,
            "winner_reward": 0.50,
            "unqualified_bid_penalty": -0.30,
            "lost_auction_penalty": 0.0,
            "eligible_abstain_penalty": -0.05,

            "use_action_mask": True,
            "force_eligible_to_bid": True,
            "debug_action_mask": True,

            "eval_mask_print_every_n": 0,
            "human_cost_offset": 2,

            "specialist_agent_ids": [0, 2, 3, 4],
            "specialist_cost_multiplier": 1.0,
            "specialist_cost_offset": -1,

            "resume_print_every_n": 50,
            "human_winner_reward": 0.5,

            "tie_break_preference": [4, 3],
        }
    )

    algo = PPO(config=cfg_dict, env=cfg_dict["env"])

    pol = algo.get_policy("policy_0")
    print("\n[SanityCheck] POLICY OBS SPACE (policy_0):", pol.observation_space)

    env = HeterogeneousTeamEnv(cfg_dict.get("env_config", {}))
    obs, _ = env.reset(seed=0)
    o = obs[0]

    def _extract_action(ret):
        if isinstance(ret, tuple) and len(ret) >= 1:
            return ret[0]
        return ret

    def compute_with_mask(allowed_action: int) -> int:
        o2 = copy.deepcopy(o)
        mask = np.zeros((env.action_space.n,), dtype=np.float32)
        mask[int(allowed_action)] = 1.0
        o2["action_mask"] = mask
        act = pol.compute_single_action(o2, explore=False)
        return int(_extract_action(act))

    a0 = compute_with_mask(0)
    a10 = compute_with_mask(env.action_space.n - 1)

    print("\n[SanityCheck] Forced mask -> only 0 allowed   => action:", a0)
    print("[SanityCheck] Forced mask -> only 10 allowed  => action:", a10)

    if a0 != 0 or a10 != (env.action_space.n - 1):
        print("\n❌ ACTION MASK CHECK FAILED")
        print("This means RLlib/model is NOT enforcing the mask the way we expect.")
    else:
        print("\n✅ ACTION MASK CHECK PASSED")

    algo.stop()
    ray.shutdown()
    logger.info("Sanity check done")


def train_single_experiment():
    logger.info("Initializing Ray")
    ray.init(ignore_reinit_error=True)

    ds = os.path.abspath("./datasets/tasks_v1.npz")
    if not os.path.exists(ds):
        raise FileNotFoundError(f"Dataset not found: {ds}. Generate it first with build_dataset()+save_dataset_npz().")

    register_env("heterogeneous_team_env", lambda cfg: HeterogeneousTeamEnv(cfg))

    train_metric = "env_runners/episode_return_mean"
    eval_metric = "evaluation/env_runners/episode_return_mean"

    algo_config = build_single_trial_config(
        {
            "max_steps": 50,
            "num_env_runners": 4,
            "num_envs_per_env_runner": 1,
            "num_gpus": 0.0,

            "gamma": 0.99,
            "lambda_": 0.95,
            "lr": 3e-5,
            "entropy_coeff": 0.08,
            "entropy_coeff_schedule": None,
            "kl_target": 0.015,
            "clip_param": 0.2,
            "train_batch_size": 4096,
            "minibatch_size": 512,
            "num_epochs": 10,

            "evaluation_interval": 5,
            "evaluation_duration": 5,

            "dataset_mode": True,
            "dataset_sampling": "category_weighted",
            "dataset_path": os.path.abspath("./datasets/tasks_v1.npz"),

            "eligible_bid_bonus": 0.10,
            "bid_shaping_coeff": 1.2,
            "missed_assignment_penalty": -0.50,
            "winner_reward": 0.2,

            "unqualified_bid_penalty": -0.55,
            "lost_auction_penalty": -0.15,
            "eligible_abstain_penalty": -0.20,

            "human_cost_offset": 17,

            "use_action_mask": True,
            "mask_curriculum": True,
            "mask_warmup_tasks": 25000,
            "mask_decay_tasks": 150000,
            "mask_min_prob": 0.0,
            "force_eligible_to_bid": False,
            "debug_action_mask": False,
            "debug_specialist_cost": False,
            "eval_use_action_mask": False,

            "eval_mask_print_every_n": 5,

            "specialist_agent_ids": [0, 2, 3, 4],
            "specialist_cost_multiplier": 0.75,
            "specialist_cost_offset": -2,
            "arm2_cost_multiplier": 0.65,
            "arm2_cost_offset": 0,

            "human_winner_reward": 0.5,
            "resume_print_every_n": 20000,
            "eval_resume_print_every_n": 250,

            "tie_break_preference": [4, 3],
        }
    )

    storage_path = os.path.abspath("./results_mappo_bidding")
    os.makedirs(storage_path, exist_ok=True)

    stop = {"training_iteration": int(os.environ.get("STOP_ITERS", "250"))}

    checkpoint_kwargs = {}
    if CheckpointConfig is not None:
        checkpoint_kwargs["checkpoint_config"] = CheckpointConfig(
            checkpoint_frequency=int(os.environ.get("CKPT_FREQ", "10")),
            checkpoint_at_end=True,
            num_to_keep=3,
        )

    tuner = tune.Tuner(
        "PPO",
        param_space=algo_config,
        tune_config=tune.TuneConfig(
            metric=train_metric,
            mode="max",
            num_samples=1,
        ),
        run_config=RunConfig(
            name="mappo_style_ppo_single",
            storage_path=storage_path,
            stop=stop,
            verbose=1,
            **checkpoint_kwargs,
        ),
    )

    logger.info("Starting MAPPO-style PPO training (centralized critic, train/eval templates split)")
    results = tuner.fit()

    # -----------------------
    # Print global totals once at the end (across all workers)
    # -----------------------
    try:
        trial_res_for_totals = results.get_best_result(metric=train_metric, mode="max")
        df = getattr(trial_res_for_totals, "metrics_dataframe", None)

        train_tasks_total = _metric_total_from_df(
            df,
            lifetime_col="num_env_steps_sampled_lifetime",
            this_iter_col="num_env_steps_sampled_this_iter",
        )

        eval_tasks_total = _metric_total_from_df(
            df,
            lifetime_col="evaluation/num_env_steps_sampled_lifetime",
            this_iter_col="evaluation/num_env_steps_sampled_this_iter",
        )

        print("\n" + "=" * 90)
        print("[GLOBAL TASK TOTALS] (across all workers)")
        print(f"  train_tasks_total = {train_tasks_total}")
        if eval_tasks_total > 0:
            print(f"  eval_tasks_total  = {eval_tasks_total}")
        else:
            print("  eval_tasks_total  = (not reported as steps; depends on Ray/RLlib version)")
        print("=" * 90 + "\n")
    except Exception as e:
        print(f"[GLOBAL TASK TOTALS] Could not extract totals from metrics_dataframe: {repr(e)}")

    # -----------------------
    # Print per-agent resume tables (TRAIN + EVAL if present)
    # -----------------------
    agent_names = {
        0: "Mobile Robot",
        1: "Mobile Manipulator",
        2: "Legged Robot",
        3: "Robotic Arm1",
        4: "Robotic Arm2",
        5: "Human",
    }

    try:
        trial_res_train, _ = _safe_get_best_result(results, metric=train_metric, mode="max")
        df_train = getattr(trial_res_train, "metrics_dataframe", None)
        _print_resume_table_from_df(df_train, "bidding_resume/train", agent_names)

        trial_res_eval, used_eval_metric = _safe_get_best_result(
            results, metric=eval_metric, mode="max", fallback_metric=train_metric
        )
        df_eval = getattr(trial_res_eval, "metrics_dataframe", None)

        if used_eval_metric == eval_metric:
            _print_resume_table_from_df(df_eval, "evaluation/bidding_resume/eval", agent_names)
        else:
            print("[BIDDING RESUME TABLE] Eval disabled/absent -> skipping eval resume table.\n")
    except Exception as e:
        print(f"[BIDDING RESUME TABLE] Failed to print tables: {repr(e)}")

    best, used_best_metric = _safe_get_best_result(
        results, metric=eval_metric, mode="max", fallback_metric=train_metric
    )

    best_eval = best.metrics.get(eval_metric)
    if best_eval is None:
        best_eval = best.metrics.get("evaluation", {}).get("env_runners", {}).get("episode_return_mean")

    best_train = best.metrics.get(train_metric)
    if best_train is None:
        best_train = best.metrics.get("env_runners", {}).get("episode_return_mean")

    logger.info(f"Best result (by {used_best_metric}) -> eval_metric_value={best_eval}")
    logger.info(f"Best result train metric {train_metric}: {best_train}")
    logger.info(f"Best result path: {best.path}")

    export_dir = os.path.abspath("./exported_checkpoints/best_checkpoint")
    os.makedirs(export_dir, exist_ok=True)

    try:
        ckpt = getattr(best, "checkpoint", None)
        if ckpt is not None:
            out_dir = ckpt.to_directory(export_dir)
            logger.info(f"Exported BEST checkpoint to: {out_dir}")
            print(f"\n✅ Load this folder in task_allocation_interface.py:\n{out_dir}\n")
        else:
            logger.warning("No best.checkpoint object found; use a checkpoint_* directory inside the best trial folder.")
            print(f"\nℹ️ Open this folder and pick a checkpoint_*/ directory for task_allocation_interface.py:\n{best.path}\n")
    except Exception as e:
        logger.exception("Failed to export best checkpoint")
        print(f"\n⚠️ Could not export best checkpoint automatically: {repr(e)}")
        print(f"Try loading a checkpoint_*/ directory inside:\n{best.path}\n")

    ray.shutdown()
    logger.info("Ray shutdown")


def quick_env_test(
    episodes: int = 2,
    max_steps: int = 20,
    train_mode: bool = True,
    dataset_path: Optional[str] = None,
    dataset_mode: bool = True,
    dataset_sampling: str = "permute",
    eligible_bid_bonus: float = 0.0,
    bid_shaping_coeff: float = 0.50,
    missed_assignment_penalty: float = -0.50,
    winner_reward: float = 0.10,
    unqualified_bid_penalty: float = -1.00,
    lost_auction_penalty: float = -0.05,
    eligible_abstain_penalty: float = -0.10,
    use_action_mask: bool = False,
    force_eligible_to_bid: bool = False,
    human_cost_offset: int = 2,
):
    env = HeterogeneousTeamEnv(
        {
            "max_steps": max_steps,
            "train_mode": train_mode,
            "show_detailed_episodes": True,
            "dataset_mode": dataset_mode,
            "dataset_sampling": dataset_sampling,
            "dataset_path": dataset_path,
            "eligible_bid_bonus": eligible_bid_bonus,
            "bid_shaping_coeff": bid_shaping_coeff,
            "missed_assignment_penalty": missed_assignment_penalty,
            "winner_reward": winner_reward,
            "unqualified_bid_penalty": unqualified_bid_penalty,
            "lost_auction_penalty": lost_auction_penalty,
            "eligible_abstain_penalty": eligible_abstain_penalty,
            "use_action_mask": use_action_mask,
            "force_eligible_to_bid": force_eligible_to_bid,
            "human_cost_offset": human_cost_offset,
            "tie_break_preference": [4, 3],
        }
    )

    for ep in range(episodes):
        obs, _info = env.reset()
        total = 0.0
        steps = 0

        for _ in range(max_steps):
            actions = {aid: env.action_space.sample() for aid in obs.keys()}
            obs, rew, terminateds, truncateds, _infos = env.step(actions)

            total += float(np.mean(list(rew.values())))
            steps += 1

            if terminateds.get("__all__", False) or truncateds.get("__all__", False):
                break

        mode = "TRAIN" if train_mode else "EVAL"
        print(f"[EnvTest:{mode}] episode={ep} avg_reward_per_step={total/max(1,steps):.3f} steps={steps}")


def forced_bid_debug(
    forced_bid: int = 5,
    max_steps: int = 1,
    train_mode: bool = False,
    dataset_path: Optional[str] = None,
):
    env = HeterogeneousTeamEnv(
        {
            "max_steps": max_steps,
            "train_mode": train_mode,
            "show_detailed_episodes": True,
            "dataset_mode": True,
            "dataset_sampling": "permute",
            "dataset_path": dataset_path,
            "eligible_bid_bonus": 0.03,
            "bid_shaping_coeff": 0.10,
            "missed_assignment_penalty": -0.05,
            "winner_reward": 0.50,
            "unqualified_bid_penalty": -0.30,
            "lost_auction_penalty": 0.0,
            "eligible_abstain_penalty": -0.05,
            "use_action_mask": True,
            "force_eligible_to_bid": True,
            "human_cost_offset": 2,
            "tie_break_preference": [4, 3],
        }
    )

    obs, _ = env.reset()

    actions = {aid: int(forced_bid) for aid in obs.keys()}
    _obs2, rew, terminateds, truncateds, _infos = env.step(actions)

    print("\n" + "=" * 120)
    print(f"[FORCED BID DEBUG] forced_bid={forced_bid} train_mode={train_mode}")
    print("Per-agent reward (after forcing bid):")
    for aid, r in rew.items():
        print(f"  {aid}: {r:.3f}")

    tracker = getattr(env, "bidding_tracker", None)
    if tracker is not None and getattr(tracker, "current_step_data", None):
        tracker.print_step_summary(tracker.current_step_data[-1])
    else:
        print("No tracker data. Ensure show_detailed_episodes=True in env_config.")
    print("=" * 120 + "\n")


def print_menu():
    print("\n🤖 Multi-Agent Task Bidding (MAPPO-style PPO) CLI")
    print("============================================================")
    print("1. Full Training (Single Trial, PPO with Tune)")
    print("2. Quick Environment Test (TRAIN templates)")
    print("3. Quick Environment Test (EVAL templates)")
    print("4. Forced Bid Debug (1 step, bid=5 for all agents)")
    print("5. RLlib Action Mask Sanity Check (forces masks, verifies chosen action)")
    print("0. Exit\n")


def main():
    logger.info("Script started")

    while True:
        print_menu()
        try:
            choice = input("Select an option (0-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if choice == "1":
            try:
                train_single_experiment()
            except Exception:
                logger.exception("Error during training")
        elif choice == "2":
            quick_env_test(
                train_mode=True,
                dataset_path=os.path.abspath("./datasets/tasks_v1.npz"),
                dataset_mode=True,
                dataset_sampling="permute",
                eligible_bid_bonus=0.0,
                bid_shaping_coeff=0.50,
                missed_assignment_penalty=-0.50,
                winner_reward=0.10,
                unqualified_bid_penalty=-1.00,
                lost_auction_penalty=-0.05,
                eligible_abstain_penalty=-0.10,
                use_action_mask=False,
                force_eligible_to_bid=False,
                human_cost_offset=2,
            )
        elif choice == "3":
            quick_env_test(
                train_mode=False,
                dataset_path=os.path.abspath("./datasets/tasks_v1.npz"),
                dataset_mode=True,
                dataset_sampling="permute",
                eligible_bid_bonus=0.0,
                bid_shaping_coeff=0.50,
                missed_assignment_penalty=-0.50,
                winner_reward=0.10,
                unqualified_bid_penalty=-1.00,
                lost_auction_penalty=-0.05,
                eligible_abstain_penalty=-0.10,
                use_action_mask=False,
                force_eligible_to_bid=False,
                human_cost_offset=2,
            )
        elif choice == "4":
            forced_bid_debug(
                forced_bid=5,
                max_steps=1,
                train_mode=False,
                dataset_path=os.path.abspath("./datasets/tasks_v1.npz"),
            )
        elif choice == "5":
            try:
                rllib_action_mask_sanity_check()
            except Exception:
                logger.exception("Error during RLlib action mask sanity check")
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, 3, 4, 5, or 0.")


if __name__ == "__main__":
    main()