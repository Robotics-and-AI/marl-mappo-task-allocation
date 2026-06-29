"""
evaluate_checkpoint_on_evalset.py

Evaluate an RLlib multi-agent PPO/MAPPO checkpoint on a fixed eval dataset (.npz),
WITHOUT GUI, with better parity.

Implementation notes:
- Uses Algorithm.compute_single_action(..., policy_id=...) instead of policy.compute_single_action(...)
  (this matches RLlib's evaluation pipeline much better).
- Uses env.step(action_dict) as the canonical source of rewards/outcomes (history/step state advances).
- Injects eval tasks sequentially (task 0..N-1 exactly once), split into episodes of length --episode_len.
- Avoids calling env._calculate_rewards() except as a last-resort fallback if env.step() infos lack fields.

How to run
----------

python scripts/evaluate_checkpoint_on_evalset.py \
  --checkpoint exported_checkpoints/best_checkpoint \
  --eval_npz datasets/tasks_v1_eval_only.npz \
  --out_csv ./results/eval_rows.csv \
  --out_json ./results/eval_summary.json \
  --episode_len 50



TIP (parity): If your env internally depends on extra dataset keys (beyond eval_*),
pass the ORIGINAL dataset instead:
  --eval_npz datasets/tasks_v1.npz
(this script still reads eval_* keys from it)
"""
# Allow running this script from the repository root as:
#   python scripts/<script_name>.py
# Python otherwise puts scripts/ on sys.path, not the project root.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



import argparse
import csv
import json
import os
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import ray

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from heterogeneous_team_env import HeterogeneousTeamEnv
from train import MAPPOCentralCriticModel


# -----------------------------------------------------------------------------
# RLlib registries (needed so Algorithm.from_checkpoint can rebuild)
# -----------------------------------------------------------------------------
def ensure_rllib_registries() -> None:
    register_env("heterogeneous_team_env", lambda cfg: HeterogeneousTeamEnv(cfg))
    ModelCatalog.register_custom_model("mappo_cc_model", MAPPOCentralCriticModel)


# -----------------------------------------------------------------------------
# Checkpoint resolution
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

    # Case 1: path is already a checkpoint dir
    if os.path.isdir(ckpt) and _looks_like_rllib_checkpoint_dir(ckpt):
        return ckpt

    # Case 2: path contains checkpoint_* subdirs
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
                print(f"[CKPT] Provided path contains checkpoints; using latest: {pick}")
                return pick

    raise FileNotFoundError(
        f"Could not resolve a valid RLlib checkpoint directory from: {checkpoint_path}\n"
        f"Tip: pass a checkpoint_*/ directory, or a folder containing checkpoint_*/ subfolders."
    )


def load_algorithm_from_checkpoint(checkpoint_path: str):
    ensure_rllib_registries()
    ckpt_dir = resolve_checkpoint_dir(checkpoint_path)
    print(f"[CKPT] Using checkpoint dir: {ckpt_dir}")

    from ray.rllib.algorithms.algorithm import Algorithm
    algo = Algorithm.from_checkpoint(ckpt_dir)
    return algo


# -----------------------------------------------------------------------------
# Dataset loader (reads eval_* arrays)
# -----------------------------------------------------------------------------
def load_eval_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)

    required = ["eval_instructions", "eval_categories", "eval_requirements"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}. Keys found: {list(data.files)}")

    instr = data["eval_instructions"]
    cat = data["eval_categories"]
    req = data["eval_requirements"].astype(np.float32, copy=False)

    if not (len(instr) == len(cat) == len(req)):
        raise ValueError(
            f"Eval arrays misaligned: len(instr)={len(instr)}, len(cat)={len(cat)}, len(req)={len(req)}"
        )
    if req.ndim != 2 or req.shape[1] != 3:
        raise ValueError(f"Expected eval_requirements shape (N,3), got {req.shape}")

    return instr, cat, req


def infer_req_scale(req: np.ndarray) -> str:
    mn = float(np.nanmin(req))
    mx = float(np.nanmax(req))
    if mn >= 0.95 and mx <= 4.05:
        return "raw_1to4"
    if mn >= -0.05 and mx <= 1.05:
        return "normalized_0to1"
    return "raw_1to4"


# -----------------------------------------------------------------------------
# Env config parity (from checkpoint)
# -----------------------------------------------------------------------------
def build_eval_env_config_from_checkpoint(
    algo,
    *,
    episode_len: int,
    use_action_mask: Optional[bool],
    force_eligible_to_bid: Optional[bool],
) -> Dict[str, Any]:
    cfg = getattr(algo, "config", None) or {}

    eval_cfg = dict(cfg.get("evaluation_config", {}) or {})
    base_env_cfg = dict(cfg.get("env_config", {}) or {})
    eval_env_cfg = dict(eval_cfg.get("env_config", {}) or {})

    merged = dict(base_env_cfg)
    merged.update(eval_env_cfg)

    # We inject tasks manually from eval_npz => prevent env from sampling dataset internally.
    merged["dataset_mode"] = False

    merged["train_mode"] = False
    merged["show_detailed_episodes"] = False
    merged["max_steps"] = int(episode_len)

    if use_action_mask is not None:
        merged["use_action_mask"] = bool(use_action_mask)
    if force_eligible_to_bid is not None:
        merged["force_eligible_to_bid"] = bool(force_eligible_to_bid)

    return merged


# -----------------------------------------------------------------------------
# Task injection
# -----------------------------------------------------------------------------
def inject_task(
    env: HeterogeneousTeamEnv,
    instruction: str,
    category: str,
    req_vec: np.ndarray,
    task_id: int,
    *,
    req_scale: str,
    requirements_mode: str,  # auto/raw/normalized
) -> None:
    req_vec = np.asarray(req_vec, dtype=np.float32).reshape(3,)

    # Build both raw + normalized
    if req_scale == "raw_1to4":
        req_raw = req_vec
        try:
            req_norm = env._normalize(req_raw).astype(np.float32, copy=False)
        except Exception:
            req_norm = req_raw
    else:
        req_norm = req_vec
        req_raw = req_vec

    mode = requirements_mode.lower().strip()
    if mode == "auto":
        env_mode = getattr(env, "requirements_mode", None)
        if isinstance(env_mode, str) and env_mode.lower() in ("raw", "normalized"):
            mode = env_mode.lower()
        else:
            # safest default for policy inputs (you can override via CLI)
            mode = "normalized"

    req_primary = req_raw if mode == "raw" else req_norm

    env.current_task = {
        "id": int(task_id),
        "instruction": str(instruction),
        "category": str(category),
        "requirements": np.asarray(req_primary, dtype=np.float32).reshape(3,),
        "requirements_raw": np.asarray(req_raw, dtype=np.float32).reshape(3,),
        "requirements_norm": np.asarray(req_norm, dtype=np.float32).reshape(3,),
    }


# -----------------------------------------------------------------------------
# Agent name lookup (agent_types may be dict OR list)
# -----------------------------------------------------------------------------
def get_agent_name(env: HeterogeneousTeamEnv, aid: int) -> str:
    agent_type_names = getattr(env, "agent_type_names", {}) or {}
    agent_types = getattr(env, "agent_types", None)

    at = None
    if isinstance(agent_types, dict):
        at = agent_types.get(aid, None)
    elif isinstance(agent_types, (list, tuple, np.ndarray)):
        if 0 <= int(aid) < len(agent_types):
            at = agent_types[int(aid)]

    if at is None:
        return f"Agent{aid}"
    return agent_type_names.get(at, str(at))


# -----------------------------------------------------------------------------
# Robust step unpacking (gym vs gymnasium)
# -----------------------------------------------------------------------------
def unpack_step(ret):
    # gymnasium: obs, rewards, terminateds, truncateds, infos
    if isinstance(ret, tuple) and len(ret) == 5:
        return ret
    # gym: obs, rewards, dones, infos
    if isinstance(ret, tuple) and len(ret) == 4:
        obs, rewards, dones, infos = ret
        terminateds = dones
        truncateds = {k: False for k in dones} if isinstance(dones, dict) else {"__all__": False}
        return obs, rewards, terminateds, truncateds, infos
    raise RuntimeError(f"Unexpected env.step return shape: {type(ret)} / {getattr(ret, '__len__', lambda: 'NA')()}")


# -----------------------------------------------------------------------------
# Outcome extraction (prefer infos from env.step; fallback to _calculate_rewards once)
# -----------------------------------------------------------------------------
def extract_outcomes_from_infos(
    env: HeterogeneousTeamEnv,
    infos: Any,
    bids: Dict[int, int],
    *,
    warned: Dict[str, bool],
) -> Dict[int, Dict[str, Any]]:
    outcomes: Dict[int, Dict[str, Any]] = {}

    def normalize_outcome_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        # Some envs put fields at top-level, some nested under "outcome"
        if "outcome" in d and isinstance(d["outcome"], dict):
            return d["outcome"]
        return d

    if isinstance(infos, dict):
        for aid, maybe in infos.items():
            if isinstance(aid, (int, np.integer)) and isinstance(maybe, dict):
                outcomes[int(aid)] = normalize_outcome_dict(maybe)

    # If we still don't have eligibility keys, fallback once (some envs return empty infos)
    need_fallback = False
    for aid in bids.keys():
        od = outcomes.get(int(aid), {})
        if "eligible" not in od or "won" not in od:
            need_fallback = True
            break

    if need_fallback:
        if not warned.get("fallback_calculate_rewards", False):
            print("[WARN] env.step() infos missing eligible/won fields; falling back to env._calculate_rewards() (warn once).")
            warned["fallback_calculate_rewards"] = True
        try:
            _, outs = env._calculate_rewards(bids, return_outcomes=True, raw_actions=bids)
            for aid in bids.keys():
                if int(aid) in outs:
                    outcomes[int(aid)] = dict(outs[int(aid)])
        except Exception as e:
            if not warned.get("fallback_failed", False):
                print(f"[WARN] fallback env._calculate_rewards() failed: {e} (warn once)")
                warned["fallback_failed"] = True

    return outcomes


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
def evaluate(
    *,
    checkpoint_path: str,
    eval_npz_path: str,
    out_csv: str,
    out_json: Optional[str],
    max_tasks: Optional[int],
    episode_len: int,
    use_action_mask: Optional[bool],
    force_eligible_to_bid: Optional[bool],
    requirements_mode: str,
    debug_first_k: int,
) -> None:
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    algo = load_algorithm_from_checkpoint(checkpoint_path)

    env_cfg = build_eval_env_config_from_checkpoint(
        algo,
        episode_len=episode_len,
        use_action_mask=use_action_mask,
        force_eligible_to_bid=force_eligible_to_bid,
    )
    env = HeterogeneousTeamEnv(env_cfg)

    # Load eval dataset
    instr, cat, req = load_eval_npz(eval_npz_path)
    n_total = len(instr)
    n = n_total if max_tasks is None else min(n_total, int(max_tasks))

    req_scale = infer_req_scale(req[: min(n, 256)])

    agent_ids = list(getattr(env, "_agent_ids", [0, 1, 2, 3, 4, 5]))

    print(f"[PARITY] explore: False")
    print(f"[PARITY] episode_len: {episode_len}")
    print(f"[PARITY] use_action_mask: {env_cfg.get('use_action_mask')}")
    print(f"[PARITY] force_eligible_to_bid: {env_cfg.get('force_eligible_to_bid')}")
    print(f"[PARITY] requirements_mode: {requirements_mode} (dataset_scale={req_scale})")
    print(f"[TASKS] evaluating {n}/{n_total} eval tasks sequentially")

    # Policy mapping (default: policy_{agent_id})
    def policy_id_for(aid: int) -> str:
        return f"policy_{int(aid)}"

    # Stats
    stats = {
        "tasks": 0,
        "tasks_no_eligible": 0,
        "tasks_all_abstain": 0,
        "tasks_no_eligible_bid": 0,
        "tasks_only_ineligible_bids": 0,
        "tasks_no_winner": 0,
        "winner_counts": {str(aid): 0 for aid in agent_ids},
        "eligible_counts": {str(aid): 0 for aid in agent_ids},
        "eligible_abstain_counts": {str(aid): 0 for aid in agent_ids},
        "ineligible_bid_counts": {str(aid): 0 for aid in agent_ids},
        "sum_bid_eligible": {str(aid): 0 for aid in agent_ids},
        "cnt_bid_eligible": {str(aid): 0 for aid in agent_ids},
        "sum_bid_ineligible": {str(aid): 0 for aid in agent_ids},
        "cnt_bid_ineligible": {str(aid): 0 for aid in agent_ids},
        "action_counts": {str(aid): {str(b): 0 for b in range(11)} for aid in agent_ids},
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    warned = {}

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_idx",
                "category",
                "instruction",
                "req_mob",
                "req_man",
                "req_pay",
                "winner_agent_id",
                "winning_bid",
                "agent_id",
                "agent_name",
                "eligible",
                "bid",
                "won",
                "payment",
                "profit",
                "slack_min",
                "reward",
            ],
        )
        writer.writeheader()

        # Episode loop (reset every episode_len tasks)
        for t in range(n):
            if t % episode_len == 0:
                # reset episode but DO NOT reset task sequence (we control that)
                try:
                    r = env.reset(seed=0)
                    if isinstance(r, tuple) and len(r) == 2:
                        _ = r[0]
                except Exception:
                    try:
                        env.reset()
                    except Exception:
                        pass
                try:
                    env.current_step = 0
                except Exception:
                    pass

            # Inject task t
            instruction = str(instr[t])
            category = str(cat[t])
            req_vec = np.asarray(req[t], dtype=np.float32).reshape(3,)

            inject_task(
                env,
                instruction,
                category,
                req_vec,
                task_id=t,
                req_scale=req_scale,
                requirements_mode=requirements_mode,
            )

            # Build obs from env AFTER injection
            obs = env._get_obs()

            # Compute actions using ALGORITHM API (important parity fix)
            bids: Dict[int, int] = {}
            action_dict: Dict[int, int] = {}
            for aid in agent_ids:
                pid = policy_id_for(aid)
                act = algo.compute_single_action(obs[aid], policy_id=pid, explore=False)
                # compute_single_action may return tuple in older Ray
                if isinstance(act, tuple):
                    act = act[0]
                bid = int(np.asarray(act).reshape(-1)[0])
                bid = int(np.clip(bid, 0, 10))
                bids[int(aid)] = bid
                action_dict[int(aid)] = bid
                stats["action_counts"][str(aid)][str(bid)] += 1

            # Step env (canonical)
            ret = env.step(action_dict)
            next_obs, rewards, terminateds, truncateds, infos = unpack_step(ret)

            if not isinstance(rewards, dict):
                # Some envs return scalar; convert
                rewards = {int(aid): float(rewards) for aid in agent_ids}

            outcomes = extract_outcomes_from_infos(env, infos, bids, warned=warned)

            # Determine winner
            winner_aid = None
            winning_bid = None
            for aid in agent_ids:
                od = outcomes.get(int(aid), {})
                if bool(od.get("won", False)):
                    winner_aid = int(aid)
                    winning_bid = int(bids[int(aid)])
                    stats["winner_counts"][str(aid)] += 1
                    break

            # Task-level counters
            stats["tasks"] += 1
            any_eligible = any(bool(outcomes.get(int(aid), {}).get("eligible", False)) for aid in agent_ids)
            any_bid = any(int(bids[int(aid)]) > 0 for aid in agent_ids)
            any_eligible_bid = any(
                bool(outcomes.get(int(aid), {}).get("eligible", False)) and int(bids[int(aid)]) > 0
                for aid in agent_ids
            )

            if not any_eligible:
                stats["tasks_no_eligible"] += 1
            if not any_bid:
                stats["tasks_all_abstain"] += 1
            if not any_eligible_bid:
                stats["tasks_no_eligible_bid"] += 1
            if any_bid and (not any_eligible_bid):
                stats["tasks_only_ineligible_bids"] += 1
            if winner_aid is None:
                stats["tasks_no_winner"] += 1

            if debug_first_k > 0 and t < debug_first_k:
                print(f"\n[DEBUG task={t}] cat={category} req={req_vec.tolist()} winner={winner_aid} win_bid={winning_bid}")
                for aid in agent_ids:
                    od = outcomes.get(int(aid), {})
                    am = obs[aid].get("action_mask", None)
                    am_sum = int(np.sum(am)) if isinstance(am, np.ndarray) else None
                    print(
                        f"  aid={aid} eligible={int(bool(od.get('eligible', False)))} "
                        f"bid={bids[int(aid)]} won={int(bool(od.get('won', False)))} "
                        f"action_mask_sum={am_sum}"
                    )

            # Per-agent counters + CSV rows
            for aid in agent_ids:
                od = outcomes.get(int(aid), {})
                eligible = bool(od.get("eligible", False))
                won = bool(od.get("won", False))
                bid = int(bids[int(aid)])

                if eligible:
                    stats["eligible_counts"][str(aid)] += 1
                    if bid == 0:
                        stats["eligible_abstain_counts"][str(aid)] += 1
                    else:
                        stats["sum_bid_eligible"][str(aid)] += bid
                        stats["cnt_bid_eligible"][str(aid)] += 1
                else:
                    if bid > 0:
                        stats["ineligible_bid_counts"][str(aid)] += 1
                        stats["sum_bid_ineligible"][str(aid)] += bid
                        stats["cnt_bid_ineligible"][str(aid)] += 1

                writer.writerow(
                    {
                        "task_idx": t,
                        "category": category,
                        "instruction": instruction,
                        "req_mob": float(req_vec[0]),
                        "req_man": float(req_vec[1]),
                        "req_pay": float(req_vec[2]),
                        "winner_agent_id": winner_aid,
                        "winning_bid": winning_bid,
                        "agent_id": int(aid),
                        "agent_name": get_agent_name(env, int(aid)),
                        "eligible": int(eligible),
                        "bid": bid,
                        "won": int(won),
                        "payment": od.get("payment", None),
                        "profit": od.get("profit", None),
                        "slack_min": od.get("slack_min", None),
                        "reward": float(rewards.get(int(aid), 0.0)),
                    }
                )

    # Print summary
    print(f"\nSaved per-agent rows to CSV: {out_csv}")
    print(f"Tasks evaluated: {stats['tasks']}")
    print(f"Tasks with no eligible agent: {stats['tasks_no_eligible']}")
    print(f"Tasks where ALL agents abstained (all bids=0): {stats['tasks_all_abstain']}")
    print(f"Tasks with NO eligible bid (no eligible agent bid>0): {stats['tasks_no_eligible_bid']}")
    print(f"Tasks with only ineligible bids: {stats['tasks_only_ineligible_bids']}")
    print(f"Tasks with no winner: {stats['tasks_no_winner']}\n")

    print("Per-agent summary (wins/eligible + behavior rates):")
    for aid in agent_ids:
        s = str(aid)
        elig = stats["eligible_counts"][s]
        wins = stats["winner_counts"][s]
        elig_abst = stats["eligible_abstain_counts"][s]
        inelig_bid = stats["ineligible_bid_counts"][s]

        win_rate_given_elig = wins / max(1, elig)
        elig_abst_rate = elig_abst / max(1, elig)

        mean_bid_elig = (
            stats["sum_bid_eligible"][s] / max(1, stats["cnt_bid_eligible"][s])
            if stats["cnt_bid_eligible"][s] > 0 else 0.0
        )
        mean_bid_inelig = (
            stats["sum_bid_ineligible"][s] / max(1, stats["cnt_bid_ineligible"][s])
            if stats["cnt_bid_ineligible"][s] > 0 else 0.0
        )

        name = get_agent_name(env, int(aid))
        print(
            f"  {aid} ({name}) | eligible={elig} | wins={wins} | win%|elig={win_rate_given_elig*100:.2f}% "
            f"| eligible_abstain%={elig_abst_rate*100:.2f}% | ineligible_bids={inelig_bid} "
            f"| mean_bid_elig={mean_bid_elig:.2f} | mean_bid_inelig={mean_bid_inelig:.2f}"
        )

    # Optional JSON
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(stats, jf, indent=2)
        print(f"\nSaved summary JSON: {out_json}")

    try:
        algo.stop()
    except Exception:
        pass
    ray.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="RLlib checkpoint dir OR folder containing checkpoint_*/")
    ap.add_argument("--eval_npz", required=True, help="NPZ containing eval_* arrays (eval-only NPZ is OK)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_json", default=None, help="Optional output JSON summary path")
    ap.add_argument("--max_tasks", type=int, default=None, help="Optional cap on number of tasks evaluated")

    ap.add_argument("--episode_len", type=int, default=50, help="Episode length parity (default 50)")
    ap.add_argument("--use_action_mask", type=int, default=-1, help="Override use_action_mask (0/1), -1=from checkpoint")
    ap.add_argument("--force_eligible_to_bid", type=int, default=-1, help="Override force_eligible_to_bid (0/1), -1=from checkpoint")
    ap.add_argument("--requirements_mode", choices=["auto", "raw", "normalized"], default="auto")

    ap.add_argument("--debug_first_k", type=int, default=0)

    args = ap.parse_args()

    use_action_mask = None if args.use_action_mask == -1 else bool(args.use_action_mask)
    force_eligible_to_bid = None if args.force_eligible_to_bid == -1 else bool(args.force_eligible_to_bid)

    evaluate(
        checkpoint_path=args.checkpoint,
        eval_npz_path=args.eval_npz,
        out_csv=args.out_csv,
        out_json=args.out_json,
        max_tasks=args.max_tasks,
        episode_len=args.episode_len,
        use_action_mask=use_action_mask,
        force_eligible_to_bid=force_eligible_to_bid,
        requirements_mode=args.requirements_mode,
        debug_first_k=args.debug_first_k,
    )


if __name__ == "__main__":
    main()