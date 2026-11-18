"""
IaC Drift Reconciler — Baseline Inference Script
=================================================

MANDATORY env vars:
  API_BASE_URL     LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME       Model id      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN         HF / API key
  ENV_BASE_URL     Running env server (default: http://localhost:8000)

STDOUT FORMAT  (evaluator parses these — do not change):
  [START] task=<task_id> env=IaCDriftReconciler model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

All debug output goes to stderr, including a per-episode [SUMMARY] line:
  [SUMMARY] task=<id> success=<bool> steps=<n> final_drift=<0.000> violations=<n>
            rejected_actions=<n> parse_errors=<n> llm_errors=<n>
"""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── env client & models ────────────────────────────────────────────────────
try:
    from IaCDriftReconciler.client import IaCDriftReconcilerEnv
    from IaCDriftReconciler.models import (
        IaCDriftReconcilerAction,
        IaCDriftReconcilerObservation,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import IaCDriftReconcilerEnv  # type: ignore[no-redef]
    from models import IaCDriftReconcilerAction, IaCDriftReconcilerObservation  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))

BENCHMARK = "IaCDriftReconciler"
TASK_IDS = ["easy", "medium", "hard"]

SCORE_DECIMALS = 3
MIN_TASK_SCORE = 10 ** (-SCORE_DECIMALS)  # 0.001
MAX_TASK_SCORE = 1.0 - MIN_TASK_SCORE  # 0.999

# Only the last few turns are sent back to the model.  Every user prompt
# already contains the full current drift list, so older turns add cost
# without adding information.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))

# Rewards with a fixed meaning in the environment.  The OpenEnv server strips
# observation.metadata before sending it, so over HTTP/WebSocket the reward
# is the only signal of *why* a step went wrong.
GUARDRAIL_REWARD = -1.0
REJECTED_ACTION_REWARD = -0.05

# ---------------------------------------------------------------------------
# OpenAI-compatible client (HF Router uses the same API surface)
# ---------------------------------------------------------------------------

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "placeholder")

# ---------------------------------------------------------------------------
# Stdout log helpers  (exact format the evaluator parses)
# ---------------------------------------------------------------------------


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _debug(msg: str) -> None:
    print(f"[debug] {msg}", file=sys.stderr, flush=True)


def _normalise_task_score(raw_score: float) -> float:
    """Return a finite score strictly inside (0, 1), rounded for log output."""
    if not math.isfinite(raw_score):
        return MIN_TASK_SCORE
    clamped = max(MIN_TASK_SCORE, min(raw_score, MAX_TASK_SCORE))
    return round(clamped, SCORE_DECIMALS)


# ---------------------------------------------------------------------------
# Compact action string for [STEP] log
# ---------------------------------------------------------------------------


def _action_str(action: IaCDriftReconcilerAction) -> str:
    """Return a short, evaluator-safe single-line representation of the action."""
    t = action.action_type
    if t == "update_resource":
        return f"update_resource({action.resource_name},{action.attribute},{action.new_value})"
    if t == "create_missing_resource":
        return f"create_missing_resource({action.resource_name})"
    if t == "delete_extra_resource":
        return f"delete_extra_resource({action.resource_name})"
    if t in ("attach_volume", "detach_volume"):
        return f"{t}({action.instance_name},{action.volume_name})"
    return "no_op()"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_ACTION_SCHEMA = textwrap.dedent("""\
    {"action_type": "update_resource",          "resource_name": "...", "attribute": "...", "new_value": "..."}
    {"action_type": "create_missing_resource",  "resource_type": "...", "resource_name": "...", "properties": "{...}"}
    {"action_type": "delete_extra_resource",    "resource_name": "..."}
    {"action_type": "attach_volume",            "instance_name": "...", "volume_name": "..."}
    {"action_type": "detach_volume",            "instance_name": "...", "volume_name": "..."}
    {"action_type": "no_op"}""")

SYSTEM_PROMPT = textwrap.dedent(f"""\
    You are an infrastructure reconciliation agent.
    Your goal is to fix drifted cloud resources by choosing ONE action per turn.

    RULES:
    1. Respond with a single valid JSON object — no markdown fences, no prose.
    2. Use only the action_type values below. Include only fields required for that type.
    3. new_value must always be a string (e.g. "true", "false", "t3.medium").
    4. NEVER violate a guardrail constraint — that ends the episode with reward = -1.0.
    5. For create_missing_resource, encode properties as a JSON string of the fields dict.
    6. If uncertain, emit: {{"action_type": "no_op"}}

    ACTION SCHEMAS (one per line):
    {_ACTION_SCHEMA}

    EXAMPLES:
    {{"action_type": "update_resource", "resource_name": "aws_instance.web", "attribute": "instance_type", "new_value": "t3.medium"}}
    {{"action_type": "delete_extra_resource", "resource_name": "aws_security_group.old"}}
    {{"action_type": "no_op"}}""")


def _build_user_prompt(
    obs: IaCDriftReconcilerObservation, step: int, last_error: Optional[str] = None
) -> str:
    drift_lines = (
        "\n".join(
            f"  {i + 1}. {d.resource_id}.{d.field}: "
            f"actual={d.actual_value!r} → desired={d.desired_value!r} [{d.severity}]"
            for i, d in enumerate(obs.drift_items)
        )
        or "  (none — fully reconciled)"
    )

    guardrail_lines = "\n".join(f"  - {r}" for r in obs.holy_grail_rules) or "  (none)"
    feedback = f"LAST ACTION FAILED: {last_error}\n" if last_error else ""

    return feedback + textwrap.dedent(f"""\
        STEP {step} | drift_score={obs.drift_score:.4f} | remaining_drifts={len(obs.drift_items)}

        DRIFT ITEMS TO FIX:
        {drift_lines}

        GUARDRAIL CONSTRAINTS (never violate):
        {guardrail_lines}

        Respond with exactly one JSON action object.""")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_llm(user_prompt: str, conversation_history: list) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": user_prompt},
    ]
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


class ParseError(ValueError):
    """The model's reply could not be turned into a valid action."""


def _parse_action(raw: str) -> IaCDriftReconcilerAction:
    """Parse the model's reply into an action.

    Accepts bare JSON, JSON inside ``` fences, or JSON surrounded by prose
    (we take the outermost {...}).  Raises ParseError otherwise.
    """
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ParseError(f"no JSON object in reply: {text[:60]!r}")
    try:
        data = json.loads(text[start : end + 1])
        return IaCDriftReconcilerAction(**data)
    except Exception as exc:  # JSONDecodeError, pydantic ValidationError, ...
        raise ParseError(str(exc).splitlines()[0]) from exc


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def _step_error(obs: IaCDriftReconcilerObservation, reward: float, done: bool) -> Optional[str]:
    """Classify what went wrong in a step, if anything.

    Uses observation.metadata when it is available (in-process env) and falls
    back to the reward, because the OpenEnv server drops metadata on the wire.
    """
    meta = obs.metadata or {}
    if meta.get("guardrail_violated") or (done and reward <= GUARDRAIL_REWARD):
        rule = meta.get("violated_rule")
        return f"guardrail_violated: {rule}" if rule else "guardrail_violated"
    if meta.get("last_action_valid") is False or math.isclose(reward, REJECTED_ACTION_REWARD):
        detail = meta.get("error")
        return f"action_rejected: {detail}" if detail else "action_rejected"
    return None


def run_episode(env, task_id: str) -> dict:
    """Run one full episode; returns summary dict."""
    result = env.reset(task_id=task_id)
    obs: IaCDriftReconcilerObservation = result.observation

    log_start(task=task_id, model=MODEL_NAME)

    rewards: List[float] = []
    conversation_history: list = []
    steps_taken = 0
    success = False
    score = MIN_TASK_SCORE
    counts = {"violations": 0, "rejected_actions": 0, "parse_errors": 0, "llm_errors": 0}
    last_error: Optional[str] = None

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            user_prompt = _build_user_prompt(obs, step, last_error)
            raw_text = ""
            action: IaCDriftReconcilerAction
            error_msg: Optional[str] = None

            try:
                raw_text = _call_llm(user_prompt, conversation_history)
            except Exception as exc:
                counts["llm_errors"] += 1
                error_msg = f"llm_error: {exc}"
                raw_text = "<llm error>"
            if error_msg is None:
                try:
                    action = _parse_action(raw_text)
                except ParseError as exc:
                    counts["parse_errors"] += 1
                    error_msg = f"parse_error: {exc}"
            if error_msg is not None:
                _debug(f"  step {step}: {error_msg} -> no_op fallback")
                action = IaCDriftReconcilerAction(action_type="no_op")
            else:
                _debug(f"  step {step}: {_action_str(action)}")

            # Keep only the last HISTORY_TURNS turns (bounded prompt size).
            conversation_history.append({"role": "user", "content": user_prompt})
            conversation_history.append({"role": "assistant", "content": raw_text})
            del conversation_history[: -2 * HISTORY_TURNS]

            # Step the environment
            result = env.step(action)
            obs = result.observation
            reward = float(
                result.reward
                if result.reward is not None
                else obs.metadata.get("reward", 0.0)
            )
            done = bool(result.done or obs.done)

            env_error = _step_error(obs, reward, done)
            if env_error:
                if env_error.startswith("guardrail_violated"):
                    counts["violations"] += 1
                else:
                    counts["rejected_actions"] += 1
                error_msg = env_error
            last_error = error_msg

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=_action_str(action),
                reward=reward,
                done=done,
                error=_one_line(error_msg),
            )

            if done:
                break

        # Success means fully reconciled without breaking a guardrail.
        # Score = 1 - remaining drift, normalised to strict (0, 1); an episode
        # that ended on a guardrail violation gets the minimum score.
        success = obs.drift_score == 0.0 and counts["violations"] == 0
        if counts["violations"]:
            score = MIN_TASK_SCORE
        else:
            score = _normalise_task_score(1.0 - float(obs.drift_score))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        _debug(
            f"[SUMMARY] task={task_id} success={str(success).lower()} "
            f"steps={steps_taken} final_drift={float(obs.drift_score):.3f} "
            + " ".join(f"{k}={v}" for k, v in counts.items())
        )

    return {
        "task_id": task_id,
        "score": score,
        "steps_taken": steps_taken,
        "success": success,
        "final_drift": float(obs.drift_score),
        **counts,
    }


def _one_line(msg: Optional[str]) -> Optional[str]:
    """Keep [STEP] lines single-line and short for the evaluator."""
    if not msg:
        return None
    return " ".join(msg.split())[:160]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _debug(f"API_BASE_URL = {API_BASE_URL}")
    _debug(f"MODEL_NAME   = {MODEL_NAME}")
    _debug(f"ENV_BASE_URL = {ENV_BASE_URL}")
    if not HF_TOKEN:
        _debug("WARNING: HF_TOKEN not set — requests will likely fail with 401.")

    results = []
    env = IaCDriftReconcilerEnv(base_url=ENV_BASE_URL)

    with env.sync() as sync_env:
        for task_id in TASK_IDS:
            try:
                summary = run_episode(sync_env, task_id)
                results.append(summary)
            except Exception as exc:
                _debug(f"Episode crashed for task_id={task_id!r}: {exc!r}")
                fallback_score = _normalise_task_score(0.0)
                log_end(success=False, steps=0, score=fallback_score, rewards=[])
                results.append(
                    {
                        "task_id": task_id,
                        "score": fallback_score,
                        "steps_taken": 0,
                        "success": False,
                        "violations": 0,
                        "parse_errors": 0,
                    }
                )

    # Human-readable summary to stderr only
    print("\n" + "=" * 60, file=sys.stderr)
    print(
        f"  {'task':<10} {'model':<30} {'score':>6} {'steps':>6} {'viol':>5} {'parse':>5}",
        file=sys.stderr,
    )
    print("=" * 60, file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']:<10} {MODEL_NAME:<30} "
            f"{r['score']:>6.3f} {r['steps_taken']:>6} "
            f"{r['violations']:>5} {r['parse_errors']:>5}",
            file=sys.stderr,
        )
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
