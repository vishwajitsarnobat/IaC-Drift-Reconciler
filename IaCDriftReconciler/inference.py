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

All debug output goes to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── env client & models ────────────────────────────────────────────────────
try:
    from IaCDriftReconciler.client import IaCDriftReconcilerEnv
    from IaCDriftReconciler.models import IaCDriftReconcilerAction, IaCDriftReconcilerObservation
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import IaCDriftReconcilerEnv          # type: ignore[no-redef]
    from models import IaCDriftReconcilerAction, IaCDriftReconcilerObservation  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN     = os.getenv("HF_TOKEN")     or os.getenv("API_KEY") or ""
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"
MAX_STEPS    = int(os.getenv("MAX_STEPS", "30"))

BENCHMARK = "IaCDriftReconciler"
TASK_IDS  = ["easy", "medium", "hard"]

SUCCESS_SCORE_THRESHOLD = 0.99   # drift_score == 0.0 ↔ score == 1.0

# ---------------------------------------------------------------------------
# OpenAI-compatible client (HF Router uses the same API surface)
# ---------------------------------------------------------------------------

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "placeholder")

# ---------------------------------------------------------------------------
# Stdout log helpers  (exact format the evaluator parses)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
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


def _build_user_prompt(obs: IaCDriftReconcilerObservation, step: int) -> str:
    drift_lines = "\n".join(
        f"  {i+1}. {d.resource_id}.{d.field}: "
        f"actual={d.actual_value!r} → desired={d.desired_value!r} [{d.severity}]"
        for i, d in enumerate(obs.drift_items)
    ) or "  (none — fully reconciled)"

    guardrail_lines = "\n".join(
        f"  - {r}" for r in obs.holy_grail_rules
    ) or "  (none)"

    return textwrap.dedent(f"""\
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


def _parse_action(raw: str) -> IaCDriftReconcilerAction:
    text = raw.strip()
    # Strip markdown fences if model wraps in ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines[1:] if not l.strip().startswith("```")).strip()
    data = json.loads(text)
    return IaCDriftReconcilerAction(**data)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, task_id: str) -> dict:
    """Run one full episode; returns summary dict."""
    result = env.reset(task_id=task_id)
    obs: IaCDriftReconcilerObservation = result.observation

    log_start(task=task_id, model=MODEL_NAME)

    rewards:  List[float] = []
    conversation_history: list = []
    steps_taken = 0
    success     = False
    score       = 0.0

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            user_prompt = _build_user_prompt(obs, step)
            raw_text    = ""
            action: IaCDriftReconcilerAction
            error_msg: Optional[str] = None

            try:
                raw_text = _call_llm(user_prompt, conversation_history)
                action   = _parse_action(raw_text)
                _debug(f"  step {step}: {_action_str(action)}")
            except Exception as exc:
                _debug(f"  step {step}: parse/call error ({exc!r}) → no_op fallback")
                action    = IaCDriftReconcilerAction(action_type="no_op")
                error_msg = str(exc)[:120]
                raw_text  = raw_text or "<error>"

            # Append to history so the model has sequential context
            conversation_history.append({"role": "user",      "content": user_prompt})
            conversation_history.append({"role": "assistant", "content": raw_text})

            # Step the environment
            result = env.step(action)
            obs    = result.observation
            reward = float(
                result.reward
                if result.reward is not None
                else obs.metadata.get("reward", 0.0)
            )

            done = obs.done
            if not obs.metadata.get("last_action_valid", True):
                error_msg = obs.metadata.get("error", error_msg)
            if obs.metadata.get("guardrail_violated"):
                error_msg = f"guardrail_violated: {obs.metadata.get('violated_rule', '')}"

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=_action_str(action),
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

        # Score = 1 - remaining drift (1.0 = fully reconciled)
        score   = round(1.0 - float(obs.drift_score), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "steps_taken": steps_taken,
            "success": success}


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
                log_end(success=False, steps=0, score=0.0, rewards=[])
                results.append({"task_id": task_id, "score": 0.0,
                                 "steps_taken": 0, "success": False})

    # Human-readable summary to stderr only
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"  {'task':<10} {'model':<30} {'score':>6} {'steps':>6}",
          file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']:<10} {MODEL_NAME:<30} "
            f"{r['score']:>6.3f} {r['steps_taken']:>6}",
            file=sys.stderr,
        )
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
