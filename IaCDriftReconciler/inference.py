"""
IaC Drift Reconciler — Baseline Inference Script.

Runs a full episode for each of the three task IDs using an LLM as the agent.
The LLM is called via the OpenAI-compatible chat completions API.

Environment variables (required)
---------------------------------
  API_BASE_URL   OpenAI-compatible endpoint, e.g. https://api.openai.com/v1
  MODEL_NAME     Model identifier, e.g. gpt-4o or meta-llama/Llama-3-70b-instruct
  HF_TOKEN       Hugging Face token (used as the API key when hitting HF endpoints)

Optional
--------
  ENV_BASE_URL   Running environment server (default: http://localhost:8000)
  MAX_STEPS      Episode step cap override  (default: 30)

Log format (stdout, one JSON per line)
---------------------------------------
  [START]  {"marker": "[START]", "task_id": ..., "model": ..., "episode": ...}
  [STEP]   {"marker": "[STEP]",  "task_id": ..., "step": ..., "action_type": ...,
             "reward": ..., "drift_score": ..., "done": ...}
  [END]    {"marker": "[END]",   "task_id": ..., "model": ...,
             "score": ..., "steps_taken": ..., "status": ...}

The evaluator parses these lines — any deviation breaks scoring.
All debug / error output goes to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

# ── Third-party ────────────────────────────────────────────────────────────
from openai import OpenAI

# ── Environment client + models ────────────────────────────────────────────
# Support both `python inference.py` (from the IaCDriftReconciler/ directory)
# and `python IaCDriftReconciler/inference.py` (from repo root).
try:
    from IaCDriftReconciler.client import IaCDriftReconcilerEnv
    from IaCDriftReconciler.models import IaCDriftReconcilerAction, IaCDriftReconcilerObservation
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from client import IaCDriftReconcilerEnv  # type: ignore[no-redef]
    from models import IaCDriftReconcilerAction, IaCDriftReconcilerObservation  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Configuration — read once at import time
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN:     str = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
MAX_STEPS:    int = int(os.environ.get("MAX_STEPS", "30"))

TASK_IDS = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# OpenAI client (OpenAI-compatible)
# ---------------------------------------------------------------------------

llm = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "placeholder",   # HF Inference API uses the token as key
)

# ---------------------------------------------------------------------------
# Logging helpers — all structured output goes to stdout, debug to stderr
# ---------------------------------------------------------------------------

def _log(record: Dict[str, Any]) -> None:
    """Emit one JSON log line to stdout (evaluator-facing)."""
    print(json.dumps(record), flush=True)


def _debug(msg: str) -> None:
    """Emit a human-readable debug line to stderr (not parsed by evaluator)."""
    print(f"[debug] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_ACTION_SCHEMA = {
    "action_type": "one of: update_resource | create_missing_resource | delete_extra_resource | attach_volume | detach_volume | no_op",
    "resource_name": "(str) required for: update_resource, create_missing_resource, delete_extra_resource",
    "attribute":     "(str) required for: update_resource — the field name to change",
    "new_value":     "(str) required for: update_resource — the new value (always as a string)",
    "resource_type": "(str) required for: create_missing_resource",
    "properties":    "(str) required for: create_missing_resource — JSON-encoded dict of fields",
    "instance_name": "(str) required for: attach_volume, detach_volume",
    "volume_name":   "(str) required for: attach_volume, detach_volume",
}

def _build_system_prompt() -> str:
    return (
        "You are an infrastructure reconciliation agent.\n"
        "Your job is to fix drifted infrastructure by submitting exactly ONE action per turn.\n\n"
        "RULES:\n"
        "1. You MUST respond with a single valid JSON object — no markdown, no explanation.\n"
        "2. The JSON must match the IaCDriftReconcilerAction schema below.\n"
        "3. Only include the fields required for the chosen action_type.\n"
        "4. Do NOT violate any guardrail constraint. Constraints are hard rules: "
        "   violating one ends the episode immediately with reward = -1.0.\n"
        "5. For create_missing_resource, encode 'properties' as a JSON string of the fields dict.\n"
        "6. If you are unsure, emit {\"action_type\": \"no_op\"}.\n\n"
        f"ACTION SCHEMA:\n{json.dumps(_ACTION_SCHEMA, indent=2)}\n\n"
        "EXAMPLE RESPONSES:\n"
        '  {"action_type": "update_resource", "resource_name": "aws_instance.web", '
        '"attribute": "instance_type", "new_value": "t3.medium"}\n'
        '  {"action_type": "delete_extra_resource", "resource_name": "aws_security_group.legacy"}\n'
        '  {"action_type": "no_op"}'
    )


def _build_user_prompt(obs: IaCDriftReconcilerObservation) -> str:
    drift_lines = []
    for item in obs.drift_items:
        drift_lines.append(
            f"  - {item.resource_id}.{item.field}: "
            f"actual={item.actual_value!r} → desired={item.desired_value!r} "
            f"[severity={item.severity}]"
        )

    drift_section = "\n".join(drift_lines) if drift_lines else "  (none — fully reconciled)"

    guardrail_section = (
        "\n".join(f"  - {r}" for r in obs.holy_grail_rules)
        if obs.holy_grail_rules
        else "  (none)"
    )

    return (
        f"STEP {obs.step_count} | drift_score={obs.drift_score:.4f} | done={obs.done}\n\n"
        f"REMAINING DRIFT ({len(obs.drift_items)} items):\n{drift_section}\n\n"
        f"GUARDRAIL CONSTRAINTS (never violate these):\n{guardrail_section}\n\n"
        "Submit your next action as a JSON object and nothing else."
    )


# ---------------------------------------------------------------------------
# LLM call + action parsing
# ---------------------------------------------------------------------------

def _call_llm(
    system_prompt: str,
    user_prompt: str,
    conversation_history: list,
) -> str:
    """Call the LLM and return the raw text response."""
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": user_prompt},
    ]
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def _parse_action(raw_text: str) -> Optional[IaCDriftReconcilerAction]:
    """Parse LLM output into an IaCDriftReconcilerAction.  Returns None on failure."""
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (```json or ```) and closing fence
        inner = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(inner).strip()

    data = json.loads(text)   # raises ValueError / json.JSONDecodeError on failure
    return IaCDriftReconcilerAction(**data)


def _fallback_action() -> IaCDriftReconcilerAction:
    return IaCDriftReconcilerAction(action_type="no_op")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: IaCDriftReconcilerEnv,
    task_id: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Run one full episode for *task_id*.

    Returns a summary dict with keys: task_id, score, steps_taken, status.
    """
    _debug(f"Starting episode for task_id={task_id!r}")

    # reset
    result = env.reset(task_id=task_id)
    obs: IaCDriftReconcilerObservation = result.observation

    _log({
        "marker": "[START]",
        "task_id": task_id,
        "model": MODEL_NAME,
        "episode": obs.metadata.get("episode_id", "unknown"),
        "initial_drift_score": obs.drift_score,
        "num_drift_items": len(obs.drift_items),
        "guardrail_count": len(obs.holy_grail_rules),
        "timestamp": time.time(),
    })

    conversation_history: list = []
    total_reward = 0.0
    steps_taken  = 0
    status       = "timeout"

    for step_num in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        user_prompt = _build_user_prompt(obs)

        # ── LLM call with fallback ─────────────────────────────────────
        raw_text = ""
        action: IaCDriftReconcilerAction
        try:
            raw_text = _call_llm(system_prompt, user_prompt, conversation_history)
            action = _parse_action(raw_text)
            _debug(f"  step {step_num}: LLM → {action.action_type}")
        except Exception as exc:
            _debug(f"  step {step_num}: parse/call error ({exc!r}) → no_op fallback")
            action = _fallback_action()
            raw_text = raw_text or "<error>"

        # Append to conversation so the model has context across steps
        conversation_history.append({"role": "user",      "content": user_prompt})
        conversation_history.append({"role": "assistant", "content": raw_text})

        # ── Environment step ───────────────────────────────────────────
        result = env.step(action)
        obs    = result.observation
        reward = result.reward if result.reward is not None else obs.metadata.get("reward", 0.0)

        total_reward += reward
        steps_taken   = step_num

        _log({
            "marker":      "[STEP]",
            "task_id":     task_id,
            "step":        step_num,
            "action_type": action.action_type,
            "resource":    (
                action.resource_name
                or action.instance_name
                or None
            ),
            "reward":         round(float(reward), 4),
            "drift_score":    round(float(obs.drift_score), 4),
            "done":           obs.done,
            "valid":          obs.metadata.get("last_action_valid", True),
            "guardrail_hit":  obs.metadata.get("guardrail_violated", False),
            "timestamp":      time.time(),
        })

        if obs.done:
            if obs.metadata.get("guardrail_violated"):
                status = "guardrail_violation"
            elif obs.drift_score == 0.0:
                status = "success"
            else:
                status = "max_steps"
            break

    # Final score = 1 - remaining drift_score (higher is better)
    final_score = round(1.0 - float(obs.drift_score), 4)

    _log({
        "marker":      "[END]",
        "task_id":     task_id,
        "model":       MODEL_NAME,
        "score":       final_score,
        "steps_taken": steps_taken,
        "total_reward": round(float(total_reward), 4),
        "final_drift_score": round(float(obs.drift_score), 4),
        "status":      status,
        "timestamp":   time.time(),
    })

    return {
        "task_id":     task_id,
        "score":       final_score,
        "steps_taken": steps_taken,
        "status":      status,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _debug(f"API_BASE_URL = {API_BASE_URL}")
    _debug(f"MODEL_NAME   = {MODEL_NAME}")
    _debug(f"ENV_BASE_URL = {ENV_BASE_URL}")

    if not HF_TOKEN and "openai.com" not in API_BASE_URL:
        _debug("WARNING: HF_TOKEN is not set. Requests may fail if the endpoint requires auth.")

    system_prompt = _build_system_prompt()
    results = []

    env = IaCDriftReconcilerEnv(base_url=ENV_BASE_URL)

    with env.sync() as sync_env:
        for task_id in TASK_IDS:
            try:
                summary = run_episode(sync_env, task_id, system_prompt)
                results.append(summary)
            except Exception as exc:
                _debug(f"Episode failed for task_id={task_id!r}: {exc!r}")
                _log({
                    "marker":      "[END]",
                    "task_id":     task_id,
                    "model":       MODEL_NAME,
                    "score":       0.0,
                    "steps_taken": 0,
                    "status":      "error",
                    "error":       str(exc),
                    "timestamp":   time.time(),
                })
                results.append({
                    "task_id": task_id,
                    "score":   0.0,
                    "steps_taken": 0,
                    "status":  "error",
                })

    # ── Results table to stderr (human-readable, not parsed by evaluator) ──
    print("\n" + "=" * 56, file=sys.stderr)
    print(f"  {'task_id':<10} {'model':<20} {'score':>6} {'steps':>6}  status", file=sys.stderr)
    print("=" * 56, file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']:<10} {MODEL_NAME:<20} {r['score']:>6.4f} "
            f"{r['steps_taken']:>6}  {r['status']}",
            file=sys.stderr,
        )
    print("=" * 56, file=sys.stderr)


if __name__ == "__main__":
    main()
