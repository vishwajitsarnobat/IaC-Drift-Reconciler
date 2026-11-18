"""Tests for inference.py's agent loop with a fake LLM and an in-process env.

No network and no API keys: the OpenAI client is replaced by FakeLLM, and the
env server is replaced by WireEnv, which runs the real environment and passes
every observation through the same serialiser the real server uses (which
drops `metadata`) and the real client parser.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import List

import pytest
from openenv.core.env_server.serialization import deserialize_action, serialize_observation

from IaCDriftReconciler import inference
from IaCDriftReconciler.client import IaCDriftReconcilerEnv
from IaCDriftReconciler.models import IaCDriftReconcilerAction
from IaCDriftReconciler.server.IaCDriftReconciler_environment import (
    IaCDriftReconcilerEnvironment,
)

from conftest import OPTIMAL


class WireEnv:
    """Sync stand-in for IaCDriftReconcilerEnv.sync() with server-faithful payloads."""

    def __init__(self) -> None:
        self.env = IaCDriftReconcilerEnvironment()

    @staticmethod
    def _over_the_wire(obs):
        return IaCDriftReconcilerEnv._parse_result(None, serialize_observation(obs))

    def reset(self, task_id: str):
        return self._over_the_wire(self.env.reset(task_id=task_id))

    def step(self, action: IaCDriftReconcilerAction):
        payload = IaCDriftReconcilerEnv._step_payload(None, action)
        server_action = deserialize_action(payload, IaCDriftReconcilerAction)
        return self._over_the_wire(self.env.step(server_action))


class FakeLLM:
    """Mimics llm.chat.completions.create; replies from a scripted list."""

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[list] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        content = self.replies.pop(0) if self.replies else '{"action_type": "no_op"}'
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def as_json(action: IaCDriftReconcilerAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True, exclude={"metadata"}))


def run(monkeypatch, task_id: str, replies: List[str]) -> dict:
    fake = FakeLLM(replies)
    monkeypatch.setattr(inference, "llm", fake)
    summary = inference.run_episode(WireEnv(), task_id)
    summary["_fake"] = fake
    return summary


def end_line(out: str) -> str:
    return [l for l in out.splitlines() if l.startswith("[END]")][-1]


def step_lines(out: str) -> List[str]:
    return [l for l in out.splitlines() if l.startswith("[STEP]")]


# ── Good JSON ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_good_json_solves_every_task(monkeypatch, capsys, task_id):
    s = run(monkeypatch, task_id, [as_json(a) for a in OPTIMAL[task_id]])
    out = capsys.readouterr().out
    assert s["success"] is True
    assert s["steps_taken"] == len(OPTIMAL[task_id])
    assert s["parse_errors"] == 0
    assert s["final_drift"] == 0.0
    assert end_line(out).startswith(f"[END] success=true steps={len(OPTIMAL[task_id])} score=0.999")
    assert all("error=null" in l for l in step_lines(out))


def test_stdout_format_is_unchanged(monkeypatch, capsys):
    run(monkeypatch, "easy", [as_json(a) for a in OPTIMAL["easy"]])
    lines = capsys.readouterr().out.splitlines()
    assert re.fullmatch(r"\[START\] task=easy env=IaCDriftReconciler model=\S+", lines[0])
    for l in lines[1:-1]:
        assert re.fullmatch(
            r"\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d\d done=(true|false) error=.+", l
        )
    assert re.fullmatch(
        r"\[END\] success=(true|false) steps=\d+ score=\d\.\d{3} rewards=[-\d.,]*", lines[-1]
    )


# ── Fenced JSON ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("wrap", [
    "```json\n{}\n```",
    "```\n{}\n```",
    "Here is my action:\n```json\n{}\n```",
    "Sure! {} Hope that helps.",
])
def test_fenced_or_wrapped_json_is_parsed(monkeypatch, capsys, wrap):
    replies = [wrap.replace("{}", as_json(a)) for a in OPTIMAL["easy"]]
    s = run(monkeypatch, "easy", replies)
    assert s["success"] is True
    assert s["parse_errors"] == 0


# ── Garbage ─────────────────────────────────────────────────────────────────

def test_garbage_is_logged_counted_and_does_not_crash(monkeypatch, capsys):
    replies = ["I think we should fix the instance.", '{"action_type": "teleport"}']
    replies += [as_json(a) for a in OPTIMAL["easy"]]
    s = run(monkeypatch, "easy", replies)
    captured = capsys.readouterr()
    steps = step_lines(captured.out)

    assert s["parse_errors"] == 2
    assert s["success"] is True
    assert "action=no_op()" in steps[0] and "error=parse_error" in steps[0]
    assert "error=parse_error" in steps[1]
    assert "parse_errors=2" in captured.err  # per-episode summary on stderr


def test_llm_exception_is_counted_separately(monkeypatch, capsys):
    def boom(**kwargs):
        raise RuntimeError("503 from router")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    monkeypatch.setattr(inference, "llm", fake)
    monkeypatch.setattr(inference, "MAX_STEPS", 3)
    s = inference.run_episode(WireEnv(), "easy")
    out = capsys.readouterr().out
    assert s["llm_errors"] == 3
    assert s["parse_errors"] == 0
    assert s["success"] is False
    assert all("error=llm_error" in l for l in step_lines(out))


# ── Guardrail violation over the wire ───────────────────────────────────────

def test_guardrail_violation_is_reported_and_scores_minimum(monkeypatch, capsys):
    """The real server strips observation.metadata, so the old loop never saw
    'guardrail_violated' and gave partial credit (score 0.5) to an episode that
    ended by deleting a protected resource."""
    replies = [
        as_json(OPTIMAL["medium"][0]),
        '{"action_type": "delete_extra_resource", "resource_name": "aws_instance.bastion"}',
    ]
    s = run(monkeypatch, "medium", replies)
    out = capsys.readouterr().out
    assert s["violations"] == 1
    assert s["success"] is False
    assert s["score"] == inference.MIN_TASK_SCORE
    assert "error=guardrail_violated" in step_lines(out)[-1]
    assert "done=true" in step_lines(out)[-1]


def test_rejected_action_is_reported(monkeypatch, capsys):
    replies = ['{"action_type": "update_resource", "resource_name": "aws_route.default", '
               '"attribute": "status", "new_value": "active"}']
    replies += [as_json(a) for a in OPTIMAL["hard"]]
    s = run(monkeypatch, "hard", replies)
    out = capsys.readouterr().out
    assert s["rejected_actions"] == 1
    assert "error=action_rejected" in step_lines(out)[0]
    assert s["success"] is True


def test_previous_error_is_shown_to_the_model(monkeypatch):
    s = run(monkeypatch, "easy", ["garbage", '{"action_type": "no_op"}'])
    second_prompt = s["_fake"].calls[1][-1]["content"]
    assert "LAST ACTION" in second_prompt
    assert "parse_error" in second_prompt


# ── Conversation history is bounded ─────────────────────────────────────────

def test_history_is_capped(monkeypatch):
    s = run(monkeypatch, "easy", [])  # all no_op until MAX_STEPS
    lengths = [len(m) for m in s["_fake"].calls]
    assert len(lengths) == inference.MAX_STEPS
    cap = 2 + 2 * inference.HISTORY_TURNS  # system + current user + kept turns
    assert max(lengths) <= cap
    assert s["_fake"].calls[-1][0]["role"] == "system"


# ── Score helper keeps the (0, 1) clamp ─────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (1.0, 0.999), (0.0, 0.001), (-3.0, 0.001), (0.5, 0.5), (float("nan"), 0.001),
])
def test_score_is_clamped_to_open_interval(raw, expected):
    assert inference._normalise_task_score(raw) == expected
