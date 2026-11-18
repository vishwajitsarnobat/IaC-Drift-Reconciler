"""validate_tasks.py on the shipped tasks, and an in-process server smoke test."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from IaCDriftReconciler import validate_tasks
from IaCDriftReconciler.server.app import app


def test_validate_tasks_passes_on_shipped_tasks(capsys):
    validate_tasks.ERRORS.clear()
    validate_tasks.WARNINGS.clear()
    validate_tasks.main()  # calls sys.exit(1) on any error
    assert validate_tasks.ERRORS == []
    assert "All task files passed validation" in capsys.readouterr().out


def test_validate_tasks_rejects_bad_guardrail():
    validate_tasks.ERRORS.clear()
    ok = validate_tasks.check_guardrails(["aws_instance.web ~= 3"], {}, "t")
    assert ok is False
    assert validate_tasks.ERRORS
    validate_tasks.ERRORS.clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_tasks(client):
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/tasks").json() == ["easy", "medium", "hard"]


def test_websocket_episode_solves_easy(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"task_id": "easy"}})
        msg = ws.receive_json()
        assert msg["data"]["observation"]["drift_score"] == 1.0

        for attribute, resource, value in [
            ("instance_type", "aws_instance.web", "t3.medium"),
            ("versioning_enabled", "aws_s3_bucket.logs", "true"),
        ]:
            ws.send_json({
                "type": "step",
                "data": {
                    "action_type": "update_resource",
                    "resource_name": resource,
                    "attribute": attribute,
                    "new_value": value,
                },
            })
            msg = ws.receive_json()

        assert msg["data"]["done"] is True
        assert msg["data"]["reward"] == pytest.approx(1.5)
        assert msg["data"]["observation"]["drift_score"] == 0.0
