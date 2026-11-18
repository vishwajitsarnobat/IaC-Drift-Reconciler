"""Regression tests for bugs found in the environment.

Each test failed on the original code and passes after the matching fix.
"""

from __future__ import annotations

import pytest

from IaCDriftReconciler.models import IaCDriftReconcilerAction
from IaCDriftReconciler.server.IaCDriftReconciler_environment import MAX_STEPS

from conftest import OPTIMAL, update

NO_OP = IaCDriftReconcilerAction(action_type="no_op")


# ── Bug A: stepping after the episode ended ─────────────────────────────────

def test_no_reward_after_success(env):
    """Before the fix every no_op after success paid another +1.0 bonus."""
    env.reset(task_id="easy")
    for action in OPTIMAL["easy"]:
        env.step(action)
    extra = [env.step(NO_OP) for _ in range(3)]
    assert all(o.reward == 0.0 for o in extra)
    assert all(o.done for o in extra)
    assert all(o.metadata["last_action_valid"] is False for o in extra)
    assert all(o.step_count == 2 for o in extra)


def test_no_actions_applied_after_guardrail_violation(env):
    """Before the fix the agent could keep acting (and earn reward) after a -1."""
    env.reset(task_id="medium")
    env.step(IaCDriftReconcilerAction(
        action_type="delete_extra_resource", resource_name="aws_instance.bastion"))
    obs = env.step(OPTIMAL["medium"][0])
    assert obs.done
    assert obs.reward == 0.0
    assert obs.actual_state["aws_security_group.web_sg"]["fields"]["ingress_port_22"] is True


# ── Bug B: invalid actions did not consume a step ───────────────────────────

def test_invalid_action_counts_as_a_step_and_is_penalised(env):
    env.reset(task_id="easy")
    obs = env.step(IaCDriftReconcilerAction(action_type="update_resource"))
    assert obs.metadata["last_action_valid"] is False
    assert "missing required fields" in obs.metadata["error"]
    assert obs.step_count == 1
    assert obs.reward == pytest.approx(-0.05)


def test_endless_invalid_actions_still_hit_max_steps(env):
    """Before the fix an agent sending malformed actions never reached done."""
    env.reset(task_id="easy")
    bad = IaCDriftReconcilerAction(action_type="delete_extra_resource")
    for _ in range(MAX_STEPS):
        obs = env.step(bad)
    assert obs.done
    assert obs.step_count == MAX_STEPS


# ── Bug C: actions on unknown resources were reported as valid ─────────────

@pytest.mark.parametrize("action", [
    update("aws_instance.ghost", "instance_type", "t3.medium"),
    IaCDriftReconcilerAction(action_type="delete_extra_resource",
                             resource_name="aws_instance.ghost"),
    IaCDriftReconcilerAction(action_type="attach_volume",
                             instance_name="aws_instance.ghost", volume_name="vol"),
    IaCDriftReconcilerAction(action_type="detach_volume",
                             instance_name="aws_instance.ghost", volume_name="vol"),
])
def test_unknown_resource_is_rejected(env, action):
    env.reset(task_id="easy")
    obs = env.step(action)
    assert obs.metadata["last_action_valid"] is False
    assert "aws_instance.ghost" in obs.metadata["error"]
    assert obs.reward == pytest.approx(-0.05)
    assert not obs.done
    assert obs.drift_score == 1.0


# ── Bug D: malformed create_missing_resource ────────────────────────────────

def _create(name: str, props: str) -> IaCDriftReconcilerAction:
    return IaCDriftReconcilerAction(
        action_type="create_missing_resource",
        resource_type="aws_instance",
        resource_name=name,
        properties=props,
    )


@pytest.mark.parametrize("props", ["[1, 2]", "\"text\"", "not json"])
def test_create_with_bad_properties_is_rejected_not_crash(env, props):
    """'[1, 2]' used to crash _compute_drift with AttributeError on list.get;
    'not json' silently created a resource with no fields."""
    env.reset(task_id="hard")
    del env.actual_state["aws_eip.nat"]  # make it genuinely missing
    obs = env.step(_create("aws_eip.nat", props))
    assert obs.metadata["last_action_valid"] is False
    assert "properties" in obs.metadata["error"]
    assert "aws_eip.nat" not in obs.actual_state


def test_create_existing_resource_is_rejected(env):
    """Creating over an existing resource silently replaced it."""
    env.reset(task_id="easy")
    obs = env.step(_create("aws_instance.web", '{"instance_type": "t3.medium"}'))
    assert obs.metadata["last_action_valid"] is False
    assert obs.actual_state["aws_instance.web"]["fields"]["ami"] == "ami-0c55b159cbfafe1f0"


def test_create_missing_resource_happy_path(env):
    """A genuinely missing resource can be created and its drift resolved."""
    env.reset(task_id="easy")
    del env.actual_state["aws_s3_bucket.logs"]
    obs = env.step(_create("aws_s3_bucket.logs",
                           '{"versioning_enabled": true, "region": "us-east-1"}'))
    assert obs.metadata["last_action_valid"] is True
    assert obs.actual_state["aws_s3_bucket.logs"]["fields"]["versioning_enabled"] is True


# ── Bug E: string -> bool coercion ──────────────────────────────────────────

@pytest.mark.parametrize("value", ["no", "0", "disabled"])
def test_non_boolean_string_does_not_become_true(env, value):
    """bool('no') is True in Python, so 'no' used to *enable* versioning and
    earn reward for fixing the drift."""
    env.reset(task_id="easy")
    obs = env.step(update("aws_s3_bucket.logs", "versioning_enabled", value))
    assert obs.actual_state["aws_s3_bucket.logs"]["fields"]["versioning_enabled"] is not True
    assert obs.reward <= 0.0
    assert obs.drift_score == 1.0


@pytest.mark.parametrize("value", ["true", "True", "TRUE", " true "])
def test_boolean_strings_are_accepted(env, value):
    env.reset(task_id="easy")
    obs = env.step(update("aws_s3_bucket.logs", "versioning_enabled", value))
    assert obs.actual_state["aws_s3_bucket.logs"]["fields"]["versioning_enabled"] is True
    assert obs.drift_score == pytest.approx(0.5)
