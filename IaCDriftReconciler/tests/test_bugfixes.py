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
