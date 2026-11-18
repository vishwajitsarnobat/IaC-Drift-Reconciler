"""Behavioural tests for IaCDriftReconcilerEnvironment (reset / step / reward)."""

from __future__ import annotations

import pytest

from IaCDriftReconciler.models import IaCDriftReconcilerAction
from IaCDriftReconciler.server.IaCDriftReconciler_environment import MAX_STEPS

from conftest import OPTIMAL, update


# ── Each task is solvable by a scripted optimal sequence ────────────────────

@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_optimal_sequence_solves_task(env, task_id):
    obs = env.reset(task_id=task_id)
    assert obs.drift_score == 1.0
    assert not obs.done

    for action in OPTIMAL[task_id]:
        obs = env.step(action)
        assert obs.metadata["last_action_valid"] is True, obs.metadata
        assert obs.metadata["guardrail_violated"] is False

    assert obs.done
    assert obs.drift_score == 0.0
    assert obs.drift_items == []
    assert obs.metadata["success"] is True
    assert obs.step_count == len(OPTIMAL[task_id])


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_episode_not_done_before_last_fix(env, task_id):
    env.reset(task_id=task_id)
    for action in OPTIMAL[task_id][:-1]:
        obs = env.step(action)
        assert not obs.done
        assert obs.drift_score > 0.0


# ── Reward sign and magnitude ───────────────────────────────────────────────

def test_reward_for_drift_reducing_action(env):
    env.reset(task_id="hard")  # 5 drift items -> each fix is worth 1/5
    obs = env.step(OPTIMAL["hard"][0])
    assert obs.drift_score == pytest.approx(0.8)
    assert obs.reward == pytest.approx(0.2)
    assert obs.reward > 0


def test_final_fix_gets_success_bonus(env):
    env.reset(task_id="easy")
    env.step(OPTIMAL["easy"][0])
    obs = env.step(OPTIMAL["easy"][1])
    # 0.5 drift reduction + 1.0 success bonus
    assert obs.reward == pytest.approx(1.5)


def test_total_reward_of_optimal_episode_is_two(env):
    # drift goes 1.0 -> 0.0 (sum of reductions = 1.0) plus the +1.0 bonus
    for task_id in ("easy", "medium", "hard"):
        env.reset(task_id=task_id)
        total = sum(env.step(a).reward for a in OPTIMAL[task_id])
        assert total == pytest.approx(2.0), task_id


def test_no_op_gives_zero_reward_and_counts_a_step(env):
    env.reset(task_id="easy")
    obs = env.step(IaCDriftReconcilerAction(action_type="no_op"))
    assert obs.reward == 0.0
    assert obs.step_count == 1
    assert obs.drift_score == 1.0


def test_wrong_value_does_not_reduce_drift(env):
    env.reset(task_id="easy")
    obs = env.step(update("aws_instance.web", "instance_type", "t3.large"))
    assert obs.reward == 0.0
    assert obs.drift_score == 1.0


# ── Guardrails ──────────────────────────────────────────────────────────────

def test_deleting_unmanaged_bastion_violates_guardrail(env):
    """Medium task: delete-vs-import. Deleting the shadow resource is the trap."""
    env.reset(task_id="medium")
    obs = env.step(
        IaCDriftReconcilerAction(
            action_type="delete_extra_resource",
            resource_name="aws_instance.bastion",
        )
    )
    assert obs.done
    assert obs.reward == -1.0
    assert obs.metadata["guardrail_violated"] is True
    assert obs.metadata["violated_rule"] == "aws_instance.bastion.managed == true"
    # The violating action must NOT have been applied.
    assert "aws_instance.bastion" in obs.actual_state


def test_changing_guarded_field_violates_guardrail(env):
    env.reset(task_id="easy")
    obs = env.step(update("aws_instance.web", "ami", "ami-evil"))
    assert obs.done
    assert obs.reward == -1.0
    assert obs.metadata["violated_rule"].startswith("aws_instance.web.fields.ami")
    assert obs.actual_state["aws_instance.web"]["fields"]["ami"] == "ami-0c55b159cbfafe1f0"


def test_not_equal_guardrail_on_hard_task(env):
    env.reset(task_id="hard")
    obs = env.step(update("aws_subnet.main", "cidr_block", "10.0.0.0/8"))
    assert obs.done and obs.reward == -1.0


def test_bool_guardrail_with_string_new_value(env):
    """new_value arrives as a string; 'false' must satisfy '== false' on a bool."""
    env.reset(task_id="medium")
    obs = env.step(update("aws_security_group.web_sg", "ingress_port_22", "false"))
    assert not obs.done
    assert obs.reward == pytest.approx(0.5)
    assert obs.actual_state["aws_security_group.web_sg"]["fields"]["ingress_port_22"] is False


# ── Hard task: dependency ordering ──────────────────────────────────────────

def test_dependent_before_dependency_is_rejected(env):
    env.reset(task_id="hard")
    obs = env.step(update("aws_nat_gateway.main", "connectivity_type", "public"))
    assert obs.metadata["last_action_valid"] is False
    assert "unresolved dependencies" in obs.metadata["error"]
    assert obs.reward == pytest.approx(-0.05)
    assert obs.drift_score == 1.0
    assert obs.actual_state["aws_nat_gateway.main"]["fields"]["connectivity_type"] == "private"
    assert not obs.done


def test_reverse_order_never_succeeds(env):
    env.reset(task_id="hard")
    for action in reversed(OPTIMAL["hard"]):
        obs = env.step(action)
    assert not obs.done
    assert obs.drift_score > 0.0
    # Only the two root resources (subnet, eip) were accepted.
    assert obs.drift_score == pytest.approx(0.6)


# ── Reset isolation ─────────────────────────────────────────────────────────

def test_reset_restores_initial_state(env):
    first = env.reset(task_id="easy")
    for action in OPTIMAL["easy"]:
        env.step(action)
    second = env.reset(task_id="easy")
    assert second.actual_state == first.actual_state
    assert second.drift_score == 1.0
    assert second.step_count == 0
    assert not second.done


def test_mutating_observation_does_not_leak_into_env(env):
    obs = env.reset(task_id="easy")
    obs.actual_state["aws_instance.web"]["fields"]["instance_type"] = "t3.medium"
    obs.desired_state.clear()
    fresh = env.reset(task_id="easy")
    assert fresh.actual_state["aws_instance.web"]["fields"]["instance_type"] == "t2.micro"
    assert fresh.desired_state


def test_two_env_instances_are_independent():
    from IaCDriftReconciler.server.IaCDriftReconciler_environment import (
        IaCDriftReconcilerEnvironment,
    )

    a, b = IaCDriftReconcilerEnvironment(), IaCDriftReconcilerEnvironment()
    a.reset(task_id="easy")
    b.reset(task_id="easy")
    a.step(OPTIMAL["easy"][0])
    assert b.state.step_count == 0
    assert b.actual_state["aws_instance.web"]["fields"]["instance_type"] == "t2.micro"


def test_reset_gives_new_episode_id(env):
    env.reset(task_id="easy")
    first_id = env.state.episode_id
    env.reset(task_id="easy")
    assert env.state.episode_id != first_id


def test_unknown_task_id_raises(env):
    with pytest.raises(ValueError):
        env.reset(task_id="impossible")


# ── Max steps ───────────────────────────────────────────────────────────────

def test_episode_ends_at_max_steps(env):
    env.reset(task_id="easy")
    for i in range(MAX_STEPS):
        obs = env.step(IaCDriftReconcilerAction(action_type="no_op"))
        assert obs.done == (i == MAX_STEPS - 1)
    assert obs.step_count == MAX_STEPS
    assert obs.drift_score == 1.0
