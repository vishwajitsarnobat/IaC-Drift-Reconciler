"""
IaC Drift Reconciler — Core Environment.

Class: IaCDriftReconcilerEnvironment

Public methods
--------------
reset(task_id)          Load task JSON, initialise state, return first observation.
step(action)            Execute one agent action following the fixed 8-step sequence.
state (property)        Return current episode State (episode_id, step_count).

Private helpers
---------------
_compute_drift          Diff desired vs actual; return List[DriftItem] and score.
_validate_action_fields Ensure required fields are present for the given action_type.
_check_dependency_order Reject actions on resources whose dependencies are unresolved.
_check_guardrail_violation
                        Check ALL constraints on projected (post-action) state;
                        only raise a violation for a resource the action touches.
_project_action         Deep-copy actual_state and apply the action — used by the
                        guardrail pre-flight so real state is never mutated early.
_apply_action_to        Core mutator shared by projection and actual application.
_is_reconciled          True when a resource has zero drift vs desired_state.
_get_action_resource    Return the primary resource_id targeted by an action.

Guardrail semantics
-------------------
Guardrails are in the format "<resource_type>.<name>.<field_path> <op> <value>".
A constraint terminates the episode (reward=-1) only when:
  * The projected post-action state violates it, AND
  * The action targeted the same resource the constraint references.
This prevents a pre-existing violation (already-drifted field) from blocking every
unrelated action while still catching the agent choosing a destructive path.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        DriftItem,
        IaCDriftReconcilerAction,
        IaCDriftReconcilerObservation,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        DriftItem,
        IaCDriftReconcilerAction,
        IaCDriftReconcilerObservation,
    )

# Constants

MAX_STEPS: int = 30

TASKS_DIR = Path(__file__).parent.parent / "tasks"
TASK_FILE_MAP: Dict[str, Path] = {
    "easy":   TASKS_DIR / "task_easy.json",
    "medium": TASKS_DIR / "task_medium.json",
    "hard":   TASKS_DIR / "task_hard.json",
}

# Required action fields per action_type
_ACTION_REQUIRED: Dict[str, List[str]] = {
    "update_resource":         ["resource_name", "attribute", "new_value"],
    "create_missing_resource": ["resource_type", "resource_name", "properties"],
    "delete_extra_resource":   ["resource_name"],
    "attach_volume":           ["instance_name", "volume_name"],
    "detach_volume":           ["instance_name", "volume_name"],
    "no_op":                   [],
}

# Action types that must pass a dependency-order check
_DEPENDENCY_CHECKED = {"update_resource", "create_missing_resource", "attach_volume"}

# Severity by resource type (used when building DriftItems)
_SEVERITY_MAP: Dict[str, str] = {
    "aws_security_group": "critical",
    "aws_nat_gateway":    "high",
    "aws_route":          "high",
    "aws_instance":       "medium",
    "aws_subnet":         "medium",
    "aws_eip":            "low",
    "aws_s3_bucket":      "low",
    "aws_route_table":    "low",
}
_DEFAULT_SEVERITY = "medium"

_VALID_OPS = {"==", "!=", ">", ">=", "<", "<="}


# Pure helper functions (module-level, no side-effects)

def _coerce(value: Any) -> Any:
    """Coerce a string to bool / int / float / str.  Non-strings pass through."""
    if not isinstance(value, str):
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _coerce_to_target(raw: str, target: Any) -> Any:
    """Coerce *raw* to the same Python type as *target*.  Falls back to _coerce."""
    if target is None:
        return _coerce(raw)
    if isinstance(target, bool):
        c = _coerce(raw)
        return c if isinstance(c, bool) else bool(raw)
    if isinstance(target, int):
        try:
            return int(raw)
        except (ValueError, TypeError):
            return raw
    if isinstance(target, float):
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw
    return raw  # keep as string


def _get_field(resource: Dict[str, Any], field_path: str) -> Tuple[bool, Any]:
    """Navigate a dot-separated path inside a resource dict.

    Example paths:
      "fields.ingress_port_22"  → resource["fields"]["ingress_port_22"]
      "managed"                 → resource["managed"]
    Returns (found, value).
    """
    parts = field_path.split(".")
    node: Any = resource
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _eval_op(lhs: Any, op: str, rhs_str: str) -> bool:
    """Evaluate ``lhs <op> rhs`` after coercing types."""
    lhs_c = _coerce(lhs)
    rhs_c = _coerce(rhs_str)
    try:
        if op == "==":  return lhs_c == rhs_c
        if op == "!=":  return lhs_c != rhs_c
        if op == ">":   return lhs_c > rhs_c   # type: ignore[operator]
        if op == ">=":  return lhs_c >= rhs_c  # type: ignore[operator]
        if op == "<":   return lhs_c < rhs_c   # type: ignore[operator]
        if op == "<=":  return lhs_c <= rhs_c  # type: ignore[operator]
    except TypeError:
        if op == "==":  return str(lhs_c) == str(rhs_c)
        if op == "!=":  return str(lhs_c) != str(rhs_c)
    return False


def _parse_constraint(c: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse '<resource_type>.<name>.<field_path> <op> <value>'.

    Returns (resource_id, field_path, operator, rhs) or None on failure.
    resource_id is always the first two dot-segments joined by '.'.
    """
    parts = c.strip().split()
    if len(parts) != 3:
        return None
    lhs, op, rhs = parts
    if op not in _VALID_OPS:
        return None
    dot_parts = lhs.split(".")
    if len(dot_parts) < 3:
        return None
    resource_id = ".".join(dot_parts[:2])
    field_path  = ".".join(dot_parts[2:])
    return resource_id, field_path, op, rhs


def _constraint_satisfied(
    state: Dict[str, Any],
    resource_id: str,
    field_path: str,
    op: str,
    rhs: str,
) -> bool:
    """Return True when the constraint holds in *state*."""
    resource = state.get(resource_id)
    if resource is None:
        # Resource absent: equality / ordering operators → not satisfied.
        return op == "!="
    found, lhs = _get_field(resource, field_path)
    if not found:
        return op == "!="
    return _eval_op(lhs, op, rhs)


# Environment

class IaCDriftReconcilerEnvironment(Environment):
    """
    RL environment for infrastructure-drift reconciliation.

    An episode begins with reset(task_id) and proceeds via step(action) calls
    until done=True (full reconciliation, guardrail violation, or MAX_STEPS).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._episode_id: str = str(uuid4())
        self.step_count: int = 0

        # Set by reset()
        self.desired_state: Dict[str, Any] = {}
        self.actual_state: Dict[str, Any] = {}
        self.guardrail_constraints: List[str] = []
        self.initial_drift_total: int = 1        # prevents /0 before first reset
        self._task_id: Optional[str] = None
        self._current_drift_items: List[DriftItem] = []
        self._current_drift_score: float = 1.0

    # ── Public API ─────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> IaCDriftReconcilerObservation:
        """Load *task_id* from disk, initialise all state, return first observation."""
        if task_id not in TASK_FILE_MAP:
            raise ValueError(
                f"Unknown task_id {task_id!r}. Valid: {sorted(TASK_FILE_MAP)}"
            )

        task_path = TASK_FILE_MAP[task_id]
        with task_path.open() as fh:
            task_data = json.load(fh)

        self._episode_id   = str(uuid4())
        self._task_id      = task_id
        self.step_count    = 0

        # Deep-copy so mutations never corrupt the on-disk data
        self.desired_state         = copy.deepcopy(task_data["desired_state"])
        self.actual_state          = copy.deepcopy(task_data["actual_state"])
        self.guardrail_constraints = list(task_data["guardrail_constraints"])

        # Baseline drift — used for score normalisation throughout the episode
        initial_items          = self._compute_drift(self.desired_state, self.actual_state)
        self.initial_drift_total   = max(len(initial_items), 1)
        self._current_drift_items  = initial_items
        self._current_drift_score  = len(initial_items) / self.initial_drift_total  # → 1.0

        return self._make_obs(
            done=False,
            metadata={"task_id": task_id, "episode_id": self._episode_id, "reward": 0.0},
        )

    def step(self, action: IaCDriftReconcilerAction) -> IaCDriftReconcilerObservation:
        """Execute one step.  Fixed 8-step sequence — do not reorder.

        1. Validate required action fields.
        2. Check dependency ordering.
        3. Check guardrail violation on PROJECTED post-action state.
        4. Apply action to self.actual_state.
        5. Recompute drift.
        6. Compute reward.
        7. Increment step_count; check MAX_STEPS.
        8. Return observation.
        """
        old_score = self._current_drift_score

        # ── 1. Field validation ────────────────────────────────────────
        valid, field_err = self._validate_action_fields(action)
        if not valid:
            return self._make_obs(
                done=False,
                metadata={
                    "last_action_valid": False,
                    "error": field_err,
                    "reward": 0.0,
                    "step": self.step_count,
                },
            )

        # ── 2. Dependency ordering ─────────────────────────────────────
        dep_ok, dep_err = self._check_dependency_order(action)
        if not dep_ok:
            self.step_count += 1
            return self._make_obs(
                done=self.step_count >= MAX_STEPS,
                metadata={
                    "last_action_valid": False,
                    "error": dep_err,
                    "reward": -0.05,
                    "step": self.step_count,
                },
            )

        # ── 3. Guardrail pre-flight on projected state ─────────────────
        violated, violated_rule = self._check_guardrail_violation(action)
        if violated:
            self.step_count += 1
            return self._make_obs(
                done=True,
                metadata={
                    "last_action_valid": True,
                    "guardrail_violated": True,
                    "violated_rule": violated_rule,
                    "reward": -1.0,
                    "step": self.step_count,
                },
            )

        # ── 4. Apply action ────────────────────────────────────────────
        apply_warn = self._apply_action_to(action, self.actual_state)

        # ── 5. Recompute drift ─────────────────────────────────────────
        new_items = self._compute_drift(self.desired_state, self.actual_state)
        new_score = min(len(new_items) / self.initial_drift_total, 1.0)
        self._current_drift_items = new_items
        self._current_drift_score = new_score

        # ── 6. Reward ──────────────────────────────────────────────────
        reward  = old_score - new_score          # positive when drift reduces
        success = new_score == 0.0
        if success:
            reward += 1.0

        # ── 7. Step count / done ───────────────────────────────────────
        self.step_count += 1
        done = success or (self.step_count >= MAX_STEPS)

        # ── 8. Return ──────────────────────────────────────────────────
        meta: Dict[str, Any] = {
            "last_action_valid": True,
            "guardrail_violated": False,
            "success": success,
            "reward": reward,
            "step": self.step_count,
        }
        if apply_warn:
            meta["apply_warning"] = apply_warn

        return self._make_obs(done=done, metadata=meta)

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self.step_count)

    # ── _compute_drift ─────────────────────────────────────────────────

    def _compute_drift(
        self,
        desired: Dict[str, Any],
        actual: Dict[str, Any],
    ) -> List[DriftItem]:
        """Compare desired vs actual state; return every discrepancy as a DriftItem.

        Three drift classes:
          * Missing resource  — in desired, absent from actual  (field="__exists__")
          * Field drift       — resource present in both; specific fields differ
          * Extra resource    — in actual, absent from desired   (field="__exists__")
        """
        items: List[DriftItem] = []

        # Pass 1 — missing resources + field drift
        for res_id, des_res in desired.items():
            severity = _SEVERITY_MAP.get(des_res.get("resource_type", ""), _DEFAULT_SEVERITY)

            if res_id not in actual:
                items.append(DriftItem(
                    resource_id=res_id,
                    field="__exists__",
                    desired_value=True,
                    actual_value=False,
                    severity=severity,
                ))
                continue

            act_res = actual[res_id]

            # managed flag
            if des_res.get("managed") != act_res.get("managed"):
                items.append(DriftItem(
                    resource_id=res_id,
                    field="managed",
                    desired_value=des_res.get("managed"),
                    actual_value=act_res.get("managed"),
                    severity=severity,
                ))

            # fields dict
            des_fields: Dict[str, Any] = des_res.get("fields", {})
            act_fields: Dict[str, Any] = act_res.get("fields", {})
            for fname, des_val in des_fields.items():
                act_val = act_fields.get(fname)
                if des_val != act_val:
                    items.append(DriftItem(
                        resource_id=res_id,
                        field=f"fields.{fname}",
                        desired_value=des_val,
                        actual_value=act_val,
                        severity=severity,
                    ))

        # Pass 2 — extra resources
        for res_id, act_res in actual.items():
            if res_id not in desired:
                severity = _SEVERITY_MAP.get(act_res.get("resource_type", ""), _DEFAULT_SEVERITY)
                items.append(DriftItem(
                    resource_id=res_id,
                    field="__exists__",
                    desired_value=False,
                    actual_value=True,
                    severity=severity,
                ))

        return items

    # ── _validate_action_fields ────────────────────────────────────────

    def _validate_action_fields(
        self, action: IaCDriftReconcilerAction
    ) -> Tuple[bool, str]:
        required = _ACTION_REQUIRED.get(action.action_type, [])
        missing  = [f for f in required if getattr(action, f, None) is None]
        if missing:
            return False, (
                f"Action '{action.action_type}' missing required fields: {missing}"
            )
        return True, ""

    # ── _check_dependency_order ────────────────────────────────────────

    def _check_dependency_order(
        self, action: IaCDriftReconcilerAction
    ) -> Tuple[bool, str]:
        """Reject an action if the target resource's dependencies are not yet
        reconciled in actual_state.  Only applied to action types in
        _DEPENDENCY_CHECKED.
        """
        if action.action_type not in _DEPENDENCY_CHECKED:
            return True, ""

        resource_id = (
            action.instance_name
            if action.action_type == "attach_volume"
            else action.resource_name
        )
        des_res = self.desired_state.get(resource_id)
        if des_res is None:
            return True, ""

        unresolved = [
            dep for dep in des_res.get("dependencies", [])
            if not self._is_reconciled(dep)
        ]
        if unresolved:
            return False, (
                f"Cannot act on '{resource_id}': "
                f"unresolved dependencies {unresolved}. "
                "Reconcile dependencies first."
            )
        return True, ""

    def _is_reconciled(self, resource_id: str) -> bool:
        """True when resource_id has zero drift against desired_state."""
        if resource_id not in self.desired_state:
            return True   # unmanaged — nothing expected
        if resource_id not in self.actual_state:
            return False  # missing

        des = self.desired_state[resource_id]
        act = self.actual_state[resource_id]

        if des.get("managed") != act.get("managed"):
            return False

        des_fields = des.get("fields", {})
        act_fields = act.get("fields", {})
        return all(act_fields.get(k) == v for k, v in des_fields.items())

    # ── _check_guardrail_violation ─────────────────────────────────────

    def _check_guardrail_violation(
        self, action: IaCDriftReconcilerAction
    ) -> Tuple[bool, str]:
        """Check constraints against the PROJECTED post-action state.

        A constraint causes episode termination only when:
          1. It is violated in the projected state, AND
          2. The constrained resource is the same resource the action targets.

        Rationale: a pre-existing violation in a *different* resource should not
        block the agent from taking any action on a different resource.  The agent
        is penalised for that pre-existing drift via drift_score, not terminated.
        """
        projected      = self._project_action(action)
        action_res_id  = self._get_action_resource(action)

        for constraint in self.guardrail_constraints:
            parsed = _parse_constraint(constraint)
            if parsed is None:
                continue

            res_id, field_path, op, rhs = parsed

            # Only enforce this constraint when the action targets the same resource
            if res_id != action_res_id:
                continue

            if not _constraint_satisfied(projected, res_id, field_path, op, rhs):
                return True, constraint

        return False, ""

    def _get_action_resource(self, action: IaCDriftReconcilerAction) -> Optional[str]:
        """Return the primary resource_id touched by *action*."""
        t = action.action_type
        if t in ("update_resource", "create_missing_resource", "delete_extra_resource"):
            return action.resource_name
        if t in ("attach_volume", "detach_volume"):
            return action.instance_name
        return None  # no_op

    def _project_action(
        self, action: IaCDriftReconcilerAction
    ) -> Dict[str, Any]:
        """Return a deep-copy of actual_state with the action applied.
        Never mutates self.actual_state.
        """
        projected = copy.deepcopy(self.actual_state)
        self._apply_action_to(action, projected)
        return projected

    # ── _apply_action_to ──────────────────────────────────────────────

    def _apply_action_to(
        self,
        action: IaCDriftReconcilerAction,
        state: Dict[str, Any],
    ) -> str:
        """Mutate *state* in-place according to *action*.
        Returns '' on success, or a warning string on a non-fatal problem.
        """
        t = action.action_type

        # ── update_resource ────────────────────────────────────────────
        if t == "update_resource":
            res_id    = action.resource_name
            attribute = action.attribute
            raw_val   = action.new_value

            if res_id not in state:
                return f"update_resource: '{res_id}' not in actual_state"

            res = state[res_id]

            # Top-level fields (managed, resource_type, dependencies)
            if attribute in ("managed", "resource_type"):
                res[attribute] = _coerce(raw_val)

            elif attribute == "dependencies":
                try:
                    res["dependencies"] = json.loads(raw_val)
                except (json.JSONDecodeError, TypeError):
                    res["dependencies"] = []

            else:
                # Field inside the 'fields' dict
                res.setdefault("fields", {})
                desired_val = (
                    self.desired_state.get(res_id, {})
                    .get("fields", {})
                    .get(attribute)
                )
                res["fields"][attribute] = _coerce_to_target(raw_val, desired_val)

            return ""

        # ── create_missing_resource ────────────────────────────────────
        if t == "create_missing_resource":
            res_id    = action.resource_name
            res_type  = action.resource_type
            props_raw = action.properties or "{}"

            try:
                fields: Dict[str, Any] = json.loads(props_raw)
            except (json.JSONDecodeError, TypeError):
                fields = {}

            state[res_id] = {
                "resource_type": res_type,
                "fields":        fields,
                "managed":       True,
                "dependencies":  [],
            }
            return ""

        # ── delete_extra_resource ──────────────────────────────────────
        if t == "delete_extra_resource":
            res_id = action.resource_name
            if res_id not in state:
                return f"delete_extra_resource: '{res_id}' not in actual_state"
            del state[res_id]
            return ""

        # ── attach_volume ──────────────────────────────────────────────
        if t == "attach_volume":
            inst_id = action.instance_name
            vol_id  = action.volume_name
            if inst_id not in state:
                return f"attach_volume: instance '{inst_id}' not in actual_state"
            inst = state[inst_id]
            inst.setdefault("fields", {})
            attached: List[str] = inst["fields"].get("attached_volumes", [])
            if not isinstance(attached, list):
                attached = []
            if vol_id not in attached:
                attached.append(vol_id)
            inst["fields"]["attached_volumes"] = attached
            return ""

        # ── detach_volume ──────────────────────────────────────────────
        if t == "detach_volume":
            inst_id = action.instance_name
            vol_id  = action.volume_name
            if inst_id not in state:
                return f"detach_volume: instance '{inst_id}' not in actual_state"
            inst = state[inst_id]
            inst.setdefault("fields", {})
            attached = inst["fields"].get("attached_volumes", [])
            if isinstance(attached, list) and vol_id in attached:
                attached.remove(vol_id)
            inst["fields"]["attached_volumes"] = attached
            return ""

        # ── no_op ──────────────────────────────────────────────────────
        if t == "no_op":
            return ""

        return f"Unknown action_type: {t}"

    # ── _make_obs ──────────────────────────────────────────────────────

    def _make_obs(
        self,
        done: bool,
        metadata: Dict[str, Any],
    ) -> IaCDriftReconcilerObservation:
        return IaCDriftReconcilerObservation(
            actual_state=copy.deepcopy(self.actual_state),
            desired_state=copy.deepcopy(self.desired_state),
            drift_items=list(self._current_drift_items),
            drift_score=self._current_drift_score,
            holy_grail_rules=list(self.guardrail_constraints),
            step_count=self.step_count,
            done=done,
            metadata=metadata,
        )


# Standalone smoke test (python server/IaCDriftReconciler_environment.py)

if __name__ == "__main__":
    import sys

    env = IaCDriftReconcilerEnvironment()
    all_passed = True

    for task_id, actions in [
        (
            "easy",
            [
                IaCDriftReconcilerAction(
                    action_type="update_resource",
                    resource_name="aws_instance.web",
                    attribute="instance_type",
                    new_value="t3.medium",
                ),
                IaCDriftReconcilerAction(
                    action_type="update_resource",
                    resource_name="aws_s3_bucket.logs",
                    attribute="versioning_enabled",
                    new_value="true",
                ),
            ],
        ),
        (
            "medium",
            [
                IaCDriftReconcilerAction(
                    action_type="update_resource",
                    resource_name="aws_security_group.web_sg",
                    attribute="ingress_port_22",
                    new_value="false",
                ),
                IaCDriftReconcilerAction(
                    action_type="update_resource",
                    resource_name="aws_instance.bastion",
                    attribute="managed",
                    new_value="true",
                ),
            ],
        ),
        (
            "hard",
            [
                IaCDriftReconcilerAction(        # fix root dep (no dep)
                    action_type="update_resource",
                    resource_name="aws_subnet.main",
                    attribute="cidr_block",
                    new_value="10.0.1.0/24",
                ),
                IaCDriftReconcilerAction(        # fix root dep (no dep)
                    action_type="update_resource",
                    resource_name="aws_eip.nat",
                    attribute="domain",
                    new_value="vpc",
                ),
                IaCDriftReconcilerAction(        # dep: aws_subnet.main ✓
                    action_type="update_resource",
                    resource_name="aws_route_table.main",
                    attribute="propagating_vgws",
                    new_value="false",
                ),
                IaCDriftReconcilerAction(        # dep: aws_subnet.main ✓, aws_eip.nat ✓
                    action_type="update_resource",
                    resource_name="aws_nat_gateway.main",
                    attribute="connectivity_type",
                    new_value="public",
                ),
                IaCDriftReconcilerAction(        # dep: aws_nat_gateway.main ✓, aws_route_table.main ✓
                    action_type="update_resource",
                    resource_name="aws_route.default",
                    attribute="status",
                    new_value="active",
                ),
            ],
        ),
    ]:
        print(f"\n{'='*60}")
        print(f"  Task: {task_id}")
        print(f"{'='*60}")

        obs = env.reset(task_id=task_id)
        print(f"  reset → drift_score={obs.drift_score:.3f}  "
              f"drift_items={len(obs.drift_items)}")

        # For hard task: verify dependency rejection first
        if task_id == "hard":
            bad = IaCDriftReconcilerAction(
                action_type="update_resource",
                resource_name="aws_nat_gateway.main",
                attribute="connectivity_type",
                new_value="public",
            )
            bad_obs = env.step(bad)
            dep_rejected = bad_obs.metadata.get("last_action_valid") is False
            print(f"  dep-order rejection on nat_gateway before subnet: "
                  f"{'PASS ✓' if dep_rejected else 'FAIL ✗'}")
            if not dep_rejected:
                all_passed = False
            # reset again so the real sequence starts clean
            obs = env.reset(task_id=task_id)

        cumulative_reward = 0.0
        for action in actions:
            result = env.step(action)
            r = result.metadata.get("reward", 0.0)
            cumulative_reward += r
            valid = result.metadata.get("last_action_valid", True)
            print(
                f"  step {result.step_count:2d}  "
                f"action={action.action_type}({action.resource_name or action.instance_name})  "
                f"reward={r:+.3f}  drift={result.drift_score:.3f}  "
                f"done={result.done}  valid={valid}"
            )

        success = result.done and result.drift_score == 0.0  # type: ignore[possibly-undefined]
        print(f"  Episode success: {'PASS ✓' if success else 'FAIL ✗'}  "
              f"total_reward={cumulative_reward:+.3f}")
        if not success:
            all_passed = False

        # Guardrail violation test for medium
        if task_id == "medium":
            obs = env.reset(task_id="medium")
            bad = IaCDriftReconcilerAction(
                action_type="delete_extra_resource",
                resource_name="aws_instance.bastion",
            )
            viol_obs = env.step(bad)
            viol_ok = viol_obs.metadata.get("guardrail_violated") is True
            print(f"  guardrail fires on delete bastion: "
                  f"{'PASS ✓' if viol_ok else 'FAIL ✗'}")
            if not viol_ok:
                all_passed = False

    print(f"\n{'='*60}")
    print(f"  Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print(f"{'='*60}\n")
    sys.exit(0 if all_passed else 1)
