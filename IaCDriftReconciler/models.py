"""
Data models for the IaC Drift Reconciler Environment.

Defines four models in dependency order:
  1. DriftItem                   — a single detected drift between actual and desired state
  2. IaCDriftReconcilerAction    — the atomic operation an agent submits at each step
  3. IaCDriftReconcilerObservation — the full observation returned after reset() / step()
  4. IaCDriftReconcilerReward    — structured reward signal wrapping a scalar value

All models inherit from the appropriate OpenEnv base class so that the HTTP
server (server/app.py) can serialise/deserialise them automatically.

Design notes
------------
* `dict` is never used as a field type — always `Dict[str, Any]` so Pydantic
  can validate and serialise the generic form correctly.
* IaCDriftReconcilerAction is deliberately permissive: the environment layer is
  responsible for validating which optional fields are required for each
  action_type.
* drift_score must be in [0.0, 1.0]; 0.0 means the episode is fully reconciled.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator


# 1. DriftItem

class DriftItem(BaseModel):
    """
    A single detected drift between the actual and desired infrastructure state.

    severity lets the reward function weight fixes differently — resolving a
    'critical' drift should yield a larger positive signal than resolving a
    'low' one.
    """

    resource_id: str = Field(
        ...,
        description="Unique identifier of the drifted resource (e.g. 'aws_instance.web').",
    )
    field: str = Field(
        ...,
        description="The attribute / field path that has drifted (e.g. 'instance_type').",
    )
    desired_value: Any = Field(
        ...,
        description="The value that the desired (Terraform) state specifies.",
    )
    actual_value: Any = Field(
        ...,
        description="The value currently observed in the actual infrastructure state.",
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description=(
            "Severity tier of this drift item. "
            "Used by the reward function to weight fixes: "
            "critical > high > medium > low."
        ),
    )


# 2. IaCDriftReconcilerAction

class IaCDriftReconcilerAction(Action):
    """
    The atomic operation the agent submits at each environment step.

    action_type selects which operation to perform; the remaining fields are
    all Optional because different action types require different subsets of
    them.  The environment validates that the correct fields are populated for
    the chosen action_type — the model itself stays permissive so that the
    HTTP layer never rejects a structurally-valid but semantically-incomplete
    payload before the environment has a chance to explain the error.

    Action types
    ------------
    update_resource        : Change an attribute of an existing resource.
                             Requires: resource_name, attribute, new_value.
    create_missing_resource: Create a resource present in desired but absent
                             from actual state.
                             Requires: resource_type, resource_name, properties.
    delete_extra_resource  : Delete a resource present in actual but absent
                             from desired state.
                             Requires: resource_name.
    attach_volume          : Attach a volume to an instance (dependency-ordered).
                             Requires: instance_name, volume_name.
    detach_volume          : Detach a volume, subject to guardrail constraints.
                             Requires: instance_name, volume_name.
    no_op                  : Take no action this step.
                             Requires: nothing.
    """

    action_type: Literal[
        "update_resource",
        "create_missing_resource",
        "delete_extra_resource",
        "attach_volume",
        "detach_volume",
        "no_op",
    ] = Field(
        ...,
        description="The type of infrastructure operation to perform.",
    )

    # --- update_resource / delete_extra_resource ---
    resource_name: Optional[str] = Field(
        default=None,
        description=(
            "Name of the target resource. "
            "Required for: update_resource, create_missing_resource, delete_extra_resource."
        ),
    )

    # --- update_resource ---
    attribute: Optional[str] = Field(
        default=None,
        description="The attribute to modify. Required for: update_resource.",
    )
    new_value: Optional[str] = Field(
        default=None,
        description="The new value to assign to the attribute. Required for: update_resource.",
    )

    # --- create_missing_resource ---
    resource_type: Optional[str] = Field(
        default=None,
        description=(
            "Terraform resource type of the resource to create "
            "(e.g. 'aws_instance'). Required for: create_missing_resource."
        ),
    )
    properties: Optional[str] = Field(
        default=None,
        description=(
            "JSON-encoded string of resource properties for creation. "
            "Required for: create_missing_resource."
        ),
    )

    # --- attach_volume / detach_volume ---
    instance_name: Optional[str] = Field(
        default=None,
        description="Name of the EC2 instance. Required for: attach_volume, detach_volume.",
    )
    volume_name: Optional[str] = Field(
        default=None,
        description="Name of the EBS volume. Required for: attach_volume, detach_volume.",
    )


# 3. IaCDriftReconcilerObservation

class IaCDriftReconcilerObservation(Observation):
    """
    The full observation returned after every reset() and step() call.

    drift_score is the canonical progress signal read by the reward function
    between consecutive steps: it is always present, always a float in
    [0.0, 1.0], where 0.0 means the episode is fully reconciled.
    """

    # Current snapshot of the real infrastructure.
    actual_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full JSON representation of the current actual infrastructure state.",
    )

    # Target snapshot — does NOT change during an episode.
    desired_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="The target infrastructure state. Constant for the duration of an episode.",
    )

    # Structured list of remaining drifts.
    drift_items: List[DriftItem] = Field(
        default_factory=list,
        description=(
            "Detected drifts; each entry records the resource, field, "
            "desired value, actual value, and severity."
        ),
    )

    # Normalised progress metric.  Must stay in [0.0, 1.0].
    drift_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Normalised measure of remaining drift. "
            "0.0 means fully reconciled; 1.0 means maximum drift. "
            "The reward function reads this value between consecutive steps."
        ),
    )

    # Human-readable guardrail constraints in scope for this task.
    holy_grail_rules: List[str] = Field(
        default_factory=list,
        description=(
            "Immutable policy constraints for this task. "
            "Any action that violates one terminates the episode immediately with reward = -1.0."
        ),
    )

    # Housekeeping.
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken so far in this episode.",
    )
    done: bool = Field(
        default=False,
        description="True if the episode has ended (success, guardrail violation, or max steps).",
    )

    # Flexible diagnostic payload (action results, error messages, etc.).
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic info, e.g. the result of the last action.",
    )

    @field_validator("drift_score")
    @classmethod
    def drift_score_must_be_normalised(cls, v: float) -> float:  # noqa: N805
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"drift_score must be in [0.0, 1.0], got {v!r}")
        return v


# 4. IaCDriftReconcilerReward

class IaCDriftReconcilerReward(BaseModel):
    """
    Structured reward signal wrapping a scalar value.

    Wrapping the reward in a model is an OpenEnv spec requirement and makes
    it straightforward for logging pipelines to decompose the signal into its
    constituent parts without re-parsing episode logs.

    The scalar `value` is computed by the environment as:
        value = (old_drift_score − new_drift_score) × α
                + guardrail_violation_penalty          # -1.0 if violated, else 0.0
                + success_bonus                        # +1.0 if drift_score == 0.0
                + inefficiency_penalty_per_step        # small negative, optional
    """

    value: float = Field(
        ...,
        description="The scalar reward signal for the current step.",
    )
    drift_resolved: int = Field(
        ...,
        ge=0,
        description="Number of DriftItems resolved (drift_score reached 0 for) this step.",
    )
    drift_total: int = Field(
        ...,
        ge=0,
        description="Total number of DriftItems at the start of the episode.",
    )
    guardrail_violated: bool = Field(
        ...,
        description="True if the last action violated a guardrail constraint.",
    )
    done: bool = Field(
        ...,
        description=(
            "True if the episode has ended — mirrors "
            "IaCDriftReconcilerObservation.done for convenience."
        ),
    )
