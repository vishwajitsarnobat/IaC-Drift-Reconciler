"""IaC Drift Reconciler Environment Client.

Provides ``IaCDriftReconcilerEnv``, a typed wrapper around ``EnvClient``
that handles serialisation of ``IaCDriftReconcilerAction`` and
deserialisation of ``IaCDriftReconcilerObservation``.

The connection logic, WebSocket handling, and ``from_env`` / ``from_docker_image``
classmethods are inherited unchanged from the base class — they are
environment-agnostic.

Typical usage (async)
----------------------
>>> async with IaCDriftReconcilerEnv(base_url="http://localhost:8000") as env:
...     result = await env.reset(task_id="easy")
...     print(result.observation.drift_score)  # 1.0
...
...     action = IaCDriftReconcilerAction(
...         action_type="update_resource",
...         resource_name="aws_instance.web",
...         attribute="instance_type",
...         new_value="t3.medium",
...     )
...     result = await env.step(action)
...     print(result.reward, result.observation.drift_score)

Sync usage
----------
>>> client = IaCDriftReconcilerEnv(base_url="http://localhost:8000")
>>> with client.sync() as env:
...     result = env.reset(task_id="easy")
...     result = env.step(IaCDriftReconcilerAction(...))

Docker / HF Spaces usage
------------------------
>>> env = await IaCDriftReconcilerEnv.from_env("my-org/iac-drift-reconciler")
>>> try:
...     result = await env.reset(task_id="hard")
...     result = await env.step(IaCDriftReconcilerAction(...))
... finally:
...     await env.close()
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    DriftItem,
    IaCDriftReconcilerAction,
    IaCDriftReconcilerObservation,
)


class IaCDriftReconcilerEnv(
    EnvClient[IaCDriftReconcilerAction, IaCDriftReconcilerObservation, State]
):
    """Typed client for the IaC Drift Reconciler environment.

    Inherits all connection logic, WebSocket handling, ``from_env``,
    ``from_docker_image``, context-manager support, and the ``.sync()``
    wrapper from ``EnvClient``.  Only the three payload methods are
    environment-specific.

    ``reset()`` accepts ``task_id`` as a keyword argument which is forwarded
    to the server via the WebSocket ``reset`` message body.

    Parameters
    ----------
    base_url:
        HTTP base URL of a running environment server, e.g.
        ``"http://localhost:8000"`` or a deployed HF Space URL.
    """

    # reset() — inherited from EnvClient; **kwargs are forwarded as-is.
    # The base class sends:
    #   {"type": "reset", "data": kwargs}
    # so passing task_id="easy" is all that is needed.
    #
    # Signature reminder (for IDE hints):
    #   async def reset(self, *, task_id: str = "easy") -> StepResult[IaCDriftReconcilerObservation]

    # Abstract method implementations

    def _step_payload(self, action: IaCDriftReconcilerAction) -> Dict[str, Any]:
        """Serialise an ``IaCDriftReconcilerAction`` to the JSON body expected
        by ``POST /step`` and the ``"step"`` WebSocket message.

        Every field is included; ``None`` values are omitted so the server
        does not trip over unexpected nulls for action types that don't use
        certain optional fields.
        """
        payload: Dict[str, Any] = {"action_type": action.action_type}

        # Conditionally include optional fields when set
        optional_fields = (
            "resource_name",
            "attribute",
            "new_value",
            "resource_type",
            "properties",
            "instance_name",
            "volume_name",
        )
        for field in optional_fields:
            value = getattr(action, field, None)
            if value is not None:
                payload[field] = value

        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[IaCDriftReconcilerObservation]:
        """Parse the server's JSON response into a typed ``StepResult``.

        The server returns a flat dict with ``observation``, ``reward``,
        and ``done`` at the top level.  The ``observation`` sub-dict maps
        directly to ``IaCDriftReconcilerObservation`` fields.
        """
        obs_data = payload.get("observation", {})

        # Reconstruct DriftItem objects from the raw list
        raw_drift_items = obs_data.get("drift_items", [])
        drift_items = [
            DriftItem(
                resource_id=item["resource_id"],
                field=item["field"],
                desired_value=item["desired_value"],
                actual_value=item["actual_value"],
                severity=item["severity"],
            )
            for item in raw_drift_items
            if isinstance(item, dict)
        ]

        observation = IaCDriftReconcilerObservation(
            actual_state=obs_data.get("actual_state", {}),
            desired_state=obs_data.get("desired_state", {}),
            drift_items=drift_items,
            drift_score=float(obs_data.get("drift_score", 1.0)),
            holy_grail_rules=obs_data.get("holy_grail_rules", []),
            step_count=int(obs_data.get("step_count", 0)),
            done=bool(payload.get("done", obs_data.get("done", False))),
            metadata=obs_data.get("metadata", {}),
        )

        # Reward: prefer top-level field; fall back to observation.metadata["reward"]
        # (the environment stores it there when the WebSocket server doesn't hoist it)
        metadata = obs_data.get("metadata", {})
        raw_reward = payload.get("reward")
        if raw_reward is None:
            raw_reward = metadata.get("reward")

        return StepResult(
            observation=observation,
            reward=raw_reward,
            done=bool(payload.get("done", obs_data.get("done", False))),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse the server's ``/state`` response into a ``State`` object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
        )
