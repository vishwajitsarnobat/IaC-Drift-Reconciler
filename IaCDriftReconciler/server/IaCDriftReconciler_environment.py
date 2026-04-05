"""
Iacdriftreconciler Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.

TODO: Define what the environment is doing
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import IacdriftreconcilerAction, IacdriftreconcilerObservation
except ImportError:
    from models import IacdriftreconcilerAction, IacdriftreconcilerObservation


class IacdriftreconcilerEnvironment(Environment):
    """
    This environment is designed for testing the HTTP server infrastructure.
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the IaCDriftReconciler environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> IacdriftreconcilerObservation:
        """
        Reset the environment.

        Returns:
            IacdriftreconcilerObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return IacdriftreconcilerObservation(
            echoed_message="Iacdriftreconciler environment ready!",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: IacdriftreconcilerAction) -> IacdriftreconcilerObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: IacdriftreconcilerAction containing the message to echo

        Returns:
            IacdriftreconcilerObservation with the echoed message and its length
        """
        self._state.step_count += 1

        message = action.message
        length = len(message)

        # Simple reward: longer messages get higher rewards
        reward = length * 0.1

        return IacdriftreconcilerObservation(
            echoed_message=message,
            message_length=length,
            done=False,
            reward=reward,
            metadata={"original_message": message, "step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
