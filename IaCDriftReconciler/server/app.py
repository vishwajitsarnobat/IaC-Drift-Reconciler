"""
FastAPI application for the IaC Drift Reconciler Environment.

The OpenEnv create_app factory generates these endpoints automatically:
    POST /reset          Reset the environment (accepts task_id in body)
    POST /step           Execute an action
    GET  /state          Get current episode state
    GET  /schema         Action / observation JSON schemas
    WS   /ws             Persistent WebSocket session

One custom route is added manually:
    GET  /tasks          List available task IDs (required by the validator)

Usage
-----
Development (auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Direct execution:
    python -m server.app
"""

from __future__ import annotations

from typing import List

# ── OpenEnv factory ────────────────────────────────────────────────────────
try:
    from openenv import create_app  # preferred short import
except ImportError:
    from openenv.core.env_server.http_server import create_app  # type: ignore[no-redef]

# ── Local imports (relative when run as a package, absolute as __main__) ───
try:
    from ..models import IaCDriftReconcilerAction, IaCDriftReconcilerObservation
    from .IaCDriftReconciler_environment import IaCDriftReconcilerEnvironment
except ImportError:
    from models import IaCDriftReconcilerAction, IaCDriftReconcilerObservation  # type: ignore[no-redef]
    from server.IaCDriftReconciler_environment import IaCDriftReconcilerEnvironment  # type: ignore[no-redef]


# ── Create the app via the OpenEnv factory ─────────────────────────────────
app = create_app(
    IaCDriftReconcilerEnvironment,
    IaCDriftReconcilerAction,
    IaCDriftReconcilerObservation,
    env_name="IaCDriftReconciler",
    # Keep at 1 for submission (2 vCPU constraint).
    # Raise to e.g. 4 for parallel inference once tested.
    max_concurrent_envs=1,
)


# ── Custom route: GET /tasks ───────────────────────────────────────────────
@app.get(
    "/tasks",
    summary="List available task IDs",
    response_model=List[str],
    tags=["tasks"],
)
async def list_tasks() -> List[str]:
    """Return the list of task IDs that can be passed to ``/reset``.

    The environment currently ships three tasks of increasing difficulty:
    * ``"easy"``   — two independent attribute fixes, no guardrail danger.
    * ``"medium"`` — guardrail urgency + import-vs-delete decision.
    * ``"hard"``   — five-resource dependency chain; order enforcement.
    """
    return ["easy", "medium", "hard"]


# ── Entry point for direct execution ──────────────────────────────────────
def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the uvicorn server programmatically.

    Equivalent CLI commands::

        uv run --project . server
        uvicorn server.app:app --host 0.0.0.0 --port 8000
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IaC Drift Reconciler server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # main() called here — required by openenv validate
    main(host=args.host, port=args.port)
