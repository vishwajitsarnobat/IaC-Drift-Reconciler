---
title: IaC Drift Reconciler
emoji: 🥇
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# IaC Drift Reconciler Environment

An [OpenEnv](https://github.com/meta-pytorch/openenv)-compatible reinforcement learning environment where an agent learns to reconcile infrastructure drift by transforming a drifted **actual state** into a **desired state**, while strictly respecting a set of **immutable guardrail constraints**. This is the first open benchmark for safe, sequential infrastructure repair.

## Quick Start

The simplest way to use the IaC Drift Reconciler environment is through the `IaCDriftReconcilerEnv` class:

```python
import asyncio
from IaCDriftReconciler.client import IaCDriftReconcilerEnv
from IaCDriftReconciler.models import IaCDriftReconcilerAction

async def run():
    # from_docker_image builds and runs the container automatically!
    async with IaCDriftReconcilerEnv.from_docker_image("iac-drift-reconciler") as env:
        result = await env.reset(task_id="easy")
        print(f"Initial drift score: {result.observation.drift_score}")
        
        step_result = await env.step(IaCDriftReconcilerAction(
            action_type="no_op"
        ))
        print(f"Reward: {step_result.reward}")

asyncio.run(run())
```

That's it! The `IaCDriftReconcilerEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
docker build -t iac-drift-reconciler .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for a Hugging Face Docker Space (enables the web interface)
3. Upload to Hugging Face (prompts for login if not already authenticated)

### Prerequisites

- Authenticate with Hugging Face: the command will prompt for login if not already authenticated.
- Set the required environment variables in your Space settings:

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | The API endpoint for LLM inference (e.g., `https://api.openai.com/v1`). |
| `MODEL_NAME` | The model identifier to use (e.g., `gpt-4o`). |
| `HF_TOKEN` | Your Hugging Face API token. |

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory).
- `--repo-id`, `-r`: Repository ID in format `username/repo-name` (defaults to `username/env-name` from `openenv.yaml`).
- `--base-image`, `-b`: Base Docker image to use (overrides the Dockerfile `FROM`).
- `--private`: Deploy the Space as private (default: public).

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/iac-drift-reconciler

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private Space
openenv push --private

# Combine options
openenv push --repo-id my-org/iac-drift-reconciler --base-image custom-base:latest --private
```

After deployment, your Space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed Space includes:
- **Web Interface** at `/web`: interactive UI for exploring and manually testing tasks.
- **API Documentation** at `/docs`: full OpenAPI / Swagger interface.
- **Health Check** at `/health`: container health monitoring.
- **WebSocket** at `/ws`: persistent session endpoint for low-latency episodes.

## Environment Details

### Action

**`IaCDriftReconcilerAction`**: the atomic operation the agent submits at each step.

| Action Type | Parameters | Description |
|-------------|------------|-------------|
| `update_resource` | `resource_name`, `attribute`, `new_value` | Change an attribute of an existing resource (e.g., instance type). |
| `create_missing_resource` | `resource_type`, `name`, `properties` | Create a resource present in the desired state but absent from the actual state. |
| `delete_extra_resource` | `resource_name` | Delete a resource present in the actual state but absent from the desired state. |
| `attach_volume` | `instance_name`, `volume_name` | Attach a volume to an instance, respecting dependency order. |
| `detach_volume` | `instance_name`, `volume_name` | Detach a volume, subject to guardrail constraints. |
| `no_op` | N/A | Take no action. Useful when the agent requires more context before committing. |

> Note: See `models.py` for the complete Pydantic definitions and nested structures.

### Observation

**`IaCDriftReconcilerObservation`**: returned after every `reset()` and `step()` call.

| Field | Type | Description |
|-------|------|-------------|
| `actual_state` | `dict` | Full JSON representation of the current actual infrastructure. |
| `desired_state` | `dict` | The target infrastructure. Does not change during an episode. |
| `drift_items` | `list[DriftItem]` | Detected drifts, where each entry records the resource, attribute, desired value, and actual value. |
| `drift_score` | `float` | Normalised measure of remaining drift. `0.0` means fully reconciled. |
| `holy_grail_rules` | `list[str]` | Human-readable list of immutable constraints in scope for this task. |
| `step_count` | `int` | Number of steps taken so far in this episode. |
| `done` | `bool` | `True` if the episode has ended by success, violation, or max steps. |
| `metadata` | `dict` | Additional diagnostic info such as the last action result. |

> Note: See `models.py` for the complete Pydantic definitions and nested structures.

### Reward

The reward at each step is computed as:

```
drift_score = remaining_drift_items / drift_items_at_reset      (capped at 1.0)
reward      = (old_drift_score − new_drift_score)  +  success_bonus
```

| Term | Value | Condition |
|------|-------|-----------|
| Drift reduction | `old_drift_score − new_drift_score` | Positive when drift decreases (e.g. `+0.2` per fix on the 5-item hard task). |
| Success bonus | `+1.0` | Awarded once, on the step where `drift_score` reaches `0.0`. |
| Guardrail violation | `-1.0` | The action would break a guardrail; it is not applied and the episode ends. |
| Rejected action | `-0.05` | Missing fields, unknown resource, or dependencies not yet reconciled. The step is still used up. |

A perfect episode therefore earns a total of `2.0`. Episodes end on success, on a guardrail violation, or after 30 steps.

> Note: See `server/IaCDriftReconciler_environment.py` for the exact reward calculation logic.

### Guardrail Constraints

Guardrail constraints are loaded per task and represent non-negotiable organisational policy. Any action that violates a constraint immediately terminates the episode with `reward = -1.0`. The full constraint set is provided to the agent in every observation via `holy_grail_rules`.

Example rules:

```
"aws_s3_bucket.data must have block_public_access = true"
"aws_db_instance.main must have backup_retention_days >= 7"
"aws_security_group.web_sg must NOT contain a rule with port 22"
```

### Tasks

The environment ships with three pre-defined tasks of increasing difficulty. Each task specifies a desired state, a drifted actual state, and a set of guardrail constraints.

| Task | Description | Key Challenge |
|------|-------------|---------------|
| **Easy** | An instance has the wrong type and an S3 bucket has versioning off. | Simple attribute updates with no resource dependencies. |
| **Medium** | SSH (port 22) is open on a security group, and a bastion instance exists but is not managed by Terraform (shadow resource). | Agent must close the port and decide whether to delete or import the bastion; deleting it violates a guardrail, importing it (`managed = true`) is correct. |
| **Hard** | Five drifted network resources form a dependency chain (subnet, EIP → route table, NAT gateway → route). | Agent must fix resources in dependency order; acting on a resource before its dependencies are reconciled is rejected. |

> Note: See the JSON files located inside `tasks/` directory for full representation of the task resources.

## Advanced Usage

### Connecting to an Existing Server

If you already have an IaC Drift Reconciler environment server running, you can connect directly:

```python
import asyncio
from IaCDriftReconciler.client import IaCDriftReconcilerEnv

async def connect():
    env = IaCDriftReconcilerEnv(base_url="http://localhost:8000")
    result = await env.reset(task_id="easy")
    print("Connected!", result.observation.drift_score)

asyncio.run(connect())
```

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
import asyncio
from IaCDriftReconciler.client import IaCDriftReconcilerEnv

async def run():
    async with IaCDriftReconcilerEnv(base_url="http://localhost:8000") as env:
        res = await env.reset(task_id="hard")
        # step loops here...
        pass
```

The client uses WebSocket connections for:
- **Lower latency**: no HTTP connection overhead per request.
- **Persistent session**: the server maintains your environment state across steps.
- **Efficient for episodes**: better performance for many sequential steps.

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this, modify `server/app.py` to use factory mode:

```python
# In server/app.py, use factory mode for concurrent sessions
app = create_app(
    IaCDriftReconcilerEnvironment,  # Pass class, not instance
    IaCDriftReconcilerAction,
    IaCDriftReconcilerObservation,
    max_concurrent_envs=4,          # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
# Setup 4 background tasks handling tasks async
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
python3 server/IaCDriftReconciler_environment.py
```

This verifies that:
- Environment resets correctly for all three tasks.
- `step()` executes actions and updates the actual state properly.
- Guardrail violation detection terminates episodes correctly.
- Drift score and reward are calculated correctly at each step.

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

Then connect using the client with `base_url="http://localhost:8000"`.

### Running the Baseline Inference Script

```bash
# Set required environment variables, then:
python inference.py
```

```text
============================================================
  task       model                           score  steps
============================================================
  easy       Qwen/Qwen2.5-72B-Instruct       1.000      2
  medium     Qwen/Qwen2.5-72B-Instruct       1.000      2
  hard       Qwen/Qwen2.5-72B-Instruct       1.000      7
============================================================
```

## Project Structure

```
iac-drift-reconciler/
├── .dockerignore                           # Docker build exclusions
├── __init__.py                             # Module exports
├── README.md                               # This file
├── openenv.yaml                            # OpenEnv manifest
├── pyproject.toml                          # Project metadata and dependencies
├── uv.lock                                 # Locked dependencies (generated)
├── inference.py                            # Baseline inference script (OpenAI client)
├── client.py                               # IaCDriftReconcilerEnv client
├── models.py                               # Action, Observation, DriftItem models
├── tasks/
│   ├── task_easy.json                      # Desired + actual state + guardrail constraints
│   ├── task_medium.json
│   └── task_hard.json
├── tests/                                  # pytest suite (no network, fake LLM)
├── Dockerfile                              # Container image definition
└── server/
    ├── __init__.py                         # Server module exports
    ├── IaCDriftReconciler_environment.py   # Core environment: step / reset / state / reward
    └── app.py                              # FastAPI + WebSocket endpoints
```

## Testing

The test suite runs entirely in-process: no Docker, no network and no real LLM.

```bash
uv sync --extra dev   # installs pytest and pytest-cov into .venv
uv run pytest         # run from this directory
```

| File | What it checks |
|------|----------------|
| `tests/test_environment.py` | Every task is solved by a scripted optimal action sequence (drift 0, success, no violation); reward sign and size; guardrail termination with `-1`; hard-task dependency order (reverse order never succeeds); reset isolation; max-steps termination. |
| `tests/test_bugfixes.py` | One regression test per bug that was found and fixed (see the commit history). |
| `tests/test_inference.py` | The agent loop with a fake LLM that returns good JSON, fenced JSON, prose-wrapped JSON and garbage. The fake environment sends observations through the same serialiser the real server uses, so tests see exactly what a deployed client sees. |
| `tests/test_validate_and_server.py` | `validate_tasks.py` passes on the shipped tasks; the FastAPI app answers `/health` and `/tasks` and solves the easy task over WebSocket. |

## Design decisions

- **Guardrails are checked before an action is applied.** The environment applies the action to a deep copy of the state, checks the rules there, and only then touches the real state. A violating action is never applied; the episode ends with `-1`.
- **Only the action that causes a violation is blamed.** A rule fires only if the action targets the guarded resource *and* changes the guarded value (or deletes the resource). A rule that is already broken at reset counts as drift to fix, not as a mistake by the agent.
- **Dependencies are enforced, not just scored.** In the hard task, acting on a resource whose dependencies are not yet reconciled is rejected with a small penalty, so a wrong order cannot reach success.
- **Every action uses a step, including rejected ones.** An agent stuck emitting bad actions still hits the 30-step limit, and stepping after the episode has ended changes nothing and earns nothing.
- **`new_value` is always a string** and is converted to the type of the desired value. Only `"true"`/`"false"` become booleans, so a value like `"no"` cannot accidentally count as `true`.
- **Observing the agent.** `inference.py` prints the evaluator's `[START]/[STEP]/[END]` lines on stdout, and a `[SUMMARY]` line per episode on stderr with steps, final drift, guardrail violations, rejected actions, parse errors and LLM errors. The OpenEnv server does not send `observation.metadata` to clients, so the loop works out each step's outcome from the reward (`-1.0` and done means a violation, `-0.05` means a rejected action).
- **Scoring.** Task score is `1 − final drift_score`, clamped to the open interval (0, 1) as the hackathon validator requires. An episode that ends on a guardrail violation gets the minimum score, and success means drift reached 0 with no violation.
- **Bounded prompts.** Each prompt already contains the full current drift list, so only the last 4 turns of history (`HISTORY_TURNS`) are sent to the model. The previous step's error is included in the next prompt so the model can correct itself.
