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
# <!-- CODE PLACEHOLDER -->
# Minimal working example demonstrating:
#   - IaCDriftReconcilerEnv.from_docker_image()
#   - env.reset(task_id=...)
#   - env.step(IaCDriftReconcilerAction(...))
#   - env.close()
# Source: client.py
# Status: pending finalization of client API and Docker image name.
```

That's it! The `IaCDriftReconcilerEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# <!-- CODE PLACEHOLDER -->
# docker build command with the correct -f path and image tag.
# Source: server/Dockerfile
# Status: pending finalization of Dockerfile.
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

```python
# <!-- CODE PLACEHOLDER -->
# Full IaCDriftReconcilerAction Pydantic model definition.
# Source: models.py
# Status: pending finalization of action set.
```

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

```python
# <!-- CODE PLACEHOLDER -->
# Full IaCDriftReconcilerObservation and DriftItem Pydantic model definitions.
# Source: models.py
# Status: pending finalization of observation schema.
```

### Reward

The reward at each step is computed as:

```
reward = (old_drift_score − new_drift_score) × α  +  violation_penalty  +  success_bonus
```

| Term | Value | Condition |
|------|-------|-----------|
| Drift reduction | `(old_drift_score − new_drift_score) × α` | Positive when drift decreases. `α` is a scaling coefficient. |
| Guardrail violation | `-1.0` | Any action that violates a guardrail constraint; episode terminates immediately. |
| Success bonus | `+1.0` | Awarded when `drift_score == 0.0` (fully reconciled). |
| Inefficiency penalty | Small negative per step | Optional; encourages shorter action sequences. |

```python
# <!-- CODE PLACEHOLDER -->
# Full reward computation implementation.
# Source: server/ia_cdrift_reconciler_environment.py
# Status: coefficients (α and inefficiency penalty weight) pending finalization.
```

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
| **Easy** | Two resources have the wrong instance sizes. | Simple attribute updates with no resource dependencies. |
| **Medium** | A security group rule was added manually outside Terraform (shadow resource). | Agent must decide whether to delete or import the unmanaged rule; the incorrect choice violates a guardrail constraint. |
| **Hard** | Cascading drift: a missing EBS volume makes the instance-type drift unresolvable until the volume is created and attached first. | Agent must discover the dependency, sequence fixes correctly, and re-plan mid-episode. |

```jsonc
// <!-- CODE PLACEHOLDER -->
// Task JSON definitions for easy, medium, and hard scenarios.
// Source: tasks/task_easy.json, tasks/task_medium.json, tasks/task_hard.json
// Status: pending finalization of resource schemas.
```

## Advanced Usage

### Connecting to an Existing Server

If you already have an IaC Drift Reconciler environment server running, you can connect directly:

```python
# <!-- CODE PLACEHOLDER -->
# Example showing IaCDriftReconcilerEnv(base_url="...") direct connection.
# Note: env.close() will NOT stop the server when connecting this way.
# Source: client.py
# Status: pending finalization of client API.
```

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
# <!-- CODE PLACEHOLDER -->
# Example showing `with IaCDriftReconcilerEnv(...) as env:` pattern.
# Demonstrate reset() + multiple step() calls inside the context.
# Source: client.py
# Status: pending finalization of client API.
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
# <!-- CODE PLACEHOLDER -->
# Concurrent episode example using ThreadPoolExecutor.
# Source: client.py
# Status: pending finalization of client API.
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
python3 server/ia_cdrift_reconciler_environment.py
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

```
# <!-- CODE PLACEHOLDER -->
# Baseline scores produced by inference.py against all three tasks.
# Format: task_id | model | score | steps_taken
# Status: pending completion of inference.py and HF Space deployment.
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
├── Dockerfile                              # Container image definition
└── server/
    ├── __init__.py                         # Server module exports
    ├── ia_cdrift_reconciler_environment.py # Core environment: step / reset / state / reward
    └── app.py                              # FastAPI + WebSocket endpoints
```