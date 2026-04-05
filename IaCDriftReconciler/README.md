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

An OpenEnv reinforcement learning environment where an agent learns to reconcile infrastructure drift – transforming a drifted **actual state** into the **desired state** – while strictly respecting **immutable guardrails** (the “Holy Grail”). This is the first open benchmark for safe, sequential infrastructure repair.

## Quick Start

The simplest way to use the environment is through the `IaCDriftReconcilerEnv` class:

```python
from iac_drift_reconciler import IaCDriftReconcilerAction, IaCDriftReconcilerEnv

# Create environment from Docker image (built locally or pulled)
env = IaCDriftReconcilerEnv.from_docker_image("iac-drift-reconciler:latest")

# Reset to a specific task (easy / medium / hard)
result = env.reset(task_id="easy")
print(f"Initial drift score: {result.observation.drift_score}")

# Take a reconciliation action
action = IaCDriftReconcilerAction(
    action_type="update_resource",
    resource_name="aws_instance.web",
    attribute="instance_type",
    new_value="t3.micro"
)
result = env.step(action)
print(f"Reward: {result.reward}, Drift remaining: {result.observation.drift_score}")

# Always clean up
env.close()
```

The client handles container startup, WebSocket connection, and automatic cleanup.

## Building the Docker Image

Before using the environment, build the Docker image:

```bash
# From project root (where server/Dockerfile is located)
docker build -t iac-drift-reconciler:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

Deploy your environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the directory containing openenv.yaml
openenv push

# Or specify a custom repository
openenv push --repo-id your-username/iac-drift-reconciler --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` – Interactive UI to test tasks manually.
- **API Documentation** at `/docs` – Full OpenAPI/Swagger interface.
- **Health Check** at `/health` – Container health monitoring.
- **WebSocket** at `/ws` – Persistent session for low‑latency episodes.

## Environment Details

### Action Space (`IaCDriftReconcilerAction`)

The agent can choose from the following atomic actions (each is a Pydantic model):

| Action Type | Parameters | Description |
|-------------|------------|-------------|
| `update_resource` | `resource_name`, `attribute`, `new_value` | Change an attribute of an existing resource (e.g., instance type). |
| `create_missing_resource` | `resource_type`, `name`, `properties` | Create a resource that exists in desired state but is missing in actual state. |
| `delete_extra_resource` | `resource_name` | Delete a resource that exists in actual state but is not in desired state. |
| `attach_volume` | `instance_name`, `volume_name` | Attach a volume to an instance (respects dependency order). |
| `detach_volume` | `instance_name`, `volume_name` | Detach a volume (if allowed by holy grail rules). |
| `no_op` | – | Do nothing (useful for testing or when agent is stuck). |

> **Placeholder**: The final action set may be extended. See `models.py` for the exact definitions.

### Observation Space (`IaCDriftReconcilerObservation`)

Each step returns an observation containing:

| Field | Type | Description |
|-------|------|-------------|
| `actual_state` | `dict` | Full JSON representation of the current actual infrastructure. |
| `desired_state` | `dict` | The target infrastructure (never changes during an episode). |
| `drift_items` | `list[DriftItem]` | List of detected drifts (resource, attribute, desired value, actual value). |
| `drift_score` | `float` | Normalized measure of remaining drift (0.0 = fully reconciled). |
| `holy_grail_rules` | `list[str]` | Human‑readable list of immutable constraints. |
| `step_count` | `int` | Number of steps taken so far in this episode. |
| `done` | `bool` | True if episode ended (success or violation). |
| `metadata` | `dict` | Additional info (e.g., last action result). |

### Reward Function

The reward at each step is computed as:

```
reward = (old_drift_score - new_drift_score) * α + violation_penalty + success_bonus
```

- **Drift reduction**: Positive reward proportional to how much the drift score decreased.
- **Holy grail violation**: `-1.0` and episode terminates immediately.
- **Success bonus**: `+1.0` when `drift_score == 0.0` (fully reconciled).
- **Optional inefficiency penalty**: Small negative reward per step to encourage shorter sequences.

> **Placeholder**: The exact coefficients and any additional shaping terms will be finalized in `server/ia_cdrift_reconciler_environment.py`.

### Holy Grail Rules (Immutable Guardrails)

These rules are loaded per task and cannot be violated by any action. Example rules:

- `"aws_s3_bucket.data must have block_public_access = true"`
- `"aws_db_instance.main must have backup_retention_days >= 7"`
- `"aws_security_group.web_sg must NOT contain a rule with port 22"`

If an action would break any rule, the environment returns `done=True`, `reward=-1.0`, and the episode ends.

## Tasks and Difficulty Progression

The environment provides three pre‑defined tasks (easy / medium / hard). Each task includes:
- A **desired state** (JSON)
- An **actual state** (JSON, drifted)
- A set of **holy grail rules**

| Task | Description | Key Challenge |
|------|-------------|----------------|
| **Easy** | Two resources have wrong instance sizes. | Simple attribute updates, no dependencies. |
| **Medium** | An extra security group rule was added manually (shadow resource). | Agent must delete the extra rule, not just update. |
| **Hard** | Cascading drift: missing a volume makes the instance size drift unresolvable until the volume is created and attached. | Agent must discover dependency and re‑plan mid‑episode. |

> **Placeholder**: Exact JSON definitions for each task will be placed in `tasks/` directory.

## Real‑World Use Cases (Why This Matters)

A trained agent can be deployed to:

1. **Compliance enforcement** – Automatically fix drift while never violating security or audit rules (e.g., keep S3 buckets private).
2. **Safe database migrations** – Sequence upgrades (create replica → test → promote → delete old) without downtime.
3. **Cascading fix after emergency patching** – Recognise that a manually added firewall rule is legitimate and propose adding it to the desired state instead of deleting it.
4. **Cost‑aware reconciliation** – Choose cheaper sequences (spot instances, off‑peak changes) when multiple options exist.

## Novelty

This environment is the first open‑source RL benchmark for infrastructure drift reconciliation with **immutable guardrails**. Unlike static policy engines or linear remediation scripts, an RL agent must learn to:

- **Sequence actions** safely (e.g., create before delete).
- **Discover hidden dependencies** (cascading drift).
- **Generalise** to unseen drift patterns.
- **Trade off** cost, speed, and safety.

## Development & Testing

### Run Core Environment Logic (without HTTP server)

```bash
python server/ia_cdrift_reconciler_environment.py
```

This runs a quick test of reset/step/reward logic.

### Run the FastAPI Server Locally

```bash
uvicorn server.app:app --reload
```

Then connect using the client with `base_url="http://localhost:8000"`.

## Project Structure (Placeholder)

```
iac-drift-reconciler/
├── .dockerignore
├── README.md                 # This file
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml            # Dependencies (pydantic, fastapi, etc.)
├── models.py                 # Action, Observation, DriftItem models
├── client.py                 # IaCDriftReconcilerEnv client
├── tasks/                    # [TODO] JSON files for easy/medium/hard tasks
├── server/
│   ├── __init__.py
│   ├── ia_cdrift_reconciler_environment.py   # Core environment class
│   ├── app.py                                # FastAPI + WebSocket endpoints
│   └── Dockerfile
└── tests/                    # [TODO] Unit tests for graders and state transitions
```

## Next Steps (To Be Completed)

- [ ] Finalize action space and reward coefficients.
- [ ] Implement the three task JSONs with realistic resource schemas.
- [ ] Write the `openenv.yaml` metadata file.
- [ ] Add baseline inference script (`inference.py`) using OpenAI client.
- [ ] Run `openenv validate` and fix any spec violations.
- [ ] Deploy to Hugging Face Spaces and test with the validation script.

---

For questions or contributions, please refer to the [OpenEnv documentation](https://github.com/meta-pytorch/openenv).