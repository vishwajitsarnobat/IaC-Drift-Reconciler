#!/usr/bin/env python3
"""
Standalone validator for IaC Drift Reconciler task JSON files.

Run from anywhere:
    python validate_tasks.py

Checks performed per task file:
  1. File exists and is valid JSON.
  2. Top-level keys are exactly: desired_state, actual_state, guardrail_constraints.
  3. Every resource in both states conforms to the resource schema.
  4. Every dependency referenced in a resource exists in that state dict.
  5. No circular dependencies exist (topological sort check).
  6. Every guardrail string parses as '<path> <operator> <value>'.
  7. The resource_id in each guardrail exists in desired_state.
  8. Desired vs actual: every resource in desired_state exists in actual_state
     (or does not — both are valid; we just report the diff for review).
  9. Each resource's fields dict has at least one key.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ── locate tasks dir relative to this script ─────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
TASKS_DIR = SCRIPT_DIR / "tasks"
TASK_FILES = ["task_easy.json", "task_medium.json", "task_hard.json"]

REQUIRED_TOP_LEVEL = {"desired_state", "actual_state", "guardrail_constraints"}
REQUIRED_RESOURCE_KEYS = {"resource_type", "fields", "managed", "dependencies"}
VALID_OPERATORS = {"==", "!=", ">", ">=", "<", "<="}

ERRORS: list[str] = []
WARNINGS: list[str] = []


# ── helpers ───────────────────────────────────────────────────────────────────

def err(task: str, msg: str) -> None:
    ERRORS.append(f"[{task}] ERROR: {msg}")


def warn(task: str, msg: str) -> None:
    WARNINGS.append(f"[{task}] WARN:  {msg}")


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


# ── check 1: valid JSON + top-level keys ─────────────────────────────────────

def load_json(path: Path, task: str) -> dict | None:
    if not path.exists():
        err(task, f"File not found: {path}")
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        err(task, f"Invalid JSON: {e}")
        return None
    ok(f"Valid JSON ({path.stat().st_size} bytes)")
    return data


def check_top_level(data: dict, task: str) -> bool:
    keys = set(data.keys())
    missing = REQUIRED_TOP_LEVEL - keys
    extra = keys - REQUIRED_TOP_LEVEL
    valid = True
    if missing:
        err(task, f"Missing top-level keys: {missing}")
        valid = False
    if extra:
        err(task, f"Unexpected top-level keys: {extra}")
        valid = False
    if valid:
        ok("Top-level keys: desired_state, actual_state, guardrail_constraints")
    return valid


# ── check 2: resource schema ──────────────────────────────────────────────────

def check_resource_schema(state: dict, state_name: str, task: str) -> bool:
    if not isinstance(state, dict):
        err(task, f"{state_name} must be a dict, got {type(state).__name__}")
        return False
    if len(state) == 0:
        err(task, f"{state_name} is empty")
        return False

    valid = True
    for res_id, res in state.items():
        if not isinstance(res, dict):
            err(task, f"{state_name}.{res_id}: resource must be a dict")
            valid = False
            continue

        missing = REQUIRED_RESOURCE_KEYS - set(res.keys())
        if missing:
            err(task, f"{state_name}.{res_id}: missing keys {missing}")
            valid = False

        # resource_type
        if "resource_type" in res and not isinstance(res["resource_type"], str):
            err(task, f"{state_name}.{res_id}.resource_type must be a string")
            valid = False

        # fields
        if "fields" in res:
            if not isinstance(res["fields"], dict):
                err(task, f"{state_name}.{res_id}.fields must be a dict")
                valid = False
            elif len(res["fields"]) == 0:
                warn(task, f"{state_name}.{res_id}.fields is empty")

        # managed
        if "managed" in res and not isinstance(res["managed"], bool):
            err(task, f"{state_name}.{res_id}.managed must be a boolean")
            valid = False

        # dependencies
        if "dependencies" in res:
            if not isinstance(res["dependencies"], list):
                err(task, f"{state_name}.{res_id}.dependencies must be a list")
                valid = False
            elif not all(isinstance(d, str) for d in res["dependencies"]):
                err(task, f"{state_name}.{res_id}.dependencies must be a list of strings")
                valid = False

    if valid:
        ok(f"{state_name}: {len(state)} resources all conform to schema")
    return valid


# ── check 3: dependency references ───────────────────────────────────────────

def check_dependency_refs(state: dict, state_name: str, task: str) -> bool:
    valid = True
    for res_id, res in state.items():
        for dep in res.get("dependencies", []):
            if dep not in state:
                err(task, f"{state_name}.{res_id}: dependency '{dep}' not found in {state_name}")
                valid = False
    if valid:
        ok(f"{state_name}: all dependency references resolve")
    return valid


# ── check 4: no circular dependencies (toposort) ─────────────────────────────

def check_no_cycles(state: dict, state_name: str, task: str) -> bool:
    visited: set[str] = set()
    in_progress: set[str] = set()

    def dfs(node: str) -> bool:
        if node in in_progress:
            return False  # cycle
        if node in visited:
            return True
        in_progress.add(node)
        for dep in state.get(node, {}).get("dependencies", []):
            if not dfs(dep):
                return False
        in_progress.discard(node)
        visited.add(node)
        return True

    for res_id in state:
        if not dfs(res_id):
            err(task, f"{state_name}: circular dependency detected involving '{res_id}'")
            return False

    ok(f"{state_name}: no circular dependencies")
    return True


# ── check 5: guardrail format + resource exists in desired_state ──────────────

def parse_guardrail(constraint: str) -> tuple[str, str, str, str] | None:
    """
    Returns (resource_id, field_path, operator, value) or None on parse failure.
    Resource IDs always have the form '<type>.<name>' (exactly one dot in the type+name pair).
    Full path: '<type>.<name>.<field_path...>'
    We split on spaces first to isolate the operator, then parse the LHS path.
    """
    parts = constraint.strip().split()
    if len(parts) != 3:
        return None
    lhs, operator, value = parts
    if operator not in VALID_OPERATORS:
        return None
    # LHS: first two dot-segments form the resource_id
    dot_parts = lhs.split(".")
    if len(dot_parts) < 3:
        return None
    resource_id = ".".join(dot_parts[:2])
    field_path = ".".join(dot_parts[2:])
    return resource_id, field_path, operator, value


def check_guardrails(constraints: Any, desired: dict, task: str) -> bool:
    if not isinstance(constraints, list):
        err(task, "guardrail_constraints must be a list")
        return False
    if len(constraints) == 0:
        warn(task, "guardrail_constraints is empty")
        return True

    valid = True
    for i, c in enumerate(constraints):
        if not isinstance(c, str):
            err(task, f"guardrail_constraints[{i}] must be a string, got {type(c).__name__}")
            valid = False
            continue

        parsed = parse_guardrail(c)
        if parsed is None:
            err(
                task,
                f"guardrail_constraints[{i}] could not be parsed: {c!r}\n"
                f"    Expected: '<resource_type>.<resource_name>.<field_path> <operator> <value>'\n"
                f"    Valid operators: {VALID_OPERATORS}",
            )
            valid = False
            continue

        resource_id, field_path, operator, value = parsed
        if resource_id not in desired:
            warn(
                task,
                f"guardrail_constraints[{i}]: resource_id '{resource_id}' not found "
                f"in desired_state (may be intentional for resources absent from desired)",
            )

    if valid:
        ok(f"guardrail_constraints: {len(constraints)} rules parse successfully")
    return valid


# ── check 6: desired vs actual diff summary ───────────────────────────────────

def report_state_diff(desired: dict, actual: dict, task: str) -> None:
    desired_ids = set(desired.keys())
    actual_ids = set(actual.keys())

    only_desired = desired_ids - actual_ids
    only_actual = actual_ids - desired_ids
    in_both = desired_ids & actual_ids

    if only_desired:
        warn(task, f"Resources in desired_state but NOT in actual_state: {only_desired}")
    if only_actual:
        warn(task, f"Resources in actual_state but NOT in desired_state: {only_actual}")

    drifted = []
    for res_id in in_both:
        d_fields = desired[res_id].get("fields", {})
        a_fields = actual[res_id].get("fields", {})
        d_managed = desired[res_id].get("managed")
        a_managed = actual[res_id].get("managed")

        diffed_fields = {k for k in d_fields if d_fields.get(k) != a_fields.get(k)}
        managed_drifted = d_managed != a_managed

        if diffed_fields or managed_drifted:
            drift_summary = []
            for k in sorted(diffed_fields):
                drift_summary.append(f"fields.{k}: {a_fields.get(k)!r} → {d_fields.get(k)!r}")
            if managed_drifted:
                drift_summary.append(f"managed: {a_managed!r} → {d_managed!r}")
            drifted.append(f"  {res_id}: " + ", ".join(drift_summary))

    if drifted:
        ok(f"Drift detected in {len(drifted)} resource(s):")
        for line in drifted:
            print(f"      {line}")
    else:
        warn(task, "No drift detected between desired_state and actual_state — task would be trivially solved.")


# ── main ──────────────────────────────────────────────────────────────────────

def validate_task(filename: str) -> None:
    task = filename.replace(".json", "")
    path = TASKS_DIR / filename

    print(f"\n{'='*60}")
    print(f"  Validating {filename}")
    print(f"{'='*60}")

    data = load_json(path, task)
    if data is None:
        return

    if not check_top_level(data, task):
        return  # can't proceed without the right keys

    desired = data.get("desired_state", {})
    actual = data.get("actual_state", {})
    constraints = data.get("guardrail_constraints", [])

    check_resource_schema(desired, "desired_state", task)
    check_resource_schema(actual, "actual_state", task)
    check_dependency_refs(desired, "desired_state", task)
    check_dependency_refs(actual, "actual_state", task)
    check_no_cycles(desired, "desired_state", task)
    check_no_cycles(actual, "actual_state", task)
    check_guardrails(constraints, desired, task)
    report_state_diff(desired, actual, task)


def main() -> None:
    print(f"\nIaC Drift Reconciler — Task JSON Validator")
    print(f"Tasks directory: {TASKS_DIR}")

    if not TASKS_DIR.exists():
        print(f"\nFATAL: tasks directory not found at {TASKS_DIR}", file=sys.stderr)
        sys.exit(1)

    for filename in TASK_FILES:
        validate_task(filename)

    print(f"\n{'='*60}")
    if WARNINGS:
        print(f"\n⚠  {len(WARNINGS)} warning(s):")
        for w in WARNINGS:
            print(f"   {w}")

    if ERRORS:
        print(f"\n✗  {len(ERRORS)} error(s):")
        for e in ERRORS:
            print(f"   {e}")
        print()
        sys.exit(1)
    else:
        print(f"\n✓  All task files passed validation.")
        if WARNINGS:
            print("   (review warnings above before proceeding)")
        print()


if __name__ == "__main__":
    main()
