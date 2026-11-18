"""Shared fixtures and helpers for the IaC Drift Reconciler test suite.

Everything runs in-process: no Docker, no network, no real LLM.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from IaCDriftReconciler.models import IaCDriftReconcilerAction
from IaCDriftReconciler.server.IaCDriftReconciler_environment import (
    IaCDriftReconcilerEnvironment,
)


def update(resource: str, attribute: str, value: str) -> IaCDriftReconcilerAction:
    return IaCDriftReconcilerAction(
        action_type="update_resource",
        resource_name=resource,
        attribute=attribute,
        new_value=value,
    )


# Scripted optimal action sequences for each shipped task.
OPTIMAL: Dict[str, List[IaCDriftReconcilerAction]] = {
    "easy": [
        update("aws_instance.web", "instance_type", "t3.medium"),
        update("aws_s3_bucket.logs", "versioning_enabled", "true"),
    ],
    "medium": [
        # Close SSH on the security group, then IMPORT the unmanaged bastion
        # (managed=true) instead of deleting it.
        update("aws_security_group.web_sg", "ingress_port_22", "false"),
        update("aws_instance.bastion", "managed", "true"),
    ],
    "hard": [
        # Roots first, then dependents: subnet, eip -> route_table, nat -> route.
        update("aws_subnet.main", "cidr_block", "10.0.1.0/24"),
        update("aws_eip.nat", "domain", "vpc"),
        update("aws_route_table.main", "propagating_vgws", "false"),
        update("aws_nat_gateway.main", "connectivity_type", "public"),
        update("aws_route.default", "status", "active"),
    ],
}


@pytest.fixture
def env() -> IaCDriftReconcilerEnvironment:
    return IaCDriftReconcilerEnvironment()
