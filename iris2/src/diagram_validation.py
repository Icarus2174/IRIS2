from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import Allocation, FeedbackResponse, IR, ParsedIntent, Schedule, Topology


def _compute_nodes(topology: Topology) -> List[Dict[str, Any]]:
    return [n for n in topology.nodes if "compute" in n.get("roles", [])]


def _compute_ids(topology: Topology) -> set[str]:
    return {n["id"] for n in _compute_nodes(topology)}


def _compute_backbone_links(topology: Topology) -> List[Dict[str, Any]]:
    compute_ids = _compute_ids(topology)
    return [
        e for e in topology.links
        if e.get("src") in compute_ids and e.get("dst") in compute_ids
    ]


def _storage_links(topology: Topology) -> List[Dict[str, Any]]:
    compute_ids = _compute_ids(topology)
    return [
        e for e in topology.links
        if e.get("src") in compute_ids and e.get("dst") == "object_store"
    ]


def validate_topology(
    parsed: ParsedIntent,
    ir: IR,
    allocation: Allocation,
    schedule: Schedule,
    topology: Topology,
    feedback: Optional[FeedbackResponse] = None,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, details: Dict[str, Any]) -> None:
        checks.append({
            "name": name,
            "passed": passed,
            "details": details,
        })

    compute_nodes = _compute_nodes(topology)
    backbone_links = _compute_backbone_links(topology)
    storage_links = _storage_links(topology)

    all_link_types = sorted({e.get("type", "packet") for e in topology.links})
    optical_backbone = any(e.get("type") == "optical" for e in backbone_links)

    # Basic structural validity
    add_check(
        "diagram_has_required_nodes",
        passed=(len(compute_nodes) > 0) if parsed.accelerator_count > 0 else True,
        details={
            "compute_nodes": [n["id"] for n in compute_nodes],
            "requested_accelerators": parsed.accelerator_count,
        },
    )

    add_check(
        "storage_paths_present",
        passed=(len(storage_links) == len(compute_nodes)),
        details={
            "compute_node_count": len(compute_nodes),
            "storage_link_count": len(storage_links),
        },
    )

    # Workload-aware checks
    if ir.communication_intensity == "high":
        add_check(
            "high_comm_has_compute_backbone",
            passed=(len(backbone_links) >= 1),
            details={
                "backbone_link_count": len(backbone_links),
                "backbone_link_types": sorted({e.get("type", "packet") for e in backbone_links}),
            },
        )

    if ir.latency_class == "strict":
        add_check(
            "strict_latency_stays_earth",
            passed=(allocation.placement == "earth"),
            details={
                "placement": allocation.placement,
            },
        )

    # Runtime downgrade visibility
    if feedback and feedback.effects.get("recompiled_topology"):
        add_check(
            "runtime_packet_downgrade_visible",
            passed=("optical" not in all_link_types),
            details={
                "link_types_after_feedback": all_link_types,
                "feedback_effects": feedback.effects,
                "feedback_event": feedback.rationale.get("event"),
            },
        )

    # Proxy objectives
    allocated = sum(int(s.get("accelerators", 0)) for s in allocation.selected_clusters)
    requested = int(parsed.accelerator_count)
    allocation_satisfaction = 1.0 if requested == 0 else min(1.0, allocated / requested)
    packing_score = 1.0 if allocation.packed else 0.85
    same_domain_score = 1.0 if allocation.placement != "hybrid" else 0.70
    dense_backbone_score = 1.0 if (ir.communication_intensity != "high" or len(backbone_links) >= 1) else 0.30
    optical_backbone_score = 1.0 if optical_backbone else (0.70 if len(backbone_links) >= 1 else 0.40)

    low_cct_proxy = round(
        0.50 * dense_backbone_score
        + 0.30 * optical_backbone_score
        + 0.20 * same_domain_score,
        3,
    )

    gpu_utilization_proxy = round(
        0.60 * allocation_satisfaction
        + 0.40 * packing_score,
        3,
    )

    cross_domain_penalty = 0.25 if allocation.placement == "hybrid" else 0.00
    same_domain_proxy = round(1.0 - cross_domain_penalty, 3)

    goals = [g.lower() for g in parsed.objectives]
    objective_weights: Dict[str, float] = {}
    if any(("collective" in g) or ("cct" in g) for g in goals):
        objective_weights["low_cct_proxy"] = 0.50
    if any(("gpu utilization" in g) or ("utilization" in g) for g in goals):
        objective_weights["gpu_utilization_proxy"] = 0.35
    if any(("cross-domain" in g) or ("cross domain" in g) for g in goals):
        objective_weights["same_domain_proxy"] = 0.15

    if not objective_weights:
        objective_weights = {
            "low_cct_proxy": 0.50 if ir.communication_intensity == "high" else 0.20,
            "gpu_utilization_proxy": 0.35,
            "same_domain_proxy": 0.15,
        }

    total_weight = sum(objective_weights.values())
    weighted_score = round(
        (
            objective_weights.get("low_cct_proxy", 0.0) * low_cct_proxy
            + objective_weights.get("gpu_utilization_proxy", 0.0) * gpu_utilization_proxy
            + objective_weights.get("same_domain_proxy", 0.0) * same_domain_proxy
        ) / total_weight,
        3,
    )

    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "objective_proxies": {
            "low_cct_proxy": low_cct_proxy,
            "gpu_utilization_proxy": gpu_utilization_proxy,
            "same_domain_proxy": same_domain_proxy,
            "cross_domain_penalty": cross_domain_penalty,
            "weighted_objective_score": weighted_score,
            "weights_used": objective_weights,
        },
    }