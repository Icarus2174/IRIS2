from __future__ import annotations

import json
from pathlib import Path

from .allocator import allocate
from .diagram_validation import validate_topology
from .feedback import apply_feedback
from .ir import lower_to_ir
from .main import _runtime_event_from_dict, run_case
from .models import Topology
from .parser import parse_intent
from .scheduler import schedule as schedule_fn
from .topology_compiler import compile_topology


def _compile_case(case_path: str):
    parsed, runtime_event_dict = parse_intent(case_path)
    ir = lower_to_ir(parsed)
    allocation = allocate(ir)
    runtime_event = _runtime_event_from_dict(parsed.case_name, runtime_event_dict)
    schedule = schedule_fn(ir, allocation, runtime_event=runtime_event)
    topology = compile_topology(ir, allocation, schedule)
    feedback, schedule2, topology2 = apply_feedback(ir, allocation, schedule, topology, runtime_event)

    validation = validate_topology(
        parsed,
        ir,
        allocation,
        schedule2,
        topology2,
        feedback=feedback,
    )

    return {
        "parsed": parsed,
        "ir": ir,
        "allocation": allocation,
        "schedule": schedule2,
        "topology": topology2,
        "feedback": feedback,
        "validation": validation,
    }


def _remove_compute_backbone(topology: Topology) -> Topology:
    compute_ids = {
        n["id"]
        for n in topology.nodes
        if "compute" in n.get("roles", [])
    }

    kept_links = [
        e for e in topology.links
        if not (e.get("src") in compute_ids and e.get("dst") in compute_ids)
    ]

    new_rationale = dict(topology.rationale)
    new_rationale["mutation"] = "removed compute-to-compute backbone links"

    return Topology(
        case_name=topology.case_name + "_broken",
        nodes=topology.nodes,
        links=kept_links,
        groups=topology.groups,
        rationale=new_rationale,
    )


def main() -> None:
    cases = [
        "inputs/case_inference_earth.yaml",
        "inputs/case_runtime_alert.yaml",
        "inputs/case_training_100gpu.yaml",
    ]

    # Ensure normal case outputs are generated
    for case in cases:
        run_case(case)

    results = []

    for case in cases:
        bundle = _compile_case(case)
        results.append({
            "scenario": case,
            "valid": bundle["validation"]["passed"],
            "failed_checks": [
                c["name"] for c in bundle["validation"]["checks"] if not c["passed"]
            ],
            "objective_proxies": bundle["validation"]["objective_proxies"],
        })

    # Add one deliberately incorrect topology
    bundle_100 = _compile_case("inputs/case_training_100gpu.yaml")
    bad_topology = _remove_compute_backbone(bundle_100["topology"])
    bad_validation = validate_topology(
        bundle_100["parsed"],
        bundle_100["ir"],
        bundle_100["allocation"],
        bundle_100["schedule"],
        bad_topology,
        feedback=bundle_100["feedback"],
    )

    results.append({
        "scenario": "mutated/case_training_100gpu_no_backbone",
        "valid": bad_validation["passed"],
        "failed_checks": [
            c["name"] for c in bad_validation["checks"] if not c["passed"]
        ],
        "objective_proxies": bad_validation["objective_proxies"],
    })

    out_dir = Path("outputs") / "diagram_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "experiment_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Diagram Validation Experiment Summary",
        "",
        "| Scenario | Valid? | Failed checks | low_cct_proxy | gpu_utilization_proxy | same_domain_proxy |",
        "|---|---:|---|---:|---:|---:|",
    ]

    for row in results:
        obj = row["objective_proxies"]
        failed = ", ".join(row["failed_checks"]) if row["failed_checks"] else "—"
        lines.append(
            f"| {row['scenario']} | {'PASS' if row['valid'] else 'FAIL'} | {failed} | "
            f"{obj['low_cct_proxy']:.3f} | {obj['gpu_utilization_proxy']:.3f} | {obj['same_domain_proxy']:.3f} |"
        )

    (out_dir / "experiment_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote experiment outputs to: {out_dir}")


if __name__ == "__main__":
    main()