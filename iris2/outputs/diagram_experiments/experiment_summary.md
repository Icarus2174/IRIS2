# Diagram Validation Experiment Summary

| Scenario | Valid? | Failed checks | low_cct_proxy | gpu_utilization_proxy | same_domain_proxy |
|---|---:|---|---:|---:|---:|
| inputs/case_inference_earth.yaml | PASS | — | 0.820 | 1.000 | 1.000 |
| inputs/case_runtime_alert.yaml | PASS | — | 0.910 | 1.000 | 1.000 |
| inputs/case_training_100gpu.yaml | PASS | — | 1.000 | 1.000 | 1.000 |
| mutated/case_training_100gpu_no_backbone | FAIL | high_comm_has_compute_backbone | 0.470 | 1.000 | 1.000 |
