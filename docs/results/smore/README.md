# SMORE + DAMPS reproduction results

Task 4 completed 45 runs on Baby, Sports, Clothing, Electronics and MicroLens. Raw SMORE and SMORE + DAMPS have three paired seeds (999, 2024, 2025); the three component ablations have seed 999 only. The implementation inserts DAMPS after modality projection and before SMORE's original frequency fusion.

| Dataset | SMORE Recall@20 | SMORE + DAMPS Recall@20 | Relative change |
| --- | ---: | ---: | ---: |
| Baby | 0.101833 | 0.101667 | -0.164% |
| Sports | 0.113700 | 0.113033 | -0.586% |
| Clothing | 0.097133 | 0.097367 | +0.240% |
| Electronics | 0.063267 | 0.063167 | -0.158% |
| MicroLens | 0.109933 | 0.110000 | +0.061% |

These are three-seed means, not best-seed results. Across all 20 mean metrics, 7 increase, 12 decrease and 1 ties. The experiments do not reproduce the paper's consistent improvement, and three seeds do not establish statistical significance.

- [Full four-metric table, sample standard deviations, paired results, time and memory](../../../task4/results/three_seed/RESULTS.md)
- [Single-seed component ablations](../../../task4/results/RESULTS.md)
- [Run evidence and effective configurations](../../../task4/results/three_seed/evidence.json)
- [Fixed source, parameters and data fingerprints](../../../task4/audit_20261031/fixed_protocol.json)
- [Implementation audit](../../../task4/audit_20261031/INTEGRATION_AUDIT.md)
- [Code and reproducible runtime preparation](../../../task4/README.md)

The measurements come from the preserved KDD2026_DAMPS/MMRec runtime. The submitted source and results are byte-identical to the executed Task 4 submission; they are not presented as new runs through COMP5331's `mmrecsys` trainer. The main project trainer and other backbones are unchanged.
