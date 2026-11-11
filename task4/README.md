# Task 4: SMORE baseline, DAMPS and ablations

This is the Task 4 submission for **ustc-mkh/COMP5331**. It contains the code and results from 45 completed runs: five datasets × five variants at seed 999, followed by Raw/Full at seeds 2024 and 2025. Three-seed means do not show a consistent DAMPS improvement. The ablations remain single-seed results.

- [Three-seed comparison, uncertainty, timing and memory](results/three_seed/RESULTS.md)
- [First 25 configurations and ablations](results/RESULTS.md)
- [Portable evidence for all 30 Raw/Full runs](results/three_seed/evidence.json)
- [Author integration audit](audit_20261031/INTEGRATION_AUDIT.md)
- [Source migration manifest](migration_manifest.json)

## Preserved experimental implementation

The reported experiments used the KDD2026_DAMPS/MMRec runner, not COMP5331's `mmrecsys` runner. This submission preserves that implementation so the results remain attributable to the executed code. It does not register a new model in `mmrecsys` or alter the existing backbones, trainer, datasets or shared DAMPS module.

The four added runtime files are under `runtime/src/`: SMORE, the projected-feature DAMPS adapter, the pinned Task 1 module and the training entry point. Plans, fixed configurations, queue scripts, tests and results remain under `task4/`. All 35 files in the original submission are preserved byte for byte; their original paths and SHA256 hashes are listed in `migration_manifest.json`. The original README is retained as [historical protocol documentation](runtime/ORIGINAL_README.md); its old PR destination is superseded by this README.

DAMPS is applied after SMORE's modality projections and before its original frequency fusion. The adapter uses the Task 1 projection-level interface from COMP5331 commit `9395c29b4a1cb05996f55a143ddea7a1f71929d0`. The pinned snapshot records what was trained, independently of subsequent changes to the team module. The author's exact SMORE+DAMPS implementation was absent from the audited public history and Zenodo release.

## Reconstruct and run

Run preparation on the selected server. It uses the server's direct network, disables proxy variables and Git proxy configuration, and clears SSH agent/forwarding paths. It checks out the pinned public runtime base and adds only the archived Task 4 files. It refuses an existing output path, installs nothing and starts no training.

```sh
# From the COMP5331 repository root; standard-library Python only.
python3 task4/prepare_runtime.py --verify-only
python3 task4/prepare_runtime.py --output /absolute/new-smore-runtime

# Offline alternative, using an existing local KDD2026_DAMPS Git checkout:
python3 task4/prepare_runtime.py --source /absolute/existing-kdd-checkout \
  --output /absolute/another-new-smore-runtime
```

Preparation checks all 35 archived files and all seven source/configuration hashes in the fixed protocol. Before new training, create a local commit of the reconstructed overlay so run metadata records a clean source revision:

```sh
cd /absolute/new-smore-runtime
git add src task4
git commit -m "Reconstruct fixed Task 4 SMORE experiment source"
```

Use an isolated Python environment matching `task4/requirements-lock.txt`; follow the direct-network environment setup in the historical protocol. Do not install dependencies into a teammate's environment. Data and run outputs belong outside either repository.

```sh
# From the reconstructed runtime, using the isolated environment:
python task4/test_adapter.py
python task4/test_integration_route.py
python task4/test_seed_summary.py
python task4/test_dispatch.py
python task4/verify_fixed_protocol.py --data-root /absolute/task-root/data

python src/run_task4.py --dataset baby --variant raw --seed 999 --gpu 0 \
  --data-root /absolute/task-root/data --output /absolute/new-runs/baby__raw__seed999
```

For the original 25-run queue, first create `data/`, `runs/`, `logs/` and `tmp/` under a new task root and follow the [queue instructions](runtime/ORIGINAL_README.md#running). The fixed follow-up plan is `task4/plan_seeds_20261031.json`; it requires a separate follow-up root and the original reference root. Choose GPU IDs only after checking other users' jobs. The dispatcher waits for idle GPUs, CPU/RAM/disk headroom and yields only its own workers on a GPU conflict. No experiments are launched by this repository migration.

## Protocol and interpretation

The experiments preserve the published `x_label` splits (nominal per-user 80/10/10 with the documented low-count exception), use full ranking, select the checkpoint by validation Recall@20, and evaluate Recall/NDCG@10/20 on the test set. Raw/Full results use seeds 999, 2024 and 2025 with sample standard deviations; component ablations use seed 999. Parameters were fixed before the paired repeats, with no search to match paper numbers.

Training commit IDs and archived environment records retain their original meaning. The migration does not represent a new training run or a native COMP5331 port. See the reports for the unreproduced paper gains, unpublished Electronics/MicroLens parameter choices, released-code AVRF/IMCF behavior and memory-accounting limits.
