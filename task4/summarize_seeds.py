"""Validate paired runs, then report every seed, sample SD and paired differences."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

DATASETS = ('baby', 'sports', 'clothing', 'elec', 'microlens')
VARIANTS = ('raw', 'full')
SEEDS = (999, 2024, 2025)
METRICS = ('recall@10', 'recall@20', 'ndcg@10', 'ndcg@20')
IGNORED_CONFIG = {'seed', 'gpu_id', 'graph_cache_path', 'data_path', 'damps_variant'}


def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else None}


def aggregate(records):
    indexed = {(r['dataset'], r['variant'], r['seed']): r for r in records}
    if len(indexed) != len(records):
        raise ValueError('Duplicate dataset/variant/seed')
    summary, paired = [], []
    for dataset in DATASETS:
        for metric in METRICS:
            for variant in VARIANTS:
                rows = [indexed[dataset, variant, seed] for seed in SEEDS
                        if (dataset, variant, seed) in indexed]
                if rows:
                    summary.append(dict(dataset=dataset, variant=variant, metric=metric,
                                        seeds=','.join(str(r['seed']) for r in rows),
                                        **stats([r[metric] for r in rows])))
            paired_seeds = [s for s in SEEDS if all((dataset, v, s) in indexed for v in VARIANTS)]
            if paired_seeds:
                raw = [indexed[dataset, 'raw', s][metric] for s in paired_seeds]
                full = [indexed[dataset, 'full', s][metric] for s in paired_seeds]
                delta = [f-r for r, f in zip(raw, full)]
                paired.append(dict(dataset=dataset, metric=metric,
                                   seeds=','.join(map(str, paired_seeds)),
                                   **stats(delta), raw_mean=statistics.mean(raw),
                                   full_mean=statistics.mean(full),
                                   relative_mean_change_pct=100*statistics.mean(delta)/statistics.mean(raw),
                                   positive=sum(d > 0 for d in delta),
                                   negative=sum(d < 0 for d in delta), ties=sum(d == 0 for d in delta),
                                   **{f'delta_seed{s}': indexed[dataset, 'full', s][metric]-indexed[dataset, 'raw', s][metric]
                                      if s in paired_seeds else None for s in SEEDS}))
    return summary, paired


def read_runs(root, reference_root):
    records, missing, baselines = [], [], {}
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                directory = (reference_root if seed == 999 else root)/'runs'/f'{dataset}__{variant}__seed{seed}'
                if not (directory/'result.json').exists():
                    missing.append(directory.name)
                    continue
                result = json.loads((directory/'result.json').read_text())
                metadata = json.loads((directory/'metadata.json').read_text())
                manifest = json.loads((directory/'data_manifest.json').read_text())
                epochs = [json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
                assert (result['dataset'], result['variant'], result['seed']) == (dataset, variant, seed), directory
                config = metadata['config']
                assert (config['dataset'], config['damps_variant'], config['seed']) == (dataset, variant, seed), directory
                fixed = {k: v for k, v in config.items() if k not in IGNORED_CONFIG}
                environment = {k: metadata[k] for k in ('python', 'torch', 'cuda', 'numpy', 'pandas', 'scipy', 'gpu')}
                fingerprint = (fixed, manifest, environment)
                if dataset in baselines:
                    assert fingerprint == baselines[dataset], f'Protocol/data/environment mismatch: {directory}'
                else:
                    baselines[dataset] = fingerprint
                assert result['commit'] == metadata['commit'], directory
                assert len(epochs) == result['epochs_completed'], directory
                best = max(epochs, key=lambda e: e['validation']['recall@20'])
                assert best['epoch'] == result['best_epoch'], directory
                assert best['validation'] == result['validation'], directory
                assert all(math.isfinite(result['test'][m]) for m in METRICS), directory
                keys = ('dataset', 'variant', 'seed', 'best_epoch', 'epochs_completed', 'setup_seconds',
                        'total_seconds', 'median_epoch_seconds', 'peak_gpu_allocated_bytes',
                        'peak_gpu_reserved_bytes', 'peak_cpu_rss_kib', 'commit')
                records.append({**{k: result[k] for k in keys}, **result['test'],
                                'git_status_at_launch': metadata['git_status'].strip(),
                                'run_directory': str(directory)})
    return records, missing


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.root/'results'
    out.mkdir(parents=True, exist_ok=True)
    # An interrupted invocation cannot leave a stale success marker.
    (out/'status.json').write_text(json.dumps({'state': 'checking', 'expected_new_runs': 20}))
    try:
        records, missing = read_runs(args.root, args.reference_root)
        summary, paired = aggregate(records)
    except Exception as error:
        (out/'status.json').write_text(json.dumps({'state': 'integrity_failed', 'error': str(error)}, indent=2))
        raise
    write_csv(out/'per_run.csv', records)
    write_csv(out/'summary.csv', summary)
    write_csv(out/'paired.csv', paired)
    complete = not missing
    lines = ['# SMORE: three-seed validation', '',
             '**Status: '+('complete (30/30 paired raw/full runs)' if complete else 'partial; not a final three-seed result')+'**', '',
             'Seeds: 999, 2024, 2025. Values are mean ± sample standard deviation. '
             'Checkpoint selection uses validation Recall@20. All runs must share fixed parameters, data fingerprints and environment. '
             'Each paired difference is Full − Raw for the same seed; no seed selection or significance claim.', '',
             '| Dataset | Metric | Raw | Full | Paired difference | Pairs |',
             '|---|---|---:|---:|---:|---:|']
    def display(mean, std):
        return f'{mean:.6f} ± {std:.6f}' if std is not None else f'{mean:.6f} (one seed)'
    for row in paired:
        group = {s['variant']: s for s in summary if s['dataset'] == row['dataset'] and s['metric'] == row['metric']}
        # Paired mean may differ from marginal mean while an incomplete pair is running.
        lines.append(f"| {row['dataset']} | {row['metric']} | {display(group['raw']['mean'], group['raw']['sample_std'])} | "
                     f"{display(group['full']['mean'], group['full']['sample_std'])} | {display(row['mean'], row['sample_std'])} | {row['n']} |")
    lines += ['', 'Per-run time and memory are in `per_run.csv`. GPU peaks exclude graph-construction setup; '
              'CPU peak RSS includes setup. Metrics retain the original evaluator’s four-decimal precision. '
              'The existing 25-run ablation report is single-seed; these extra runs cover only Raw and Full. '
              'The author’s exact SMORE+DAMPS integration remains unavailable in the audited public release.', '',
              'Partial summaries may contain unequal marginal sample counts; use `paired.csv` for matched pairs. '
              'See `summary.csv` for the seed list and sample count of each marginal mean.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Draw error bars only for the complete paired matrix.
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for ax, metric in zip(axes, ('recall@20', 'ndcg@20')):
            for index, variant in enumerate(VARIANTS):
                values = [next(r for r in summary if r['dataset'] == d and r['metric'] == metric and r['variant'] == variant) for d in DATASETS]
                ax.bar([i+(index-.5)*.36 for i in range(len(DATASETS))],
                       [v['mean'] for v in values], yerr=[v['sample_std'] for v in values],
                       width=.36, capsize=3, label=variant)
            ax.set_xticks(range(len(DATASETS)), DATASETS)
            ax.set_ylabel(metric)
            ax.set_ylim(bottom=0)
        axes[0].legend()
        fig.suptitle('SMORE: 3 paired seeds; mean ± sample SD')
        fig.tight_layout()
        fig.savefig(out/'three_seed_comparison.png', dpi=180)
        plt.close(fig)
    status = dict(state='complete' if complete else 'partial', integrity_checks_passed=True,
                  completed_new_runs=sum(r['seed'] != 999 for r in records), expected_new_runs=20,
                  reference_runs=sum(r['seed'] == 999 for r in records), total_expected=30, missing=missing)
    (out/'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status))


if __name__ == '__main__':
    main()
