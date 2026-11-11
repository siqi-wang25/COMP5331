"""Check paired statistics, incomplete pairs and protocol-drift rejection."""
import json
import tempfile
import unittest
from pathlib import Path
from summarize_seeds import METRICS, aggregate, read_runs


class SeedSummaryTest(unittest.TestCase):
    def test_paired_statistics(self):
        records = []
        for seed, raw, full in [(999, .1, .11), (2024, .2, .19), (2025, .3, .33)]:
            for variant, value in [('raw', raw), ('full', full)]:
                records.append(dict(dataset='baby', seed=seed, variant=variant,
                                    **{metric: value for metric in METRICS}))
        summary, paired = aggregate(records)
        self.assertEqual(len(summary), 8)
        row = paired[0]
        self.assertEqual(row['n'], 3)
        self.assertAlmostEqual(row['mean'], .01)
        self.assertAlmostEqual(row['sample_std'], .02)
        self.assertAlmostEqual(row['relative_mean_change_pct'], 5)
        self.assertEqual((row['positive'], row['negative'], row['ties']), (2, 1, 0))
        _, partial = aggregate(records[:-1])
        self.assertEqual(partial[0]['n'], 2)
        self.assertAlmostEqual(partial[0]['mean'], 0)
        self.assertIsNone(partial[0]['delta_seed2025'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate(records+[records[0]])

    def test_checkpoint_and_protocol_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def make_run(variant):
                path = root/'runs'/f'baby__{variant}__seed999'
                path.mkdir(parents=True)
                metric = {m: .1 for m in METRICS}
                result = dict(dataset='baby', variant=variant, seed=999, validation=metric, test=metric,
                              best_epoch=0, epochs_completed=2, commit='abc', setup_seconds=1,
                              total_seconds=2, median_epoch_seconds=.5, peak_gpu_allocated_bytes=100,
                              peak_gpu_reserved_bytes=110, peak_cpu_rss_kib=200)
                metadata = dict(config=dict(dataset='baby', damps_variant=variant, seed=999, learning_rate=.001),
                                commit='abc', git_status='', **{k:'fixed' for k in ('python','torch','cuda','numpy','pandas','scipy','gpu')})
                for name, value in [('result',result), ('metadata',metadata), ('data_manifest',dict(files={'sha256':'fixed'}))]:
                    (path/f'{name}.json').write_text(json.dumps(value))
                # A tie must retain the first checkpoint.
                (path/'epochs.jsonl').write_text(''.join(json.dumps(dict(epoch=i,validation=metric))+'\n' for i in range(2)))
                return path
            make_run('raw')
            full = make_run('full')
            records, missing = read_runs(root, root)
            self.assertEqual(len(records), 2)
            self.assertEqual(len(missing), 28)
            metadata = json.loads((full/'metadata.json').read_text())
            metadata['config']['learning_rate'] = .002
            (full/'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(AssertionError, 'Protocol/data/environment mismatch'):
                read_runs(root, root)


if __name__ == '__main__':
    unittest.main()
