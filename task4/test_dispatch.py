"""Simulate an occupied GPU and a later collision; never launch or signal real jobs."""
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class DispatchTest(unittest.TestCase):
    def test_wait_then_yield_only_own_worker_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root/'runs').mkdir()
            (root/'data/baby').mkdir(parents=True)
            for name in ('baby.inter', 'image_feat.npy', 'text_feat.npy'):
                (root/'data/baby'/name).touch()
            plan = root/'test-plan.json'
            plan.write_text(json.dumps([dict(dataset='baby',variant='raw',seed=2024)]))
            out = root/'runs/baby__raw__seed2024'
            state = dict(sleeps=0, starts=0, worker=None)

            def output(command, **kwargs):
                if '--query-gpu=index,uuid,memory.used,utilization.gpu' in command:
                    return '4, GPU-test, 0, 0\n'
                if '--query-compute-apps=gpu_uuid,pid' in command:
                    if state['sleeps'] == 0:
                        return 'GPU-test, 9999\n'  # Existing workload prevents launch.
                    if state['starts'] == 1 and state['worker'].ret is None:
                        return 'GPU-test, 4444\nGPU-test, 9999\n'
                    return ''
                raise AssertionError(command)

            class Worker:
                pid = 4444
                ret = None
                def poll(self): return self.ret
                def wait(self): return self.ret

            def start(command, **kwargs):
                self.assertGreater(state['sleeps'], 0)
                state['starts'] += 1
                worker = Worker()
                state['worker'] = worker
                if state['starts'] == 2:
                    (out/'result.json').write_text('{}')
                    worker.ret = 0
                return worker

            def signal_own(pid, sig):
                self.assertEqual(pid, 4444)
                state['worker'].ret = -15

            def sleep(seconds):
                state['sleeps'] += 1
                self.assertLess(state['sleeps'], 8, 'Queue did not finish')

            original_read, original_stat, original_bytes = Path.read_text, Path.stat, Path.read_bytes
            def read(path, *args, **kwargs):
                if str(path) == '/proc/meminfo': return 'MemAvailable: 1000000000 kB\n'
                return original_read(path, *args, **kwargs)
            def stat(path, *args, **kwargs):
                if str(path) == '/proc/4444': return SimpleNamespace(st_uid=os.getuid())
                return original_stat(path, *args, **kwargs)
            def read_bytes(path):
                if str(path) == '/proc/4444/cmdline': return str(out).encode()
                return original_bytes(path)

            with patch.dict(os.environ), \
                 patch('sys.argv',['dispatch.py','--root',str(root),'--gpus','4','--plan',str(plan),'--reference-root',str(root)]), \
                 patch('subprocess.check_output',side_effect=output), \
                 patch('subprocess.Popen',side_effect=start), \
                 patch('subprocess.run') as summary, \
                 patch('os.kill',side_effect=signal_own) as kill, \
                 patch('time.sleep',side_effect=sleep), \
                 patch('os.getloadavg',return_value=(0,0,0)), \
                 patch('shutil.disk_usage',return_value=SimpleNamespace(free=1024**4)), \
                 patch.object(Path,'read_text',read), \
                 patch.object(Path,'read_bytes',read_bytes), \
                 patch.object(Path,'stat',stat):
                runpy.run_path(str(Path(__file__).with_name('dispatch.py')),run_name='__main__')
            self.assertEqual(state['starts'],2)
            kill.assert_called_once()
            summary.assert_called_once()
            self.assertEqual(len(list((root/'interrupted_attempts').glob('*/interrupted.json'))),1)
            status = json.loads((root/'queue-status.json').read_text())
            self.assertEqual(status['completed'],['baby__raw__seed2024'])
            self.assertEqual(status['failed'],[])


if __name__ == '__main__':
    unittest.main()
