"""Conservative, resumable background queue; never controls other users' jobs."""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--gpus', default='4,5')
p.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
p.add_argument('--reference-root', type=Path)
a = p.parse_args()
root = Path(a.root).resolve()
repo = Path(__file__).resolve().parents[1]
lock = (root / 'queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
for k in list(os.environ):
    if k.lower().endswith('_proxy') or k == 'SSH_AUTH_SOCK': os.environ.pop(k, None)
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  PYTHONHASHSEED='999', MPLBACKEND='Agg', MPLCONFIGDIR=str(root/'tmp/matplotlib'),
                  NO_PROXY='*', no_proxy='*')
plan = json.loads(a.plan.read_text())
assert len({(j['dataset'],j['variant'],j['seed']) for j in plan}) == len(plan)
(root/'plan.json').write_text(json.dumps(plan,indent=2))
running = {}
failed = {p.parent.name for p in (root/'runs').glob('*/failure.json')}

def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return {x[0]: x[1:] for line in raw.splitlines() if (x := [s.strip() for s in line.split(',')])}

def resources_ok():
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line and line.split()[1].isdigit()}
    return os.getloadavg()[0] < os.cpu_count()*0.5 and mem['MemAvailable'] > 64*1024**2 and shutil.disk_usage(root).free > 100*1024**3

def compute_apps():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    pairs=[tuple(p.strip() for p in row.split(',')) for row in rows.splitlines() if row.strip()]
    return [(uuid,int(pid)) for uuid,pid in pairs]

def run_id(job): return f"{job['dataset']}__{job['variant']}__seed{job['seed']}"

def ready(job):
    d = root/'data'/job['dataset']
    return all((d/f).exists() for f in [job['dataset']+'.inter', 'image_feat.npy', 'text_feat.npy'])

class ExistingProcess:
    """Reconnect to this queue's own worker after dispatcher restart."""
    def __init__(self, pid, job): self.pid, self.job = pid, job
    def poll(self):
        cmd=Path('/proc')/str(self.pid)/'cmdline'
        try:
            if str(root/'runs'/run_id(self.job)).encode() in cmd.read_bytes(): return None
        except FileNotFoundError: pass
        return 0 if (root/'runs'/run_id(self.job)/'result.json').exists() else 1
    def wait(self):
        while (ret := self.poll()) is None: time.sleep(10)
        return ret

if (root/'queue-status.json').exists():
    previous=json.loads((root/'queue-status.json').read_text())
    for gpu, record in previous.get('running',{}).items():
        job={k:record[k] for k in ('dataset','variant','seed')}
        proc=ExistingProcess(record['pid'],job)
        if proc.poll() is None:
            running[gpu]=(proc,open(os.devnull,'w'),job)
(root/'queue.pid').write_text(str(os.getpid()))

while True:
    for gpu, (proc, log, job) in list(running.items()):
        ret = proc.poll()
        if ret is not None:
            log.close()
            if ret != 0:
                failed.add(run_id(job))
                (root/'runs'/run_id(job)/'failure.json').write_text(json.dumps({'returncode':ret}))
            del running[gpu]
    # If another workload takes a card, yield only this queue's own worker.
    # Restart its exact configuration later; completed results are never replaced.
    if running:
        snapshot=gpu_snapshot()
        apps=compute_apps()
        for gpu,(proc,log,job) in list(running.items()):
            uuid=snapshot[gpu][0]
            outsiders=[pid for card,pid in apps if card==uuid and pid!=proc.pid]
            if not outsiders or proc.poll() is not None: continue
            process_path=Path('/proc')/str(proc.pid)
            cmd=process_path/'cmdline'
            try:
                if process_path.stat().st_uid != os.getuid() or str(root/'runs'/run_id(job)).encode() not in cmd.read_bytes():
                    raise RuntimeError('Worker identity changed; refusing to signal process')
            except FileNotFoundError:
                continue  # Finished between the GPU snapshot and identity check.
            print('YIELD',run_id(job),'GPU',gpu,'other PIDs',outsiders,flush=True)
            try: os.kill(proc.pid,signal.SIGTERM)
            except ProcessLookupError: continue
            for _ in range(20):
                if proc.poll() is not None: break
                time.sleep(0.5)
            if proc.poll() is None:
                try: os.kill(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            proc.wait(); log.close()
            out=root/'runs'/run_id(job)
            if not (out/'result.json').exists():
                (out/'interrupted.json').write_text(json.dumps({'reason':'yield_to_other_gpu_process','other_pids':outsiders}))
                archive=root/'interrupted_attempts'; archive.mkdir(exist_ok=True)
                out.rename(archive/(run_id(job)+'__'+str(time.time_ns())))
            del running[gpu]
    complete = [run_id(j) for j in plan if (root/'runs'/run_id(j)/'result.json').exists()]
    pending = [j for j in plan if run_id(j) not in complete and run_id(j) not in failed and j not in [v[2] for v in running.values()]]
    status = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  completed=complete, failed=sorted(failed), running={k:dict(v[2],pid=v[0].pid) for k,v in running.items()},
                  waiting_data=[run_id(j) for j in pending if not ready(j)], pending=[run_id(j) for j in pending])
    temporary=root/'queue-status.json.tmp'
    temporary.write_text(json.dumps(status,indent=2)); temporary.replace(root/'queue-status.json')
    if (not pending or failed) and not running: break
    if not failed and resources_ok():
        snapshot=gpu_snapshot()
        for gpu in a.gpus.split(','):
            if gpu in running or gpu not in snapshot: continue
            uuid, used, utilization=snapshot[gpu]
            # A zero-utilization card with an allocated process is still occupied.
            if any(card==uuid for card,pid in compute_apps()) or int(used)>128 or int(utilization)>5: continue
            available=[j for j in pending if ready(j)]
            if not available: continue
            job=available[0]; pending.remove(job)
            out=root/'runs'/run_id(job); out.mkdir(parents=True,exist_ok=True)
            log=(out/'console.log').open('a')
            cmd=['nice','-n','10',sys.executable,'-u',str(repo/'src/run_task4.py'),
                 '--dataset',job['dataset'],'--variant',job['variant'],'--seed',str(job['seed']),
                 '--gpu',gpu,'--data-root',str(root/'data'),'--output',str(out)]
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=repo/'src')
            running[gpu]=(proc,log,job)
            print('START',run_id(job),'GPU',gpu,'PID',proc.pid,flush=True)
    time.sleep(30)
if a.reference_root:
    subprocess.run([sys.executable,str(repo/'task4/summarize_seeds.py'),'--root',str(root),
                    '--reference-root',str(a.reference_root)],check=True)
else:
    subprocess.run([sys.executable,str(repo/'task4/summarize.py'),'--root',str(root)],check=True)
