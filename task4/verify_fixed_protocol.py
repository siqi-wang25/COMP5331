"""Refuse repeated-seed experiments if code, fixed parameters or data changed."""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
lock=json.loads((repo/'task4/audit_20261031/fixed_protocol.json').read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
for name,expected in lock['file_sha256'].items():
    assert digest(repo/name)==expected, 'Source/config changed: '+name
for dataset,manifest in lock['reference_data'].items():
    for name,info in manifest['files'].items():
        path=a.data_root/dataset/name
        assert path.stat().st_size==info['bytes'],str(path)
        assert digest(path)==info['sha256'], 'Data changed: '+str(path)
print(json.dumps({'verified':True,'files':len(lock['file_sha256']),'datasets':5,'seeds':lock['new_seeds'],'new_runs':lock['new_runs']}))
