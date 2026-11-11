"""Run on the training server: direct downloads of only required released files."""
import argparse,json,os
from pathlib import Path
for key in list(os.environ):
    if key.lower().endswith('_proxy') or key=='SSH_AUTH_SOCK': os.environ.pop(key,None)
os.environ.update(NO_PROXY='*',no_proxy='*')
import gdown
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
folders=['13cBy1EA_saTUuXxVllKgtfci2A09jyaG','14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w']
manifest=[]
for folder in folders:
    files=gdown.download_folder(id=folder,output=str(root),skip_download=True,quiet=True,use_cookies=False,timeout=30)
    for f in files:
        path=Path(f.path);d=path.parent.name
        if d not in ['baby','sports','clothing','elec','microlens']:continue
        if path.name not in [d+'.inter','image_feat.npy','text_feat.npy']:continue
        target=root/d/path.name;target.parent.mkdir(exist_ok=True)
        gdown.download(id=f.id,output=str(target),quiet=False,use_cookies=False,resume=True,timeout=60)
        manifest.append(dict(dataset=d,file=path.name,google_drive_id=f.id,bytes=target.stat().st_size))
(root/'download_sources.json').write_text(json.dumps(manifest,indent=2))
