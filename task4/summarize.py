"""Generate tables and figures from completed results only."""
import argparse, csv, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True); a=p.parse_args();root=Path(a.root)
rows=[]
for f in sorted((root/'runs').glob('*/result.json')):
    x=json.loads(f.read_text());rows.append({**{k:x[k] for k in ['dataset','variant','seed','best_epoch','epochs_completed','total_seconds','median_epoch_seconds','peak_gpu_allocated_bytes','peak_cpu_rss_kib','commit']},**x['test']})
out=root/'results';out.mkdir(exist_ok=True)
if rows:
    with (out/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    datasets=list(dict.fromkeys(r['dataset'] for r in rows));variants=['raw','full','no_apc','no_avrf','no_imcf']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['recall@20','ndcg@20']):
        for n,v in enumerate(variants):
            vals=[next((r[metric] for r in rows if r['dataset']==d and r['variant']==v),np.nan) for d in datasets]
            ax.bar(np.arange(len(datasets))+(n-2)*.15,vals,width=.15,label=v)
        ax.set_xticks(range(len(datasets)),datasets);ax.set_ylabel(metric);ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8);fig.suptitle('Task 4: measured results (missing runs omitted)');fig.tight_layout();fig.savefig(out/'ablation.png',dpi=180)
(root/'results/status.json').write_text(json.dumps({'completed':len(rows),'expected':25},indent=2))
