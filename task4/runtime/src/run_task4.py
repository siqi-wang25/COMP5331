"""One fixed Task 4 configuration; validation-only selection, test once at best epoch."""
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--variant', choices=['raw', 'full', 'no_apc', 'no_avrf', 'no_imcf'], required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--seed', type=int, default=999)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import pandas as pd
    import scipy
    import torch
    from utils.configurator import Config
    from utils.dataset import RecDataset
    from utils.dataloader import TrainDataLoader, EvalDataLoader
    from utils.utils import init_seed
    from common.trainer import Trainer
    from models.smore import SMORE
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    os.chdir(root / 'src')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise RuntimeError('Completed result exists; choose a new output directory')
    fixed = json.loads((root / 'task4/configs.json').read_text())[args.dataset]
    settings = dict(fixed, seed=args.seed, hyper_parameters=[], gpu_id=args.gpu,
                    data_path=str(Path(args.data_root).resolve()) + '/',
                    graph_cache_path=str(out / 'graph_cache'), damps_variant=args.variant,
                    epochs=args.epochs, metrics=['Recall', 'NDCG'], topk=[10, 20],
                    eval_batch_size=1024, save_recommended_topk=False,
                    valid_metric='Recall@20')
    config = Config('SMORE', args.dataset, settings)
    if config['device'].type != 'cuda':
        raise RuntimeError('CUDA required for production runs')
    init_seed(args.seed)
    started = time.time()
    metadata = dict(config=config.final_config_dict, args=vars(args), hostname=platform.node(),
                    started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True),
                    python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                    gpu=torch.cuda.get_device_name(0))
    dump(out / 'metadata.json', metadata)
    dataset = RecDataset(config)
    df = dataset.df
    uid, iid, label = config['USER_ID_FIELD'], config['ITEM_ID_FIELD'], config['inter_splitting_label']
    counts = df[label].value_counts().to_dict()
    assert set(counts) == {0, 1, 2}, counts
    assert not df.duplicated([uid, iid]).any(), 'Repeated user-item pairs can leak across splits'
    assert min(df[uid]) >= 0 and min(df[iid]) >= 0
    manifest = dict(users=dataset.user_num, items=dataset.item_num, interactions=len(df), splits=counts, files={})
    for name in [config['inter_file_name'], config['vision_feature_file'], config['text_feature_file']]:
        path = Path(config['data_path']) / args.dataset / name
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
        manifest['files'][name] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
        if name.endswith('.npy'):
            features = np.load(path, mmap_mode='r', allow_pickle=False)
            assert features.ndim == 2 and len(features) == dataset.item_num
            assert np.isfinite(features).all(), name
            manifest['files'][name]['shape'] = list(features.shape)
    dump(out / 'data_manifest.json', manifest)
    # Preserve author-provided per-user split labels (rounding changes global ratios).
    train_ds, valid_ds, test_ds = dataset.split()
    # Upstream populates this field as a side effect of printing a dataset.
    # Initialize explicitly so correctness does not depend on logging verbosity.
    for split in (train_ds, valid_ds, test_ds):
        split.inter_num = len(split)
    train = TrainDataLoader(config, train_ds, batch_size=config['train_batch_size'], shuffle=True)
    valid = EvalDataLoader(config, valid_ds, additional_dataset=train_ds, batch_size=config['eval_batch_size'])
    test = EvalDataLoader(config, test_ds, additional_dataset=train_ds, batch_size=config['eval_batch_size'])
    train.pretrain_setup()
    model = SMORE(config, train).to(config['device'])
    trainer = Trainer(config, model)
    best_score, best_epoch, stale = -1.0, -1, 0
    epoch_seconds = []
    torch.cuda.reset_peak_memory_stats()
    setup_seconds = time.time() - started
    for epoch in range(args.epochs):
        tick = time.time()
        model.pre_epoch_processing()
        loss, _ = trainer._train_epoch(train, epoch)
        if not np.isfinite(float(loss)):
            raise FloatingPointError('Non-finite training loss')
        trainer.lr_scheduler.step()
        model.post_epoch_processing()
        metrics = trainer.evaluate(valid)
        score = metrics['recall@20']
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            checkpoint = dict(model=model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                              scheduler=trainer.lr_scheduler.state_dict(), epoch=epoch,
                              config=config.final_config_dict, validation=metrics,
                              torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            torch.save(checkpoint, out / 'best.pt.tmp')
            (out / 'best.pt.tmp').replace(out / 'best.pt')
        else:
            stale += 1
        seconds = time.time() - tick
        epoch_seconds.append(seconds)
        row = dict(epoch=epoch, loss=float(loss), validation=metrics, seconds=seconds,
                   best_epoch=best_epoch, elapsed_seconds=time.time()-started,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        with (out / 'epochs.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        dump(out / 'progress.json', row)
        print(json.dumps(row), flush=True)
        if stale > config['stopping_step']: break
    checkpoint = torch.load(out / 'best.pt', map_location=config['device'], weights_only=False)
    model.load_state_dict(checkpoint['model'])
    result = dict(dataset=args.dataset, variant=args.variant, seed=args.seed,
                  validation=checkpoint['validation'], test=trainer.evaluate(test),
                  best_epoch=best_epoch, epochs_completed=epoch+1,
                  setup_seconds=setup_seconds, total_seconds=time.time()-started,
                  median_epoch_seconds=float(np.median(epoch_seconds)),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  commit=metadata['commit'])
    dump(out / 'result.json', result)
    print('COMPLETED '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
