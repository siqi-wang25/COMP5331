import os
import argparse
from utils.quick_start import quick_start
from pathlib import Path
import yaml
os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='KSMMRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=str, default='0', help='gpu id')
    
    args, _ = parser.parse_known_args()
    
    save_dir = './saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    config_dict = {
        'gpu_id': int(args.gpu_id.split(',')[0]),
        'save_dir': save_dir  
    }

    baseline_configs = Path(
        "/mnt/ngc/250010217/FREEDOM/src/configs"
    )

    candidates = [
        baseline_configs / "model" / "FREEDOM.yaml",
        baseline_configs / "models" / "FREEDOM.yaml",
    ]

    config_file = next(
        (path for path in candidates if path.is_file()), None
    )
    if config_file is None:
        raise FileNotFoundError(
            "找不到原版 FREEDOM.yaml，请检查原版仓库的 configs 目录。"
        )

    with config_file.open("r", encoding="utf-8") as f:
        freedom_config = yaml.safe_load(f)

    if not isinstance(freedom_config, dict):
        raise ValueError(f"模型配置内容无效：{config_file}")

    # 保留原有 GPU 等显式设置，补上 FREEDOM 的模型配置。
    freedom_config.update(config_dict)
    config_dict = freedom_config

    required = [
        "feat_embed_dim",
        "knn_k",
        "mm_image_weight",
        "n_mm_layers",
        "n_ui_layers",
        "dropout",
        "reg_weight",
    ]
    missing = [key for key in required if config_dict.get(key) is None]
    if missing:
        raise ValueError(f"FREEDOM 配置缺少：{missing}")

    print(f"[CONFIG] Loaded: {config_file}", flush=True)
    for key in required:
        print(f"[CONFIG] {key} = {config_dict[key]}", flush=True)
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

