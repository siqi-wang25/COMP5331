"""Both backbones reuse the existing training, checkpoint, and ablation APIs."""
from copy import deepcopy
import json

import pytest
import torch

from mmrecsys.config import load_config
from mmrecsys.data.views import TrainBatch
from mmrecsys.engine.checkpoint import load_checkpoint
from mmrecsys.experiment.ablation import run_ablation
from mmrecsys.experiment.runner import assemble, train_experiment


def config_for(tiny_config, model, enabled):
    overrides = [
        f"data.root={tiny_config['data']['root']}",
        f"runtime.output_root={tiny_config['runtime']['output_root']}",
        f"runtime.cache_root={tiny_config['runtime']['cache_root']}",
        "runtime.device=cpu", "runtime.num_threads=1",
        "train.epochs=1", "train.batch_size=4", "train.preload_to_device=false",
        "model.embedding_dim=4", f"model.damps_enabled={str(enabled).lower()}",
        "eval.topk=[1, 3]", "eval.monitor=Recall@3",
        "eval.user_batch_size=2", "eval.item_chunk_size=2",
    ]
    overrides += (["model.knn_k=2", "model.knn_chunk_size=2"] if model == "pgl" else
                  ["model.hyperedges=2", "model.contrastive_chunk=3"])
    return load_config(model=model, dataset="baby", overrides=overrides)


@pytest.mark.parametrize("model_name", ["pgl", "lgmrec"])
@pytest.mark.parametrize("enabled", [False, True])
def test_forward_connects_every_trainable_parameter(tiny_config, model_name, enabled):
    config = config_for(tiny_config, model_name, enabled)
    _, _, _, model, _, _ = assemble(config)
    model.train()
    model.on_epoch_start(1)
    batch = TrainBatch(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([[3], [4]]))
    loss = model.compute_loss(batch).total
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    assert (model.damps is not None) == enabled
    if enabled:
        assert model.damps.phase_residual.grad is not None
    model.eval()
    scorer = model.make_scorer()
    scores = scorer.score(torch.tensor([0, 1]), torch.arange(6))
    assert scores.shape == (2, 6) and torch.isfinite(scores).all()
    torch.testing.assert_close(scores, torch.cat([
        scorer.score(torch.tensor([0, 1]), torch.arange(3)),
        scorer.score(torch.tensor([0, 1]), torch.arange(3, 6)),
    ], dim=1))


@pytest.mark.parametrize("model_name", ["pgl", "lgmrec"])
@pytest.mark.parametrize("enabled", [False, True])
def test_common_trainer_and_checkpoint_roundtrip(tiny_config, model_name, enabled):
    config = config_for(tiny_config, model_name, enabled)
    run, result = train_experiment(config)
    assert result["test"]["n_users"] == 4
    assert (run / "best.pt").is_file() and (run / "last.pt").is_file()
    _, _, metadata, first, _, _ = assemble(config)
    load_checkpoint(run / "best.pt", first, config, metadata["fingerprint"])
    _, _, _, second, _, _ = assemble(config)
    load_checkpoint(run / "best.pt", second, config, metadata["fingerprint"])
    first.eval()
    second.eval()
    # LGMRec samples its global hypergraph during evaluation, as in the source.
    torch.manual_seed(321)
    a = first.make_scorer().score(torch.arange(4), torch.arange(6))
    torch.manual_seed(321)
    b = second.make_scorer().score(torch.arange(4), torch.arange(6))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("model_name", ["pgl", "lgmrec"])
@pytest.mark.parametrize("dataset", ["baby", "sports", "clothing", "elec", "microlens"])
def test_existing_dataset_configs_and_damps_switch(model_name, dataset):
    baseline = load_config(model=model_name, dataset=dataset)
    full = load_config(model=model_name, dataset=dataset, overrides=["model.damps_enabled=true"])
    expected = deepcopy(baseline)
    expected["model"]["damps_enabled"] = True
    assert full == expected


@pytest.mark.parametrize("model_name", ["pgl", "lgmrec"])
def test_existing_ablation_entry_runs_baseline_and_full(tiny_config, model_name):
    config = config_for(tiny_config, model_name, False)
    output = run_ablation(config, [999], ["baseline", "full"])
    records = json.loads((output / "runs.json").read_text())
    assert [record["variant"] for record in records] == ["baseline", "full"]
    assert all(record["result"]["test"]["n_users"] == 4 for record in records)


@pytest.mark.parametrize("enabled", [False, True])
def test_pgl_global_mode(tiny_config, enabled):
    config = config_for(tiny_config, "pgl", enabled)
    config["model"]["mode"] = "global"
    _, _, _, model, _, _ = assemble(config)
    model.on_epoch_start(1)
    batch = TrainBatch(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([[3], [4]]))
    loss = model.compute_loss(batch).total
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("model_name", ["pgl", "lgmrec"])
def test_concat_preserves_topology_and_hypergraph(tiny_config, model_name):
    baseline_config = config_for(tiny_config, model_name, False)
    enabled_config = deepcopy(baseline_config)
    enabled_config["model"]["damps_enabled"] = True
    torch.manual_seed(999)
    _, _, _, baseline, _, _ = assemble(baseline_config)
    torch.manual_seed(999)
    _, _, _, enabled, _, _ = assemble(enabled_config)
    if model_name == "pgl":
        torch.testing.assert_close(enabled.item_graph.to_dense(), baseline.item_graph.to_dense(), rtol=0, atol=0)
        graph_before = enabled.item_graph.to_dense().clone()
        with torch.no_grad():
            enabled.damps.image_trs.weight.add_(10)
        torch.testing.assert_close(enabled.item_graph.to_dense(), graph_before, rtol=0, atol=0)
    else:
        for name in ("image_projection", "text_projection", "image_attributes", "text_attributes"):
            torch.testing.assert_close(getattr(enabled, name), getattr(baseline, name), rtol=0, atol=0)
        assert enabled.image_projection.requires_grad
        assert enabled.text_projection.requires_grad
        assert enabled.image_attributes.requires_grad
        # Changing local DAMPS outputs cannot change the global branch.
        enabled.eval()
        torch.manual_seed(321)
        first = enabled.encode()[2]
        with torch.no_grad():
            enabled.damps.image_trs.weight.add_(10)
        torch.manual_seed(321)
        second = enabled.encode()[2]
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def pgl_concat_configuration(tiny_config):
    cfg=config_for(tiny_config,'pgl',True)

    return cfg


def test_pgl_concat_all_channels_gradients_and_score_decomposition(tiny_config):
    cfg=pgl_concat_configuration(tiny_config)
    _,_,_,model,_,_=assemble(cfg)
    model.on_epoch_start(1)
    batch=TrainBatch(torch.tensor([0,1]),torch.tensor([1,2]),torch.tensor([[3],[4]]))
    model.compute_loss(batch).total.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert model.image_embedding.weight.data_ptr()!=model.damps.image_embedding.weight.data_ptr()
    model.eval();u,i=model.encode(model.full_graph)
    assert u.shape==(4,16) and i.shape==(6,16)
    score=u@i.T
    split_score=sum(a@b.T for a,b in zip(u.split(4,dim=1),i.split(4,dim=1)))
    torch.testing.assert_close(score,split_score)


def test_pgl_concat_preserves_baseline_initial_parameters_and_graph(tiny_config):
    cfg=pgl_concat_configuration(tiny_config);baseline=deepcopy(cfg)
    baseline['model'].update(name='pgl',damps_enabled=False)
    torch.manual_seed(999);_,_,_,a,_,_=assemble(baseline)
    torch.manual_seed(999);_,_,_,b,_,_=assemble(cfg)
    torch.testing.assert_close(a.item_graph.to_dense(),b.item_graph.to_dense(),rtol=0,atol=0)
    for name,p in a.named_parameters():
        torch.testing.assert_close(p,dict(b.named_parameters())[name],rtol=0,atol=0)


def test_pgl_concat_checkpoint_roundtrip(tiny_config):
    cfg=pgl_concat_configuration(tiny_config);run,result=train_experiment(cfg)
    _,_,metadata,a,_,_=assemble(cfg);load_checkpoint(run/'best.pt',a,cfg,metadata['fingerprint'])
    _,_,_,b,_,_=assemble(cfg);load_checkpoint(run/'best.pt',b,cfg,metadata['fingerprint'])
    a.eval();b.eval()
    torch.testing.assert_close(a.make_scorer().score(torch.arange(4),torch.arange(6)),b.make_scorer().score(torch.arange(4),torch.arange(6)),rtol=0,atol=0)


def lgmrec_concat_configuration(tiny_config):
    config = config_for(tiny_config, 'lgmrec', True)

    return config


def test_lgmrec_concat_four_channels_and_all_parameter_gradients(tiny_config):
    cfg = lgmrec_concat_configuration(tiny_config)
    _, _, _, model, _, _ = assemble(cfg)
    model.train()
    batch = TrainBatch(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([[3], [4]]))
    model.compute_loss(batch).total.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    model.eval()
    torch.manual_seed(12)
    users, items, hyper = model.encode()
    assert users.shape == (4, 16) and items.shape == (6, 16)
    assert all(x.shape[1] == 16 for x in hyper)
    torch.manual_seed(12)
    collaborative = model._collaborative()
    ci, ct = model.damps()
    local = torch.cat([model._local(x) for x in (
        model.image @ model.image_projection, model.text @ model.text_projection, ci, ct)], -1)
    iu, ii = model._global(model.image, model.image_attributes, collaborative[4:])
    tu, ti = model._global(model.text, model.text_attributes, collaborative[4:])
    expected = collaborative + local + model.config.alpha * torch.nn.functional.normalize(
        torch.cat((iu + tu, ii + ti)), dim=-1)
    torch.testing.assert_close(torch.cat((users, items)), expected)
    # Calibrated local features must not modify raw-feature hypergraph outputs.
    with torch.no_grad():
        model.damps.image_trs.weight.add_(1)
    torch.manual_seed(12)
    changed = model.encode()[2]
    for a, b in zip(hyper, changed):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_lgmrec_concat_checkpoint_roundtrip(tiny_config):
    cfg = lgmrec_concat_configuration(tiny_config)
    run, result = train_experiment(cfg)
    assert result['test']['n_users'] == 4
    _, _, metadata, a, _, _ = assemble(cfg)
    _, _, _, b, _, _ = assemble(cfg)
    for model in (a, b):
        load_checkpoint(run / 'best.pt', model, cfg, metadata['fingerprint'])
        model.eval()
    torch.manual_seed(321)
    first = a.make_scorer().score(torch.arange(4), torch.arange(6))
    torch.manual_seed(321)
    second = b.make_scorer().score(torch.arange(4), torch.arange(6))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def official_config(tiny_config, name):
    config = config_for(tiny_config, name, True)
    config['model']['damps_mode'] = 'official'
    return config


@pytest.mark.parametrize('name',['pgl','lgmrec'])
def test_official_gradient_paths_and_two_steps(tiny_config, name):
    cfg=official_config(tiny_config,name)
    _,_,_,model,_,_=assemble(cfg)
    model.train();model.on_epoch_start(1)
    batch=TrainBatch(torch.tensor([0,1]),torch.tensor([1,2]),torch.tensor([[3],[4]]))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=model.compute_loss(batch).total
        assert torch.isfinite(loss)
        loss.backward();optimizer.step()
    dead=[]
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        if p.grad is None:dead.append(n)
        else:assert torch.isfinite(p.grad).all(),n
    if name=='pgl':
        assert set(dead)=={'image_embedding.weight','text_embedding.weight','image_projection.weight','image_projection.bias','text_projection.weight','text_projection.bias'}
        assert model.damps.phase_residual.grad is not None
    else:
        assert dead and all(n.startswith('damps.') for n in dead)
        assert model.damps.phase_residual.grad is None
        assert not model.image_embedding.weight.requires_grad
        assert model.image_projection.shape[0]==model.image_embedding.weight.shape[1]
    from mmrecsys.nn.damps import DAMPS
    assert isinstance(model.damps, DAMPS)
    assert 'damps.phase_prior' in model.state_dict()
    assert model.image_embedding.weight.shape[1] == cfg['model']['embedding_dim']


@pytest.mark.parametrize('name',['pgl','lgmrec'])
def test_official_checkpoint_roundtrip(tiny_config,name):
    cfg=official_config(tiny_config,name)
    run,result=train_experiment(cfg)
    assert result['test']['n_users']==4
    _,_,metadata,a,_,_=assemble(cfg);load_checkpoint(run/'best.pt',a,cfg,metadata['fingerprint'])
    _,_,_,b,_,_=assemble(cfg);load_checkpoint(run/'best.pt',b,cfg,metadata['fingerprint'])
    a.eval();b.eval();torch.manual_seed(123);x=a.make_scorer().score(torch.arange(4),torch.arange(6))
    torch.manual_seed(123);y=b.make_scorer().score(torch.arange(4),torch.arange(6))
    torch.testing.assert_close(x,y,rtol=0,atol=0)


@pytest.mark.parametrize("name", ["pgl", "lgmrec"])
@pytest.mark.parametrize("component", ["apc", "avrf", "imcf"])
def test_official_style_supports_shared_component_switches(tiny_config, name, component):
    cfg = official_config(tiny_config, name)
    cfg['model']['damps_' + component] = False
    _, _, _, model, _, _ = assemble(cfg)
    model.on_epoch_start(1)
    batch = TrainBatch(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([[3], [4]]))
    loss = model.compute_loss(batch).total
    assert torch.isfinite(loss)
    loss.backward()
