"""PGL backbone with optional calibration through the shared DAMPS module."""
from dataclasses import dataclass
import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch
from torch import nn
import torch.nn.functional as F

from .base import DotProductScorer, LossOutput, Recommender
from ..data.views import TrainBatch, TrainData
from ..experiment.artifacts import GraphCache
from ..nn.damps import DAMPS
from ..nn.graph import interaction_graph, knn_graph
from ..nn.losses import bpr_loss, info_nce


@dataclass(frozen=True)
class PGLConfig:
    name: str = "pgl"
    embedding_dim: int = 64
    ui_layers: int = 2
    item_layers: int = 1
    knn_k: int = 10
    knn_chunk_size: int = 512
    image_weight: float = 0.1
    mode: str = "local"
    local_ratio: float = 0.3
    global_fraction: float = 0.25
    global_threshold: float = 1e-3
    mask_ratio: float = 0.2
    ssl_weight: float = 0.0
    temperature: float = 0.2
    damps_enabled: bool = False
    damps_mode: str = "concat"
    damps_apc: bool = True
    damps_avrf: bool = True
    damps_imcf: bool = True
    damps_eps: float = 1e-6

    def __post_init__(self):
        if self.damps_mode not in ("concat", "official"):
            raise ValueError("model.damps_mode must be concat or official")
        if self.name != "pgl" or self.mode not in ("local", "global"):
            raise ValueError("PGL requires name=pgl and mode=local/global")
        for field in ("embedding_dim", "ui_layers", "item_layers", "knn_k", "knn_chunk_size"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"model.{field} must be a positive integer")
        for field in ("image_weight", "local_ratio", "global_fraction", "mask_ratio"):
            value = getattr(self, field)
            if type(value) not in (float, int) or not 0 <= value <= 1:
                raise ValueError(f"model.{field} must be in [0, 1]")
        for field in ("global_threshold", "temperature", "damps_eps"):
            value = getattr(self, field)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"model.{field} must be positive and finite")
        if type(self.ssl_weight) not in (float, int) or not math.isfinite(self.ssl_weight) or self.ssl_weight < 0:
            raise ValueError("model.ssl_weight must be nonnegative and finite")
        for field in ("damps_enabled", "damps_apc", "damps_avrf", "damps_imcf"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"model.{field} must be boolean")

    @classmethod
    def parse(cls, values):
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown PGL fields: {sorted(unknown)}")
        return cls(**values)


def _official_product_graph(full_graph, rank, fraction, threshold, block_rows=256):
    """The released PGL product of leading and trailing singular values."""
    graph = full_graph.coalesce().cpu()
    row, col = graph.indices().numpy()
    matrix = sp.csr_matrix((graph.values().numpy(), (row, col)), shape=graph.shape)
    rank = min(rank, min(matrix.shape) - 1)
    left_vectors, singular, right_vectors = svds(matrix, k=rank, which="LM")
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    left_vectors = left_vectors[:, order]
    right_vectors = right_vectors[order]
    width = max(1, int(rank * fraction))
    left = (left_vectors[:, :width] * (singular[:width] * singular[-width:])).astype(np.float32)
    right = right_vectors[:width].astype(np.float32)
    rows, cols, values = [], [], []
    for start in range(0, matrix.shape[0], block_rows):
        block = left[start:start + block_rows] @ right
        selected_row, selected_col = np.nonzero(np.abs(block) >= threshold)
        rows.append(selected_row + start)
        cols.append(selected_col)
        values.append(block[selected_row, selected_col])
    if not rows:
        raise RuntimeError("Empty global reconstruction")
    indices = torch.as_tensor(np.stack((np.concatenate(rows), np.concatenate(cols))), dtype=torch.long)
    weights = torch.as_tensor(np.concatenate(values), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, weights, matrix.shape, check_invariants=True).coalesce()


class PGL(Recommender):
    def __init__(self, config: PGLConfig, data: TrainData, cache: GraphCache):
        if config.damps_enabled and config.damps_mode == "official":
            self._init_official(config, data, cache)
            return
        super().__init__()
        self.config = config
        self.n_users, self.n_items = data.n_users, data.n_items
        self.register_buffer("full_graph", cache.get_or_build(
            {"kind": "pgl_ui", "train": data.fingerprint},
            lambda: interaction_graph(data.edges, data.n_users, data.n_items)), persistent=False)
        self.register_buffer("train_edges", torch.as_tensor(data.edges.copy(), dtype=torch.long), persistent=False)
        degree_u = torch.bincount(self.train_edges[:, 0], minlength=data.n_users).float()
        degree_i = torch.bincount(self.train_edges[:, 1], minlength=data.n_items).float()
        edge_weights = ((degree_u[self.train_edges[:, 0]] + 1e-7).rsqrt() *
                        (degree_i[self.train_edges[:, 1]] + 1e-7).rsqrt())
        self.register_buffer("edge_weights", edge_weights, persistent=False)
        self.user_image = nn.Embedding(data.n_users, config.embedding_dim)
        self.user_text = nn.Embedding(data.n_users, config.embedding_dim)
        nn.init.xavier_uniform_(self.user_image.weight)
        nn.init.xavier_uniform_(self.user_text.weight)
        image = data.features["image"]
        text = data.features["text"]
        self.image_embedding = nn.Embedding.from_pretrained(image.clone(), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(text.clone(), freeze=False)
        self.image_projection = nn.Linear(image.shape[1], config.embedding_dim)
        self.text_projection = nn.Linear(text.shape[1], config.embedding_dim)
        self.damps = None
        graph_image, graph_text = image, text
        graphs = []
        for modality, features in (("image", graph_image), ("text", graph_text)):
            options = dict(kind="pgl_item_knn", feature=data.feature_fingerprints[modality],
                           k=config.knn_k, self_loops=True, weight="binary", normalize="symmetric",
                           graph_source="raw_features",
                           damps_apc=config.damps_apc, damps_avrf=config.damps_avrf,
                           damps_imcf=config.damps_imcf, damps_eps=config.damps_eps,
                           adapter_version=5)
            graphs.append(cache.get_or_build(options, lambda features=features: knn_graph(
                features, config.knn_k, chunk_size=config.knn_chunk_size,
                self_loops=True, symmetrize=False, edge_weight="binary", normalization="symmetric")))
        self.register_buffer("item_graph", (config.image_weight * graphs[0] +
                              (1 - config.image_weight) * graphs[1]).coalesce(), persistent=False)
        if config.mode == "global":
            self.register_buffer("subgraph", cache.get_or_build(
                {"kind": "pgl_global_product", "train": data.fingerprint,
                 "rank": config.embedding_dim, "fraction": config.global_fraction,
                 "threshold": config.global_threshold},
                lambda: _official_product_graph(self.full_graph, config.embedding_dim,
                    config.global_fraction, config.global_threshold)), persistent=False)
        else:
            self.register_buffer("subgraph", None, persistent=False)
        self._epoch = 0
        if config.damps_enabled:
            self.user_damps_image = nn.Embedding(data.n_users, config.embedding_dim)
            self.user_damps_text = nn.Embedding(data.n_users, config.embedding_dim)
            nn.init.xavier_uniform_(self.user_damps_image.weight)
            nn.init.xavier_uniform_(self.user_damps_text.weight)
            self.damps = DAMPS(config.embedding_dim, apc=config.damps_apc,
                               avrf=config.damps_avrf, imcf=config.damps_imcf,
                               eps=config.damps_eps, raw_image=image.clone(), raw_text=text.clone())

    def on_epoch_start(self, epoch: int) -> None:
        self._epoch = epoch
        if self.config.mode == "local":
            generator = torch.Generator().manual_seed(epoch * 1_000_003)
            count = max(1, int(len(self.train_edges) * self.config.local_ratio))
            chosen = torch.multinomial(self.edge_weights.cpu(), count, replacement=False,
                                       generator=generator)
            edges = self.train_edges[chosen.to(self.train_edges.device)]
            self.subgraph = interaction_graph(edges.cpu().numpy(), self.n_users,
                                               self.n_items).to(self.full_graph.device)

    def encode(self, graph):
        if self.config.damps_enabled and self.config.damps_mode == "official":
            return self._encode_official(graph)
        image = self.image_projection(self.image_embedding.weight)
        text = self.text_projection(self.text_embedding.weight)
        item_channels = [image, text]
        user_channels = [self.user_image.weight, self.user_text.weight]
        if self.damps is not None:
            item_channels.extend(self.damps())
            user_channels.extend((self.user_damps_image.weight, self.user_damps_text.weight))
        initial_items = torch.cat([F.normalize(x, dim=-1) for x in item_channels], dim=-1)
        initial_users = torch.cat(user_channels, dim=-1)
        item_context = initial_items
        for _ in range(self.config.item_layers):
            item_context = torch.sparse.mm(self.item_graph, item_context)
        hidden = torch.cat((initial_users, initial_items), dim=0)
        layers = [hidden]
        for _ in range(self.config.ui_layers):
            hidden = torch.sparse.mm(graph, hidden)
            layers.append(hidden)
        fused = torch.stack(layers).mean(0)
        return fused[:self.n_users], fused[self.n_users:] + item_context

    def compute_loss(self, batch: TrainBatch) -> LossOutput:
        if batch.negative_items is None or batch.negative_items.shape[1] != 1:
            raise ValueError("PGL requires one negative item per positive")
        if self.subgraph is None:
            self.on_epoch_start(max(self._epoch, 1))
        users, items = self.encode(self.subgraph)
        u, p, n = users[batch.users], items[batch.positive_items], items[batch.negative_items]
        ranking = bpr_loss(u, p, n)
        user_views = (F.dropout(u, p=self.config.mask_ratio, training=True),
                      F.dropout(u, p=self.config.mask_ratio, training=True))
        item_views = (F.dropout(p, p=self.config.mask_ratio, training=True),
                      F.dropout(p, p=self.config.mask_ratio, training=True))
        contrastive = (info_nce(*user_views, self.config.temperature) +
                       info_nce(*item_views, self.config.temperature)) / 2
        weighted = self.config.ssl_weight * contrastive
        return LossOutput(ranking + weighted, {"bpr": ranking, "contrastive": weighted}, len(u))

    @torch.no_grad()
    def make_scorer(self):
        if self.training:
            raise RuntimeError("make_scorer requires model.eval()")
        return DotProductScorer(*self.encode(self.full_graph))


    # Official call sites using the shared COMP5331 DAMPS implementation.
    def _init_official(self, config: PGLConfig, data: TrainData, cache: GraphCache):
        Recommender.__init__(self)
        self.config = config
        self.n_users, self.n_items = data.n_users, data.n_items
        self.register_buffer("full_graph", cache.get_or_build(
            {"kind": "pgl_ui", "train": data.fingerprint},
            lambda: interaction_graph(data.edges, data.n_users, data.n_items)), persistent=False)
        self.register_buffer("train_edges", torch.as_tensor(data.edges.copy(), dtype=torch.long), persistent=False)
        degree_u = torch.bincount(self.train_edges[:, 0], minlength=data.n_users).float()
        degree_i = torch.bincount(self.train_edges[:, 1], minlength=data.n_items).float()
        edge_weights = ((degree_u[self.train_edges[:, 0]] + 1e-7).rsqrt() *
                        (degree_i[self.train_edges[:, 1]] + 1e-7).rsqrt())
        self.register_buffer("edge_weights", edge_weights, persistent=False)
        self.user_text = nn.Embedding(data.n_users, config.embedding_dim)
        self.user_image = nn.Embedding(data.n_users, config.embedding_dim)
        nn.init.xavier_uniform_(self.user_image.weight)
        nn.init.xavier_uniform_(self.user_text.weight)
        image = data.features["image"]
        text = data.features["text"]
        # Reuse COMP5331's feature-owned DAMPS; its output width is D.
        self.damps = DAMPS(config.embedding_dim, apc=config.damps_apc,
                           avrf=config.damps_avrf, imcf=config.damps_imcf,
                           eps=config.damps_eps, raw_image=image.clone(), raw_text=text.clone())
        with torch.no_grad():
            graph_image, graph_text = self.damps()
        self.image_embedding = nn.Embedding.from_pretrained(graph_image.clone(), freeze=False)
        self.image_projection = nn.Linear(graph_image.shape[1], config.embedding_dim)
        self.text_embedding = nn.Embedding.from_pretrained(graph_text.clone(), freeze=False)
        self.text_projection = nn.Linear(graph_text.shape[1], config.embedding_dim)
        graphs = []
        for modality, features in (("image", graph_image), ("text", graph_text)):
            options = dict(kind="pgl_comp5331_official_style_item_knn", feature=data.feature_fingerprints[modality],
                           k=config.knn_k, self_loops=True, weight="binary", normalize="symmetric",
                           graph_source="comp5331_initial_d_width_calibration",
                           seed=torch.initial_seed(),
                           embedding_dim=config.embedding_dim,
                           damps_apc=config.damps_apc, damps_avrf=config.damps_avrf,
                           damps_imcf=config.damps_imcf, damps_eps=config.damps_eps,
                           adapter_version=2)
            graphs.append(cache.get_or_build(options, lambda features=features: knn_graph(
                features, config.knn_k, chunk_size=config.knn_chunk_size,
                self_loops=True, symmetrize=False, edge_weight="binary", normalization="symmetric")))
        self.register_buffer("item_graph", (config.image_weight * graphs[0] +
                              (1 - config.image_weight) * graphs[1]).coalesce(), persistent=False)
        if config.mode == "global":
            self.register_buffer("subgraph", cache.get_or_build(
                {"kind": "pgl_global_product", "train": data.fingerprint,
                 "rank": config.embedding_dim, "fraction": config.global_fraction,
                 "threshold": config.global_threshold},
                lambda: _official_product_graph(self.full_graph, config.embedding_dim,
                    config.global_fraction, config.global_threshold)), persistent=False)
        else:
            self.register_buffer("subgraph", None, persistent=False)
        self._epoch = 0

    def _encode_official(self, graph):
        # Preserve the source adapter's outer projection call site. The shared
        # feature-owned DAMPS validates these D-wide inputs then recomputes its
        # own embeddings/projections, so the outer parameters have no gradient.
        image = self.image_projection(self.image_embedding.weight)
        text = self.text_projection(self.text_embedding.weight)
        image, text = self.damps(image, text)
        initial_items = torch.cat((F.normalize(image, dim=-1), F.normalize(text, dim=-1)), dim=-1)
        initial_users = torch.cat((self.user_image.weight, self.user_text.weight), dim=-1)
        item_context = initial_items
        for _ in range(self.config.item_layers):
            item_context = torch.sparse.mm(self.item_graph, item_context)
        hidden = torch.cat((initial_users, initial_items), dim=0)
        layers = [hidden]
        for _ in range(self.config.ui_layers):
            hidden = torch.sparse.mm(graph, hidden)
            layers.append(hidden)
        fused = torch.stack(layers).mean(0)
        return fused[:self.n_users], fused[self.n_users:] + item_context
