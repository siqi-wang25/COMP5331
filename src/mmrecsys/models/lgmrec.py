"""LGMRec local/global graph backbone with optional shared DAMPS calibration."""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .base import DotProductScorer, LossOutput, Recommender
from ..data.views import TrainBatch, TrainData
from ..experiment.artifacts import GraphCache
from ..nn.damps import DAMPS
from ..nn.graph import interaction_graph
from ..nn.losses import bpr_loss


@dataclass(frozen=True)
class LGMRecConfig:
    name: str = "lgmrec"
    embedding_dim: int = 64
    ui_layers: int = 2
    modal_layers: int = 2
    hyper_layers: int = 1
    hyperedges: int = 4
    keep_rate: float = 0.5
    alpha: float = 0.3
    contrastive_weight: float = 1e-4
    regularization_weight: float = 1e-6
    temperature: float = 0.2
    contrastive_chunk: int = 8192
    damps_enabled: bool = False
    damps_mode: str = "concat"
    damps_apc: bool = True
    damps_avrf: bool = True
    damps_imcf: bool = True
    damps_eps: float = 1e-6

    def __post_init__(self):
        if self.damps_mode not in ("concat", "official"):
            raise ValueError("model.damps_mode must be concat or official")
        if self.name != "lgmrec":
            raise ValueError("Expected model.name=lgmrec")
        for field in ("embedding_dim", "ui_layers", "modal_layers", "hyper_layers",
                      "hyperedges", "contrastive_chunk"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"model.{field} must be a positive integer")
        if not 0 < self.keep_rate <= 1 or self.alpha < 0:
            raise ValueError("Invalid LGMRec keep_rate or alpha")
        for field in ("temperature", "damps_eps"):
            value = getattr(self, field)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"model.{field} must be positive and finite")
        for field in ("contrastive_weight", "regularization_weight"):
            value = getattr(self, field)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"model.{field} must be nonnegative and finite")
        for field in ("damps_enabled", "damps_apc", "damps_avrf", "damps_imcf"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"model.{field} must be boolean")

    @classmethod
    def parse(cls, values):
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown LGMRec fields: {sorted(unknown)}")
        return cls(**values)


class LGMRec(Recommender):
    def __init__(self, config: LGMRecConfig, data: TrainData, cache: GraphCache):
        if config.damps_enabled and config.damps_mode == "official":
            self._init_official(config, data, cache)
            return
        super().__init__()
        self.config = config
        self.n_users, self.n_items = data.n_users, data.n_items
        self.register_buffer("ui_graph", cache.get_or_build(
            {"kind": "lgmrec_ui", "train": data.fingerprint},
            lambda: interaction_graph(data.edges, data.n_users, data.n_items)), persistent=False)
        edges = torch.as_tensor(data.edges.copy(), dtype=torch.long)
        incidence = torch.sparse_coo_tensor(edges.T, torch.ones(len(edges)),
                                             (data.n_users, data.n_items), check_invariants=True).coalesce()
        self.register_buffer("incidence", incidence, persistent=False)
        degree = torch.bincount(edges[:, 0], minlength=data.n_users).float()
        self.register_buffer("inverse_user_degree", (degree + 1e-7).reciprocal(), persistent=False)
        image, text = data.features["image"], data.features["text"]
        self.register_buffer("image", image, persistent=False)
        self.register_buffer("text", text, persistent=False)
        dim = config.embedding_dim
        self.user_id = nn.Embedding(data.n_users, dim)
        self.item_id = nn.Embedding(data.n_items, dim)
        nn.init.xavier_uniform_(self.user_id.weight)
        nn.init.xavier_uniform_(self.item_id.weight)
        image_dim = image.shape[1]
        text_dim = text.shape[1]
        self.image_projection = nn.Parameter(torch.empty(image_dim, dim))
        self.text_projection = nn.Parameter(torch.empty(text_dim, dim))
        self.image_attributes = nn.Parameter(torch.empty(image_dim, config.hyperedges))
        self.text_attributes = nn.Parameter(torch.empty(text_dim, config.hyperedges))
        for parameter in (self.image_projection, self.text_projection,
                          self.image_attributes, self.text_attributes):
            nn.init.xavier_uniform_(parameter)
        self.damps = None
        if config.damps_enabled:
            # Match the established four-channel experiment, including init order.
            self.user_id = nn.Embedding(data.n_users, 4 * dim)
            self.item_id = nn.Embedding(data.n_items, 4 * dim)
            nn.init.xavier_uniform_(self.user_id.weight)
            nn.init.xavier_uniform_(self.item_id.weight)
            self.damps = DAMPS(dim, apc=config.damps_apc, avrf=config.damps_avrf,
                               imcf=config.damps_imcf, eps=config.damps_eps,
                               raw_image=image.clone(), raw_text=text.clone(),
                               trainable_features=False)

    def _collaborative(self):
        hidden = torch.cat((self.user_id.weight, self.item_id.weight), dim=0)
        layers = [hidden]
        for _ in range(self.config.ui_layers):
            hidden = torch.sparse.mm(self.ui_graph, hidden)
            layers.append(hidden)
        return torch.stack(layers).mean(0)

    def _local(self, items):
        users = torch.sparse.mm(self.incidence, items) * self.inverse_user_degree[:, None]
        hidden = torch.cat((users, items), dim=0)
        for _ in range(self.config.modal_layers):
            hidden = torch.sparse.mm(self.ui_graph, hidden)
        return F.normalize(hidden, dim=-1)

    def _global(self, raw, attributes, collaborative_items):
        item_logits = raw @ attributes
        user_logits = torch.sparse.mm(self.incidence, item_logits)
        item_h = F.gumbel_softmax(item_logits, tau=self.config.temperature, dim=1)
        user_h = F.gumbel_softmax(user_logits, tau=self.config.temperature, dim=1)
        item_h = F.dropout(item_h, p=1 - self.config.keep_rate, training=self.training)
        user_h = F.dropout(user_h, p=1 - self.config.keep_rate, training=self.training)
        item_state = collaborative_items
        for _ in range(self.config.hyper_layers):
            latent = item_h.T @ item_state
            item_state = item_h @ latent
            user_state = user_h @ latent
        return user_state, item_state

    def encode(self):
        if self.config.damps_enabled and self.config.damps_mode == "official":
            return self._encode_official()
        collaborative = self._collaborative()
        image_items = self.image @ self.image_projection
        text_items = self.text @ self.text_projection
        image_local = self._local(image_items)
        text_local = self._local(text_items)
        if self.damps is None:
            local = image_local + text_local
        else:
            damps_image, damps_text = self.damps()
            local = torch.cat((image_local, text_local,
                               self._local(damps_image), self._local(damps_text)), dim=-1)
        image_user, image_item = self._global(self.image, self.image_attributes,
                                              collaborative[self.n_users:])
        text_user, text_item = self._global(self.text, self.text_attributes,
                                            collaborative[self.n_users:])
        global_sum = torch.cat((image_user + text_user, image_item + text_item), dim=0)
        fused = (collaborative + image_local + text_local if self.damps is None else collaborative + local) + self.config.alpha * F.normalize(global_sum, dim=-1)
        return fused[:self.n_users], fused[self.n_users:], (image_user, image_item, text_user, text_item)

    def _cross_modal(self, anchor, positive, candidates):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        candidates = F.normalize(candidates, dim=-1)
        match = (anchor * positive).sum(-1) / self.config.temperature
        denominator = None
        for start in range(0, len(candidates), self.config.contrastive_chunk):
            block = anchor @ candidates[start:start + self.config.contrastive_chunk].T / self.config.temperature
            part = torch.logsumexp(block, dim=-1)
            denominator = part if denominator is None else torch.logaddexp(denominator, part)
        return (denominator - match).sum()

    def compute_loss(self, batch: TrainBatch) -> LossOutput:
        if batch.negative_items is None or batch.negative_items.shape[1] != 1:
            raise ValueError("LGMRec requires one negative item per positive")
        users, items, hyper = self.encode()
        u, p, n = users[batch.users], items[batch.positive_items], items[batch.negative_items]
        ranking = bpr_loss(u, p, n)
        image_user, image_item, text_user, text_item = hyper
        cl = self._cross_modal(image_user[batch.users], text_user[batch.users], text_user)
        cl = cl + self._cross_modal(image_item[batch.positive_items],
                                    text_item[batch.positive_items], text_item)
        regularizer = (u.norm(p=2) + p.norm(p=2) + n.norm(p=2)) / len(u)
        weighted_cl = self.config.contrastive_weight * cl
        weighted_reg = self.config.regularization_weight * regularizer
        return LossOutput(ranking + weighted_cl + weighted_reg,
                          {"bpr": ranking, "contrastive": weighted_cl,
                           "regularization": weighted_reg}, len(u))

    @torch.no_grad()
    def make_scorer(self):
        if self.training:
            raise RuntimeError("make_scorer requires model.eval()")
        users, items, _ = self.encode()
        return DotProductScorer(users, items)


    # Official call sites using the shared COMP5331 DAMPS implementation.
    def _init_official(self, config: LGMRecConfig, data: TrainData, cache: GraphCache):
        Recommender.__init__(self)
        self.config = config
        self.n_users, self.n_items = data.n_users, data.n_items
        self.register_buffer("ui_graph", cache.get_or_build(
            {"kind": "lgmrec_ui", "train": data.fingerprint},
            lambda: interaction_graph(data.edges, data.n_users, data.n_items)), persistent=False)
        edges = torch.as_tensor(data.edges.copy(), dtype=torch.long)
        incidence = torch.sparse_coo_tensor(edges.T, torch.ones(len(edges)),
                                             (data.n_users, data.n_items), check_invariants=True).coalesce()
        self.register_buffer("incidence", incidence, persistent=False)
        degree = torch.bincount(edges[:, 0], minlength=data.n_users).float()
        self.register_buffer("inverse_user_degree", (degree + 1e-7).reciprocal(), persistent=False)
        image, text = data.features["image"], data.features["text"]
        self.register_buffer("image", image, persistent=False)
        self.register_buffer("text", text, persistent=False)
        dim = config.embedding_dim
        self.user_id = nn.Embedding(data.n_users, dim)
        self.item_id = nn.Embedding(data.n_items, dim)
        nn.init.xavier_uniform_(self.user_id.weight)
        nn.init.xavier_uniform_(self.item_id.weight)
        # Reuse COMP5331's feature-owned DAMPS; its output width is D.
        self.damps = DAMPS(config.embedding_dim, apc=config.damps_apc,
                           avrf=config.damps_avrf, imcf=config.damps_imcf,
                           eps=config.damps_eps, raw_image=image.clone(), raw_text=text.clone())
        with torch.no_grad():
            calibrated_image, calibrated_text = self.damps()
        self.image_embedding = nn.Embedding.from_pretrained(calibrated_image.clone(), freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(calibrated_text.clone(), freeze=True)
        # Shared DAMPS returns D-wide features: adapt the downstream input
        # shapes while retaining the source local/global branch call sites.
        self.image_projection = nn.Parameter(nn.init.xavier_uniform_(torch.empty(calibrated_image.shape[1], dim)))
        self.image_attributes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(calibrated_image.shape[1], config.hyperedges)))
        self.text_projection = nn.Parameter(nn.init.xavier_uniform_(torch.empty(calibrated_text.shape[1], dim)))
        self.text_attributes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(calibrated_text.shape[1], config.hyperedges)))

    def _encode_official(self):
        collaborative = self._collaborative()
        image_items = self.image_embedding.weight @ self.image_projection
        text_items = self.text_embedding.weight @ self.text_projection
        image_local = self._local(image_items)
        text_local = self._local(text_items)
        image_user, image_item = self._global(self.image_embedding.weight, self.image_attributes,
                                              collaborative[self.n_users:])
        text_user, text_item = self._global(self.text_embedding.weight, self.text_attributes,
                                            collaborative[self.n_users:])
        global_sum = torch.cat((image_user + text_user, image_item + text_item), dim=0)
        fused = collaborative + image_local + text_local + self.config.alpha * F.normalize(global_sum, dim=-1)
        return fused[:self.n_users], fused[self.n_users:], (image_user, image_item, text_user, text_item)
