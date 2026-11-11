"""DAMPS spectral operations aligned with the supplied source; see docs/damps.md."""
import math

import torch
from torch import nn


class DAMPS(nn.Module):
    """Joint calibration of all items' projected image/text representations."""

    def __init__(self, dim: int, *, apc=True, avrf=True, imcf=True, eps=1e-6,
                 raw_image=None, raw_text=None, trainable_features=True):
        super().__init__()
        if type(dim) is not int or dim < 1:
            raise ValueError("DAMPS dim must be a positive integer")
        if any(type(flag) is not bool for flag in (apc, avrf, imcf)):
            raise ValueError("DAMPS component switches must be boolean")
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("DAMPS eps must be finite and positive")
        if (raw_image is None) != (raw_text is None):
            raise ValueError("DAMPS requires both raw image and text features")
        self.owns_features = raw_image is not None
        if self.owns_features:
            if (raw_image.ndim != 2 or raw_text.ndim != 2
                    or raw_image.shape[0] != raw_text.shape[0] or raw_image.shape[0] == 0):
                raise ValueError("DAMPS raw features must have matching nonempty item rows")
            # Like the author, register separate Parameters over the supplied storage.
            self.image_embedding = nn.Embedding.from_pretrained(raw_image, freeze=not trainable_features)
            self.text_embedding = nn.Embedding.from_pretrained(raw_text, freeze=not trainable_features)
            self.image_trs = nn.Linear(raw_image.shape[1], dim).to(raw_image)
            self.text_trs = nn.Linear(raw_text.shape[1], dim).to(raw_text)
        self.dim, self.apc, self.avrf, self.imcf, self.eps = dim, apc, avrf, imcf, eps
        self.register_parameter("phase_residual", nn.Parameter(torch.zeros(dim // 2 + 1)) if apc else None)
        self.register_parameter("avrf_image", nn.Parameter(torch.zeros(dim // 2 + 1)) if avrf else None)
        self.register_parameter("avrf_text", nn.Parameter(torch.zeros(dim // 2 + 1)) if avrf else None)
        self.register_buffer("phase_prior", torch.zeros(dim // 2 + 1) if apc else None)
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("implementation_version", torch.tensor(3))
        self.register_parameter("mix_logits", nn.Parameter(torch.tensor([0.6, 0.4])) if avrf and imcf else None)

        if self.owns_features:
            self.to(raw_image)
            with torch.no_grad():
                self.initialize(*self.project_features())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = state_dict.get(prefix + "implementation_version")
        if version is None or version.numel() != 1 or version.item() != 3:
            error_msgs.append("DAMPS phase-rotation version mismatch: expected version 3; "
                              "start a new run instead of resuming a different rotation checkpoint")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def project_features(self):
        """Author DAMPS uses its own embeddings and projections for all items."""
        return (self.image_trs(self.image_embedding.weight),
                self.text_trs(self.text_embedding.weight))

    @torch.no_grad()
    def initialize(self, image, text):
        """Estimate once from initial full-item projections, before optimizer creation."""
        if self.initialized.item():
            raise RuntimeError("DAMPS statistics have already been initialized")
        self._validate_inputs(image, text)
        zi = torch.fft.rfft(image, dim=-1, norm="ortho")
        zt = torch.fft.rfft(text, dim=-1, norm="ortho")
        if self.apc:
            difference = torch.angle(zt) - torch.angle(zi)
            sine, cosine = difference.sin().mean(0), difference.cos().mean(0)
            undefined = (sine == 0) & (cosine == 0)
            theta = torch.atan2(torch.where(undefined, torch.zeros_like(sine), sine),
                                torch.where(undefined, torch.ones_like(cosine), cosine))
            self.phase_prior.copy_(theta)
        if self.avrf:
            self.avrf_image.copy_(self.initial_variance_weight(zi))
            self.avrf_text.copy_(self.initial_variance_weight(zt))
        self.initialized.fill_(True)

    def calibrate_phase(self, image, text):
        # Preserve the source's image-/text+ signs; only the residual is learned.
        rotation = self.phase_prior / 2 + self.phase_residual
        return image * torch.exp(-1j * rotation), text * torch.exp(1j * rotation)

    def initial_variance_weight(self, spectrum):
        # Author initialization: lower median, double statistics, standardized VR.
        amplitude = spectrum.abs().double()
        median = amplitude.median(dim=0).values
        mad = (amplitude - median).abs().median(dim=0).values
        noise = (1.4826 * mad).square()
        signal = (amplitude.var(dim=0, correction=0) - noise).clamp_min(0)
        ratio = (signal / (signal + noise + self.eps)).float()
        # Sample std matches the source; a single frequency has zero spread.
        spread = ratio.std() if ratio.numel() > 1 else ratio.new_zeros(())
        probability = torch.sigmoid((ratio - ratio.mean()) / (spread + 1e-6))
        return torch.log(probability / (1 - probability + 1e-8))

    @staticmethod
    def coherence_filter(image, text):
        # Source IMCF: epsilon suppresses bins with tiny joint power.
        cross = image * text.conj()
        power = image.abs().square() * text.abs().square()
        return cross.abs().square() / (power + 1e-8)

    def _validate_inputs(self, image, text):
        if image.ndim != 2 or image.shape != text.shape or image.shape[1] != self.dim or image.shape[0] == 0:
            raise ValueError("DAMPS expects matching nonempty [n_items, dim] tensors")
        if image.dtype != text.dtype or image.device != text.device or image.dtype not in (torch.float32, torch.float64):
            raise ValueError("DAMPS expects float32/float64 inputs on the same device with the same dtype")

    def forward(self, image=None, text=None):
        if (image is None) != (text is None):
            raise ValueError("DAMPS requires both image and text inputs")
        if image is not None:
            self._validate_inputs(image, text)
        elif not self.owns_features:
            raise ValueError("DAMPS without owned features requires image and text inputs")
        if self.owns_features:
            image, text = self.project_features()
        if (self.apc or self.avrf) and not self.initialized.item():
            raise RuntimeError("Initialize DAMPS from full-item projections before training")
        if not (self.apc or self.avrf or self.imcf):
            return image, text
        zi, zt = torch.fft.rfft(image, dim=-1, norm="ortho"), torch.fft.rfft(text, dim=-1, norm="ortho")
        if self.apc:
            zi, zt = self.calibrate_phase(zi, zt)
        branches = []
        if self.avrf:
            branches.append((self.avrf_image * zi, self.avrf_text * zt))
        if self.imcf:
            coherence = self.coherence_filter(zi, zt)
            branches.append((coherence * zi, coherence * zt))
        if len(branches) == 2:
            weights = self.mix_logits.softmax(dim=0)
            zi, zt = (weights[0] * branches[0][m] + weights[1] * branches[1][m] for m in (0, 1))
        elif branches:
            zi, zt = branches[0]
        # irfft discards imaginary DC/Nyquist components, as before.
        return torch.fft.irfft(zi, n=self.dim, dim=-1, norm="ortho"), torch.fft.irfft(zt, n=self.dim, dim=-1, norm="ortho")
