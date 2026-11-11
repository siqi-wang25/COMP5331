import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# coding: utf-8
import os, random, numpy as np, torch

class DAMPS(nn.Module):
    """
    General spectral filtering plugin: supports VR-Mask, IMCF, Phase Alignment, etc., suitable for multimodal feature filtering.
    """
    def __init__(self, embedding_dim, device, raw_image, raw_txt):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.freq_dim = embedding_dim // 2 + 1
        self.raw_image = raw_image
        self.raw_txt = raw_txt
        self.image_embedding = nn.Embedding.from_pretrained(self.raw_image, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.raw_txt, freeze=False)
        self.image_trs = nn.Linear(self.raw_image.shape[1], self.embedding_dim).to(device)
        self.text_trs = nn.Linear(self.raw_txt.shape[1], self.embedding_dim).to(device)
    
        # inint_AVRF
        with torch.no_grad():
            if raw_image is not None:
                proj_image = self.image_trs(raw_image.to(device))
                AVRF_image = self._compute_AVRF(proj_image).to(device)
                self.AVRF_image = nn.Parameter(torch.log(AVRF_image / (1 - AVRF_image + 1e-8)))
            else:
                self.AVRF_img = nn.Parameter(torch.zeros(self.freq_dim, device=device))
            if raw_txt is not None:
                proj_txt = self.text_trs(raw_txt.to(device))
                AVRF_txt = self._compute_AVRF(proj_txt).to(device)
                self.AVRF_txt = nn.Parameter(torch.log(AVRF_txt / (1 - AVRF_txt + 1e-8)))
            else:
                self.AVRF_txt = nn.Parameter(torch.zeros(self.freq_dim, device=device))

        self.lambda_weights = nn.Parameter(torch.tensor([0.6, 0.4], dtype=torch.float32, device=device))

        # inint_Phase Alignment
        with torch.no_grad():
            proj_image_fft, proj_txt_fft = None, None
            if raw_image is not None:
                proj_image = self.image_trs(raw_image.to(device))
                proj_image_fft = torch.fft.rfft(proj_image, dim=1, norm='ortho')
            if raw_txt is not None:
                proj_txt = self.text_trs(raw_txt.to(device))
                proj_txt_fft = torch.fft.rfft(proj_txt, dim=1, norm='ortho')
            if (proj_image_fft is not None) and (proj_txt_fft is not None):
                phase_diff = torch.angle(proj_txt_fft) - torch.angle(proj_image_fft)
                sin_mean = torch.mean(torch.sin(phase_diff), dim=0)   # [F]
                cos_mean = torch.mean(torch.cos(phase_diff), dim=0)   # [F]
                avg_R = torch.atan2(sin_mean, cos_mean)   
            else:
                avg_R = torch.zeros(self.freq_dim, device=device)
        self.avg_R = avg_R.to(device)
        self.psi = nn.Parameter(torch.zeros(self.freq_dim, device=device))


    def _compute_AVRF(self, real_feats: torch.Tensor):
        fft_feats = torch.fft.rfft(real_feats, dim=1, norm='ortho')
        mag = torch.abs(fft_feats).double()
        sigma_tot_sq   = mag.var(dim=0, unbiased=False)
        median_mag     = mag.median(dim=0).values
        mad            = (mag - median_mag).abs().median(dim=0).values
        sigma_intra_sq = (1.4826 * mad)**2
        sigma_inter_sq = torch.clamp(sigma_tot_sq - sigma_intra_sq, min=0.)
        vr = sigma_inter_sq / (sigma_inter_sq + sigma_intra_sq + 1e-6)
        vr = vr.float()
        vr = (vr - vr.mean()) / (vr.std() + 1e-6)
        vr = torch.sigmoid(vr)
        return vr.detach()

    def _imcf_filter(self, APC_embedding_image, APC_embedding_text):
        cross = APC_embedding_image * torch.conj(APC_embedding_text)
        msc = (torch.abs(cross) ** 2) / (torch.abs(APC_embedding_image) ** 2 * torch.abs(APC_embedding_text) ** 2 + 1e-8)
        IMCF_embedding_image = APC_embedding_image * msc
        IMCF_embedding_txt = APC_embedding_text * msc
        return IMCF_embedding_image, IMCF_embedding_txt

    def forward(self, image_embeds, text_embeds):
        """
        input: image_embeds, text_embeds: [N, D]
        output: image_conv, text_conv, fused_conv: [N, D]
        """
        proj_image = self.image_trs(self.image_embedding.weight)  # [N, D]
        proj_txt = self.text_trs(self.text_embedding.weight)     # [N, D]
        proj_image_fft = torch.fft.rfft(proj_image, dim=1, norm='ortho')
        proj_txt_fft = torch.fft.rfft(proj_txt, dim=1, norm='ortho')
        phi = self.avg_R / 2 + self.psi
        rot_image = torch.exp(-1j * phi).view(1, -1)
        rot_txt = torch.exp(+1j * phi).view(1, -1)
        APC_image = proj_image_fft  * rot_image
        APC_txt = proj_txt_fft * rot_txt
        vr_mask_img = self.AVRF_image.view(1, -1)
        vr_mask_txt = self.AVRF_txt.view(1, -1)
        img_fft_vr = APC_image  * vr_mask_img
        txt_fft_vr = APC_txt * vr_mask_txt
        img_fft_tilde, txt_fft_tilde = self._imcf_filter(APC_image, APC_txt)
        lambda_vr, lambda_msc = torch.softmax(self.lambda_weights, dim=0)
        img_fft_late = lambda_vr * img_fft_vr + lambda_msc * img_fft_tilde
        txt_fft_late = lambda_vr * txt_fft_vr + lambda_msc * txt_fft_tilde
        image_conv = torch.fft.irfft(img_fft_late, n=image_embeds.shape[1], dim=1, norm='ortho')
        text_conv = torch.fft.irfft(txt_fft_late, n=text_embeds.shape[1], dim=1, norm='ortho')

         
        return image_conv, text_conv



