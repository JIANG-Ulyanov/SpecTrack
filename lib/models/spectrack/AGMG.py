import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class Mlp(nn.Module):

    def __init__(self, in_features: int, hidden_features: Union[int, None]=None, out_features: Union[int, None]=None, act_layer=nn.GELU, drop: float=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ExtremeModalityGenerator(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, kdim=hidden_dim, vdim=hidden_dim) for _ in range(2)])
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, exist_feats: list[torch.Tensor], mask_feat: torch.Tensor):
        context = torch.cat(exist_feats, dim=1)
        for attn in self.cross_attn:
            (mask_feat, _) = attn(mask_feat, context, context)
        gate = self.gate(torch.cat([mask_feat.mean(dim=1), context.mean(dim=1)], dim=-1))
        return gate.unsqueeze(1) * mask_feat

class MoE_AQA(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.in_proj = nn.ModuleDict({'rgb': self._build_proj(in_dim, hidden_dim), 'dte': self._build_proj(in_dim, hidden_dim)})
        self.alpha_logits = nn.ParameterDict({'rgb': nn.Parameter(torch.tensor(0.0)), 'dte': nn.Parameter(torch.tensor(0.0))})
        self.experts = nn.ModuleDict({'rgb': self._build_expert(2 * hidden_dim, hidden_dim), 'dte': self._build_expert(2 * hidden_dim, hidden_dim)})
        self.generators = nn.ModuleDict({'rgb': ExtremeModalityGenerator(hidden_dim), 'dte': ExtremeModalityGenerator(hidden_dim)})
        self.router = Mlp(hidden_dim, hidden_dim // 2, 2, drop=dropout)
        self.fusion = nn.Sequential(nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True))
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def _build_proj(self, in_dim: int, hidden_dim: int):
        mid = max(1, in_dim // 2)
        return nn.Sequential(nn.Conv1d(in_dim, mid, kernel_size=1), nn.BatchNorm1d(mid), nn.ReLU(inplace=True), nn.Conv1d(mid, hidden_dim, kernel_size=1), nn.BatchNorm1d(hidden_dim))

    def _build_expert(self, in_dim: int, out_dim: int):
        mid = max(1, out_dim // 2)
        return nn.Sequential(nn.Conv1d(in_dim, mid, kernel_size=3, padding=1), nn.BatchNorm1d(mid), nn.GELU(), nn.Conv1d(mid, out_dim, kernel_size=1), nn.BatchNorm1d(out_dim))

    def _proj(self, x: torch.Tensor, key: str):
        return self.in_proj[key](x.transpose(1, 2)).transpose(1, 2)

    def _expert(self, x: torch.Tensor, key: str):
        return self.experts[key](x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x_rgb, x_dte, rgb_mask, dte_mask):
        (B, L, _) = x_rgb.shape
        rgb_m = rgb_mask.view(B, 1, 1)
        dte_m = dte_mask.view(B, 1, 1)
        proj_rgb = self._proj(x_rgb, 'rgb')
        proj_dte = self._proj(x_dte, 'dte')
        proj_rgb = proj_rgb * rgb_m
        proj_dte = proj_dte * dte_m
        rgb_gen = self.generators['rgb']([proj_dte], torch.zeros_like(proj_rgb))
        dte_gen = self.generators['dte']([proj_rgb], torch.zeros_like(proj_dte))
        feat_rgb = proj_rgb * rgb_m + rgb_gen * (1.0 - rgb_m)
        feat_dte = proj_dte * dte_m + dte_gen * (1.0 - dte_m)
        alpha_rgb = torch.sigmoid(self.alpha_logits['rgb'])
        alpha_dte = torch.sigmoid(self.alpha_logits['dte'])
        h_rgb = self._expert(torch.cat([alpha_rgb * feat_rgb, (1.0 - alpha_rgb) * feat_dte], dim=-1), 'rgb')
        h_dte = self._expert(torch.cat([alpha_dte * feat_rgb, (1.0 - alpha_dte) * feat_dte], dim=-1), 'dte')
        w_rgb = F.softmax(self.router(feat_rgb), dim=-1)
        fused_rgb = w_rgb[..., 0:1] * h_rgb + w_rgb[..., 1:2] * h_dte
        w_dte = F.softmax(self.router(feat_dte), dim=-1)
        fused_dte = w_dte[..., 0:1] * h_rgb + w_dte[..., 1:2] * h_dte
        cat = torch.cat([fused_rgb, fused_dte], dim=-1)
        fusion = self.fusion(cat.transpose(1, 2)).transpose(1, 2)
        out = self.out_proj(fusion)
        recon_loss = torch.tensor(0.0, device=x_rgb.device, dtype=x_rgb.dtype)
        if self.training:
            both = (rgb_mask * dte_mask).view(B, 1, 1)
            n_both = both.sum().clamp(min=1.0)
            mse_rgb = (F.mse_loss(rgb_gen, proj_rgb.detach(), reduction='none') * both).sum() / (n_both * L * self.hidden_dim)
            mse_dte = (F.mse_loss(dte_gen, proj_dte.detach(), reduction='none') * both).sum() / (n_both * L * self.hidden_dim)
            recon_loss = recon_loss + mse_rgb + mse_dte
        return {'output': out, 'recon_loss': recon_loss}
