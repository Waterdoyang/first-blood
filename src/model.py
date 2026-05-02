from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerExtractor(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float, feature_dim: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.feature_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, feature_dim))
        self.cls_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        z = self.proj(x)
        z = self.pe(z)
        z = self.encoder(z)
        z = z.mean(dim=1)
        feat = self.feature_head(z)
        logits = self.cls_head(feat)
        return feat, logits
