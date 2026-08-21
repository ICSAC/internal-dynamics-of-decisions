#!/usr/bin/env python3
"""Six architectures, two families. Frozen per PREREG-CMC.md sec.2.

Every model is run causally one bin at a time and returns:
    H      : (batch, T, hidden_penult)  per-bin penultimate/hidden trajectory  (demix sees this)
    logits : (batch, T, 2)              per-bin decision logits                (commitment + accuracy)

Recurrent family carries state across bins (RNN/GRU/LSTM); feedforward family recomputes h_t from a
fixed input window ending at t (MLP/TCN/TFM), receptive field >= T so it CAN match the I/O.
Width per architecture is selected by size_to_budget to land in the frozen [60k,160k] param band,
NOT tuned to performance.
"""
import math
import torch
import torch.nn as nn

T = 40
RF = T                         # feedforward receptive field must span the task
PARAM_LO, PARAM_HI = 60_000, 160_000
FAMILY = {"RNN": "recurrent", "GRU": "recurrent", "LSTM": "recurrent",
          "MLP": "feedforward", "TCN": "feedforward", "TFM": "feedforward"}
ARCHS = ["RNN", "GRU", "LSTM", "MLP", "TCN", "TFM"]


# ---------------- recurrent ----------------
class RecNet(nn.Module):
    def __init__(self, kind, hidden):
        super().__init__()
        rnn = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[kind]
        self.rnn = rnn(2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        self.hidden = hidden

    def forward(self, u):
        H, _ = self.rnn(u)           # (B,T,hidden)
        return H, self.head(H)


# ---------------- MLP over sliding window ----------------
def _sliding(u, rf):
    """(B,T,2) -> (B,T,rf*2) left-zero-padded windows ending at each t."""
    B, Tt, C = u.shape
    pad = torch.zeros(B, rf - 1, C, device=u.device, dtype=u.dtype)
    up = torch.cat([pad, u], dim=1)                       # (B, rf-1+T, C)
    idx = torch.arange(Tt, device=u.device)[:, None] + torch.arange(rf, device=u.device)[None, :]
    w = up[:, idx, :]                                     # (B,T,rf,C)
    return w.reshape(B, Tt, rf * C)


class MLPNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.rf = RF
        self.net = nn.Sequential(nn.Linear(RF * 2, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU())
        self.head = nn.Linear(width, 2)
        self.hidden = width

    def forward(self, u):
        w = _sliding(u, self.rf)             # (B,T,rf*2)
        H = self.net(w)                       # (B,T,width)  penultimate
        return H, self.head(H)


# ---------------- dilated causal TCN ----------------
class _CausalConv(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNNet(nn.Module):
    # kernel 3, dilations [1,2,4,8,16] -> RF = 1 + 2*(1+2+4+8+16) = 63 >= T
    def __init__(self, ch):
        super().__init__()
        dils = [1, 2, 4, 8, 16]
        layers = []
        cin = 2
        for d in dils:
            layers += [_CausalConv(cin, ch, 3, d), nn.ReLU()]
            cin = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(ch, 2)
        self.hidden = ch

    def forward(self, u):
        x = u.transpose(1, 2)                 # (B,2,T)
        h = self.tcn(x)                       # (B,ch,T)
        H = h.transpose(1, 2)                 # (B,T,ch) penultimate
        return H, self.head(H)


# ---------------- causal transformer encoder ----------------
class TFMNet(nn.Module):
    def __init__(self, dmodel, nlayers=2, nhead=4):
        super().__init__()
        self.embed = nn.Linear(2, dmodel)
        pe = torch.zeros(T, dmodel)
        pos = torch.arange(T).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(dmodel, nhead, dim_feedforward=2 * dmodel,
                                           batch_first=True, activation="relu")
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(dmodel, 2)
        self.hidden = dmodel

    def forward(self, u):
        B, Tt, _ = u.shape
        x = self.embed(u) + self.pe[:, :Tt, :]
        mask = torch.triu(torch.ones(Tt, Tt, device=u.device, dtype=torch.bool), 1)  # causal
        H = self.enc(x, mask=mask)            # (B,T,dmodel)
        return H, self.head(H)


# ---------------- budget-constrained construction ----------------
def _count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _build(arch, width):
    if arch in ("RNN", "GRU", "LSTM"):
        return RecNet(arch, width)
    if arch == "MLP":
        return MLPNet(width)
    if arch == "TCN":
        return TCNNet(width)
    if arch == "TFM":
        return TFMNet(width)
    raise ValueError(arch)


# candidate widths per arch (must satisfy TFM: width % nhead(4) == 0)
_CANDS = {
    "RNN":  list(range(80, 520, 8)),
    "GRU":  list(range(80, 420, 8)),
    "LSTM": list(range(64, 360, 8)),
    "MLP":  list(range(120, 420, 8)),
    "TCN":  list(range(32, 140, 4)),
    "TFM":  list(range(40, 200, 4)),
}


def size_to_budget(arch):
    """Deterministic: smallest candidate width whose param count >= PARAM_LO (and <= PARAM_HI)."""
    for w in _CANDS[arch]:
        m = _build(arch, w)
        n = _count(m)
        if PARAM_LO <= n <= PARAM_HI:
            return w, n
    # fallback: closest to band
    best = min(_CANDS[arch], key=lambda w: abs(_count(_build(arch, w)) - PARAM_LO))
    return best, _count(_build(arch, best))


def make(arch):
    w, n = size_to_budget(arch)
    return _build(arch, w), w, n


if __name__ == "__main__":
    for a in ARCHS:
        m, w, n = make(a)
        print(f"{a:4s} family={FAMILY[a]:11s} width={w:4d} params={n:,}")
