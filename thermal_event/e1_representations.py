"""E1: which event representation should the event branch use?

Every event input from the 15/11 plan goes through the same small CNN (or, for the two raw-event
methods, a network of the same size class), with the same training recipe and the same
leave-one-recording-out folds. Only the input changes.

Usage (from thermal_event/):
    python e1_representations.py --cache cache --e1-cache e1_cache --out results/e1
    python e1_representations.py ... --smoke          # 1 epoch, 1 seed, 24 samples per recording

--cache     the existing caches (labels, thermal, 40 ms count bins)
--e1-cache  output of build_e1_cache.py, one folder per recording (exact-timing inputs)

Runs are saved one by one in <out>/runs/, so rerunning the same command resumes where it stopped.
summary.md is rewritten after every seed, so a run cut short still leaves complete results for the
seeds that finished.

Writes:
    summary.md          all tables (start here)
    metrics.csv         main metrics with 95% CIs, per model
    per_class.csv       F1 per class per model
    per_recording.csv   macro F1 per test recording per model
    confusion.csv       confusion counts per model (summed over seeds)
    predictions.csv     per sample and zone: truth and each model's prediction (seed 0)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset
import representations as R
from e1_e2 import THERMAL_NOISE, ZoneNet, folds, grid_pool, macro_f1
from labels import CLASSES, ZONES

N_Z, N_C = len(ZONES), len(CLASSES)
EV_W, EV_S = (16, 32, 64, 64, 64), (2, 2, 2, 2, 1)   # event CNN, same as e1_e2.py
TH_W, TH_S = (16, 32, 64, 64, 64), (1, 2, 1, 2, 1)   # thermal CNN, same as e1_e2.py
HISTORY = 6                                          # 40 ms bins loaded per sample (240 ms)
BOOT = 1000
BLOCK = 40                                           # bootstrap block: 40 samples = 10 s
BASE = "count_40"                                    # the input we use now; every Δ is against it
HEAVY = ("aegnn", "eventmg_style")                   # run in their own notebook (e1_graphs_kaggle.ipynb)


# ---------------------------------------------------------------- spiking network

class SpikeFn(torch.autograd.Function):
    """Heaviside spike forward; sigmoid surrogate gradient backward (alpha = 4, SpikingJelly's default;
    Li et al. use "the sigmoid function as surrogate gradients")."""

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0).float()

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        sg = torch.sigmoid(4.0 * v)
        return g * 4.0 * sg * (1 - sg)


class SpikingZoneNet(nn.Module):
    """The event CNN with every ReLU replaced by leaky integrate-and-fire neurons, run over time steps.

    Input (B, T, C, H, W). Neurons as in Li et al. (SpikingJelly LIF, tau 2, threshold 1):
    v <- v + (input - v) / tau; spike when v >= 1, then reset v to 0.
    The last layer's spikes are pooled to the same 2 x 3 grid and averaged over time; a normal linear
    layer reads them out (non-spiking readout)."""

    def __init__(self, in_ch=2, widths=EV_W, strides=EV_S, tau=2.0):
        super().__init__()
        self.convs, self.bns, c = nn.ModuleList(), nn.ModuleList(), in_ch
        for w, s in zip(widths, strides):
            self.convs.append(nn.Conv2d(c, w, 3, s, 1, bias=False))
            self.bns.append(nn.BatchNorm2d(w))
            c = w
        self.tau, self.dim = tau, c * 6
        self.head = nn.Linear(self.dim, N_Z * N_C)
        self.spike_rate = float("nan")

    def embed(self, x):
        b, t = x.shape[:2]
        h, rates = x.flatten(0, 1), []
        for conv, bn in zip(self.convs, self.bns):
            cur = bn(conv(h))
            cur = cur.view(b, t, *cur.shape[1:])
            v, out = torch.zeros_like(cur[:, 0]), []
            for k in range(t):
                v = v + (cur[:, k] - v) / self.tau
                s = SpikeFn.apply(v - 1.0)
                v = v * (1 - s)                     # hard reset to 0
                out.append(s)
            h = torch.stack(out, 1).flatten(0, 1)
            if not self.training:
                rates.append(h.mean())
        if rates:
            self.spike_rate = float(torch.stack(rates).mean())
        return grid_pool(h).view(b, t, -1).mean(1)

    def forward(self, x):
        return self.head(self.embed(x)).view(-1, N_Z, N_C)


# ---------------------------------------------------------------- graph network

class EdgeConv(nn.Module):
    """DGCNN edge convolution: h_i <- max over neighbours j of MLP([h_i, h_j - h_i])."""

    def __init__(self, i, o):
        super().__init__()
        self.lin, self.bn = nn.Linear(2 * i, o, bias=False), nn.BatchNorm1d(o)

    def forward(self, h, idx):
        b, n, c = h.shape
        k = idx.shape[-1]
        nb = torch.gather(h, 1, idx.reshape(b, n * k, 1).expand(-1, -1, c)).view(b, n, k, c)
        hi = h.unsqueeze(2).expand(-1, -1, k, -1)
        e = self.lin(torch.cat([hi, nb - hi], -1))
        e = F.relu(self.bn(e.reshape(-1, e.shape[-1])).view_as(e))
        return e.max(2).values


class GraphZoneNet(nn.Module):
    """Events become graph nodes linked to their k nearest neighbours in (x, y, time); no image.

    Node features: x, y (scaled to 0-1), age in the window (0-1), polarity (+-1). Distance treats
    10 ms as 1 pixel. Three edge convolutions, then node features are averaged inside the same
    2 x 3 grid of cells the CNNs pool to, and a linear layer gives the zone predictions."""

    def __init__(self, widths=(16, 32, 64), k=16, h=240, w=320):
        super().__init__()
        self.k, self.h, self.w = k, h, w
        self.layers, c = nn.ModuleList(), 4
        for o in widths:
            self.layers.append(EdgeConv(c, o))
            c = o
        self.dim = c * 6
        self.head = nn.Linear(self.dim, N_Z * N_C)
        # cell borders identical to grid_pool's tensor_split of the 240 x 320 image
        self.cols = [int(s[0]) for s in torch.tensor_split(torch.arange(w), 3)[1:]]
        self.row = int(torch.tensor_split(torch.arange(h), 2)[1][0])

    def embed(self, pts):
        x, y, age, p = pts.unbind(-1)
        coords = torch.stack([x, y, age * R.GRAPH_T_PX], -1)
        idx = torch.cdist(coords, coords).topk(self.k, largest=False).indices
        h = torch.stack([x / self.w, y / self.h, age / R.WINDOW_S, 2 * p - 1], -1)
        for layer in self.layers:
            h = layer(h, idx)
        cell = (y >= self.row).long() * 3 + (x >= self.cols[0]).long() + (x >= self.cols[1]).long()
        onehot = F.one_hot(cell, 6).float()                                  # (B, N, 6)
        pooled = torch.einsum("bnc,bnf->bcf", h, onehot) / onehot.sum(1, keepdim=True).clamp(min=1)
        return pooled.flatten(1)

    def forward(self, pts):
        return self.head(self.embed(pts)).view(-1, N_Z, N_C)


# ---------------------------------------------------------------- AEGNN (paper version of the graph network)
# Rebuilt from the official code (github.com/uzh-rpg/aegnn, MIT): graph_res.py with its N-Cars settings,
# ncars.py preprocessing, normalization.py, event_dm.py. The Gen1 setting (kernel 8, 16-128 channels,
# 25,000 events, 128 neighbours) rebuilt the same way has 20.2 M parameters (EventMG's Table 1 lists AEGNN
# at 20 M) and ~49 G multiply-adds per sample here: about 200x our CNNs and too slow to train in our
# budget, so the smaller N-Cars setting from the same code is used. Needs torch_spline_conv and torch_cluster (the same
# kernels PyTorch Geometric's SplineConv and radius_graph use); the thin wrappers below follow PyG 2.0.4,
# the version AEGNN pins.
AEGNN_R, AEGNN_DMAX, AEGNN_BETA = 3.0, 32, 0.5e-5    # ncars.py: r=3.0, d_max=32; normalize_time beta (per µs)
AEGNN_KERNEL, AEGNN_CH = 2, (1, 8, 16, 16, 16, 32, 32, 32)  # graph_res.py, "ncars" (its 9th entry is unused)
AEGNN_POOL5, AEGNN_POOL7 = (16, 12), 4               # MaxPooling(16 x 12 px); MaxPoolingX(input // 4, size=16)
AEGNN_MICRO = 4                                      # graphs per forward pass; gradients accumulated to batch 32


def cartesian(pos, ei, max_value=None):
    """PyG Cartesian(norm=True, cat=False): relative position of each edge, scaled into [0, 1]."""
    cart = pos[ei[0]] - pos[ei[1]]
    if cart.numel() == 0:                       # graph without edges (e.g. a window with almost no events)
        return cart
    m = max_value if max_value is not None else cart.abs().max().clamp(min=1e-12)
    return cart / (2 * m) + 0.5


class SplineConvLayer(nn.Module):
    """PyG SplineConv(bias=False, root_weight=False): open B-splines of degree 1, mean over neighbours."""

    def __init__(self, i, o, dim=3, k=AEGNN_KERNEL):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k ** dim, i, o))
        bound = 1.0 / (k ** dim * i) ** 0.5                      # PyG: uniform(weight.size(0) * size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        self.register_buffer("ks", torch.full((dim,), k, dtype=torch.long))
        self.register_buffer("open", torch.ones(dim, dtype=torch.uint8))
        self.macs = 0

    def forward(self, x, ei, ea):
        if ei.shape[1] == 0 or x.shape[0] == 0:
            # no edges: every node has no neighbours, so the mean over neighbours is 0. Without this guard the
            # spline CUDA kernels launch with zero blocks and fail with "CUDA error: invalid argument".
            self.macs = 0
            return x.new_zeros(x.shape[0], self.weight.shape[2]) + 0 * self.weight.sum()
        from torch_spline_conv import spline_basis, spline_weighting
        basis, widx = spline_basis(ea, self.ks, self.open, 1)
        msg = spline_weighting(x[ei[0]], self.weight, basis, widx)
        out = torch.zeros(x.shape[0], msg.shape[1], device=x.device, dtype=msg.dtype).index_add_(0, ei[1], msg)
        self.macs = ei.shape[1] * basis.shape[1] * self.weight.shape[1] * self.weight.shape[2]
        return out / torch.bincount(ei[1], minlength=x.shape[0]).clamp(min=1).unsqueeze(1)


def voxel_pool(x, pos, batch, ei, size):
    """AEGNN MaxPooling: voxel_grid over (x, y) + PyG max_pool. Max features, mean positions, merged edges."""
    if pos.shape[0] == 0:   # a batch whose windows hold no events at all: nothing to pool (min() would fail)
        return x, pos, batch, ei
    start = pos[:, :2].min(0).values
    cell = ((pos[:, :2] - start) / torch.tensor(size, device=pos.device)).floor().long()
    ncell = cell.max(0).values + 1
    key = (batch * ncell[1] + cell[:, 1]) * ncell[0] + cell[:, 0]
    uniq, c = torch.unique(key, return_inverse=True)
    n = len(uniq)
    xo = torch.full((n, x.shape[1]), float("-inf"), device=x.device).scatter_reduce(
        0, c.unsqueeze(1).expand_as(x), x, "amax", include_self=True)
    cnt = torch.bincount(c, minlength=n).clamp(min=1).unsqueeze(1).float()
    po = torch.zeros(n, pos.shape[1], device=pos.device).index_add_(0, c, pos) / cnt
    bo = torch.zeros(n, dtype=torch.long, device=pos.device).scatter_(0, c, batch)
    e = c[ei]
    e = e[:, e[0] != e[1]]
    e = torch.unique(e, dim=1)
    return xo, po, bo, e


class AEGNN(nn.Module):
    """AEGNN's GraphRes (N-Cars settings) on our zone task; only the last layer's size differs (12 outputs).

    Input per sample: the 10,000 stored events (x, y, age, polarity, valid). As in AEGNN's code:
    node feature = polarity (0/1, as their Gen1 loader; their N-Caltech101 loader uses -1/+1); position =
    (x, y, (t - t_min) in µs x 0.5e-5); radius graph r = 3, at most 32 neighbours; edge attributes =
    relative positions scaled into [0, 1]. Assumption: N-Cars's time unit is not visible in their code
    (its loader does no conversion), so µs is used as for Gen1.
    The whole window is processed at once; AEGNN's event-by-event mode gives the same output."""

    def __init__(self, h=240, w=320):
        super().__init__()
        n = AEGNN_CH
        self.convs = nn.ModuleList(SplineConvLayer(n[i], n[i + 1]) for i in range(7))
        self.norms = nn.ModuleList(nn.BatchNorm1d(n[i + 1]) for i in range(7))
        self.cell = (w // AEGNN_POOL7, h // AEGNN_POOL7)
        self.fc = nn.Linear(n[-1] * AEGNN_POOL7 ** 2, N_Z * N_C, bias=False)
        self.extra_macs = 0

    def block(self, i, x, ei, ea):
        return self.norms[i](F.elu(self.convs[i](x, ei, ea)))

    def forward(self, pts):
        from torch_cluster import radius_graph
        b = pts.shape[0]
        valid = pts[..., 4] > 0
        batch = torch.arange(b, device=pts.device).repeat_interleave(valid.sum(1))
        p = pts[valid]
        age_us = p[:, 2] * 1e6
        oldest = torch.zeros(b, device=pts.device).scatter_reduce(0, batch, age_us, "amax", include_self=False)
        pos = torch.stack([p[:, 0], p[:, 1], (oldest[batch] - age_us) * AEGNN_BETA], 1)  # normalize_time
        x = p[:, 3:4]
        ei = radius_graph(pos, r=AEGNN_R, batch=batch, loop=False, max_num_neighbors=AEGNN_DMAX)
        ea = cartesian(pos, ei, AEGNN_R)
        x = self.block(0, x, ei, ea)
        x = self.block(1, x, ei, ea)
        sc = x
        x = self.block(2, x, ei, ea)
        x = self.block(3, x, ei, ea) + sc
        x = self.block(4, x, ei, ea)
        x, pos, batch, ei = voxel_pool(x, pos, batch, ei, AEGNN_POOL5)
        ea = cartesian(pos, ei)
        sc = x
        x = self.block(5, x, ei, ea)
        x = self.block(6, x, ei, ea) + sc
        # MaxPoolingX: max over a fixed 4 x 4 grid of cells per sample (empty cells stay 0)
        cx = (pos[:, 0] / self.cell[0]).floor().long().clamp(0, AEGNN_POOL7 - 1)
        cy = (pos[:, 1] / self.cell[1]).floor().long().clamp(0, AEGNN_POOL7 - 1)
        slot = batch * AEGNN_POOL7 ** 2 + cy * AEGNN_POOL7 + cx
        out = torch.zeros(b * AEGNN_POOL7 ** 2, x.shape[1], device=x.device).scatter_reduce(
            0, slot.unsqueeze(1).expand_as(x), x, "amax", include_self=False)
        self.extra_macs = sum(c.macs for c in self.convs)
        return self.fc(out.view(b, -1)).view(-1, N_Z, N_C)


# ---------------------------------------------------------------- EventMG-style network
# Structure from Wu et al., "EventMG" (NeurIPS 2025), Sections 3.2-3.4; the code is not released and the
# paper gives no layer sizes, so every size below is ours (chosen to stay near our other models' ~0.1 M
# parameters). Needs mamba_ssm (CUDA) for the scans at full length; selective_scan_ref is the same maths,
# step by step, used only where mamba_ssm is missing (local tests on short inputs).
MG_D, MG_STATE, MG_EXPAND, MG_CONV = 64, 8, 2, 4      # widths and Mamba settings (ours)
MG_MICRO = 4                                         # graphs per forward pass (memory); gradients accumulated


def selective_scan_ref(u, delta, A, B, C, D, z):
    """Mamba's selective scan (same arguments and maths as mamba_ssm.selective_scan_fn with
    delta_softplus=True): h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t;  y_t = C_t h_t + D u_t;  y * silu(z).
    u, delta, z: (b, d, L); A: (d, n); B, C: (b, n, L)."""
    delta = F.softplus(delta)
    h = torch.zeros(u.shape[0], u.shape[1], A.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for t in range(u.shape[2]):
        h = torch.exp(delta[:, :, t, None] * A) * h + (delta[:, :, t] * u[:, :, t])[..., None] * B[:, None, :, t]
        ys.append((h * C[:, None, :, t]).sum(-1))
    return (torch.stack(ys, 2) + D[None, :, None] * u) * F.silu(z)


def selective_scan(u, delta, A, B, C, D, z):
    if u.is_cuda:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        return selective_scan_fn(u, delta, A, B, C, D, z, delta_softplus=True)
    return selective_scan_ref(u, delta, A, B, C, D, z)


class MambaBlock(nn.Module):
    """A standard Mamba block (Gu & Dao): in-projection, causal depthwise conv, selective scan, gate."""

    def __init__(self, d=MG_D, state=MG_STATE, expand=MG_EXPAND, conv=MG_CONV):
        super().__init__()
        di, self.r = expand * d, max(1, d // 16)
        self.inp = nn.Linear(d, 2 * di, bias=False)
        self.conv = nn.Conv1d(di, di, conv, groups=di, padding=conv - 1)
        self.xproj = nn.Linear(di, self.r + 2 * state, bias=False)
        self.dt = nn.Linear(self.r, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state + 1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out = nn.Linear(di, d, bias=False)
        self.state = state

    def forward(self, x):                                        # (b, L, d)
        L = x.shape[1]
        u, z = self.inp(x).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L])          # (b, di, L)
        dt, Bm, Cm = self.xproj(u.transpose(1, 2)).split([self.r, self.state, self.state], -1)
        delta = self.dt(dt).transpose(1, 2)
        y = selective_scan(u.contiguous(), delta.contiguous(), -torch.exp(self.A_log), Bm.transpose(1, 2).contiguous(),
                           Cm.transpose(1, 2).contiguous(), self.D, z.transpose(1, 2).contiguous())
        return self.out(y.transpose(1, 2))


class EventMGStyle(nn.Module):
    """EventMG's structure on our zone task.

    Input per sample: the most recent 20,000 events as nodes, their neighbours (eq. 1) and connected
    components (eq. 6), from representations.eventmg_graph.
      1. lightweight event GNN: one edge convolution over the event graph (sec. 3.2.1)
      2. APN (eq. 2-3): Conv1D over the time-ordered nodes + global MLP -> tanh -> (dx, dy, dt), scaled by
         delta_st and added to the coordinates
      3. MSM + EventsMamba (eq. 4-5): sort nodes along t, x, y forwards and backwards on the perturbed
         coordinates, run one shared Mamba block on each order, average the six results
      4. component graph (eq. 7 + 3.3.2): supernode = mean of its nodes, fully connected, one
         message-passing layer
      5. gated fusion (eq. 8): g = sigmoid(MLP([h_v, f(h_C)])), h = g h_v + (1 - g) f(h_C); events outside
         a kept component keep h_v
      6. zone head: mean of node features in the same 2 x 3 cells as the CNNs, then a linear layer."""

    def __init__(self, d=MG_D, w=320, h=240):
        super().__init__()
        self.w, self.h = w, h
        self.embed = nn.Linear(4, d)
        self.gnn = EdgeConv(d, d)
        self.apn_local = nn.Conv1d(d, d, 3, padding=1)
        self.apn_global = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.apn_final = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))
        self.norm = nn.LayerNorm(d)
        self.mamba = MambaBlock(d)
        self.comp_net = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.fmap = nn.Linear(d, d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.head = nn.Linear(d * 6, N_Z * N_C)
        cols = [int(c[0]) for c in torch.tensor_split(torch.arange(w), 3)[1:]]
        self.cols, self.row = cols, int(torch.tensor_split(torch.arange(h), 2)[1][0])
        self.comp_stats = (float("nan"), float("nan"))

    def forward(self, x):
        feat, nbr, comp, valid = x
        b, n, _ = feat.shape
        nbr = nbr.long()
        h = self.gnn(self.embed(feat), nbr)                                    # 1
        order_t = feat[..., 2].argsort(1)                                       # 2: time-ordered for Conv1D
        ht = torch.gather(h, 1, order_t[..., None].expand_as(h))
        loc = torch.empty_like(h).scatter_(1, order_t[..., None].expand_as(h),
                                           self.apn_local(ht.transpose(1, 2)).transpose(1, 2))
        shift = torch.tanh(self.apn_final(loc + self.apn_global(h))) * R.EVENTMG_DST
        coords = torch.stack([feat[..., 0], feat[..., 1] * self.h / self.w, feat[..., 2]], -1) + shift
        coords = torch.where(valid[..., None] > 0, coords, torch.full_like(coords, 2.0))  # padding sorts last
        hn, outs = self.norm(h), []                                             # 3
        for dim in range(3):
            for flip in (False, True):
                order = coords[..., dim].argsort(1, descending=flip)
                seq = torch.gather(hn, 1, order[..., None].expand_as(hn))
                outs.append(torch.empty_like(hn).scatter_(1, order[..., None].expand_as(hn), self.mamba(seq)))
        hv = h + torch.stack(outs).mean(0)
        ncomp = int(comp.max().item()) + 1 if (comp >= 0).any() else 0          # 4
        hv_out = hv
        if ncomp > 0:
            cid = torch.where(comp >= 0, comp.long() + torch.arange(b, device=comp.device)[:, None] * ncomp,
                              torch.full_like(comp.long(), b * ncomp))
            flat, idx = hv.reshape(-1, hv.shape[-1]), cid.reshape(-1)
            sums = torch.zeros(b * ncomp + 1, hv.shape[-1], device=hv.device).index_add_(0, idx, flat)
            cnt = torch.bincount(idx, minlength=b * ncomp + 1).float()
            hc = (sums / cnt.clamp(min=1)[:, None])[:-1].view(b, ncomp, -1)
            present = (cnt[:-1] > 0).view(b, ncomp, 1).float()
            others = ((hc * present).sum(1, keepdim=True) - hc * present) / (present.sum(1, keepdim=True) - present).clamp(min=1)
            hc = hc + self.comp_net(torch.cat([hc, others], -1))                # fully connected, one layer
            ctx = self.fmap(torch.cat([hc.reshape(-1, hc.shape[-1]), torch.zeros(1, hc.shape[-1], device=hc.device)])[idx]).view_as(hv)
            g = torch.sigmoid(self.gate(torch.cat([hv, ctx], -1)))             # 5
            hv_out = torch.where((comp >= 0)[..., None], g * hv + (1 - g) * ctx, hv)
            if not self.training:
                self.comp_stats = (float(present.sum(1).mean()), float(((comp >= 0).float().sum(1) / valid.sum(1).clamp(min=1)).mean()))
        x_px, y_px = feat[..., 0] * self.w, feat[..., 1] * self.h               # 6
        cell = (y_px >= self.row).long() * 3 + (x_px >= self.cols[0]).long() + (x_px >= self.cols[1]).long()
        onehot = F.one_hot(cell, 6).float() * valid[..., None]
        pooled = torch.einsum("bnc,bnf->bcf", hv_out, onehot) / onehot.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled.flatten(1)).view(-1, N_Z, N_C)


def eventmg_build(rec, idx, rng):
    """Graphs are built once per sample (about 0.5 s each) and kept with the recording."""
    memo = rec.setdefault("eventmg_memo", {})
    for i in idx:
        if int(i) not in memo:
            memo[int(i)] = R.eventmg_graph(rec["e1"]["eventmg"][i], int(rec["e1"]["eventmg_n"][i]))
    parts = [memo[int(i)] for i in idx]
    return tuple(np.stack([p[k] for p in parts]) for k in range(4))


def eventmg_flip(x, m):
    """Mirror left/right: x -> 1 - x for the flipped samples (the graph itself is unchanged by a mirror)."""
    feat = x[0].copy()
    feat[m, :, 0] = np.where(x[3][m] > 0, 1.0 - 1.0 / 320 - feat[m, :, 0], 0.0)
    return (feat,) + tuple(x[1:])


# ---------------------------------------------------------------- thermal + events (extra rows)

def embed(net, x):
    return grid_pool(net.features(x)) if isinstance(net, ZoneNet) else net.embed(x)


class Hybrid(nn.Module):
    """Thermal CNN + an event branch, features joined before the zone heads (Li et al. with thermal
    in place of their RGB frames; their branches also exchange features mid-network, this doesn't)."""

    def __init__(self, event_net):
        super().__init__()
        self.thermal, self.event = ZoneNet(1, TH_W, TH_S), event_net
        self.thermal.head, self.event.head = nn.Identity(), nn.Identity()
        self.head = nn.Linear(64 * 6 + 64 * 6, N_Z * N_C)

    @property
    def spike_rate(self):
        return getattr(self.event, "spike_rate", float("nan"))

    def forward(self, x):
        th, ev = x
        return self.head(torch.cat([embed(self.thermal, th), embed(self.event, ev)], 1)).view(-1, N_Z, N_C)


# ---------------------------------------------------------------- the models compared

def thermal_raw(rec, idx, rng):
    x = rec["thermal"][idx][:, None]
    return x + rng.normal(0, THERMAL_NOISE, x.shape).astype(np.float32)


def e1(rec, key):
    return rec["e1"][key]


def same(x):
    return x.float()


SPECS = {}


def spec(name, group, net, build, prep, shape, source, graph=False):
    SPECS[name] = dict(group=group, net=net, build=build, prep=prep, shape=shape, source=source, graph=graph)


for ms in (40, 80, 120, 240):
    spec(f"count_{ms}", "E1", lambda: ZoneNet(2, EV_W, EV_S),
         lambda rec, idx, rng, n=ms // 40: R.count_window(rec["hist"][idx], n), R.log_counts,
         "2 × 240 × 320", "40 ms cache")
spec("stacked_6x40", "E1", lambda: ZoneNet(12, EV_W, EV_S),
     lambda rec, idx, rng: R.stack_bins(rec["hist"][idx], 6), R.log_counts, "12 × 240 × 320", "40 ms cache")
spec("voxel_6", "E1", lambda: ZoneNet(R.VOXEL_BINS, EV_W, EV_S),
     lambda rec, idx, rng: e1(rec, "voxel")[idx], R.signed_log, "6 × 240 × 320", "bag")
spec("time_surface", "E1", lambda: ZoneNet(2, EV_W, EV_S),
     lambda rec, idx, rng: e1(rec, "ts_age_ms")[idx], R.time_surface, "2 × 240 × 320", "bag")
for k, n in enumerate(R.FIXED_N):
    spec(f"fixed_n_{round(n / 1000)}k", "paper" if n == 26_880 else "E1", lambda: ZoneNet(2, EV_W, EV_S),
         lambda rec, idx, rng, k=k: e1(rec, "fixed_n")[idx, k], R.log_counts, "2 × 240 × 320", "bag")
spec(f"snn_{R.SNN_STEPS}x20", "E1", lambda: SpikingZoneNet(2),
     lambda rec, idx, rng: e1(rec, "snn")[idx], R.log_counts, f"{R.SNN_STEPS} steps × 2 × 240 × 320", "bag")
spec("graph", "E1", GraphZoneNet,
     lambda rec, idx, rng: R.graph_points(e1(rec, "graph")[idx], e1(rec, "graph_n")[idx], rng), same,
     f"{R.GRAPH_POINTS} events × 4", "bag", graph=True)
# paper settings (see representations.py for the source of each)
spec("count_40_raw", "paper", lambda: ZoneNet(2, EV_W, EV_S),
     lambda rec, idx, rng: R.count_window(rec["hist"][idx], 1), R.raw_counts, "2 × 240 × 320", "40 ms cache")
spec("stacked_rvt_10x5", "paper", lambda: ZoneNet(2 * R.RVT_BINS, EV_W, EV_S),
     lambda rec, idx, rng: e1(rec, "rvt")[idx], R.rvt_input, "20 × 240 × 320 (last 50 ms)", "bag")
spec("voxel5_norm", "paper", lambda: ZoneNet(R.VOXEL5_BINS, EV_W, EV_S),
     lambda rec, idx, rng: e1(rec, "voxel5")[idx], R.voxel_norm, "5 × 240 × 320", "bag")
for tau in (25, 100):
    spec(f"time_surface_tau{tau}", "paper", lambda: ZoneNet(2, EV_W, EV_S),
         lambda rec, idx, rng: e1(rec, "ts_age_ms")[idx], lambda x, tau=tau: R.time_surface(x, tau),
         "2 × 240 × 320", "bag")
spec("time_surface_zubic", "paper", lambda: ZoneNet(2 * R.ZUBIC_SAMPLES, EV_W, EV_S),
     lambda rec, idx, rng: e1(rec, "ts_zubic")[idx], R.time_surface_zubic, "12 × 240 × 320", "bag")
spec("snn_voxel5", "paper", lambda: SpikingZoneNet(1),
     lambda rec, idx, rng: e1(rec, "voxel5")[idx], lambda x: R.voxel_norm(x).unsqueeze(2),
     "5 steps × 1 × 240 × 320", "bag")
spec("time_surface_tevio", "paper", lambda: ZoneNet(1, EV_W, EV_S),
     lambda rec, idx, rng: e1(rec, "ts_age_ms")[idx], R.time_surface_tevio, "1 × 240 × 320", "bag")
spec("aegnn", "paper", AEGNN,
     lambda rec, idx, rng: R.aegnn_points(e1(rec, "graph")[idx], e1(rec, "graph_n")[idx]), same,
     f"up to {R.GRAPH_STORE:,} events × 4", "bag", graph=True)
SPECS["aegnn"].update(micro=AEGNN_MICRO, cpu_on_mps=True, needs=("torch_spline_conv", "torch_cluster"))
spec("eventmg_style", "paper", EventMGStyle, eventmg_build,
     lambda x: (x[0].float(), x[1], x[2], x[3].float()), f"{R.EVENTMG_N:,} most recent events × 4", "bag")
SPECS["eventmg_style"].update(micro=MG_MICRO, flip_fn=eventmg_flip, needs_cuda=("mamba_ssm",))
spec("thermal", "reference", lambda: ZoneNet(1, TH_W, TH_S), thermal_raw, same, "1 × 24 × 32", "40 ms cache")
spec("thermal_x2", "reference", lambda: ZoneNet(1, tuple(2 * w for w in TH_W), TH_S), thermal_raw, same,
     "1 × 24 × 32", "40 ms cache")
spec("thermal+snn", "extra", lambda: Hybrid(SpikingZoneNet(2)),
     lambda rec, idx, rng: (thermal_raw(rec, idx, rng), e1(rec, "snn")[idx]),
     lambda x: (x[0].float(), R.log_counts(x[1])), "thermal + spiking steps", "bag")
spec("thermal+stacked_cnn", "extra", lambda: Hybrid(ZoneNet(12, EV_W, EV_S)),
     lambda rec, idx, rng: (thermal_raw(rec, idx, rng), R.stack_bins(rec["hist"][idx], 6)),
     lambda x: (x[0].float(), R.log_counts(x[1])), "thermal + 12 × 240 × 320", "40 ms cache")

E1_KEYS = {"voxel_6": ["voxel"], "time_surface": ["ts_age_ms"], "graph": ["graph", "graph_n"],
           f"snn_{R.SNN_STEPS}x20": ["snn"], "thermal+snn": ["snn"],
           "stacked_rvt_10x5": ["rvt"], "voxel5_norm": ["voxel5"], "snn_voxel5": ["voxel5"],
           "time_surface_tau25": ["ts_age_ms"], "time_surface_tau100": ["ts_age_ms"],
           "time_surface_zubic": ["ts_zubic"], "time_surface_tevio": ["ts_age_ms"], "aegnn": ["graph", "graph_n"],
           "eventmg_style": ["eventmg", "eventmg_n"],
           **{f"fixed_n_{round(n / 1000)}k": ["fixed_n"] for n in R.FIXED_N}}


# ---------------------------------------------------------------- data

def flip(x, mask, graph, width):
    """Mirror the samples in `mask` left/right (numpy, in place on a copy)."""
    if isinstance(x, tuple):
        return tuple(flip(v, mask, graph and i == 1, width) for i, v in enumerate(x))
    x = x.copy()
    x[mask] = R.flip_points(x[mask], width) if graph else x[mask][..., ::-1]
    return x


def to_device(x, device):
    return tuple(v.to(device) for v in x) if isinstance(x, tuple) else x.to(device)


def batches(recs, s, rows, batch, rng, train, width):
    order = rng.permutation(len(rows)) if train else np.arange(len(rows))
    for b in range(0, len(rows), batch):
        part = [rows[i] for i in order[b:b + batch]]
        xs, ys = [], []
        for r in sorted({r for r, _ in part}):
            idx = np.array([i for rr, i in part if rr == r])
            xs.append(s["build"](recs[r], idx, rng))
            ys.append(recs[r]["y"][idx])
        x = tuple(np.concatenate(v) for v in zip(*xs)) if isinstance(xs[0], tuple) else np.concatenate(xs)
        y = np.concatenate(ys)
        if train:  # mirror left/right: flip the input and swap the left and right zone labels
            m = rng.random(len(y)) < 0.5
            x, y = (s["flip_fn"](x, m) if "flip_fn" in s else flip(x, m, s["graph"], width)), y.copy()
            y[m] = y[m][:, ::-1]
        tens = (tuple(torch.from_numpy(np.ascontiguousarray(v)) for v in x) if isinstance(x, tuple)
                else torch.from_numpy(np.ascontiguousarray(x)))
        yield tens, torch.from_numpy(np.ascontiguousarray(y))


def train_predict(recs, name, train_rows, test_rows, seed, device, epochs, width, batch=32):
    """Same recipe for every model: AdamW 1e-3, cosine schedule, class-weighted cross-entropy, flips."""
    s = SPECS[name]
    if s.get("cpu_on_mps") and device == "mps":
        device = "cpu"  # torch_spline_conv / torch_cluster have no Apple GPU kernels
    micro = s.get("micro", batch)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = s["net"]().to(device)
    ytr = np.concatenate([recs[r]["y"][[i]] for r, i in train_rows])
    freq = np.bincount(ytr.ravel(), minlength=N_C) + 1
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(freq.sum() / (N_C * freq), dtype=torch.float32).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        model.train()
        for x, y in batches(recs, s, train_rows, batch, rng, True, width):
            opt.zero_grad()
            for k in range(0, len(y), micro):  # large graphs: split the batch, accumulate the gradients
                xs = tuple(v[k:k + micro] for v in x) if isinstance(x, tuple) else x[k:k + micro]
                xs, ys = s["prep"](to_device(xs, device)), y[k:k + micro].to(device)
                loss = loss_fn(model(xs).reshape(-1, N_C), ys.reshape(-1)) * len(ys) / len(y)
                loss.backward()
            opt.step()
        sched.step()
    model.eval()
    probs, rates, comps = [], [], []
    with torch.no_grad():
        for x, _ in batches(recs, s, test_rows, micro, np.random.default_rng(10_000 + seed), False, width):
            probs.append(torch.softmax(model(s["prep"](to_device(x, device))), -1).cpu().numpy())
            rates.append(getattr(model, "spike_rate", float("nan")))
            comps.append(getattr(model, "comp_stats", (float("nan"), float("nan"))))
            if hasattr(model, "comp_stats"):
                model.comp_stats = (float("nan"), float("nan"))
    comps = np.array(comps, float)
    comp = np.nanmean(comps, 0) if np.isfinite(comps).any() else np.array([np.nan, np.nan])
    return np.concatenate(probs), float(np.mean(rates)), comp


def cost(name, recs):
    """Parameters, MFLOPs (2 x multiply-adds of conv/linear layers, + the graph's distance matrix) and
    CPU ms for one update (batch 1, 1 thread). Building the input from raw events is not timed."""
    s = SPECS[name]
    model = s["net"]().cpu().eval()
    x = s["prep"](to_device(next(batches(recs, s, [(0, 0)], 1, np.random.default_rng(0), False,
                                             recs[0]["events"].shape[-1]))[0], "cpu"))
    macs = []

    def hook(m, _, out):
        if isinstance(m, nn.Conv2d):
            macs.append(out.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1])
        else:
            macs.append(out.numel() // m.out_features * m.in_features * m.out_features)

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(x)
        for h in hs:
            h.remove()
        if name == "graph":
            macs.append(3 * R.GRAPH_POINTS ** 2)  # pairwise distances in (x, y, t)
        macs.append(getattr(model, "extra_macs", 0))  # AEGNN's spline convolutions (per edge)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        for _ in range(5):
            model(x)
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1e3)
        torch.set_num_threads(threads)
    return sum(p.numel() for p in model.parameters()), 2 * sum(macs) / 1e6, float(np.median(times))


# ---------------------------------------------------------------- metrics

def class_f1(t, p, c):
    tp, fp, fn = np.sum((p == c) & (t == c)), np.sum((p == c) & (t != c)), np.sum((p != c) & (t == c))
    return 2 * tp / max(2 * tp + fp + fn, 1) if (t == c).any() or (p == c).any() else np.nan


def scores(t, p):
    """t, p: (n, zones). Main metric = per-zone macro F1 averaged over zones."""
    blocked = t != 0
    return dict(macro_f1=float(np.mean([macro_f1(t[:, z], p[:, z]) for z in range(t.shape[1])])),
                zones_correct=float(np.mean(t == p)),
                missed=float(np.mean(p[blocked] == 0)) if blocked.any() else np.nan)


def block_draws(groups, sample_idx, rng, n=BOOT):
    """Bootstrap draws of 10 s blocks, resampled within each test recording (so every draw keeps
    each place's share). Returns a list of index arrays; the same draws are used for every model,
    which makes the differences between models paired."""
    blocks = {}
    for j, (g, i) in enumerate(zip(groups, sample_idx)):
        blocks.setdefault(g, {}).setdefault(i // BLOCK, []).append(j)
    per_rec = [[np.array(v) for v in b.values()] for b in blocks.values()]
    draws = []
    for _ in range(n):
        sel = [blks[k] for blks in per_rec for k in rng.integers(0, len(blks), len(blks))]
        draws.append(np.concatenate(sel))
    return draws


def summarize(t, preds, draws):
    """Mean over seeds, with 95% CIs from the block bootstrap."""
    point = {k: np.mean([scores(t, p)[k] for p in preds]) for k in ("macro_f1", "zones_correct", "missed")}
    boot = {k: [] for k in point}
    for sel in draws:
        s = [scores(t[sel], p[sel]) for p in preds]
        for k in point:
            boot[k].append(np.mean([v[k] for v in s]))
    return {k: (point[k], *np.nanpercentile(boot[k], [2.5, 97.5])) for k in point}, boot["macro_f1"]


# ---------------------------------------------------------------- main

def load(args):
    recs = dataset.load_all(args.cache, history=HISTORY, every=args.every)
    if args.recordings:
        recs = [r for r in recs if r["name"] in args.recordings.split(",")]
    for r in recs:
        if args.max_samples:
            keep = np.arange(len(r["y"])) < args.max_samples
            r.update({k: v[keep] for k, v in r.items() if isinstance(v, np.ndarray) and len(v) == len(keep)})
        r["e1"] = {}
    return recs


def need_e1(recs, names, e1_root):
    keys = sorted({k for n in names for k in E1_KEYS.get(n, [])})
    for r in recs:
        missing = [k for k in keys if k not in r["e1"]]
        if missing:
            r["e1"].update(dataset.load_e1(Path(e1_root) / r["name"], r, missing))


def report(out, recs, models, t, P, rows_all, groups, args, costs, rates, seeds_done, device, split_note,
           comp_stats=None):
    rng = np.random.default_rng(0)
    sample_idx = np.array([i for _, i in rows_all])
    draws = block_draws(groups, sample_idx, rng)
    preds = {m: [p.argmax(-1) for p in P[m]] for m in models}
    # reference: always predict the most common class per zone in the training recordings
    preds["majority"] = [P["__majority"]] * len(next(iter(P.values())))
    res, boots = {}, {}
    for m in ["majority"] + models:
        res[m], boots[m] = summarize(t, preds[m], draws)
    base = np.array(boots[BASE]) if BASE in boots else None

    def delta(m):
        if base is None or m == BASE:
            return ""
        d = np.array(boots[m]) - base
        point = res[m]["macro_f1"][0] - res[BASE]["macro_f1"][0]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        mark = " ✓" if lo > 0 else " ✗" if hi < 0 else ""
        return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}"

    fmt = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
    pct = lambda v: f"{100 * v[0]:.1f}% [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"
    md = ["# E1: event representation", "",
          f"Recordings: {', '.join(f'{r['name']} ({len(r['y'])})' for r in recs)}; {len(t)} test samples × 3 zones. "
          f"Task: clear / person / furniture / other per zone. Split: {split_note}. "
          f"Seeds finished: {seeds_done} of {args.seeds}; {args.epochs} epochs; device {device}.",
          f"Brackets: 95% CI from resampling 10 s blocks within each test recording ({BOOT} draws). "
          f"Δ F1 is paired against `{BASE}` (the input used so far): ✓ = CI above 0 (better), ✗ = below 0.", ""]
    head = ("| Input | Source | Network input | Macro F1 (main) | Δ F1 vs count_40 | Zones correct | "
            "Missed obstacles | Params | MFLOPs | CPU ms |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"

    def row(m):
        s = SPECS.get(m)
        p, fl, ms = costs.get(m, (0, 0, 0))
        src, shp = (s["source"], s["shape"]) if s else ("—", "—")
        return (f"| {m} | {src} | {shp} | {fmt(res[m]['macro_f1'])} | {delta(m)} | {pct(res[m]['zones_correct'])} | "
                f"{pct(res[m]['missed'])} | {p:,} | {fl:.1f} | {ms:.2f} |")

    for title, group in (("E1 comparison (events only, our settings: every input sees the same 240 ms)", "E1"),
                         ("E1 with each paper's own settings", "paper"), ("References", "reference"),
                         ("Extra: thermal + events (action item: spiking network with thermal instead of RGB)", "extra")):
        ms = [m for m in models if SPECS[m]["group"] == group] + (["majority"] if group == "reference" else [])
        if ms:
            md += [f"## {title}", "", head, sep] + [row(m) for m in ms] + [""]
    md += ["Missed obstacles = share of zones that hold a person / furniture / other object but were called clear.",
           "MFLOPs for the spiking network are dense-equivalent (every step computed as a normal conv); "
           "CPU ms is one forward pass at batch 1 on one thread, excluding building the input from events.", ""]

    md += ["## F1 per class", "", "| Input | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * N_C]
    per_class = []
    for m in ["majority"] + models:
        f = [np.nanmean([class_f1(t.ravel(), p.ravel(), c) for p in preds[m]]) for c in range(N_C)]
        per_class.append([m] + f)
        md.append(f"| {m} | " + " | ".join(f"{v:.3f}" for v in f) + " |")
    md += [""]

    names = [r["name"] for r in recs]
    md += ["## Macro F1 per test recording (place)", "", "| Input | " + " | ".join(names) + " |",
           "|---|" + "---|" * len(names)]
    per_rec = []
    for m in ["majority"] + models:
        v = [np.mean([scores(t[groups == g], p[groups == g])["macro_f1"] for p in preds[m]])
             if (groups == g).any() else np.nan for g in names]
        per_rec.append([m] + v)
        md.append(f"| {m} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    md += [""]

    if "thermal" in P:
        md += ["## How much each event input adds to thermal (preview for E3)", "",
               "Late average = mean of the thermal and event models' probabilities. Oracle = a zone counts as "
               "right if either model got it right (the ceiling for any fusion of the two).", "",
               "| Event input | Event alone F1 | Late average F1 | Oracle zones correct |", "|---|---|---|---|"]
        for m in [m for m in models if SPECS[m]["group"] in ("E1", "paper")]:
            late = np.mean([scores(t, ((a + b) / 2).argmax(-1))["macro_f1"] for a, b in zip(P["thermal"], P[m])])
            orc = np.mean([np.mean((a.argmax(-1) == t) | (b.argmax(-1) == t)) for a, b in zip(P["thermal"], P[m])])
            md.append(f"| {m} | {res[m]['macro_f1'][0]:.3f} | {late:.3f} | {100 * orc:.1f}% |")
        md += [f"| (thermal alone) | {res['thermal']['macro_f1'][0]:.3f} | | |", ""]

    rate_rows = [(m, r) for m, r in rates.items() if not np.isnan(r)]
    if rate_rows:
        md += ["Spiking networks, mean share of neurons spiking per step on test data: " +
               ", ".join(f"{m} {100 * r:.1f}%" for m, r in rate_rows) + ".", ""]
    for m, (nc, share) in (comp_stats or {}).items():
        md += [f"{m}: on test windows, {nc:.1f} kept components per window on average, holding {100 * share:.0f}% "
               f"of the events (if this is ~1 component with ~100% of events, EventMG's component level does "
               f"little on our data).", ""]
    fx = [(r["name"], k, r["e1"]["fixed_n_window_s"][:, k] if "fixed_n_window_s" in r["e1"] else None)
          for r in recs for k in range(len(R.FIXED_N))]
    if any(v is not None for *_, v in fx):
        md += ["## Fixed number of events: how long the windows were", "",
               "Median window length in ms (in brackets: share of samples where the 1 s cap was hit first).", "",
               "| Recording | " + " | ".join(f"{n:,} events" for n in R.FIXED_N) + " |",
               "|---|" + "---|" * len(R.FIXED_N)]
        for r in recs:
            if "fixed_n_window_s" in r["e1"]:
                w, ev = r["e1"]["fixed_n_window_s"], r["e1"]["fixed_n_events"]
                md.append(f"| {r['name']} | " + " | ".join(
                    f"{1e3 * np.median(w[:, k]):.0f} ({np.mean(ev[:, k] < n):.0%})" for k, n in enumerate(R.FIXED_N)) + " |")
        md += [""]
    counts = np.bincount(t.ravel(), minlength=N_C)
    md += ["Test labels (all zones): " + ", ".join(f"{c} {n} ({n / counts.sum():.0%})" for c, n in zip(CLASSES, counts))]
    (out / "summary.md").write_text("\n".join(md) + "\n")

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "group", "macro_f1", "f1_lo", "f1_hi", "zones_correct", "zc_lo", "zc_hi",
                    "missed", "missed_lo", "missed_hi", "params", "mflops", "cpu_ms"])
        for m in ["majority"] + models:
            w.writerow([m, SPECS[m]["group"] if m in SPECS else "reference"] +
                       [f"{v:.4f}" for k in ("macro_f1", "zones_correct", "missed") for v in res[m][k]] +
                       list(costs.get(m, ("", "", ""))))
    for fname, hdr, rows in (("per_class.csv", ["model"] + list(CLASSES), per_class),
                             ("per_recording.csv", ["model"] + names, per_rec)):
        with open(out / fname, "w", newline="") as f:
            csv.writer(f).writerows([hdr] + [[r[0]] + [f"{v:.4f}" for v in r[1:]] for r in rows])
    with open(out / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "truth"] + [f"pred_{c}" for c in CLASSES])
        for m in ["majority"] + models:
            cm = sum(np.bincount(t.ravel() * N_C + p.ravel(), minlength=N_C * N_C) for p in preds[m]).reshape(N_C, N_C)
            w.writerows([[m, CLASSES[i]] + list(cm[i]) for i in range(N_C)])
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "sample", "zone", "truth"] + [f"pred_{m}" for m in ["majority"] + models])
        for j, (rec, i) in enumerate(rows_all):
            for z, zone in enumerate(ZONES):
                w.writerow([rec, i, zone, CLASSES[t[j, z]]] + [CLASSES[preds[m][0][j, z]] for m in ["majority"] + models])
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--e1-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--split", choices=("recording", "blocks", "place"), default="recording",
                    help="place: hold out one place type (room / hall / door / lift) at a time")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth sample (2 = 2 per second; less memory)")
    ap.add_argument("--models", default="all", help="comma-separated names, or 'all'")
    ap.add_argument("--skip", default="", help="comma-separated names to leave out (e.g. the heavy graph models)")
    ap.add_argument("--recordings", default="", help="comma-separated; default all in --cache")
    ap.add_argument("--max-samples", type=int, default=0, help="first N samples per recording (testing)")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch, 24 samples: checks every model runs")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.max_samples = 1, 1, args.max_samples or 24
    models = list(SPECS) if args.models == "all" else args.models.split(",")
    models = [m for m in models if m not in args.skip.split(",")]
    unknown = [m for m in models if m not in SPECS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {list(SPECS)}")
    import importlib.util
    for m in list(models):
        missing = [p for p in SPECS[m].get("needs", ()) if importlib.util.find_spec(p) is None]
        if torch.cuda.is_available():  # full-length Mamba scans need the CUDA kernels; elsewhere the slow
            missing += [p for p in SPECS[m].get("needs_cuda", ()) if importlib.util.find_spec(p) is None]
        elif SPECS[m].get("needs_cuda") and not args.smoke:  # reference scan is only for short smoke tests
            missing += ["a CUDA GPU"]
        if missing:
            print(f"SKIPPING {m}: needs {', '.join(missing)} (not installed)")
            models.remove(m)
    out = Path(args.out).expanduser()
    (out / "runs").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    recs = load(args)
    for r in recs:  # fixed-n window lengths, for the report
        if any(m.startswith("fixed_n") for m in models):
            r["e1"].update(dataset.load_e1(Path(args.e1_cache) / r["name"], r, ["fixed_n_window_s", "fixed_n_events"]))
    width = recs[0]["events"].shape[-1]
    fold_list = list(folds(recs, args.split))
    split_note = ("hold out one place type at a time (" + ", ".join(n for n, _, _ in fold_list) + ")"
                  if args.split == "place" else
                  "leave one recording out" if args.split == "recording" and len(recs) > 1 else
                  "5 held-out time blocks per recording, 1 s gap (seen places)")
    if args.every > 1:
        split_note += f"; every {args.every}. sample ({4 / args.every:g} per second)"
    print(f"{len(recs)} recordings: " + ", ".join(f"{r['name']} ({len(r['y'])})" for r in recs) +
          f"; {len(fold_list)} folds; {len(models)} models; device {device}")
    (out / "config.json").write_text(json.dumps(vars(args) | {"models": models, "device": device}, indent=1))

    t = np.concatenate([np.stack([recs[r]["y"][i] for r, i in te]) for _, _, te in fold_list])
    groups = np.array([recs[r]["name"] for _, _, te in fold_list for r, _ in te])
    rows_all = [(recs[r]["name"], i) for _, _, te in fold_list for r, i in te]
    major = []
    for _, tr, te in fold_list:
        ytr = np.stack([recs[r]["y"][i] for r, i in tr])
        major.append(np.tile([np.bincount(ytr[:, z], minlength=N_C).argmax() for z in range(N_Z)], (len(te), 1)))
    costs = {}
    P, rates, comp_stats = {m: [] for m in models}, {}, {}

    for seed in range(args.seeds):  # seed outermost: every finished seed is a complete result
        for j, m in enumerate(models):
            # load only this model's inputs, and free the previous model's (all of them at once would
            # not fit in memory on the large dataset)
            nxt = set(E1_KEYS.get(m, []))
            for r in recs:
                for k in [k for k in r["e1"] if k not in nxt and k not in ("fixed_n_window_s", "fixed_n_events")]:
                    del r["e1"][k]
            need_e1(recs, [m], args.e1_cache)
            if m not in costs:
                costs[m] = cost(m, recs)
            parts, rs, cs = [], [], []
            for name, tr, te in fold_list:
                f = out / "runs" / f"{m}__{name}__seed{seed}.npz"
                if f.exists():
                    z = np.load(f)
                    prob, rate = z["probs"], float(z["spike_rate"])
                    comp = z["comp"] if "comp" in z.files else np.array([np.nan, np.nan])
                else:
                    t0 = time.time()
                    prob, rate, comp = train_predict(recs, m, tr, te, seed, device, args.epochs, width)
                    np.savez(f, probs=prob, spike_rate=rate, comp=comp)
                    print(f"seed {seed} {m:>20s} test {name}: {time.time() - t0:.0f} s", flush=True)
                parts.append(prob)
                rs.append(rate)
                cs.append(comp)
            P[m].append(np.concatenate(parts))
            rates[m] = float(np.nanmean(rs)) if not np.isnan(rs).all() else float("nan")
            cs = np.array(cs, float)
            if np.isfinite(cs).any():
                comp_stats[m] = np.nanmean(cs, 0)
        P["__majority"] = np.concatenate(major)
        report(out, recs, models, t, {k: v for k, v in P.items()}, rows_all, groups, args, costs, rates,
               seed + 1, device, split_note, comp_stats)
        del P["__majority"]
    if args.smoke:
        print("\nSMOKE TEST PASSED: every model trained, predicted and was reported.")


if __name__ == "__main__":
    main()
