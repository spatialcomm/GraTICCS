import math

import torch
from torch import nn
from torch_geometric.utils import softmax, degree


def segment_sum(values, index, size):
    return values.new_zeros((size,) + values.shape[1:]).index_add(0, index, values)


class EventGraphAttention(nn.Module):
    """Queries from the receiver state, keys/values from neighbour cell-type tokens and the event embedding."""

    def __init__(self, hidden=128, event_dim=32, heads=4, dropout=0.15):
        super().__init__()
        self.hidden, self.heads, self.dh = hidden, heads, hidden // heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.event_k = nn.Linear(event_dim, hidden, bias=False)
        self.event_v = nn.Linear(event_dim, hidden, bias=False)
        self.edge_bias = nn.Sequential(nn.Linear(3 + event_dim, 64), nn.GELU(), nn.Linear(64, heads))
        self.out = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, h, t, src, dst, geom, ev):
        n = len(h)
        q = self.q(h).view(-1, self.heads, self.dh)
        k = (self.k(t)[src] + self.event_k(ev)).view(-1, self.heads, self.dh)
        v = (self.v(t)[src] + self.event_v(ev)).view(-1, self.heads, self.dh)
        logits = (q[dst] * k).sum(-1) / math.sqrt(self.dh) + self.edge_bias(torch.cat([geom, ev], 1))
        w = softmax(logits, dst, num_nodes=n)
        agg = segment_sum(w[..., None] * v, dst, n).reshape(-1, self.hidden)
        h = self.norm1(h + self.drop(self.out(agg)))
        return self.norm2(h + self.drop(self.ff(h)))


class Background(nn.Module):
    """Background (nuisance) branch. graph=False gives the matched local model."""

    def __init__(self, d_in, n_types, n_out, hidden=128, heads=4, dropout=0.15, graph=True):
        super().__init__()
        self.graph = graph
        self.node_fuse = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.tok = nn.Sequential(nn.Linear(n_types, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.event = nn.Sequential(nn.Linear(2 * n_types + 3, 64), nn.GELU(), nn.Linear(64, 32), nn.LayerNorm(32))
        self.layers = nn.ModuleList([EventGraphAttention(hidden, 32, heads, dropout) for _ in range(2)])
        self.local_branch = nn.Sequential(nn.Linear(hidden, 192), nn.GELU(), nn.Dropout(dropout), nn.Linear(192, 128), nn.GELU())
        self.graph_branch = nn.Sequential(nn.Linear(hidden * 2, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(128 * 2, 128), nn.GELU(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, n_out))

    def forward(self, xin, toh, ei, eattr):
        h0 = self.node_fuse(xin)
        local_h = self.local_branch(h0)
        if not self.graph:
            return self.head(local_h), h0
        src, dst = ei
        t = self.tok(toh)
        ev = self.event(torch.cat([toh[src], toh[dst], eattr], 1))
        h = h0
        for layer in self.layers:
            h = layer(h, t, src, dst, eattr, ev)
        graph_h = self.graph_branch(torch.cat([h0, h], 1))
        g = torch.sigmoid(self.gate(torch.cat([local_h, graph_h], 1)))
        return self.head(local_h + g * graph_h), h


class LRBranch(nn.Module):
    """Per-event channel scores s * attention * gate(receiver type); delta_y = sum over events of score * V.
    graph=False replaces the attention by 1/deg."""

    def __init__(self, n_types, d_h, n_channels, n_out, heads=4, dh=16, graph=True):
        super().__init__()
        self.graph, self.H, self.dh = graph, heads, dh
        self.q = nn.Linear(n_types + d_h, heads * dh, bias=False)
        self.k = nn.Linear(n_types, heads * dh, bias=False)
        self.event = nn.Sequential(nn.Linear(2 * n_types + 2, 32), nn.GELU(), nn.LayerNorm(32))
        self.event_k = nn.Linear(32, heads * dh, bias=False)
        self.edge_bias = nn.Sequential(nn.Linear(2 + 32, 32), nn.GELU(), nn.Linear(32, heads))
        self.P = nn.Parameter(torch.zeros(n_channels, heads))
        self.G = nn.Parameter(torch.zeros(n_types, n_channels))
        self.V = nn.Parameter(torch.zeros(n_channels, n_out))
        nn.init.normal_(self.V, std=1e-3)

    def contributions(self, tsrc, tdst, hdst, eattr, s, dst, n_nodes):
        gate = torch.sigmoid(self.G[tdst.argmax(1)])
        if not self.graph:
            return (1.0 / degree(dst, n_nodes).clamp(min=1)[dst]).unsqueeze(1) * s * gate
        a = self.head_attention(tsrc, tdst, hdst, eattr, dst, n_nodes)
        return (a @ torch.softmax(self.P, 1).T) * s * gate

    def head_attention(self, tsrc, tdst, hdst, eattr, dst, n_nodes):
        ev = self.event(torch.cat([tsrc, tdst, eattr], 1))
        q = self.q(torch.cat([tdst, hdst], 1)).view(-1, self.H, self.dh)
        k = (self.k(tsrc) + self.event_k(ev)).view(-1, self.H, self.dh)
        return softmax((q * k).sum(-1) / math.sqrt(self.dh) + self.edge_bias(torch.cat([eattr, ev], 1)), dst, num_nodes=n_nodes)


class FixedRuleBranch(LRBranch):
    """Local LR branch with a fixed neighbour-weighting rule instead of attention (controls for the selective-sender tests).
    rule="type": 1/deg x 2 sigmoid(T[sender type, channel]), T learned and starting at zero (equal weights).
    rule="area": shared contact length / the receiver's total contact length.
    rule="dist": exp(-d / 15 um), normalised over the receiver's neighbours."""

    def __init__(self, n_types, d_h, n_channels, n_out, rule):
        super().__init__(n_types, d_h, n_channels, n_out, graph=False)
        self.rule = rule
        if rule == "type":
            self.T = nn.Parameter(torch.zeros(n_types, n_channels))

    def contributions(self, tsrc, tdst, hdst, eattr, s, dst, n_nodes):
        gate = torch.sigmoid(self.G[tdst.argmax(1)])
        if self.rule == "type":
            w = (1.0 / degree(dst, n_nodes).clamp(min=1)[dst]).unsqueeze(1) * 2 * torch.sigmoid(self.T[tsrc.argmax(1)])
        else:
            c = torch.exp(-eattr[:, 1].float()) if self.rule == "dist" else torch.expm1(eattr[:, 0].float())
            eps = 1e-12 if self.rule == "dist" else 1e-6
            w = (c / segment_sum(c, dst, n_nodes).clamp(min=eps)[dst]).unsqueeze(1)
        return w * s * gate


def make_lr_branch(arch, n_types, n_channels, n_out, heads=16):
    """graticcs: learned attention; local: equal weights 1/deg; local_type / local_area / local_dist: fixed rules."""
    if arch.startswith("local_"):
        return FixedRuleBranch(n_types, 128, n_channels, n_out, arch.split("_", 1)[1])
    return LRBranch(n_types, 128, n_channels, n_out, heads=heads, graph=arch == "graticcs")
