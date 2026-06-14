import torch

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    LayerNorm,
    global_add_pool,
)
from torch_geometric.utils import softmax

from layers import (
    CoAttentionLayer,
    RESCAL,
    IntraGraphAttention,
    InterGraphAttention,
)


def _graph_gate_from_batch(graph_gate, batch_index, device, dtype):
    if graph_gate is None or batch_index is None or batch_index.numel() == 0:
        return None

    gate = graph_gate.to(device=device, dtype=dtype).reshape(-1)
    if gate.numel() == 0:
        return None

    batch_index = batch_index.to(device=device)
    num_graphs = int(batch_index.max().item()) + 1

    if gate.numel() < num_graphs:
        fill_value = gate.mean() if gate.numel() > 0 else torch.tensor(1.0, device=device, dtype=dtype)
        pad = fill_value.repeat(num_graphs - gate.numel())
        gate = torch.cat([gate, pad], dim=0)
    elif gate.numel() > num_graphs:
        gate = gate[:num_graphs]

    return gate[batch_index].unsqueeze(-1)


def _graph_enrichment_from_edge_weights(edge_index, edge_weights, batch, device, dtype):
    """
    Build one enrichment scalar per graph from edge weights.
    If unavailable, fall back to ones.
    """
    if batch is None or batch.numel() == 0:
        return torch.ones(1, device=device, dtype=dtype)

    num_graphs = int(batch.max().item()) + 1

    if edge_index is None or edge_index.numel() == 0 or edge_weights is None or edge_weights.numel() == 0:
        return torch.ones(num_graphs, device=device, dtype=dtype)

    src_nodes = edge_index[0]
    src_batch = batch[src_nodes].to(device=device)

    ew = edge_weights.to(device=device, dtype=dtype)
    graph_sum = torch.zeros(num_graphs, device=device, dtype=dtype)
    graph_cnt = torch.zeros(num_graphs, device=device, dtype=dtype)

    graph_sum.index_add_(0, src_batch, ew)
    graph_cnt.index_add_(0, src_batch, torch.ones_like(ew))
    graph_cnt = torch.clamp(graph_cnt, min=1.0)

    graph_mean = graph_sum / graph_cnt
    graph_mean = torch.clamp(graph_mean, min=0.3, max=3.0)
    return graph_mean


class AtomAttentionReadout(nn.Module):
    """
    思路一-lite:
    1) 用原子级注意力代替原始 pooling
    2) 用注意力聚合后的结构摘要给 gray gate 提供结构重要性
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim

        # 节点注意力：节点特征 + 图级 enrichment 标量
        self.att_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 结构摘要 -> gate 分数
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, batch, edge_index=None, edge_weights=None):
        if x.numel() == 0 or batch is None or batch.numel() == 0:
            zero_pool = torch.zeros(1, self.input_dim, device=x.device, dtype=x.dtype)
            zero_score = torch.ones(1, device=x.device, dtype=x.dtype) * 0.5
            zero_alpha = torch.zeros(0, device=x.device, dtype=x.dtype)
            return zero_pool, zero_score, zero_alpha

        graph_enrich = _graph_enrichment_from_edge_weights(
            edge_index, edge_weights, batch, x.device, x.dtype
        )  # [B]

        node_enrich = graph_enrich[batch].unsqueeze(-1)  # [N, 1]
        att_in = torch.cat([x, node_enrich], dim=-1)     # [N, D+1]

        logits = self.att_mlp(att_in).squeeze(-1)        # [N]
        alpha = softmax(logits, batch)                   # graph-wise softmax

        pooled = global_add_pool(alpha.unsqueeze(-1) * x, batch)  # [B, D]

        # 结构重要性分数：由注意力池化后的结构摘要 + enrichment 共同决定
        gate_raw = torch.sigmoid(self.gate_mlp(pooled)).squeeze(-1)     # [B]
        enrich_score = torch.sigmoid(graph_enrich)                      # [B]
        sub_score = 0.5 * gate_raw + 0.5 * enrich_score                # [B]

        return pooled, sub_score, alpha



class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, proj_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)


class MVN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)

        self.initial_norm = LayerNorm(self.in_features)

        self.blocks = nn.ModuleList()
        current_dim = in_features
        for n_heads, head_out_feats in zip(blocks_params, heads_out_feat_params):
            block = MVN_DDI_Block(n_heads, current_dim, head_out_feats, self.hidd_dim)
            self.blocks.append(block)
            current_dim = head_out_feats * 2

        self.net_norms = nn.ModuleList([
            LayerNorm(head_out_feats * 2)
            for head_out_feats in heads_out_feat_params
        ])

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

        self.graph_emb_dim = heads_out_feat_params[-1] * 2
        self.cl_proj = ProjectionHead(self.graph_emb_dim, proj_dim=128)

    def encode_graph_reprs(self, h_data, t_data, b_graph):
        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)

        repr_h = []
        repr_t = []

        for block, norm in zip(self.blocks, self.net_norms):
            h_data, t_data, r_h, r_t = block(h_data, t_data, b_graph)

            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(norm(h_data.x, h_data.batch))
            t_data.x = F.elu(norm(t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        return repr_h, repr_t

    def get_cl_views(self, h_data, t_data, b_graph):
        repr_h, repr_t = self.encode_graph_reprs(h_data, t_data, b_graph)
        h_last = repr_h[:, -1, :]
        t_last = repr_t[:, -1, :]
        h_proj = self.cl_proj(h_last)
        t_proj = self.cl_proj(t_last)
        return h_last, t_last, h_proj, t_proj

    def forward(self, triples, return_last_repr=False):
        h_data, t_data, rels, b_graph = triples

        repr_h, repr_t = self.encode_graph_reprs(h_data, t_data, b_graph)

        attentions = self.co_attention(repr_h, repr_t)
        scores = self.KGE(repr_h, repr_t, rels, attentions)

        if return_last_repr:
            h_last = repr_h[:, -1, :]
            t_last = repr_t[:, -1, :]
            return scores, h_last, t_last

        return scores


class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats // n_heads, n_heads)

        self.intraAtt = IntraGraphAttention(head_out_feats)
        self.interAtt = InterGraphAttention(head_out_feats)

        # 思路一-lite：
        # 1) gate_readout: 为灰色门控提供结构重要性
        # 2) readout: 替换原来的 SAGPooling + global_add_pool
        self.gate_readout = AtomAttentionReadout(head_out_feats)
        self.readout = AtomAttentionReadout(head_out_feats * 2)

        self.gray_gate_alpha = 0.3

    def forward(self, h_data, t_data, b_graph):
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)

        # ---------- 用结构注意力给灰色门控提供子结构重要性 ----------
        _, h_sub_score, _ = self.gate_readout(
            h_data.x,
            h_data.batch,
            h_data.edge_index,
            getattr(h_data, 'intra_enrichment_weights', None)
        )
        _, t_sub_score, _ = self.gate_readout(
            t_data.x,
            t_data.batch,
            t_data.edge_index,
            getattr(t_data, 'intra_enrichment_weights', None)
        )

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        # ---------- 子结构加权灰色门控 ----------
        pair_sub_score = 0.5 * (h_sub_score + t_sub_score)   # [B]
        pair_gate = 1.0 + self.gray_gate_alpha * (pair_sub_score - 0.5)
        pair_gate = torch.clamp(pair_gate, min=0.7, max=1.3)

        h_gate = _graph_gate_from_batch(pair_gate, h_data.batch, h_interRep.device, h_interRep.dtype)
        t_gate = _graph_gate_from_batch(pair_gate, t_data.batch, t_interRep.device, t_interRep.dtype)

        if h_gate is not None:
            h_interRep = h_interRep * h_gate
        if t_gate is not None:
            t_interRep = t_interRep * t_gate

        # ---------- 拼接 intra / inter ----------
        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = h_rep
        t_data.x = t_rep

        # ---------- 子结构注意力池化（替代 SAGPooling + global_add_pool） ----------
        h_global_graph_emb, _, _ = self.readout(
            h_data.x,
            h_data.batch,
            h_data.edge_index,
            getattr(h_data, 'intra_enrichment_weights', None)
        )
        t_global_graph_emb, _, _ = self.readout(
            t_data.x,
            t_data.batch,
            t_data.edge_index,
            getattr(t_data, 'intra_enrichment_weights', None)
        )

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb
