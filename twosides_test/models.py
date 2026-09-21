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


def _graph_enrichment_from_edge_weights(edge_index, edge_weights, batch, device, dtype):
    """
    Aggregate edge-level functional-group enrichment weights into one
    structure-prior scalar for each molecular graph in a mini-batch.

    This helper is shared by SP-SAR and CD-SAGG. It does not alter the
    original FG-DDI intra-/inter-graph attention modules.
    """
    if batch is None or batch.numel() == 0:
        return torch.ones(1, device=device, dtype=dtype)

    num_graphs = int(batch.max().item()) + 1

    if (
        edge_index is None
        or edge_index.numel() == 0
        or edge_weights is None
        or edge_weights.numel() == 0
    ):
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
    return torch.clamp(graph_mean, min=0.3, max=3.0)


def _graph_gate_from_batch(graph_gate, batch_index, device, dtype):
    """
    Expand one graph-level gate scalar per molecule to node-level gates so
    that the cross-drug (inter-graph) atom representations can be modulated.
    """
    if graph_gate is None or batch_index is None or batch_index.numel() == 0:
        return None

    gate = graph_gate.to(device=device, dtype=dtype).reshape(-1)
    if gate.numel() == 0:
        return None

    batch_index = batch_index.to(device=device)
    num_graphs = int(batch_index.max().item()) + 1

    if gate.numel() < num_graphs:
        fill_value = (
            gate.mean()
            if gate.numel() > 0
            else torch.tensor(1.0, device=device, dtype=dtype)
        )
        gate = torch.cat([gate, fill_value.repeat(num_graphs - gate.numel())], dim=0)
    elif gate.numel() > num_graphs:
        gate = gate[:num_graphs]

    return gate[batch_index].unsqueeze(-1)


class SPSARReadout(nn.Module):
    """
    SP-SAR: Structure-Prior-Guided Substructure-Aware Readout.

    Stage 01 component retained unchanged in Stage 02. It uses the
    functional-group structural prior together with atom representations
    to estimate graph-wise atom attention and then performs weighted
    graph-level aggregation.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.att_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, batch, edge_index=None, edge_weights=None):
        if x.numel() == 0 or batch is None or batch.numel() == 0:
            pooled = torch.zeros(1, self.input_dim, device=x.device, dtype=x.dtype)
            alpha = torch.zeros(0, device=x.device, dtype=x.dtype)
            return pooled, alpha

        graph_enrich = _graph_enrichment_from_edge_weights(
            edge_index=edge_index,
            edge_weights=edge_weights,
            batch=batch,
            device=x.device,
            dtype=x.dtype,
        )

        node_enrich = graph_enrich[batch].unsqueeze(-1)
        att_input = torch.cat([x, node_enrich], dim=-1)
        logits = self.att_mlp(att_input).squeeze(-1)
        alpha = softmax(logits, batch)

        pooled = global_add_pool(alpha.unsqueeze(-1) * x, batch)
        return pooled, alpha


class CDSAGGController(nn.Module):
    """
    CD-SAGG: cross-drug structure-aware gray-gating controller.

    This module is isolated from SP-SAR so Stage 02 adds only the gray-gating
    mechanism found in the legacy SGAR-DDI implementation:
      1) form a structure-aware graph summary from atom features + FG prior;
      2) convert each drug summary into a substructure-importance score;
      3) combine the two drug scores into a pair-level gray gate;
      4) modulate only the inter-graph atom representations.

    No contrastive-learning branch and no new interaction scorer is added here.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim

        # Legacy SGAR-DDI logic: structure-aware atom attention.
        self.att_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Structure summary -> graph-level importance score.
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, batch, edge_index=None, edge_weights=None):
        if x.numel() == 0 or batch is None or batch.numel() == 0:
            sub_score = torch.full(
                (1,), 0.5, device=x.device, dtype=x.dtype
            )
            alpha = torch.zeros(0, device=x.device, dtype=x.dtype)
            return sub_score, alpha

        graph_enrich = _graph_enrichment_from_edge_weights(
            edge_index=edge_index,
            edge_weights=edge_weights,
            batch=batch,
            device=x.device,
            dtype=x.dtype,
        )

        node_enrich = graph_enrich[batch].unsqueeze(-1)
        att_input = torch.cat([x, node_enrich], dim=-1)
        logits = self.att_mlp(att_input).squeeze(-1)
        alpha = softmax(logits, batch)

        pooled = global_add_pool(alpha.unsqueeze(-1) * x, batch)

        # Preserve the legacy SGAR-DDI score construction.
        gate_raw = torch.sigmoid(self.gate_mlp(pooled)).squeeze(-1)
        enrich_score = torch.sigmoid(graph_enrich)
        sub_score = 0.5 * gate_raw + 0.5 * enrich_score

        return sub_score, alpha


class BCSIEEnhancer(nn.Module):
    """
    BC-SIE: Bidirectional Cross-layer Structural Interaction Enhancer.

    Stage 03 extends the original FG-DDI multi-block co-attention scorer
    instead of renaming it. Given the stacked graph representations from
    SP-SAR/CD-SAGG, BC-SIE adds three genuinely new operations:
      1) relation-conditioned H->T cross-layer attention;
      2) relation-conditioned T->H cross-layer attention;
      3) bounded residual cross-layer fusion before the original RESCAL score.

    The returned ``delta_attention`` is a relation-aware correction to the
    original FG-DDI CoAttentionLayer. If all BC-SIE parameters are zero, the
    representation corrections and attention correction are exactly zero, so
    the scoring path reduces to the Stage-02 FG-DDI co-attention + RESCAL path.
    """

    def __init__(self, input_dim, rel_total, att_dim=None, repr_scale=0.20, attn_scale=0.20):
        super().__init__()
        self.input_dim = input_dim
        self.rel_total = rel_total
        self.att_dim = att_dim or max(16, input_dim // 2)
        self.repr_scale = float(repr_scale)
        self.attn_scale = float(attn_scale)

        # Relation context used only by BC-SIE.  This is distinct from the
        # relation matrix used by the baseline RESCAL scorer.
        self.rel_context = nn.Embedding(rel_total, input_dim)
        nn.init.xavier_uniform_(self.rel_context.weight)

        # H -> T directional interaction.
        self.q_h = nn.Linear(input_dim, self.att_dim, bias=False)
        self.k_t = nn.Linear(input_dim, self.att_dim, bias=False)
        self.r_ht = nn.Linear(input_dim, self.att_dim, bias=False)
        self.v_ht = nn.Parameter(torch.empty(self.att_dim))

        # T -> H directional interaction; parameters are intentionally not
        # shared so the two directions are not forced to be symmetric.
        self.q_t = nn.Linear(input_dim, self.att_dim, bias=False)
        self.k_h = nn.Linear(input_dim, self.att_dim, bias=False)
        self.r_th = nn.Linear(input_dim, self.att_dim, bias=False)
        self.v_th = nn.Parameter(torch.empty(self.att_dim))

        # Direction-specific residual fusion.  The candidate uses
        # [self, cross-layer context, elementwise interaction], while the
        # gate additionally sees the relation context.
        self.fuse_h = nn.Linear(input_dim * 3, input_dim)
        self.gate_h = nn.Linear(input_dim * 4, input_dim)
        self.fuse_t = nn.Linear(input_dim * 3, input_dim)
        self.gate_t = nn.Linear(input_dim * 4, input_dim)

        nn.init.xavier_uniform_(self.v_ht.view(1, -1))
        nn.init.xavier_uniform_(self.v_th.view(1, -1))

        # Verification caches only; they never feed back into prediction.
        self.last_delta_attention = None
        self.last_h_delta = None
        self.last_t_delta = None

    @staticmethod
    def _prepare_rel_ids(rels, batch_size):
        rel_ids = rels.reshape(-1).long()
        if rel_ids.numel() == batch_size:
            return rel_ids
        if rel_ids.numel() == 1 and batch_size > 1:
            return rel_ids.repeat(batch_size)
        if rel_ids.numel() > batch_size:
            return rel_ids[:batch_size]
        raise ValueError(
            f"BC-SIE relation count mismatch: got {rel_ids.numel()} relation ids "
            f"for batch size {batch_size}."
        )

    def forward(self, heads, tails, rels):
        # heads/tails: [B, L, D]
        if heads.dim() != 3 or tails.dim() != 3:
            raise ValueError(
                "BC-SIE expects stacked graph representations with shape [B, L, D]."
            )

        batch_size = heads.size(0)
        rel_ids = self._prepare_rel_ids(rels, batch_size)
        rel_ctx = self.rel_context(rel_ids)  # [B, D]

        # ---------- Head -> Tail ----------
        qh = self.q_h(heads) + self.r_ht(rel_ctx).unsqueeze(1)
        kt = self.k_t(tails)
        e_ht = torch.tanh(qh.unsqueeze(2) + kt.unsqueeze(1))
        e_ht = torch.matmul(e_ht, self.v_ht)  # [B, L_h, L_t]
        a_ht = torch.softmax(e_ht, dim=-1)
        ctx_h = torch.matmul(a_ht, tails)

        # ---------- Tail -> Head ----------
        qt = self.q_t(tails) + self.r_th(rel_ctx).unsqueeze(1)
        kh = self.k_h(heads)
        e_th = torch.tanh(qt.unsqueeze(2) + kh.unsqueeze(1))
        e_th = torch.matmul(e_th, self.v_th)  # [B, L_t, L_h]
        a_th = torch.softmax(e_th, dim=-1)
        ctx_t = torch.matmul(a_th, heads)

        # ---------- Bidirectional residual fusion ----------
        h_interaction = heads * ctx_h
        t_interaction = tails * ctx_t

        h_candidate = torch.tanh(
            self.fuse_h(torch.cat([heads, ctx_h, h_interaction], dim=-1))
        )
        t_candidate = torch.tanh(
            self.fuse_t(torch.cat([tails, ctx_t, t_interaction], dim=-1))
        )

        rel_h = rel_ctx.unsqueeze(1).expand(-1, heads.size(1), -1)
        rel_t = rel_ctx.unsqueeze(1).expand(-1, tails.size(1), -1)

        h_gate = torch.sigmoid(
            self.gate_h(torch.cat([heads, ctx_h, h_interaction, rel_h], dim=-1))
        )
        t_gate = torch.sigmoid(
            self.gate_t(torch.cat([tails, ctx_t, t_interaction, rel_t], dim=-1))
        )

        h_delta = self.repr_scale * h_gate * h_candidate
        t_delta = self.repr_scale * t_gate * t_candidate
        enhanced_h = heads + h_delta
        enhanced_t = tails + t_delta

        # Combine the two directional layer-pair scores into a correction with
        # the same [B, L_h, L_t] shape as the original FG-DDI co-attention.
        delta_attention = self.attn_scale * 0.5 * (
            e_ht + e_th.transpose(-2, -1)
        )

        self.last_delta_attention = delta_attention.detach()
        self.last_h_delta = h_delta.detach()
        self.last_t_delta = t_delta.detach()

        return enhanced_h, enhanced_t, delta_attention


class ProjectionHead(nn.Module):
    """Auxiliary projection head used only by Stage-04 contrastive learning."""

    def __init__(self, input_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


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

        # Original FG-DDI scorer retained as the Stage-02 reference path.
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

        # Stage 03: genuinely extend the baseline scorer with relation-conditioned
        # bidirectional cross-layer interaction instead of renaming the baseline.
        self.bcsie = BCSIEEnhancer(
            input_dim=self.kge_dim,
            rel_total=self.rel_total,
            att_dim=max(16, self.kge_dim // 2),
            repr_scale=0.20,
            attn_scale=0.20,
        )

        # Stage 04: auxiliary contrastive-learning projection head.
        # It does not participate in the DDI score path.
        self.graph_emb_dim = heads_out_feat_params[-1] * 2
        self.cl_proj = ProjectionHead(self.graph_emb_dim, proj_dim=128)

    def encode_graph_reprs(self, h_data, t_data, b_graph):
        """Encode the two molecular graphs and return all block-level graph representations."""
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
        """Return final-layer graph representations and their normalized projections."""
        repr_h, repr_t = self.encode_graph_reprs(h_data, t_data, b_graph)
        h_last = repr_h[:, -1, :]
        t_last = repr_t[:, -1, :]
        h_proj = self.cl_proj(h_last)
        t_proj = self.cl_proj(t_last)
        return h_last, t_last, h_proj, t_proj

    def forward(self, triples, return_last_repr=False):
        h_data, t_data, rels, b_graph = triples

        repr_h, repr_t = self.encode_graph_reprs(h_data, t_data, b_graph)

        # Original FG-DDI cross-block co-attention.
        base_attentions = self.co_attention(repr_h, repr_t)

        # Stage 03 BC-SIE remains unchanged.
        bcsie_h, bcsie_t, delta_attentions = self.bcsie(
            repr_h, repr_t, rels
        )
        attentions = base_attentions + delta_attentions
        scores = self.KGE(bcsie_h, bcsie_t, rels, attentions)

        if return_last_repr:
            return scores, repr_h[:, -1, :], repr_t[:, -1, :]

        return scores


class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats // n_heads, n_heads)

        # Keep the FG-DDI intra-/inter-graph encoders unchanged.
        self.intraAtt = IntraGraphAttention(head_out_feats)
        self.interAtt = InterGraphAttention(head_out_feats)

        # Stage 01: SP-SAR remains unchanged.
        self.readout = SPSARReadout(head_out_feats * 2)

        # Stage 02: add CD-SAGG only.
        self.cdsagg = CDSAGGController(head_out_feats)
        self.gray_gate_alpha = 0.3

        # Debug/verification cache only; not used by prediction.
        self.last_pair_gate = None

    def forward(self, h_data, t_data, b_graph):
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)

        # ---------- CD-SAGG: estimate pair-level structural gate ----------
        h_sub_score, _ = self.cdsagg(
            h_data.x,
            h_data.batch,
            h_data.edge_index,
            getattr(h_data, 'intra_enrichment_weights', None),
        )
        t_sub_score, _ = self.cdsagg(
            t_data.x,
            t_data.batch,
            t_data.edge_index,
            getattr(t_data, 'intra_enrichment_weights', None),
        )

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        # Legacy SGAR-DDI gray-gating rule, now isolated as Stage 02.
        pair_sub_score = 0.5 * (h_sub_score + t_sub_score)
        pair_gate = 1.0 + self.gray_gate_alpha * (pair_sub_score - 0.5)
        pair_gate = torch.clamp(pair_gate, min=0.7, max=1.3)
        self.last_pair_gate = pair_gate.detach()

        h_gate = _graph_gate_from_batch(
            pair_gate, h_data.batch, h_interRep.device, h_interRep.dtype
        )
        t_gate = _graph_gate_from_batch(
            pair_gate, t_data.batch, t_interRep.device, t_interRep.dtype
        )

        if h_gate is not None:
            h_interRep = h_interRep * h_gate
        if t_gate is not None:
            t_interRep = t_interRep * t_gate

        # Concatenate intra- and gated inter-graph representations.
        h_rep = torch.cat([h_intraRep, h_interRep], dim=1)
        t_rep = torch.cat([t_intraRep, t_interRep], dim=1)
        h_data.x = h_rep
        t_data.x = t_rep

        # Stage 01 SP-SAR graph readout.
        h_global_graph_emb, _ = self.readout(
            h_data.x,
            h_data.batch,
            h_data.edge_index,
            getattr(h_data, 'intra_enrichment_weights', None),
        )
        t_global_graph_emb, _ = self.readout(
            t_data.x,
            t_data.batch,
            t_data.edge_index,
            getattr(t_data, 'intra_enrichment_weights', None),
        )

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb
