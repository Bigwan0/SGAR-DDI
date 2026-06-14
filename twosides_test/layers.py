import math
import datetime

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv,SAGPooling,global_add_pool,GATConv



class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        return attentions

class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
      
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        
        rels = rels.view(-1, self.n_features, self.n_features)
        # print(heads.size(),rels.size(),tails.size())
        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
          scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
       
        return scores 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim, 32, 2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # UPDATED: Increased enrichment scale from 0.001 to 0.5
        self.enrichment_scale = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, data):
        input_feature, edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        input_feature = self.dropout(input_feature)
        intra_rep = self.intra(input_feature, edge_index)
        
        # Apply enrichment with increased scale
        if (hasattr(data, 'intra_enrichment_weights') and 
            data.intra_enrichment_weights.numel() > 0 and 
            edge_index.numel() > 0):
            
            # UPDATED: Increased scale factor from 0.01 to 0.3 (max ~0.3)
            scale_factor = 0.3 * torch.tanh(self.enrichment_scale)
            
            # Simple average enrichment per node
            src_nodes = edge_index[0]
            node_weights = torch.zeros(intra_rep.size(0), device=intra_rep.device)
            node_weights.index_add_(0, src_nodes, data.intra_enrichment_weights)
            
            # Normalize by edge count per node
            node_counts = torch.zeros_like(node_weights)
            node_counts.index_add_(0, src_nodes, torch.ones_like(data.intra_enrichment_weights))
            node_counts = torch.clamp(node_counts, min=1)
            node_weights = node_weights / node_counts
            
            # UPDATED: Widened clamping range from [0.8, 1.5] to [0.3, 3.0]
            node_weights = torch.clamp(node_weights, min=0.3, max=3.0)
            
            # Multiplicative enrichment instead of additive for more stability
            enrichment_factor = 1.0 + scale_factor * (node_weights.unsqueeze(-1) - 1.0)
            intra_rep = intra_rep * enrichment_factor
            
        return intra_rep


class InterGraphAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim, input_dim), 32, 2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # UPDATED: Increased enrichment scale from 0.001 to 0.5
        self.enrichment_scale = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        
        h_input = self.dropout(h_input)
        t_input = self.dropout(t_input)
        
        # Compute attention in parallel
        t_rep = self.inter((h_input, t_input), edge_index)
        h_rep = self.inter((t_input, h_input), edge_index[[1,0]])
        
        # Apply enrichment with increased scale
        if (hasattr(b_graph, 'enrichment_weights') and 
            b_graph.enrichment_weights is not None and 
            b_graph.enrichment_weights.numel() > 0 and 
            edge_index.numel() > 0):
            
            # UPDATED: Increased scale factor from 0.01 to 0.3 (max ~0.3)
            scale_factor = 0.5 * torch.tanh(self.enrichment_scale)
            
            # More sophisticated enrichment: use weighted average instead of simple average
            # UPDATED: Widened clamping range from [0.8, 1.5] to [0.3, 3.0]
            edge_weights = torch.clamp(b_graph.enrichment_weights, min=0.3, max=3.0)
            
            # Weight by edge importance (can be enhanced further)
            attention_weights = F.softmax(edge_weights, dim=0)
            weighted_enrichment = torch.sum(attention_weights * edge_weights)
            
            # Multiplicative enrichment for more stability
            enrichment_factor = 1.0 + scale_factor * (weighted_enrichment - 1.0)
            h_rep = h_rep * enrichment_factor
            t_rep = t_rep * enrichment_factor
            
        return h_rep, t_rep