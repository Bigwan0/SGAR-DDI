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


# intra rep with SIMPLY stronger enrichment (just increased scale)
class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim, 32, 2)
        # ONLY CHANGE: Increased scale from 0.001 to 0.01 (10x stronger)
        self.enrichment_scale = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, data):
        input_feature, edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature, edge_index)
        
        # Apply conservative enrichment (SAME AS BEFORE, just stronger scale)
        if (hasattr(data, 'intra_enrichment_weights') and 
            data.intra_enrichment_weights.numel() > 0 and 
            edge_index.numel() > 0):
            
            # ONLY CHANGE: 0.1 instead of 0.01 (10x stronger max impact)
            scale_factor = 0.1 * torch.sigmoid(self.enrichment_scale)
            
            # Simple average enrichment per node (SAME AS CONSERVATIVE)
            src_nodes = edge_index[0]
            node_weights = torch.zeros(intra_rep.size(0), device=intra_rep.device)
            node_weights.index_add_(0, src_nodes, data.intra_enrichment_weights)
            
            # Normalize and clamp (SAME AS CONSERVATIVE)
            node_counts = torch.zeros_like(node_weights)
            node_counts.index_add_(0, src_nodes, torch.ones_like(data.intra_enrichment_weights))
            node_counts = torch.clamp(node_counts, min=1)
            node_weights = node_weights / node_counts
            
            # Slightly wider clamping for stronger effect
            node_weights = torch.clamp(node_weights, min=0.2, max=5.0)  # Was 0.5-2.0
            
            # Additive enrichment (SAME APPROACH AS CONSERVATIVE)
            enrichment_bias = scale_factor * (node_weights.unsqueeze(-1) - 1.0)
            intra_rep = intra_rep + enrichment_bias
            
        return intra_rep

# inter rep with SIMPLY stronger enrichment (just increased scale)
class InterGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim, input_dim), 32, 2)
        # ONLY CHANGE: Increased scale from 0.001 to 0.01 (10x stronger)
        self.enrichment_scale = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        
        # Compute attention in parallel (SAME AS CONSERVATIVE)
        t_rep = self.inter((h_input, t_input), edge_index)
        h_rep = self.inter((t_input, h_input), edge_index[[1,0]])
        
        # Apply conservative enrichment (SAME AS BEFORE, just stronger scale)
        if (hasattr(b_graph, 'enrichment_weights') and 
            b_graph.enrichment_weights is not None and 
            b_graph.enrichment_weights.numel() > 0 and 
            edge_index.numel() > 0):
            
            # ONLY CHANGE: 0.1 instead of 0.01 (10x stronger max impact)
            scale_factor = 0.1 * torch.sigmoid(self.enrichment_scale)
            
            # Simple approach: average enrichment (SAME AS CONSERVATIVE)
            avg_enrichment = torch.clamp(b_graph.enrichment_weights.mean(), min=0.2, max=5.0)  # Wider range
            
            # Additive enrichment (SAME APPROACH AS CONSERVATIVE)
            enrichment_bias = scale_factor * (avg_enrichment - 1.0)
            h_rep = h_rep + enrichment_bias
            t_rep = t_rep + enrichment_bias
            
        return h_rep, t_rep