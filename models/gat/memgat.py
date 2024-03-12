import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm
from models.gat.mem_att_layer import GraphSelfAttentionLayer
from models.backbones.mlp import Concat


class GAT_Encoder(nn.Module):
    def __init__(self, lay_num, in_feat_dim, out_feat_dim, multi_view,
                 nongt_dim, dropout=0.15, m=10,
                 num_heads=16, pos_emb_dim=-1):
        super(GAT_Encoder, self).__init__()
        self.lay_num = lay_num
        self.multi_view = multi_view
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.dropout = nn.Dropout(dropout)
        self.feat_fc = weight_norm(nn.Linear(in_feat_dim, out_feat_dim, bias=True), dim=None)
        self.bias = weight_norm(nn.Linear(1, 1, bias=True), dim=None)
        self.nongt_dim = nongt_dim
        self.pos_emb_dim = pos_emb_dim
        self.g_atts = nn.ModuleList([GraphSelfAttentionLayer(pos_emb_dim=pos_emb_dim,
                                                             num_heads=num_heads,
                                                             feat_dim=out_feat_dim,
                                                             nongt_dim=nongt_dim,
                                                             m=m)
                                     for _ in range(lay_num)])
        self.lay_num = lay_num
        self.fusion_layer = Concat(out_feat_dim, out_feat_dim)

    def forward(self, obj_feat, adj_matrix, pos_emb, aux_feat):
        batch_size, num_objs, feat_dim = obj_feat.shape
        nongt_dim = self.nongt_dim
        adj_matrix = adj_matrix.float()

        # Self - looping edges
        # [batch_size,num_rois, out_feat_dim]
        self_feat = self.feat_fc(self.dropout(obj_feat))  # (B,N,768)

        # [batch_size,num_rois, nongt_dim,label_num]
        input_adj_matrix = adj_matrix[:, :, :nongt_dim, :]
        condensed_adj_matrix = torch.sum(input_adj_matrix, dim=-1)

        # [batch_size,num_rois, nongt_dim]
        v_biases_neighbors = self.bias(input_adj_matrix).squeeze(-1)

        # [batch_size,num_rois, out_feat_dim]
        out = self_feat
        for i, l in enumerate(self.g_atts):
            if i == 0:
                out = l(self_feat, condensed_adj_matrix, pos_emb, v_biases_neighbors, aux_feat)
                if self.multi_view:
                    self_feat = self_feat.unsqueeze(0).repeat(out.size(0) // batch_size, 1, 1, 1).view(out.size(0),
                                                                                                       num_objs, -1)
                    aux_feat = aux_feat.repeat(out.size(0) // batch_size, 1, 1)
                out = self_feat + out
            else:
                out = l(out, condensed_adj_matrix, None, v_biases_neighbors, aux_feat)

        output = self.fusion_layer(self_feat, out)

        return output
