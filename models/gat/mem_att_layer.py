import torch
import torch.nn as nn
from models.gat.fc import FCNet
import math
from torch.nn.utils.weight_norm import weight_norm
import pdb


class GraphSelfAttentionLayer(nn.Module):
    def __init__(self, feat_dim, nongt_dim=20, pos_emb_dim=-1, m=10,
                 num_heads=8, dropout=0.15):
        """ Attetion module with vectorized version

        Args:
            position_embedding: [num_rois, nongt_dim, pos_emb_dim]
                                used in implicit relation
            pos_emb_dim: set as -1 if explicit relation
            nongt_dim: number of objects consider relations per image
            fc_dim: should be same as num_heads
            feat_dim: dimension of roi_feat
            num_heads: number of attention heads
            m: dimension of memory matrix
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GraphSelfAttentionLayer, self).__init__()
        # multi head
        self.fc_dim = num_heads
        self.feat_dim = feat_dim
        self.dim = (feat_dim, feat_dim, feat_dim)
        self.dim_group = (int(self.dim[0] / num_heads),
                          int(self.dim[1] / num_heads),
                          int(self.dim[2] / num_heads))
        self.num_heads = num_heads
        self.pos_emb_dim = pos_emb_dim
        if self.pos_emb_dim > 0:
            self.pair_pos_fc1 = FCNet([pos_emb_dim, self.fc_dim], None, dropout)
        self.query = FCNet([feat_dim, self.dim[0]], None, dropout)
        self.nongt_dim = nongt_dim

        self.key = FCNet([feat_dim, self.dim[1]], None, dropout)

        self.linear_out_ = weight_norm(
            nn.Conv2d(in_channels=self.fc_dim * feat_dim,
                      out_channels=self.dim[2],
                      kernel_size=(1, 1),
                      groups=self.fc_dim), dim=None)

        self.m = m
        if self.m > 0:
            self.m_k = nn.Parameter(torch.FloatTensor(1, m, feat_dim))
            self.m_v = nn.Parameter(torch.FloatTensor(1, m, feat_dim))
            nn.init.normal_(self.m_k, 0, 1 / self.feat_dim / self.num_heads)
            nn.init.normal_(self.m_v, 0, 1 / self.m)

    def adding_goemetric_features(self, position_embedding, nongt_dim, weighted_aff):
        # Adding goemetric features
        bs = position_embedding.size(0)
        position_embedding = position_embedding.float()
        # [batch_size,num_rois * nongt_dim, emb_dim]
        position_embedding_reshape = position_embedding.view(
            (bs, -1, self.pos_emb_dim))  # (B,nongt_dim*nongt_dim,48)

        # position_feat_1, [batch_size,num_rois * nongt_dim, fc_dim]
        position_feat_1 = self.pair_pos_fc1(position_embedding_reshape)
        position_feat_1_relu = nn.functional.relu(position_feat_1)  # (B,nongt_dim*nongt_dim,nhead)

        # aff_weight, [batch_size,num_rois, nongt_dim, fc_dim]
        aff_weight = position_feat_1_relu.view((bs, -1, nongt_dim, self.fc_dim))

        # aff_weight, [batch_size,num_rois, fc_dim, nongt_dim]
        aff_weight = torch.transpose(aff_weight, 2, 3)

        thresh = torch.FloatTensor([1e-6]).cuda()
        # weighted_aff, [batch_size,num_rois, fc_dim, nongt_dim+m]
        threshold_aff = torch.max(aff_weight, thresh)
        weighted_aff[:, :, :, :nongt_dim] = weighted_aff[:, :, :, :nongt_dim] + torch.log(threshold_aff)
        return weighted_aff

    def forward(self, roi_feat, adj_matrix, position_embedding, label_biases_att, aux_feat):
        """
        Args:
            roi_feat: [batch_size, N, feat_dim]
            adj_matrix: [batch_size, N, nongt_dim]
            position_embedding: [num_rois, nongt_dim, pos_emb_dim]
            aux_feat: [batch_size,1,feat_dim]
        Returns:
            output: [batch_size, num_rois, output_dim]
        """
        batch_size = roi_feat.size(0)
        num_rois = roi_feat.size(1)
        nongt_dim = self.nongt_dim if self.nongt_dim < num_rois else num_rois
        # [batch_size,nongt_dim, feat_dim]
        nongt_roi_feat = roi_feat[:, :nongt_dim, :]

        # Q
        q_data = self.query(roi_feat)
        q_data_batch = q_data.view(batch_size, num_rois, self.num_heads, self.dim_group[0])
        q_data_batch = torch.transpose(q_data_batch, 1, 2)  # [batch_size,num_heads, num_rois, feat_dim /num_heads]

        # K
        if self.m > 0:
            m_k = math.sqrt(self.feat_dim / self.num_heads) * self.m_k.expand(batch_size, self.m, self.feat_dim)
            k_data = torch.cat((self.key(nongt_roi_feat), m_k * aux_feat), 1)
        else:
            k_data = self.key(nongt_roi_feat)
        k_data_batch = k_data.view(batch_size, nongt_dim + self.m, self.num_heads, self.dim_group[1])
        k_data_batch = torch.transpose(k_data_batch, 1, 2)  # [B, num_heads,nongt_dim+m, feat_dim /num_heads]

        # V
        if self.m > 0:
            m_v = math.sqrt(self.feat_dim / self.num_heads) * self.m_v.expand(batch_size, self.m, self.feat_dim)
            v_data = torch.cat((nongt_roi_feat, m_v * aux_feat), 1)  # [B, nongt_dim+m,num_heads, feat_dim /num_heads]
        else:
            v_data = nongt_roi_feat

        aff = torch.matmul(q_data_batch, torch.transpose(k_data_batch, 2, 3))  # [B, num_heads, num_rois, nongt_dim+m]

        aff_scale = (1.0 / math.sqrt(float(self.dim_group[1]))) * aff
        aff_scale = torch.transpose(aff_scale, 1, 2)  # (B,N,headn,N+m)
        weighted_aff = aff_scale

        if position_embedding is not None and self.pos_emb_dim > 0:
            # # Adding goemetric features
            if len(position_embedding.shape) == 4:
                weighted_aff = self.adding_goemetric_features(position_embedding, nongt_dim, weighted_aff)
            else:
                view_num = position_embedding.size(0)
                was = []
                for vi in range(view_num):
                    wa = self.adding_goemetric_features(position_embedding[vi], nongt_dim, weighted_aff)
                    was.append(wa)
                weighted_aff = torch.cat(was, 0)  # (view_num*B,N,nhead,N+m)
        if adj_matrix is not None:
            if weighted_aff.size(0) > adj_matrix.size(0):
                view_num = weighted_aff.size(0) // adj_matrix.size(0)
                adj_matrix = adj_matrix.unsqueeze(0).repeat(view_num, 1, 1, 1, 1).view(weighted_aff.size(0), num_rois,
                                                                                       nongt_dim, 1)
                label_biases_att = label_biases_att.unsqueeze(0).repeat(view_num, 1, 1, 1).view(weighted_aff.size(0),
                                                                                                num_rois,
                                                                                                nongt_dim)
            weighted_aff_transposed = torch.transpose(weighted_aff, 2, 3)  # [view_num*B,N, N+m, num_heads]
            zero_vec = -9e15 * torch.ones_like(weighted_aff_transposed[:, :, :nongt_dim, :])

            adj_matrix = adj_matrix.view(adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2], 1)
            adj_matrix_expand = adj_matrix.expand((-1, -1, -1, weighted_aff_transposed.shape[-1]))
            weighted_aff_masked = torch.where(adj_matrix_expand > 0, weighted_aff_transposed[:, :, :nongt_dim, :],
                                              zero_vec)

            weighted_aff_masked = weighted_aff_masked + label_biases_att.unsqueeze(-1)

            weighted_aff_transposed[:, :, :nongt_dim, :] = weighted_aff_masked
            weighted_aff = torch.transpose(weighted_aff_transposed, 2, 3)

        # aff_softmax, [batch_size, num_rois, fc_dim, nongt_dim+m]
        aff_softmax = nn.functional.softmax(weighted_aff, 3)

        # aff_softmax_reshape, [batch_size, num_rois*fc_dim, nongt_dim+m]
        aff_softmax_reshape = aff_softmax.view((batch_size, -1, nongt_dim + self.m))  # (B,N*fc_dim,N+m)

        # output_t, [batch_size, num_rois * fc_dim, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)  # (B,N*fc_dim,N+m)*(B,N+m,768)

        # output_t, [batch_size*num_rois, fc_dim * feat_dim, 1, 1]
        output_t = output_t.view((-1, self.fc_dim * self.feat_dim, 1, 1))

        # linear_out, [batch_size*num_rois, dim[2], 1, 1]
        linear_out = self.linear_out_(output_t)
        output = linear_out.view((-1, num_rois, self.dim[2]))
        return output
