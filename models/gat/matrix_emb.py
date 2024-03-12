import numpy as np
import math
import torch
from torch.autograd import Variable


def torch_extract_position_embedding(position_mat, feat_dim, wave_length=1000,
                                     device=torch.device("cuda")):
    # position_mat, [batch_size,num_rois, nongt_dim, 3]
    feat_range = torch.arange(0, feat_dim / 6)  # [0,1,2,3,4,5,6,7]
    dim_mat = torch.pow(torch.ones((1,)) * wave_length,
                        (6 / feat_dim) * feat_range)  # (8,)
    dim_mat = dim_mat.view(1, 1, 1, -1).to(device)  # (1,1,1,8)
    position_mat = torch.unsqueeze(100.0 * position_mat, dim=4)  # (B,N,nongt_dim,3,1)
    div_mat = torch.div(position_mat.to(device), dim_mat)  # (B,N,nongt_dim,3,8)
    sin_mat = torch.sin(div_mat)  # (B,N,nongt_dim,3,8)
    cos_mat = torch.cos(div_mat)  # (B,N,nongt_dim,3,8)
    embedding = torch.cat([sin_mat, cos_mat], -1)  # (B,N,nongt_dim,3,16)
    # embedding, [batch_size,num_rois, nongt_dim, feat_dim]
    embedding = embedding.view(embedding.shape[0], embedding.shape[1],
                               embedding.shape[2], -1)
    return embedding


def torch_extract_position_matrix(bbox, nongt_dim):
    """ Extract position matrix

    Args:
        bbox: [batch_size, num_boxes, 6]

    Returns:
        position_matrix: [batch_size, num_boxes, nongt_dim, 4]
    """

    cx, cy, cz, lx, ly, lz = torch.split(bbox, 1, dim=-1)

    # [batch_size,num_boxes, num_boxes]
    delta_x = cx - torch.transpose(cx, 1, 2)
    delta_x = torch.div(delta_x, lx + 1e-10)
    delta_x = torch.abs(delta_x)
    threshold = 1e-3
    delta_x[delta_x < threshold] = threshold
    delta_x = torch.log(delta_x)

    delta_y = cy - torch.transpose(cy, 1, 2)
    delta_y = torch.div(delta_y, ly + 1e-10)
    delta_y = torch.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = torch.log(delta_y)

    delta_z = cz - torch.transpose(cz, 1, 2)
    delta_z = torch.div(delta_z, lz + 1e-10)
    delta_z = torch.abs(delta_z)
    delta_z[delta_z < threshold] = threshold
    delta_z = torch.log(delta_z)

    concat_list = [delta_x, delta_y, delta_z]
    for idx, sym in enumerate(concat_list):
        sym = sym[:, :nongt_dim]
        concat_list[idx] = torch.unsqueeze(sym, dim=3)
    position_matrix = torch.cat(concat_list, 3)
    return position_matrix


def prepare_graph_variables(bbox, nongt_dim, pos_emb_dim, view_num, device):
    if view_num == 1:
        bbox = bbox.to(device)
        pos_mat = torch_extract_position_matrix(bbox, nongt_dim=nongt_dim)  # (B,N,nongt_dim,3)
        pos_emb = torch_extract_position_embedding(pos_mat, feat_dim=pos_emb_dim, device=device)
        pos_emb_var = Variable(pos_emb).to(device)
    else:
        output = []
        for i in range(view_num):
            bb = bbox[:, i]
            bb = bb.to(device)
            pos_mat = torch_extract_position_matrix(bb, nongt_dim=nongt_dim)  # (B,N,nongt_dim,3)
            pos_emb = torch_extract_position_embedding(pos_mat, feat_dim=pos_emb_dim, device=device)  # (B,N,nongt_dim,48)
            output.append(pos_emb)
        pos_embs = torch.stack(output)
        pos_emb_var = Variable(pos_embs).to(device) # (4,B,N,nongt_dim,48)
    return pos_emb_var
