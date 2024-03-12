import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.utils import get_siamese_features
import math
from models.backbones.point_net_pp import PointNetPP
from transformers import BertModel, BertConfig
from models.backbones.mlp import MLP
from models.gat.memgat import GAT_Encoder
import time
from models.gat.matrix_emb import prepare_graph_variables
from torch.autograd import Variable
from models.backbones.mlp import Concat


class ReferIt3DNet_transformer(nn.Module):

    def __init__(self,
                 args,
                 n_obj_classes,
                 ignore_index):

        super().__init__()

        self.view_number = args.view_number
        self.rotate_number = args.rotate_number

        self.aggregate_type = args.aggregate_type

        self.encoder_layer_num = args.encoder_layer_num
        self.decoder_layer_num = args.decoder_layer_num
        self.decoder_nhead_num = args.decoder_nhead_num

        self.object_dim = args.object_latent_dim
        self.inner_dim = args.inner_dim

        self.dropout_rate = args.dropout_rate
        self.lang_cls_alpha = args.lang_cls_alpha
        self.obj_cls_alpha = args.obj_cls_alpha

        self.gat = args.gat
        self.multi_pos = args.multi_pos
        self.sem_encode = args.sem_encode

        self.nongt_dim = args.max_distractors + 1
        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                         sa_n_samples=[[32], [32], [None]],
                                         sa_radii=[[0.2], [0.4], [None]],
                                         sa_mlps=[[[3, 64, 64, 128]],
                                                  [[128, 128, 128, 256]],
                                                  [[256, 256, self.object_dim, self.object_dim]]])

        if self.sem_encode:
            self.semantic_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                               sa_n_samples=[[32], [32], [None]],
                                               sa_radii=[[0.2], [0.4], [None]],
                                               sa_mlps=[[[1, 64, 128]],
                                                        [[128, 128, 256]],
                                                        [[256, 256, self.object_dim]]])  # 768
            self.aux_feature_mapping = nn.Sequential(
                nn.Linear(self.object_dim, self.inner_dim),
                nn.LayerNorm(self.inner_dim),
            )
            self.fusion_layer = Concat(self.object_dim, self.object_dim)

        if self.gat:
            self.relation_encoder = GAT_Encoder(lay_num=args.lay_number,
                                                multi_view=self.view_number > 0,
                                                in_feat_dim=self.object_dim,
                                                out_feat_dim=768,
                                                nongt_dim=self.nongt_dim,
                                                pos_emb_dim=48,
                                                num_heads=8,
                                                m=args.m)

        self.language_encoder = BertModel.from_pretrained(args.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:self.encoder_layer_num]

        self.refer_encoder = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=self.inner_dim,
                                                                                    nhead=self.decoder_nhead_num,
                                                                                    dim_feedforward=2048,
                                                                                    activation="gelu"),
                                                   num_layers=self.decoder_layer_num)

        # Classifier heads
        self.language_target_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                                 nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                 nn.Linear(self.inner_dim, n_obj_classes))

        self.object_language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim),
                                                 nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                 nn.Linear(self.inner_dim, 1))

        self.obj_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, n_obj_classes],
                               dropout_rate=self.dropout_rate)
        if self.sem_encode:
            self.obj_sem_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, n_obj_classes],
                                   dropout_rate=self.dropout_rate)

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(6, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.logit_loss = nn.CrossEntropyLoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def get_adj_matrix(self, B, N, batch):
        real_nums = batch['context_size']
        adj_mats = torch.zeros((B, N, N, 1))
        for bi in range(B):
            adj_mats[bi, :real_nums[bi], :real_nums[bi]] = torch.ones((real_nums[bi], real_nums[bi], 1))
        return Variable(adj_mats)

    def get_bsize(self, r, bsize):
        B = r.shape[0]
        new_size = bsize.clone()
        for bi in range(B):
            if r[bi] == 1 or r[bi] == 3:
                new_size[bi, :, 0] = bsize[bi, :, 1]
                new_size[bi, :, 1] = bsize[bi, :, 0]
        return new_size

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
        input_points = input_points.float().to(self.device)  # (B,52,1024,7)
        box_infos = box_infos.float().to(self.device)  # (B,52,6) cx,cy,cz,lx,ly,lz
        xyz = input_points[:, :, :, :3]  # (B,52,1024,3)
        bxyz = box_infos[:, :, :3]  # B,52,3
        bsize = box_infos[:, :, -3:]
        B, N, P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor([i * 2.0 * np.pi / self.rotate_number for i in range(self.rotate_number)]).to(
            self.device)
        view_theta_arr = torch.Tensor([i * 2.0 * np.pi / self.view_number for i in range(self.view_number)]).to(
            self.device)

        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            r = torch.randint(0, self.rotate_number, (B,))
            theta = rotate_theta_arr[r]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).to(self.device)[
                None].repeat(B, 1, 1)
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[:, :, :, :3] = torch.matmul(xyz.reshape(B, N * P, 3), rotate_matrix).reshape(B, N, P, 3)
            bxyz = torch.matmul(bxyz.reshape(B, N, 3), rotate_matrix).reshape(B, N, 3)
            bsize = self.get_bsize(r, bsize)

        # multi-view
        boxs = []
        for r, theta in enumerate(view_theta_arr):
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                          [math.sin(theta), math.cos(theta), 0.0],
                                          [0.0, 0.0, 1.0]]).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B * N, 3), rotate_matrix).reshape(B, N, 3)
            new_size = self.get_bsize(torch.zeros((B,)) + r, bsize)
            boxs.append(torch.cat([rxyz, new_size], dim=-1))

        boxs = torch.stack(boxs, dim=1)  # (B，view_num,N,4)
        if self.view_number == 1:
            boxs = torch.squeeze(boxs, 1)
        return input_points, boxs

    def compute_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, AUX_LOGITS=None):
        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])
        total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss + self.lang_cls_alpha * lang_clf_loss
        return total_loss

    def forward(self, batch: dict):
        self.device = self.obj_feature_mapping[0].weight.device

        # rotation augmentation and multi_view generation
        obj_points, boxs = self.aug_input(batch['objects'], batch['box_info'])
        B, N, P = obj_points.shape[:3]

        mask = torch.zeros(B, N)
        real_nums = batch['context_size']
        for bi in range(B):
            mask[bi][:real_nums[bi]] = 1.0
        mask = mask.view(B, N, 1).to(self.device)

        ## obj_encoding
        objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack)  # (B,52,768)
        objects_features = objects_features * mask

        # obj_encoding
        obj_feats = self.obj_feature_mapping(objects_features)  # (B,N,768)
        box_infos = self.box_feature_mapping(boxs)  # (B,N,768)

        CLASS_LOGITS = self.obj_clf(obj_feats.reshape(B * N, -1)).reshape(B, N, -1)  # (B,N,40)

        if self.sem_encode:
            clogit = F.softmax(CLASS_LOGITS, -1)
            _, cl_label = torch.max(clogit, dim=-1)
            cl_label = cl_label / clogit.size(-1)
            cl_label = cl_label.view(B, N, 1, 1).repeat(1, 1, P, 1)  # (B,N,1024,1)
            obj_aux = torch.cat((obj_points[:, :, :, :3], cl_label), -1)

            aux_features = get_siamese_features(self.semantic_encoder, obj_aux, aggregator=torch.stack)  # (B,N,768)
            aux_features = aux_features * mask
            aux_feats = self.aux_feature_mapping(aux_features)

            CLASS_LOGITS = self.obj_sem_clf(aux_feats.reshape(B * N, -1)).reshape(B, N, -1)

            obj_feats = self.fusion_layer(obj_feats, aux_feats)
            # obj_feats = aux_feats

        ## language_encoding
        lang_tokens = batch['lang_tokens']
        lang_infos = self.language_encoder(**lang_tokens)[0]  # (B,len,768)
        lang_features = lang_infos[:, 0]
        # <LOSS>: lang_cls
        LANG_LOGITS = self.language_target_clf(lang_features)

        aux_lang = lang_features.unsqueeze(1)

        # gat_encoding
        if self.gat:
            adj_mats = self.get_adj_matrix(B, N, batch).to(self.device)
            if self.multi_pos:
                pos_emb = prepare_graph_variables(boxs, self.nongt_dim, 48, self.view_number, self.device)
            else:
                pos_emb = None
            re_infos = self.relation_encoder(obj_feats, adj_mats, pos_emb, aux_lang)
            if re_infos.size(0) > B:
                re_infos = re_infos.view(self.view_number, B, N, -1).transpose(0, 1)
        else:
            re_infos = obj_feats

        if self.view_number > 1:
            if len(re_infos.shape) < len(box_infos.shape):
                re_infos = re_infos[:, None].repeat(1, self.view_number, 1, 1)
            obj_infos = re_infos + box_infos
            cat_infos = obj_infos.reshape(B * self.view_number, -1, self.inner_dim)  # (B*view_num,N,768)
            mem_infos = lang_infos[:, None].repeat(1, self.view_number, 1, 1).reshape(B * self.view_number, -1,
                                                                                      self.inner_dim)
            out_feats = self.refer_encoder(cat_infos.transpose(0, 1), mem_infos.transpose(0, 1)).transpose(0, 1) \
                .reshape(B, self.view_number, -1, self.inner_dim)

            ## view_aggregation
            refer_feat = out_feats
            if self.aggregate_type == 'avg':
                agg_feats = (refer_feat / self.view_number).sum(dim=1)
            elif self.aggregate_type == 'avgmax':
                agg_feats = (refer_feat / self.view_number).sum(dim=1) + refer_feat.max(dim=1).values
            else:
                agg_feats = refer_feat.max(dim=1).values
            final_feats = agg_feats

        else:
            obj_infos = re_infos + box_infos
            mem_infos = lang_infos
            final_feats = self.refer_encoder(obj_infos.transpose(0, 1), mem_infos.transpose(0, 1)).transpose(0, 1) \
                .reshape(B, -1, self.inner_dim)

        LOGITS = self.object_language_clf(final_feats).squeeze(-1)

        # <LOSS>: semantic PC_cls ,lang_cls
        LOSS = self.compute_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOGITS)

        mm = torch.zeros(B, N).to(self.device)
        for bi in range(B):
            mm[bi][real_nums[bi]:] = -9e15
        LOGITS = LOGITS + mm
        return LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS
