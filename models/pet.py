"""
PET model and criterion classes
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

import sys
import os
import scipy.spatial
from scipy.spatial import KDTree

from models.transformer.utils import query_partition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import List, Optional

import numpy as np
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .stable_matcher import build_stable_matcher
from .backbones import *
from .transformer import *
from .position_encoding import build_position_encoding


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def inverse_sigmoid_both(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    logits =torch.log(x1 / x2)
    return torch.tanh(logits)

class QuadtreeSplitterWithAttention(nn.Module):
    def __init__(self, hidden_dim, context_h, context_w, nhead=4):
        super(QuadtreeSplitterWithAttention, self).__init__()
        self.context_h = context_h
        self.context_w = context_w
        self.hidden_dim = hidden_dim
        
        # Attention Layer
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=0.0, batch_first=True)

        # Fully connected projection to match output dimension
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        src:  torch.Tensor,      # (B,C,H,W)
        pos:  torch.Tensor,      # (B,C,H,W)
        mask: Optional[torch.Tensor],   # (B,1,H,W) or (B,H,W); False=valid, True=invalid
        train_flag=None
    ) -> torch.Tensor:
        B, C, H, W = src.shape
        src_with_pos = src + pos                             # (B,C,H,W)

        # ---------- reshape to (B, N, C) ----------
        tokens_q = src.flatten(2).permute(0, 2, 1)           # (B, N, C) N=H*W
        tokens_kv = src_with_pos.flatten(2).permute(0, 2, 1) # (B, N, C)

        # ---------- build key_padding_mask ----------
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)                       # (B,H,W)
            key_padding_mask = mask.flatten(1).bool()        # True=skip
            all_masked = key_padding_mask.all(dim=1, keepdim=True)
            key_padding_mask = key_padding_mask.masked_fill(all_masked, False)
        else:
            key_padding_mask = None

        # ---------- attention on full resolution ----------
        attn_out, _ = self.attn(
            query=tokens_q,
            key=tokens_kv,
            value=tokens_kv,
            key_padding_mask=key_padding_mask
        )                                                    # (B,N,C)

        # ---------- 1-channel projection ----------
        proj = self.fc_out(attn_out)                         # (B,N,1)
        proj = proj.transpose(1, 2).reshape(B, 1, H, W)      # (B,1,H,W)

        # ---------- *now* do pooling --------------- ### ← changed
        pooled = F.avg_pool2d(
            proj,
            kernel_size=(self.context_h, self.context_w),
            stride=(self.context_h, self.context_w)
        )                                                    # (B,1,Hp,Wp)

        return pooled
    
class FirstQuadtreeSplitterWithAttention(nn.Module):
    def __init__(self, hidden_dim, context_h, context_w, nhead=4):
        super(FirstQuadtreeSplitterWithAttention, self).__init__()
        self.context_h = context_h
        self.context_w = context_w
        self.hidden_dim = hidden_dim
        
        # Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=0.1)

        # Fully connected projection to match output dimension
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, src, pos,
                mask: Optional[torch.Tensor],   # (B,1,H,W) or (B,H,W); False=valid, True=invalid
                train_flag=None):
            """
            Args:
                src: Input feature map of shape [batch_size, hidden_dim, height, width]
                pos: Positional encoding of shape [batch_size, hidden_dim, height, width]
            Returns:
                Output of shape [batch_size, 1, pooled_height, pooled_width]
            """
            batch_size, hidden_dim, height, width = src.shape

            # Add positional encoding
            src_with_pos = src + pos 

            pooled_src = F.avg_pool2d(src, kernel_size=(self.context_h, self.context_w), stride=(self.context_h, self.context_w))
            pooled_src_with_pos = F.avg_pool2d(src_with_pos, kernel_size=(self.context_h, self.context_w), stride=(self.context_h, self.context_w))
            pooled_h, pooled_w = pooled_src.shape[-2], pooled_src.shape[-1]

            pooled_src = pooled_src.permute(0, 2, 3, 1).reshape(-1, pooled_h * pooled_w, hidden_dim).permute(1, 0, 2)  # Query
            pooled_src_with_pos = pooled_src_with_pos.permute(0, 2, 3, 1).reshape(-1, pooled_h * pooled_w, hidden_dim).permute(1, 0, 2)  # Key & Value

            attn_output, _ = self.attention(pooled_src, pooled_src_with_pos, pooled_src_with_pos)
            attn_output = attn_output.permute(1, 0, 2)  # [batch_size, pooled_h * pooled_w, hidden_dim]

            # Project back to desired output dimension
            attn_output = self.fc_out(attn_output)  # [batch_size, pooled_h * pooled_w, 1]
            attn_output = attn_output.reshape(batch_size, 1, pooled_h, pooled_w)  # Reshape to [batch_size, 1, pooled_h, pooled_w]

            return attn_output

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, relu=True, bn=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasePETCount(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim
        
        self.patch_size = args.patch_size

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)   # pred class
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)        # pred location： output 2, hidden layers num 3

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'
        self.quadtree_layer = quadtree_layer
        
        self.hidden_dim = args.hidden_dim
        
        # probability map
        # if args.prob_map_lc == 'f4x':
        #     self.prob_conv = nn.Sequential(nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=3, padding=1))
        
        # box-detr para needed:
        self.opt_query = args.opt_query_decoder
        self.opt_query_con = args.opt_query_con
        
        if args.opt_query_decoder:
            
            # patch_size = kwargs['patch_size']
            # egde_num_queries = patch_size // self.pq_stride
            # num_queries = egde_num_queries * egde_num_queries
            # self.refpoint_embed = nn.Embedding(num_queries, 2)
            
            self.dec_win_size = args.sparse_dec_win_size if quadtree_layer == 'sparse' else args.dense_dec_win_size
            # sparse_dec_win_size: [8, 4]  dense_dec_win_size: [4, 2]
            dec_w, dec_h = self.dec_win_size
            self.norm_dec = [size / self.patch_size for size in self.dec_win_size]
            self.refloc_embed = nn.Embedding(dec_w*dec_h, 2) 
            
            if not hasattr(args, 'box_setting'):
                self.box_setting = 1
            else:
                self.box_setting = args.box_setting
            
            # self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
            # self.transformer.decoder.bbox_embed = self.bbox_embed
            # nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            # nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        if args.opt_query_con:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
            # self.transformer.decoder.bbox_embed = self.bbox_embed
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
            
            # adapt:
            self.dec_win_size = args.sparse_dec_win_size if quadtree_layer == 'sparse' else args.dense_dec_win_size
            # sparse_dec_win_size: [8, 4]  dense_dec_win_size: [4, 2]
            dec_w, dec_h = self.dec_win_size
            self.norm_dec = [size / self.patch_size for size in self.dec_win_size]
            self.refloc_embed = nn.Embedding(dec_w*dec_h, 2) 
            
    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        
        sparse stride = 16
        dense stride = 8
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:]) 
        shape = (image_shape + stride//2 -1) // stride

        # generate point queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2    N=256
        h, w = shift_x.shape    # 16*16

        # get point queries embedding
        # shape:[bs, c, num_queries]
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]
        query_embed = query_embed.view(bs, c, h, w)

        # get point queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down, shift_x_down]
        query_feats = query_feats.view(bs, c, h, w)
        # print('pqs shape:', query_embed.shape, points_queries.shape, query_feats.shape)
        return query_embed, points_queries, query_feats
    
    def points_queris_embed_inference(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during inference
        """
        
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:])
        shape = (image_shape + stride//2 -1) // stride

        # generate points queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape    # 64, 128 dense
        
        # get points queries embedding 
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]   # 1, 256, num_queris

        # get points queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down, shift_x_down]
        
        or_pq = points_queries
        # window-rize
        query_embed = query_embed.reshape(bs, c, h, w)
        points_queries = points_queries.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0)
        query_feats = query_feats.reshape(bs, c, h, w)
        
        # zlt: opt box-detr query
        dec_win_w, dec_win_h = kwargs['dec_win_size']
        points_queries_scale = query_partition(or_pq, query_feats, dec_win_h, dec_win_w)
        # refbox_wish = query_partition(scaled_refbox, query_feats, dec_win_h, dec_win_w)

        query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
        points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)
        query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)
        
        # dynamic point query generation
        div = kwargs['div']
        div_win = window_partition(div.unsqueeze(1), window_size_h=dec_win_h, window_size_w=dec_win_w)
        valid_div = (div_win > 0.5).sum(dim=0)[:,0]
        v_idx = valid_div > 0
        
        query_embed_win = query_embed_win[:, v_idx]
        query_feats_win = query_feats_win[:, v_idx]
        points_queries_win = points_queries_win.cuda()
        points_queries_win = points_queries_win[:, v_idx].reshape(-1, 2)
        
        points_queries_scale = points_queries_scale.cuda()
        points_queries_scale = points_queries_scale[:, v_idx]
        
        # refbox_wish = refbox_wish.cuda()
        # refbox_wish = refbox_wish[:, v_idx]
    
        return query_embed_win, points_queries_win, query_feats_win, v_idx, points_queries_scale #refbox_scale  refbox_wish  
    
    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        # generate points queries and position embedding
        if 'train' in kwargs:
            query_embed, points_queries, query_feats = self.points_queris_embed(samples, self.pq_stride, src, **kwargs)
            query_embed = query_embed.flatten(2).permute(2,0,1) # BxCxHxW --> (HW)xBxC
            v_idx = None
            refbox_scale = None
        else:
            query_embed, points_queries, query_feats, v_idx, refbox_scale = self.points_queris_embed_inference(samples, self.pq_stride, src, **kwargs)

        out = (query_embed, points_queries, query_feats, v_idx, refbox_scale)
        return out
    
    def predict_hxn(self, samples, points_queries, hs, **kwargs):
        outputs_class = self.class_embed(hs)
        # normalize to -1~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0 

        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        
        bs = samples.tensors.shape[0]
        points_queries_save = points_queries
        if 'test' in kwargs:
            # points_queries = points_queries.unsqueeze(0).repeat(bs, 1, 1)
        
            points_queries = points_queries.float().to(outputs_class.device)
            points_queries[ :, 0] /= (self.pq_stride // 2)
            points_queries[ :, 1] /= (self.pq_stride // 2)

            outputs_points = outputs_offsets[-1] + points_queries
            # normalize coords: 0~1
            outputs_points[ :,0] *= ((self.pq_stride // 2) / img_h) 
            outputs_points[ :,1] *= ((self.pq_stride // 2) / img_w)

            points_queries[ :, 0] *= ((self.pq_stride // 2) / img_h) 
            points_queries[ :, 1] *= ((self.pq_stride // 2) / img_w)
        else:
            points_queries = points_queries.unsqueeze(0).repeat(bs, 1, 1)

            points_queries = points_queries.float().to(outputs_class.device)
            points_queries[:, :, 0] /= (self.pq_stride // 2)
            points_queries[:, :, 1] /= (self.pq_stride // 2)

            outputs_points = outputs_offsets[-1] + points_queries
            # normalize coords: 0~1
            outputs_points[:,:,0] *= ((self.pq_stride // 2) / img_h) 
            outputs_points[:,:,1] *= ((self.pq_stride // 2) / img_w)

            points_queries[:, :, 0] *= ((self.pq_stride // 2) / img_h) 
            points_queries[:, :, 1] *= ((self.pq_stride // 2) / img_w)            
            

        out = {'pred_logits': outputs_class[-1], 
               'pred_points': outputs_points, 
               'img_shape': img_shape, 
               'pred_offsets': outputs_offsets[-1]}
    
        out['points_queries'] = points_queries_save
        out['pq_stride'] = self.pq_stride
        return out
    
    def predict(self, samples, points_queries, hs, **kwargs):
        # PET: hs[2, 8192, 256]
        outputs_class = self.class_embed(hs)
        # outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0
        
        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        points_queries = points_queries.float().cuda()
        pq_or = points_queries
        
        if self.opt_query:
            refer = kwargs['refer']
            refer_bsig = inverse_sigmoid(refer)
            points_queries = refer_bsig[-1] + points_queries
            outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0
            # tmp = self.bbox_embed(hs)
            # tmp += refer_bsig
            # outputs_offsets = tmp.sigmoid()
            points_queries[..., 0] /= img_h
            points_queries[..., 1] /= img_w
            # points_queries = outputs_coord + points_queries
        elif self.opt_query_con:
            refer = kwargs['refer'][-1]
            reference_before_sigmoid = inverse_sigmoid(refer)
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = self.bbox_embed(hs[lvl])
                tmp[..., :2] += reference_before_sigmoid
                outputs_coord = (tmp.sigmoid()- 0.5) * 2.0 # con1 only: tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_offsets = torch.stack(outputs_coords)
            points_queries[:, 0] /= img_h
            points_queries[:, 1] /= img_w
        else:
            outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0
            points_queries[:, 0] /= img_h
            points_queries[:, 1] /= img_w

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[...,0] /= (img_h / self.patch_size)
            outputs_offsets[...,1] /= (img_w / self.patch_size)
            
        outputs_points = outputs_offsets[-1] + points_queries
            
        out = {'pred_logits': outputs_class[-1], 
               'pred_points': outputs_points, 
               'img_shape': img_shape, 
               'pred_offsets': outputs_offsets[-1]
               }
        
        if self.opt_query and 'test' in kwargs:
            out['reference'] = kwargs['refer']
            rb_first_dim = refer_bsig.shape[0]
            pq_ex = pq_or.unsqueeze(0).expand(rb_first_dim, -1, -1)
            pq_ex = pq_ex + refer_bsig
            pq_ex[..., 0] /= img_h
            pq_ex[..., 1] /= img_w  
            out['pq_ex'] = pq_ex
            
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

    def forward(self, samples, features, context_info, **kwargs):
        
        encode_src, src_pos_embed, mask = context_info
        # box-detr weight:
        # refbox = self.refpoint_embed.weight if self.opt_query else None
        refloc = self.refloc_embed.weight if self.opt_query or self.opt_query_con else None
        if self.opt_query or self.opt_query_con:
            # kwargs['refbox'] = refbox
            kwargs['refloc'] = refloc
            kwargs['dec_win_size_n'] = self.norm_dec
        # get points queries for transformer
        pqs = self.get_point_query(samples, features, **kwargs) 
        # (pos, que, feat, v_dix=None, refbox_scale=None if train)
        # if 'test' in kwargs:
        #     refbox = pqs[4]
        #     kwargs['refbox'] = refbox
        
        # point querying
        # kwargs['pq_stride'] = self.pq_stride
        kwargs['quadtree_layer'] = self.quadtree_layer
        
        if self.opt_query or self.opt_query_con:
            hs, refer = self.transformer(encode_src, src_pos_embed, mask, pqs, 
                              img_shape=samples.tensors.shape[-2:], 
                              **kwargs # refbox is in the kwargs['refbox']
                              )
            kwargs['refer'] = refer
        else:
            hs = self.transformer(encode_src, src_pos_embed, mask, pqs, 
                              img_shape=samples.tensors.shape[-2:], 
                              **kwargs
                              )
        # hs.shape: [2inter, 7bs, 512patch_w, 256patch_h]
        
        # prediction
        points_queries = pqs[1]
        if kwargs['predict'] == 'hxn':
            outputs = self.predict_hxn(samples, points_queries, hs, **kwargs)       # constrain output
        elif kwargs['predict'] == 'origin':
            outputs = self.predict(samples, points_queries, hs, **kwargs)
            
        return outputs
    

class PET(nn.Module):
    """ 
    Point quEry Transformer
    """
    def __init__(self, backbone, num_classes, args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.backbone_type = args.backbone
        self.encoder_free = args.encoder_free
        
        self.predict = args.predict
        
        # zlt: object query in decoder from box-detr
        self.opt_query_decoder = args.opt_query_decoder
        self.loss_f = args.loss_f
        self.total_epochs = args.epochs
        
        # positional embedding
        self.pos_embed = build_position_encoding(args)

        # feature projection
        # model list: trans the output of the backbone to the hidden_dim of the transformer
        hidden_dim = args.hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            ]
        )   

        # context encoder
        # default encoder feature scaling = 8x
        self.encode_feats = '8x'
        if not self.encoder_free:
            print('building PET with encoder')
            enc_win_list = args.enc_win_list
            args.enc_layers = len(enc_win_list)
            self.context_encoder = build_encoder(args, enc_win_list=enc_win_list)
        else:
            print('building PET without encoder')

        # quadtree splitter
        context_patch = args.context_patch
        print(f"context_patch: {context_patch}")    # 64*32
            # context_w = context_path // 8 '8x'
        context_w, context_h = context_patch[0]//int(self.encode_feats[:-1]), context_patch[1]//int(self.encode_feats[:-1])
        
        self.attn_splitter = args.attn_splitter
        if args.attn_splitter:
            print('attn splitter')
            if hasattr(args, 'Orattn'):
                self.quadtree_splitter = FirstQuadtreeSplitterWithAttention(hidden_dim, context_h, context_w)
            else:
                self.quadtree_splitter = QuadtreeSplitterWithAttention(hidden_dim, context_h, context_w)
        else:
            self.quadtree_splitter = nn.Sequential(
                nn.AvgPool2d((context_h, context_w), stride=(context_h ,context_w)),
                nn.Conv2d(hidden_dim, 1, 1),
                nn.Sigmoid(),
            )

        # point-query quadtree
        if self.opt_query_decoder:
            print('building decoder using box-agent as opt object query from box-detr for Faster Convergence')
        else:
            print('using detr object query for decoder')
        transformer = build_decoder(args)
        self.sparse_dec_win_size = args.sparse_dec_win_size
        self.dense_dec_win_size = args.dense_dec_win_size
        self.quadtree_sparse = BasePETCount(backbone, num_classes, quadtree_layer='sparse', args=args, transformer=transformer, patch_size=args.patch_size)
        self.quadtree_dense = BasePETCount(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer, patch_size=args.patch_size)
        
        self.prob_map_lc = args.prob_map_lc
        if args.prob_map_lc == 'f4x':
            self.prob_conv = nn.Sequential(
                nn.Conv2d(in_channels=backbone.num_channels, out_channels=1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

    def compute_loss(self, outputs, criterion, targets, epoch, samples, prob=None, prob_est=None):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = 5
        max_sp = 3000
        cycles = 3 # default = 5
        
        # ln_epsilon = math.log(1e-5)
        # growth_rate = warmup_ep - ln_epsilon / growth_rate
        loss_f = self.loss_f
        
        # compute loss
        if max_sp > epoch >= warmup_ep:
            loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'], loss_f=loss_f, prob=prob, prob_est=prob_est, epoch=epoch)
            loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'], loss_f=loss_f, prob=prob, prob_est=prob_est, epoch=epoch)
        else:
            loss_dict_sparse = criterion(output_sparse, targets, loss_f=loss_f, prob=prob, prob_est=prob_est, epoch=epoch)
            loss_dict_dense = criterion(output_dense, targets, loss_f=loss_f, prob=prob, prob_est=prob_est, epoch=epoch)

        # sparse point queries loss
        loss_dict_sparse = {k+'_sp':v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k+'_sp':v for k,v in weight_dict.items()}
        loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # dense point queries loss
        loss_dict_dense = {k+'_ds':v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k+'_ds':v for k,v in weight_dict.items()}
        loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)
    
        # point queries loss
        losses = loss_pq_sparse + loss_pq_dense 

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_sparse)
        loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)
        weight_dict.update(weight_dict_dense)

        # quadtree splitter loss
        den = torch.tensor([target['density'] for target in targets])   # crowd density
        bs = len(den)
        ds_idx = den < 2 * self.quadtree_sparse.pq_stride   # dense regions index
        ds_div = outputs['split_map_raw'][ds_idx]
        sp_div = 1 - outputs['split_map_raw']

        # constrain sparse regions
        loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()

        # constrain dense regions
        if sum(ds_idx) > 0:
            ds_num = ds_div.shape[0]
            loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        else:
            loss_split_ds = outputs['split_map_raw'].sum() * 0.0

        # update quadtree splitter loss            
        loss_split = loss_split_sp + loss_split_ds
        
        if epoch <= warmup_ep or epoch > max_sp:
            weight_split = 0.0
        elif self.loss_f == 'update':
            max_split_weight = 0.3
            
            # cos update
            epoch_tensor = torch.tensor(epoch, dtype=torch.float32)
            warmup_tensor = torch.tensor(warmup_ep, dtype=torch.float32)
            total_tensor = torch.tensor(self.total_epochs, dtype=torch.float32)
            cycle_length = (total_tensor - warmup_tensor) / cycles
            progress_within_cycle = ((epoch_tensor - warmup_tensor) % cycle_length) / cycle_length
            weight_split = max_split_weight * (0.5 * (1 + torch.cos(2 * torch.pi * progress_within_cycle)))
        else:
            weight_split = 0.1

        loss_dict['loss_split'] = loss_split
        weight_dict['loss_split'] = weight_split

        # final loss
        losses += loss_split * weight_split
        return {'loss_dict':loss_dict, 'weight_dict':weight_dict, 'losses':losses}

    def forward(self, samples: NestedTensor, **kwargs):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # backbone
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples) 
        #  sample.tensors shape[bs, 3, patch_size]                       VGG 64/32
        #  features.tensors/pos shape: ['4x':BCHW, '8x':BCHW] [4x/8x: bs, c, 32/16, 32/16]
        
        # prob_map: tuple[0: b,1,32,32; 1:b,1,256,256] 0 from fpn_4x, 1 interpolated
        # kwargs['prob_map']=prob_map
        
        # prob map in feature
        if 'train' in kwargs and self.prob_map_lc == 'f4x':
            prob_map = self.prob_conv(features['4x'].tensors)
            H, W = samples.tensors.shape[-2], samples.tensors.shape[-1]
            prob_map_up = F.interpolate(prob_map, size=(H, W), mode='bilinear', align_corners=False)
            kwargs['prob_map_up'] = prob_map_up

        # positional embedding
        dense_input_embed = self.pos_embed(samples)  # B*hidden_dim*imgH*imgW
        kwargs['dense_input_embed'] = dense_input_embed

        # feature projection
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)  # channels to hidden_dim
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)
        
        # prediction head
        kwargs['predict'] = self.predict

        # forward
        if 'train' in kwargs:
            out = self.train_forward(samples, features, pos, **kwargs)
        else:
            out = self.test_forward(samples, features, pos, **kwargs)  
        return out
    
    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        if self.prob_map_lc == 'f4x':
            prob, prob_est = kwargs['probability'], kwargs['prob_map_up']
        else:
            prob, prob_est = None, None
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples, prob, prob_est)
        return losses
    
    def test_forward(self, samples, features, pos, **kwargs):
        # print(samples.tensors.shape)
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        
        if 'eval_s' in kwargs: # rsc test for precision
            criterion, targets = kwargs['criterion'], kwargs['targets'] 
        
        out_dense, out_sparse = outputs['dense'], outputs['sparse']
        thrs = 0.5  # inference threshold        

        # process sparse point queries
        if outputs['sparse'] is not None:
            out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
            valid_sparse = out_sparse_scores > thrs
            index_sparse = valid_sparse.cpu()
        else:
            index_sparse = None

        # process dense point queries
        if outputs['dense'] is not None:
            out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
            valid_dense = out_dense_scores > thrs
            index_dense = valid_dense.cpu()
        else:
            index_dense = None

        # format output
        div_out = dict()
        output_names = out_sparse.keys() if out_sparse is not None else out_dense.keys()
        for name in list(output_names): 
            if 'pred' in name or name == 'points_queries': #or name == 'pq_ex':
                if index_dense is None:
                    div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                elif index_sparse is None:
                    div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                else:
                    div_out[name] = torch.cat([out_sparse[name][index_sparse].unsqueeze(0), out_dense[name][index_dense].unsqueeze(0)], dim=1)
            elif name == 'pq_ex':
                if index_dense is None:
                    div_out[name] = out_sparse[name][:, index_sparse]
                elif index_sparse is None:
                    div_out[name] = out_dense[name][:, index_dense]
                else:
                    div_out[name] = torch.cat([
                        out_sparse[name][:, index_sparse],
                        out_dense[name][:, index_dense]
                    ], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]
        div_out['split_map_raw'] = outputs['split_map_raw']
        
        if 'eval_s' in kwargs:
            div_out['ind']=criterion.forward_ind(div_out, targets)          # RSC output_index
            
        return div_out

    def pet_forward(self, samples, features, pos, **kwargs):
        # context encoding
            # encode_feats = 8x
            # src = tensors of F, 
            # src_pos = backbone pos
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None
        # print(src.shape, src_pos_embed.shape, mask.shape, encode_src.shape)  # 8*256*32*32 sparse
        if self.encoder_free:
            context_info = (src, src_pos_embed, mask)
        else:
            encode_src = self.context_encoder(src, src_pos_embed, mask)  # features['8x'] encoding --> B*hidden_dim*H*W(32)
            context_info = (encode_src, src_pos_embed, mask)
        
        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape # _ = hidden_dim
        sp_h, sp_w = src_h, src_w                         # sparse 16, 16                        
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)       # dense = sparse * 2, which 32, 32
        
        train_flag = True if 'train' in kwargs else False
        
        src = encode_src if not self.encoder_free else src
        if self.attn_splitter:
            split_map = self.quadtree_splitter(src, src_pos_embed, mask, train_flag)  # split_map: B*1*src_h/4*src_w/8   bs*1*4*2
        else:
            split_map = self.quadtree_splitter(src)  # split_map: B*1*src_h/4*src_w/8
            
        # split_map = 1 - split_map
        
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)  # nearest # dense split       1 means waiting for further revision
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)       # sparse split      we want 0 in sparse area
        
        # quadtree decoder
        # quadtree layer0 forward (sparse)
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)  # bs*256 -> bs*h*w
            kwargs['dec_win_size'] = self.sparse_dec_win_size
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
        else:
            outputs_sparse = None
        
        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size'] = self.dense_dec_win_size
            outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)
        else:
            outputs_dense = None
        
        # format outputs
        outputs = dict()
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs

def build_density_map_from_points_with_kdtree(target_points, batch_indices, img_h, img_w, device, alpha=0.4, prob_fixed=False):
    B = int(batch_indices.max().item()) + 1  # Calculate the batch size
    density = torch.zeros((B, 1, img_h, img_w), dtype=torch.float32, device=device)  # Initialize the density map

    for b in range(B):
        mask = batch_indices == b
        points_b = target_points[mask]  # [nb, 2], get points for batch b
        if points_b.numel() == 0:
            continue  # Skip empty batches

        # Unnormalize to pixel coordinates
        pts_np = (points_b * torch.tensor([img_w, img_h], device=device)).cpu().numpy()
        pts_np = np.round(pts_np).astype(np.int32)
        pts_np[:, 0] = np.clip(pts_np[:, 0], 0, img_w - 1)
        pts_np[:, 1] = np.clip(pts_np[:, 1], 0, img_h - 1)

        num_pts = len(pts_np)
        if num_pts == 0:
            continue

        # Build KDTree (k = min(4, n))
        k = min(4, num_pts)
        tree = scipy.spatial.KDTree(pts_np.copy(), leafsize=2048)
        distances, locations = tree.query(pts_np, k=k)

        for i, (x, y) in enumerate(pts_np):
            pt_map = torch.zeros((1, 1, img_h, img_w), device=device)
            pt_map[0, 0, y, x] = 1.0

            # Estimate sigma
            if prob_fixed == 'dynamic':
                if num_pts > 1 and distances[i].shape[0] >= 2 and np.isfinite(distances[i][1]):
                    di = distances[i][1]
                    neighbor_idx = locations[i][1:]
                    neighbor_dists = []

                    for j in neighbor_idx:
                        if j < len(distances) and distances[j].shape[0] >= 2 and np.isfinite(distances[j][1]):
                            neighbor_dists.append(distances[j][1])

                    if len(neighbor_dists) > 0:
                        d_mtop3 = np.mean(neighbor_dists)
                        d = min(di, d_mtop3)
                    else:
                        d = di
                    sigma = max(alpha * d, 1.0)
                else:
                    sigma = np.average([img_h, img_w]) / 4.0  # fallback sigma
            elif prob_fixed == 'fixed':
                sigma = 15

            # Generate Gaussian kernel
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1

            kernel = generate_gaussian_kernel(kernel_size, sigma, device).unsqueeze(0).unsqueeze(0)
            response = F.conv2d(pt_map, kernel, padding=kernel_size // 2)
            peak = response[0, 0, y, x]

            if peak > 0:
                response = response / peak
                density[b, 0] = torch.maximum(density[b, 0], response[0, 0])

    return density  # shape [B, 1, H, W]

def generate_prob_map_from_points(targets, img_h, img_w, device='cuda', alpha=0.4):
    """
    (优化版) 
    使用来自 target 字典列表的点生成概率图（密度图）。
    
    此版本通过用基于坐标网格的元素级张量操作替换 N 次 2D 卷积，
    极大地提高了性能，并修复了 N=1 个点时的 KNN bug。

    Args:
        targets (list of dict): 字典列表，每个包含 'points'。
        img_h (int): 图像高度。
        img_w (int): 图像宽度。
        device (str): 计算设备 ('cuda' or 'cpu')。
        alpha (float): sigma 的缩放因子。

    Returns:
        torch.Tensor: 生成的概率图，形状为 (img_h, img_w)。
    """
    
    # --- 1. 收集点 (已优化) ---
    if not targets:
        return torch.zeros((img_h, img_w), dtype=torch.float32, device=device)
        
    all_points_list = [t['points'] for t in targets if t['points'].shape[0] > 0]
    
    if not all_points_list:
        # 如果所有 target 都为空
        return torch.zeros((img_h, img_w), dtype=torch.float32, device=device)

    all_points_tensor = torch.cat(all_points_list, dim=0)
    all_points = all_points_tensor.cpu().numpy()
    num_points = len(all_points)

    if num_points == 0:
        return torch.zeros((img_h, img_w), dtype=torch.float32, device=device)

    # --- 2. KNN 计算 (已修复 Bug) ---
    # 仅当点多于1个时才进行 KNN 查询，以避免 k=1
    distances, locations = None, None
    if num_points > 1:
        tree = KDTree(all_points)
        # k_query 将始终 >= 2, 保证 'distances' 是 2D 数组
        k_query = min(4, num_points) 
        distances, locations = tree.query(all_points, k=k_query)

    # --- 3. 创建坐标网格 (核心优化) ---
    # 创建一次网格，用于所有点的计算
    # 'ij' 索引确保 yy 对应 H (维度0)，xx 对应 W (维度1)
    yy, xx = torch.meshgrid(
        torch.arange(img_h, dtype=torch.float32, device=device),
        torch.arange(img_w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 最终的密度图，形状 (H, W)
    density = torch.zeros((img_h, img_w), dtype=torch.float32, device=device)

    # --- 4. 主循环 (内部已优化) ---
    for i, pt in enumerate(all_points):
        # pt[0] 是 x (对应 W), pt[1] 是 y (对应 H)
        # 我们直接使用 float 坐标，更精确
        x0, y0 = pt[0], pt[1]

        # --- 5. 计算 Sigma (使用修复后的逻辑) ---
        sigma = -1.0
        if distances is not None:
            # distances[i][0] 是到自身的距离(0.0)
            # distances[i][1] 是到第1近邻的距离
            if np.isfinite(distances[i][1]):
                di = distances[i][1]
                neighbor_idx = locations[i][1:] # k-1 个邻居的索引

                # (微优化) 向量化内部循环
                # 获取邻居的 "第1近邻距离"
                valid_neighbor_mask = np.isfinite(distances[neighbor_idx, 1])
                neighbor_distances = distances[neighbor_idx[valid_neighbor_mask], 1]
                
                if len(neighbor_distances) > 0:
                    d_mtop3 = np.mean(neighbor_distances)
                    d = min(di, d_mtop3)
                else:
                    d = di # 回退
                
                sigma = alpha * d

        # 回退: 适用于 N=1 或邻居距离无效的情况
        if sigma <= 0: 
            sigma = np.average([img_h, img_w]) / 4.0

        sigma = max(sigma, 1.0) # 确保最小 sigma

        # --- 6. 生成高斯斑点 (核心优化) ---
        # 替代 F.conv2d + 归一化
        # (xx - x0)**2 + (yy - y0)**2 计算所有像素到 (x0, y0) 的平方欧氏距离
        dist_sq = (xx - x0)**2 + (yy - y0)**2
        
        # exp(-0) = 1，所以高斯峰值自动为 1，
        # 这等效于你原来的 "filter / peak" 归一化操作
        g_blob = torch.exp(-dist_sq / (2.0 * sigma**2))
        
        # --- 7. 更新密度图 ---
        density = torch.maximum(density, g_blob)

    # 返回 (H, W) 张量，与你原始函数 squeeze() 后的结果一致
    return density

def generate_gaussian_kernel_prob(kernel_size, sigma, device='cuda'):
    """
    Generate a Gaussian kernel for convolution.
    
    Args:
        kernel_size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian.
        device (str): The device to run the computation on ('cuda' or 'cpu').
        
    Returns:
        torch.Tensor: The generated Gaussian kernel.
    """
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    y = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def generate_gaussian_kernel(size, sigma, device):
    ax = torch.arange(size, device=device) - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel

class SetCriterion(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, sparse_stride, dense_stride, map_loss, opt_query, prob_fixed):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef    # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        # self.div_thrs_dict = {8: 0.0, 4:0.5}
        self.div_thrs_dict = {sparse_stride: 0.0, dense_stride:0.5}
        self.map_loss = map_loss
        self.opt_query = opt_query
        self.probloss_cal = 'Linear'
        self.prob_fixed = prob_fixed
    
    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # compute classification loss
        if 'div' in kwargs:
            # get sparse / dense image index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            ds_idx = den_sort[:len(den_sort)//2]
            sp_idx = den_sort[len(den_sort)//2:]
            eps = 1e-5

            # raw cross-entropy loss
            weights = target_classes.clone().float()
            weights[weights==0] = self.empty_weight[0]
            weights[weights==1] = self.empty_weight[1]
            raw_ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, ignore_index=-1, reduction='none')

            # binarize split map
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs

            # dual supervision for sparse/dense images
            loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / ((weights * div_mask)[sp_idx].sum() + eps)
            loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / ((weights * div_mask)[ds_idx].sum() + eps)
            loss_ce = loss_ce_sp + loss_ce_ds

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_ce_nondiv = (raw_ce_loss * weights * non_div_mask).sum() / ((weights * non_div_mask).sum() + eps)
            loss_ce = loss_ce + loss_ce_nondiv
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-1)

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs
        # get indices
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        
        batch_indices = []  # which image in batch this point came from
        for i, (src_idx, _) in enumerate(indices):
            batch_indices.extend([i] * len(src_idx))
        batch_indices = torch.tensor(batch_indices, device=src_points.device, dtype=torch.long)
        epoch = kwargs['epoch']
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        
        if kwargs['loss_f'] == 'gaussion_l2':
            # loss_points_raw = weighted_smooth_l1_loss(src_points, target_points, sigma=16.0)
            pass
        elif kwargs['loss_f'] == 'prob':
            B = len(targets) 
            if target_points.numel() == 0:
                gt_prob_map = torch.zeros((B, 1, img_h, img_w), 
                                          dtype=torch.float32, 
                                          device=src_points.device)
            else:
                gt_prob_map = build_density_map_from_points_with_kdtree(
                    target_points, batch_indices, img_h, img_w, device=src_points.device, prob_fixed=self.prob_fixed
                )
            x = (src_points[:, 0] * img_w).clamp(0, img_w - 1)
            y = (src_points[:, 1] * img_h).clamp(0, img_h - 1)
            x_norm = x / (img_w - 1) * 2 - 1
            y_norm = y / (img_h - 1) * 2 - 1
            grid = torch.stack([x_norm, y_norm], dim=1).unsqueeze(1).unsqueeze(1)  # [N,1,1,2]
            prob_vals = F.grid_sample(
                gt_prob_map[batch_indices],  # [N,1,H,W]
                grid, mode='bilinear', align_corners=True
            ).squeeze(-1).squeeze(-1).squeeze(1)  # [N]
            
            eps = 1e-6
            if self.probloss_cal == 'Linear': # Linear, y = 1 - p
                loss_points_raw = 1.0 - prob_vals
            elif self.probloss_cal == 'Psq': # P_squard, y = 1 - p^2
                loss_points_raw = 1.0 - prob_vals.pow(2)
            elif self.probloss_cal == 'NLL': # Negative Log Likelihood, y = - log(p)
                loss_points_raw = -torch.log(prob_vals.clamp_min(eps))
            elif self.probloss_cal == 'Squard': # Squard, y = (1 - p)^2
                loss_points_raw = ((1.0 - prob_vals).pow(2))
            elif self.probloss_cal == 'Focal': # Focal, y = (1 - p)^γ * log(p)
                gamma = 2.0
                loss_points_raw = - (1.0 - prob_vals).pow(gamma) * torch.log(prob_vals.clamp_min(eps))
            
            # scale factor & shape adjustment
            loss_points_raw = loss_points_raw.unsqueeze(1).expand(-1, 2) * 0.05
        else:
            loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        if 'div' in kwargs:
            # get sparse / dense index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            img_ds_idx = den_sort[:len(den_sort)//2]
            img_sp_idx = den_sort[len(den_sort)//2:]
            pt_ds_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_ds_idx])
            pt_sp_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_sp_idx])

            # dual supervision for sparse/dense images
            eps = 1e-5
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs
            loss_points_div = loss_points_raw * div_mask[idx].unsqueeze(-1)
            loss_points_div_sp = loss_points_div[pt_sp_idx].sum() / (len(pt_sp_idx) + eps)
            loss_points_div_ds = loss_points_div[pt_ds_idx].sum() / (len(pt_ds_idx) + eps)

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (non_div_mask[idx].sum() + eps)   

            # final point loss
            losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
        else:
            losses['loss_points'] = loss_points_raw.sum() / num_points
        
        return losses
    
    def loss_maps(self, outputs, targets, indices, num_points, **kwargs):
        criterion = nn.MSELoss(reduction='mean').cuda()
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        gt_prob_map = generate_prob_map_from_points(targets, img_h, img_w)
        prob_est = kwargs['prob_est']
        
        prob_loss = criterion(gt_prob_map, prob_est)
        losses = {}
        losses['loss_maps'] = prob_loss
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
            'maps': self.loss_maps,
        } if self.map_loss == 'f4x' else {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                      
            kwargs: prob, prob_est
        """
        # retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses
    
    def forward_ind(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        return  indices


class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_reduce=False, use_relu=True):
        super().__init__()
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim//2**i for i in range(num_layers - 1)]
        else:
            h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.use_relu = use_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_relu:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x


def build_pet(args):
    device = torch.device(args.device)

    # build model
    num_classes = 1
    backbone = build_bockbone(args)
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )

    # build loss criterion
    if not hasattr(args, 'matcher'):
        matcher = build_matcher(args)
    else:
        matcher = build_stable_matcher(args) if args.matcher == 'stable' else build_matcher(args)
    if args.prob_map_lc == 'f4x':
        weight_dict = {'loss_ce': args.ce_loss_coef, 
                    'loss_points': args.point_loss_coef,
                    'loss_maps': 1.0}
        losses = ['labels', 'points', 'maps']
    else:
        weight_dict = {'loss_ce': args.ce_loss_coef, 
                    'loss_points': args.point_loss_coef}
        losses = ['labels', 'points']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses,
                             sparse_stride=args.sparse_stride, dense_stride=args.dense_stride,
                             map_loss=args.prob_map_lc,
                             opt_query=args.opt_query_decoder,
                             prob_fixed=args.prob_bandwidth if hasattr(args, 'prob_bandwidth') else 'fixed')
    criterion.to(device)
    return model, criterion
