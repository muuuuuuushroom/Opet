"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

if __name__ == '__main__':
    from utils import *
    # import importlib
    # utils = importlib.import_module('utils')
    # print(utils.__file__)
else:
    from .utils import *

from einops import rearrange
from fla.ops.linear_attn import (chunk_linear_attn, fused_chunk_linear_attn,
                                 fused_recurrent_linear_attn)
from .attention import MultiheadAttention

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(f"Layer {i}: output shape {x.shape}")
            if i < self.num_layers - 1:
                x = F.relu(x)
            
            # x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# utils:
def gen_sineembed_for_position(pos_tensor, d_model=256):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:   # embed for 2d: x, y
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4: # embed for 4d: x, y, w, h
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu", 
                 attn_type='softmax',
                 **kwargs):
        super().__init__()
        self.attn_type = attn_type
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, 
                                     dropout, activation, attn_type=self.attn_type)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, **kwargs)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.enc_win_list = kwargs['enc_win_list']
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False           

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def sanitize_mask(self, attn_mask: torch.Tensor) -> torch.Tensor:
        if (attn_mask.ndim == 2):            # (B_win, L)
            full = attn_mask.all(dim=-1)    # (B_win, )
            attn_mask[full, -1] = False
        elif (attn_mask.ndim == 3):          # (B_win, L, L)
            full = attn_mask.all(dim=-1).all(dim=-1)
            attn_mask[full, :, -1] = False
        return attn_mask
    
    def forward(self, src, pos_embed, mask):
        bs, c, h, w = src.shape
        
        memeory_list = []
        memeory = src
        for idx, enc_win_size in enumerate(self.enc_win_list):
            # encoder window partition
            enc_win_w, enc_win_h = enc_win_size
            memeory_win, pos_embed_win, mask_win  = enc_win_partition(memeory, pos_embed, mask, enc_win_h, enc_win_w)  # (HW)(BN)C          

            # encoder forward
            mask_win = self.sanitize_mask(mask_win)
            output = self.encoder.single_forward(memeory_win, src_key_padding_mask=mask_win, pos=pos_embed_win, layer_idx=idx)

            # reverse encoder window
            memeory = enc_win_partition_reverse(output, enc_win_h, enc_win_w, h, w)
            if self.return_intermediate:
                memeory_list.append(memeory)        
        memory_ = memeory_list if self.return_intermediate else memeory
        return memory_


class WinDecoderTransformer(nn.Module):
    """
    Transformer Decoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=2, 
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 return_intermediate_dec=False,
                 dec_win_w=16, dec_win_h=8,               
                 attn_type='softmax',
                 opt_query_decoder=False,
                 # anchor-detr patterns
                 num_patterns=0,
                 opt_query_con=False, 
                 box_setting=1,
                 ):
        super().__init__()
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, attn_type, 
                                                opt_query_decoder, opt_query_con)

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec,
                                            d_model=d_model, nhead=nhead, 
                                            opt_query_decoder=opt_query_decoder,
                                            opt_query_con=opt_query_con,
                                            box_setting=box_setting)
        self._reset_parameters()

        self.dec_win_w, self.dec_win_h = dec_win_w, dec_win_h
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_decoder_layers
        
        self.opt_query_decoder = opt_query_decoder
        self.opt_query_con = opt_query_con
             
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def decoder_forward(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        # process of decoder forward
        bs, c, h, w = src_shape
        qH, qW = query_feats.shape[-2:]
        
        # if self.opt_query_decoder:
        # refbox = kwargs['refbox']
        # refbox = refbox.unsqueeze(1).repeat(1, bs, 1)
        
    
        # point queries win
        _, points_queries, _, _, _ = kwargs['pqs']
        
        # useless?:
        # points_queries_win = query_partition(points_queries, query_feats, dec_win_h, dec_win_w)
        # kwargs['pq_w'] = points_queries_win
        
        # if self.opt_query_decoder:
        #     refbox_win = query_partition(kwargs['refbox'], query_feats, dec_win_h, dec_win_w)
        
        query_embed_ = query_embed.permute(1,2,0).reshape(bs, c, qH, qW)
        query_embed_win = window_partition(query_embed_, window_size_h=dec_win_h, window_size_w=dec_win_w)
        
        # window-rize query input
        # shape: [hw, bs, C]  ->  [bs, C, h, w]  ->  [dech*decw, bs * num_wins, C]
        # num_wins = 8 (2*4) if sp else 32 (4*8)  (dech * decw)
        tgt = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w) 
        
        # box_detr
        if self.opt_query_decoder or self.opt_query_con:
            refloc = kwargs['refloc']
            num_wins = tgt.shape[1]
            refloc_win = refloc.unsqueeze(1).repeat(1, num_wins, 1)
            # we try:
            # refloc_win = points_queries_win.to(refloc.device)
            
            hs_win, references_win = self.decoder(tgt, 
                                memory_win, 
                                memory_key_padding_mask=mask_win, 
                                pos=pos_embed_win, 
                                query_pos=query_embed_win, 
                                box_unsigmoid=refloc_win,   # box-detr
                                **kwargs)
            # reference has not been used
            # shape = torch.Size([2, 56, 32, 256])
            # input para = (hs_w, 4, 8, 16, 16)
            
            hs_tmp = [window_partition_reverse(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
            refer_tmp = [window_partition_reverse(refer_w, dec_win_h, dec_win_w, qH, qW) for refer_w in references_win]
            
            # hs_tmp[0].shape = torch.Size([256, 4, 448])  ->   delete transpose: torch.Size([256, 7, 256])
            hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
            refer = torch.vstack([refer_t.unsqueeze(0) for refer_t in refer_tmp])
            return hs, refer
        
        # decoder attention
        hs_win = self.decoder(tgt,
                            memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win, 
                            query_pos=query_embed_win, 
                            box_unsigmoid=None,   # box-detr
                            **kwargs)
        # shape = torch.Size([2, 32, 56, 256])
        # input para = (hs_w, 2, 4, 32, 32)
        hs_tmp = [window_partition_reverse(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
        # len = 2, shape = torch.Size([256, 7, 256])
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        return hs
    
    def decoder_forward_dynamic(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        """ 
        decoder forward during inference
        """       
        # decoder attention
        tgt = query_feats
        
        _, points_queries, _, _, refbox_scale = kwargs['pqs'] # test: shape 8*1024*2
        # points_queries_scale = query_partition(points_queries, query_feats, dec_win_h, dec_win_w, test=True)   # 8192, 2 -> 8, 1024, 2
        
        # if self.opt_query_decoder:
            # we try:
            # refloc_win = points_queries.reshape(-1, num_wins, 2).to(refloc.device)

            
        if self.opt_query_decoder or self.opt_query_con:
            refloc = kwargs['refloc']
            num_wins = tgt.shape[1]
            refloc_win = refloc.unsqueeze(1).repeat(1, num_wins, 1)
            
            hs_win, references_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
                            box_unsigmoid = refloc_win, #refbox,
                            query_pos=query_embed, **kwargs)
            num_layer, num_elm, num_win, dim = hs_win.shape
            hs = hs_win.reshape(num_layer, num_elm * num_win, dim)
            refer = references_win.reshape(num_layer, num_elm * num_win, 2)
            return hs, refer
            
        else:
            hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
                            box_unsigmoid = None,
                            query_pos=query_embed, **kwargs)
            
            num_layer, num_elm, num_win, dim = hs_win.shape
            hs = hs_win.reshape(num_layer, num_elm * num_win, dim)
        
    # PET: hs.shape = [2, 8192, 256]    hs_win.shape = ([2, 8, 1024, 256])
        return hs
     
    def forward(self, src, pos_embed, mask, pqs, **kwargs):
        # pos_embed =  src_pos = backbone_pos
        bs, c, h, w = src.shape
        
        kwargs['pqs'] = pqs
        # box-detr needed?
        # points_queries = pqs[1]
        query_embed, points_queries, query_feats, v_idx, points_queries_scale = pqs
        self.dec_win_w, self.dec_win_h = kwargs['dec_win_size']     # sp 8,4 de 4,2
        
        # window-rize memory input
        div_ratio = 1 if kwargs['quadtree_layer'] == 'sparse' else 2 
        # memory_win -> key
        memory_win, pos_embed_win, mask_win = enc_win_partition(src, pos_embed, mask,
                                                    int(self.dec_win_h/div_ratio), int(self.dec_win_w/div_ratio)
                                                    )
        
        
        if 'test' in kwargs:    # test (dynamic decoder forward)
            memory_win = memory_win[:,v_idx]
            pos_embed_win = pos_embed_win[:,v_idx]
            mask_win = mask_win[v_idx]
            if self.opt_query_decoder or self.opt_query_con:
                hs, refer = self.decoder_forward_dynamic(query_feats, query_embed, 
                                    memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
                return hs, refer
            else:
                hs = self.decoder_forward_dynamic(query_feats, query_embed,     # bs, c, h, w (h*w = num_queries)
                                        memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
                return hs
        else:
            # train decoder forward 
            if self.opt_query_decoder or self.opt_query_con:
                hs, refer = self.decoder_forward(query_feats, query_embed,
                                    memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
                return hs.transpose(1, 2), refer.transpose(1, 2)
            # hs.shape: torch.Size([2, 256, bs, 256])     
            else:
                hs = self.decoder_forward(query_feats, query_embed,
                                        memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
                return hs.transpose(1, 2)
            
class TransformerEncoder(nn.Module):
    """
    Base Transformer Encoder
    """
    def __init__(self, encoder_layer, num_layers, **kwargs):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

        if 'return_intermediate' in kwargs:
            self.return_intermediate = kwargs['return_intermediate']
        else:
            self.return_intermediate = False
    
    def single_forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                layer_idx=0):
        '''specific layer forward'''
        output = src
        layer = self.layers[layer_idx]
        output = layer(output, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask, pos=pos)        
        return output

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        intermediate = []
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return intermediate

        return output

# class usless_forms:
#     class GaussianPointOffset(nn.Module):
#         def __init__(self, d_model, nhead, sigma=0.05):
#             super(GaussianPointOffset, self).__init__()
#             self.d_model = d_model
#             self.nhead = nhead
#             self.sigma = sigma
#             self.offset_mlp = MLP(d_model, d_model, 2 * nhead, 2)

#         def forward(self, output, box):
#             N, B, _ = box.shape     # num queries, bs, 2
#             base_offset = self.offset_mlp(output).reshape(N, B, self.nhead, 2)
#             gaussian_noise = torch.randn(N, B, self.nhead, 2, device=box.device) * self.sigma
#             point_offset = base_offset + gaussian_noise
#             agent = box.unsqueeze(-2) + point_offset
#             return agent
        
#     class DynamicOffset(nn.Module):
#         def __init__(self, nhead, init_scale=0.1):
#             super().__init__()
#             self.nhead = nhead
#             self.scale = nn.Parameter(torch.full((nhead, 1), init_scale))

#         def forward(self, box, output):
#             N, B, _ = box.shape
#             point_offset = torch.randn(N, B, self.nhead, 2, device=box.device)
#             point_offset = point_offset * self.scale.view(1, 1, self.nhead, 1)
#             agent = box.unsqueeze(-2) + point_offset
#             return agent
        
#     class LearnableGaussian(nn.Module):
#         def __init__(self, nhead, init_sigma=0.1):
#             super().__init__()
#             self.nhead = nhead
#             self.sigma = nn.Parameter(torch.full((nhead, 1), init_sigma))  # 每个 head 独立学习

#         def forward(self, box, output):
#             N, B, _ = box.shape
#             point_offset = torch.randn(N, B, self.nhead, 2, device=box.device) * self.sigma.view(1, 1, self.nhead, 1)
#             agent = box.unsqueeze(-2) + point_offset
#             return agent

#     class FeatureDependentOffset(nn.Module):
#         def __init__(self, d_model, nhead):
#             super().__init__()
#             self.nhead = nhead
#             self.offset_generator = MLP(d_model, d_model, 2 * nhead, 2)

#         def forward(self, box, output):
#             N, B, _ = box.shape
#             offset_range = self.offset_generator(output).view(N, B, self.nhead, 2)
#             point_offset = torch.randn(N, B, self.nhead, 2, device=box.device) * offset_range
#             agent = box.unsqueeze(-2) + point_offset
#             return agent

class TransformerDecoder(nn.Module):
    """
    Base Transformer Decoder
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, nhead=8, opt_query_decoder=False, opt_query_con=False,
                 box_setting=1):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.opt_query_decoder = opt_query_decoder
        self.opt_query_con = opt_query_con
        
        if opt_query_decoder:
            # box-detr
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            self.point_offset = MLP(d_model, d_model, 2 * nhead, 2) # 4d
            self.ref_point_head = MLP(d_model, d_model, d_model, 2)  # 2*dm init
            
            self.bbox_embed = MLP(d_model, d_model, 2, 3) # None # waiting for input
            # below in new box embed
            # self.bbox_embed = MLP(d_model, d_model, 2, 3)
            
            self.d_model = d_model
            self.nhead = nhead
            
            # self.offset_generator = FeatureDependentOffset(d_model, nhead)
            self.init_offset_generator = MLP(d_model, d_model, 2, 3)
            
            self.box_setting = box_setting

            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None
                
        if opt_query_con:
            self.d_model = d_model
            self.nhead = nhead
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            self.ref_point_head = MLP(d_model, d_model, 2, 2)
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None
    
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                box_unsigmoid: Optional[Tensor] = None,
                **kwargs):

        if self.opt_query_decoder:
            return self.forward_box(tgt=tgt, 
                                    memory=memory,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos,
                                    query_pos=query_pos,
                                    box_unsigmoid=box_unsigmoid,
                                    **kwargs)


        if self.opt_query_con:
            return self.forward_con(tgt=tgt, 
                                    memory=memory,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos,
                                    query_pos=query_pos,
                                    box_unsigmoid=box_unsigmoid,
                                    **kwargs)
        
        return self.forward_pet(tgt=tgt, 
                                memory=memory,
                                memory_key_padding_mask=memory_key_padding_mask,
                                pos=pos,
                                query_pos=query_pos,)
        
    def forward_pet(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):    
            output = layer(output, # tgt
                           memory, 
                           tgt_mask=tgt_mask, # None 
                        memory_mask=memory_mask,              # None
                        tgt_key_padding_mask=tgt_key_padding_mask,  # None
                        memory_key_padding_mask=memory_key_padding_mask,    # False
                        pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:            # intermediate[0 and 1 both].shape = torch.Size([8, 1024, 256]),,, len = 2
            return torch.stack(intermediate)    # PET:after stack -> torch.Size([2, 8, 1024, 256])

        return output.unsqueeze(0)

    def forward_box(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                box_unsigmoid: Optional[Tensor] = None,
                **kwargs):
        output = tgt
        intermediate = []
        query_pos_2d = self.init_offset_generator(query_pos.reshape(-1, self.d_model))
        nums, new_bs, dimen = box_unsigmoid.shape
        query_pos_2d = query_pos_2d.reshape(nums, new_bs, dimen) # query_pos_2d -1~1
        if self.box_setting == 2: # try to convert qp to 0-1
            query_pos_2d = 0.5 * query_pos_2d + 0.5
        # query_pos_2d = (query_pos_2d.sigmoid() - 0.5) * 2.0
        box = query_pos_2d
        
        # box = box_unsigmoid.sigmoid()   # box.shape = [ decw*dech, bs*nums, C=2]
        boxes = [box]
            
        for layer_id, layer in enumerate(self.layers):
            # get sine embedding for the query vector 
            query_pos = self.ref_point_head(gen_sineembed_for_position(box))
                                            # ...,2 -> ...,256
            
            # box agent
            N, B, _ = box.shape
            point_offset = self.point_offset(output).reshape(N, B, self.nhead, 2)
            agent = box[..., :2].unsqueeze(-2).repeat(1, 1, self.nhead, 1) + point_offset
                                    
            #if +point offset directly:
            # agent = box[..., :2].unsqueeze(2).repeat(1, 1, self.nhead, 1) + (box[..., 2:].unsqueeze(2).repeat(1, 1, self.nhead, 1) / 2) * point_offset
            # agent = box[..., :2].unsqueeze(2).repeat(1, 1, self.nhead, 1) + (fixed_size / 2) * point_offset
            
            # no offset box
            # agent = box[..., :2].unsqueeze(2).repeat(1, 1, self.nhead, 1)
            
            query_sine_embed = gen_sineembed_for_position(agent.flatten(1, 2)).reshape(N, B, self.nhead * self.d_model)
            
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output).repeat(1, 1, self.nhead)

            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation
            
            output = layer(output, memory, tgt_mask=tgt_mask, # None 
                    memory_mask=memory_mask,              # None
                    tgt_key_padding_mask=tgt_key_padding_mask,  # None
                    memory_key_padding_mask=memory_key_padding_mask,    # memory_key_padding_mask = mask_win = None
                    pos=pos, query_pos=query_pos, 
                    # box-detr:
                    query_sine_embed=query_sine_embed,
                    layer_id=layer_id)
            
            # iter update            
            if self.bbox_embed is not None:
                tmp = self.bbox_embed(output)
                tmp += inverse_sigmoid(box)
                new_box = tmp.sigmoid()
                if layer_id != self.num_layers - 1:
                    boxes.append(new_box)
                box = new_box.detach()
            
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)
            
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate),  #.transpose(1, 2),      # torch.Size([2, 8, 1024, 256])
                    torch.stack(boxes)  #.transpose(1, 2),
                ]
            else: # return here
                return [
                    torch.stack(intermediate),  #.transpose(1, 2), 
                    box.unsqueeze(0)  #.transpose(1, 2)
                ]  
                        
        return output.unsqueeze(0)
    
    def forward_con(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                box_unsigmoid: Optional[Tensor] = None,
                **kwargs):
        output = tgt
        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)
                
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model)
            query_sine_embed = query_sine_embed * pos_transformation
            
            output = layer(output, memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                layer_id=layer_id)
        
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return [torch.stack(intermediate), 
                    reference_points.unsqueeze(0).permute(0, 2, 1, 3).repeat(self.num_layers, 1, 1, 1)]
        
        return output.unsqueeze(0)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu", attn_type="softmax"):
        super().__init__()
        assert attn_type in ['softmax', 'chunk_linear', 'fused_chunk_linear', 'fused_recurrent_linear'], \
            print(f"attn_type should be in ['softmax', 'chunk_linear', 'fused_chunk_linear', 'fused_recurrent_linear'], but get: {attn_type}")
        print(f"attn_type: {attn_type}")
        self.attn_type = attn_type
        # if "linear" in attn_type:
        #     pass
        # else:
        #     self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, 
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        # if self.attn_type == "chunk_linear":
        #     q = rearrange(q, 'b n (h d) -> b h n d', h=8)
        #     src2 = chunk_linear_attn(q, k, v=src, normalize=True)
        # elif self.attn_type == 'fused_chunk_linear':
        #     src2 = fused_chunk_linear_attn(q, k, v=src, normalize=True)
        # elif self.attn_type == 'fused_recurrent_linear':
        #     src2 = fused_recurrent_linear_attn(q, k, v=src, normalize=True)
        # else:
        #     src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # if src2.isnan().any():
        #     src2 = torch.nan_to_num(src2, nan=0.0, posinf=1e4, neginf=-1e4)
        src = src + src2
        src = self.norm1(src)

        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu", attn_type="softmax", opt_query_decoder=False, opt_query_con=False):
        super().__init__()
        assert attn_type in ['softmax', 'chunk_linear', 'fused_chunk_linear', 'fused_recurrent_linear'], \
            print(f"attn_type should be in ['softmax', 'chunk_linear', 'fused_chunk_linear', 'fused_recurrent_linear'], but get: {attn_type}")
        self.attn_type = attn_type
    
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.nhead = nhead
        self.d_model = d_model
        
        self.opt_query_decoder = opt_query_decoder
        self.opt_query_con = opt_query_con
        self.normalize_before = False
        
        if opt_query_decoder or opt_query_con:   # box-detr-decoder
            # Decoder Self-Attention
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)  # vdim = None in PET

            # Decoder Cross-Attention
            self.ca_qcontent_proj = nn.Linear(d_model, d_model)
            self.ca_qpos_proj = nn.Linear(d_model, d_model)
            self.ca_kcontent_proj = nn.Linear(d_model, d_model)
            self.ca_kpos_proj = nn.Linear(d_model, d_model)
            self.ca_v_proj = nn.Linear(d_model, d_model)
            self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
            
            if opt_query_decoder:
                self.ca_qpos_sine_proj = nn.Conv1d(d_model * nhead, d_model, kernel_size=1, groups=nhead)
                
            if opt_query_con:
                self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
                
                # Implementation of Feedforward model
                self.dropout = nn.Dropout(dropout)
                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
                self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     # box-detr query:
                     query_sine_embed = None,
                     layer_id = 0,
                     ):
        
        if self.opt_query_decoder:
            return self.forward_box(tgt, memory, 
                        tgt_mask=tgt_mask, # None 
                        memory_mask=memory_mask,              # None
                        tgt_key_padding_mask=tgt_key_padding_mask,  # None
                        memory_key_padding_mask=memory_key_padding_mask,    # memory_key_padding_mask = mask_win = None
                        pos=pos, 
                        query_pos=query_pos, 
                        # box-detr:
                        query_sine_embed=query_sine_embed,
                        layer_id=layer_id)
        
        
        if self.opt_query_con:
            return self.forward_con(tgt, memory, 
                        tgt_mask=tgt_mask, # None 
                        memory_mask=memory_mask,              # None
                        tgt_key_padding_mask=tgt_key_padding_mask,  # None
                        memory_key_padding_mask=memory_key_padding_mask,    # memory_key_padding_mask = mask_win = None
                        pos=pos, 
                        query_pos=query_pos, 
                        query_sine_embed=query_sine_embed,
                        layer_id=layer_id)
            
        return self.forward_pet(tgt, # tgt
                           memory, 
                           tgt_mask=tgt_mask, # None 
                        memory_mask=memory_mask,              # None
                        tgt_key_padding_mask=tgt_key_padding_mask,  # None
                        memory_key_padding_mask=memory_key_padding_mask,    # False
                        pos=pos, query_pos=query_pos)

    def forward_pet(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,):
        # orginal PET (detr) decoder 
        # decoder self attention
        q = k = self.with_pos_embed(tgt, query_pos) # tgt position encoding

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, 
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        # decoder cross attention
        query = self.with_pos_embed(tgt, query_pos)
        key = self.with_pos_embed(memory, pos)
           
        tgt2 = self.multihead_attn(query, key, # 32, 56/224/224, 256
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # feed-forward
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

    def forward_box(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     # box-detr query:
                     query_sine_embed = None,
                     layer_id = 0,
                     ):
        # box-detr decoder
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)  # tgt is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)
        # proj before pos_embed
        q = q_content + q_pos
        k = k_content + k_pos
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============
        
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        
        num_queries, bs, n_model = q_content.shape
        
        hw, _, _ = k_content.shape  # win_h*win_w, bs*8, C
        k_pos = self.ca_kpos_proj(pos)
        
        # in PET: hw, num_mul_bs, d_model = q_content.shape

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        
        if layer_id == 0:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content
            
        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed.transpose(-1, -2)).transpose(-1, -2)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)
        
        tgt2 = self.cross_attn(query=q,
                            key=k,
                            value=v, attn_mask=memory_mask,
                            key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============
        
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        
        return tgt
    
    def forward_con(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     # box-detr query:
                     query_sine_embed = None,
                     layer_id = 0,
                     ):
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if layer_id == 0:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                key=k,
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_encoder(args, **kwargs):
    return WinEncoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        attn_type=args.attn_type if hasattr(args, 'attn_type') else 'softmax',
        **kwargs,
    )


def build_decoder(args, **kwargs):
    return WinDecoderTransformer(
        d_model=args.hidden_dim,                # 256
        dropout=args.dropout,                   # 0
        nhead=args.nheads,                      # 8
        dim_feedforward=args.dim_feedforward,   # 512
        num_decoder_layers=args.dec_layers,     # 2
        return_intermediate_dec=True,          
        attn_type=args.attn_type if hasattr(args, 'attn_type') else 'softmax',  # softmax
        opt_query_decoder=args.opt_query_decoder if hasattr(args, 'opt_query_decoder') else False,
        opt_query_con=args.opt_query_con if hasattr(args, 'opt_query_con') else False,
        box_setting=1 if not hasattr(args, 'box_setting') else args.box_setting,
        # **kwargs  
    )


def _get_activation_fn(activation):
    """
    Return an activation function given a string
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# if __name__ == '__main__':
#     opt = True
#     class Configs():
#         hidden_dim = 256
#         dropout = 0.0
#         nheads = 8
#         dim_feedforward = 512
#         dec_layers = 2
#         opt_query_decoder = opt
    
#     args = Configs()
#     transformer = build_decoder(args)
#     print(transformer)
    
#     class NestedTensor(object):
#         def __init__(self, tensors, mask: Optional[Tensor]):
#             self.tensors = tensors
#             self.mask = mask

#         def to(self, device):
#             # type: (Device) -> NestedTensor # noqa # type: ignore
#             cast_tensor = self.tensors.to(device)
#             mask = self.mask
#             if mask is not None:
#                 assert mask is not None
#                 cast_mask = mask.to(device)
#             else:
#                 cast_mask = None
#             return NestedTensor(cast_tensor, cast_mask)

#         def decompose(self):
#             return self.tensors, self.mask

#         def __repr__(self):
#             return str(self.tensors)
    

#     # from typing import Dict, List
#     input = {}
    
#     input['4x'] = torch.rand((3, 512, 32, 32))
#     input['8x'] = torch.rand((3, 512, 16, 16))
#     for k, v in input.items():
#         print(k, v.shape)
    
    # hs = transformer(encode_src, src_pos_embed, mask, 
    #                 pqs, img_shape=samples.tensors.shape[-2:], **kwargs)
