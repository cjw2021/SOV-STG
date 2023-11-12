# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from util.misc import inverse_sigmoid
from ..ops.deformable_transformer_attention.modules import MSDeformAttn
from ..ops.so_attn.SOAttn import SOAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 high_dim_query_update=False, no_sine_embed=False, args=None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.use_dn_weight = args.use_dn_weight
        self.args = args
        self.arch = args.arch
        self.num_queries = args.num_queries
        self.verb_embed = None
        self.obj_embed = None
        train_enc = args.train_enc
        use_checkpoint = args.use_checkpoint

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, train_enc, use_checkpoint)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, 
                                                    d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed, use_checkpoint=use_checkpoint)
        self.sub_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, 
                                                    d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed, use_checkpoint=use_checkpoint)

        verb_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                                dropout, activation,
                                                                num_feature_levels, nhead, dec_n_points)
        
        self.vDec = DeformableTransformerDecoder(verb_decoder_layer, num_decoder_layers, return_intermediate_dec, 
                                                    d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed,
                                                    verb_dec=True, arch=args.arch, use_checkpoint=use_checkpoint)


        self.fuse_module = SOAttn(d_model=d_model, dim_feedforward=4*d_model, nhead=8, num_layers=num_decoder_layers)


        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.high_dim_query_update = high_dim_query_update
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio  # bs, 2 

    def forward(self, srcs, masks, pos_embeds, query_embed=None, attn_mask=None):
        """
        Input:
            - srcs: List([bs, c, h, w])
            - masks: List([bs, h, w])
        """
        assert query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)     # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for sub and obj decoder
        bs, _, c = memory.shape
        reference_points = query_embed[..., self.d_model:].sigmoid()  # bs, nq, 8
        tgt = query_embed[..., :self.d_model]
        init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points[..., 4:], memory,
                                            spatial_shapes, level_start_index, valid_ratios, 
                                            query_pos=None, 
                                            src_padding_mask=mask_flatten, attn_mask=attn_mask)
        sub_hs, sub_inter_references = self.sub_decoder(tgt, reference_points[..., :4], memory,
                                                        spatial_shapes, level_start_index, valid_ratios, 
                                                        query_pos=None, 
                                                        src_padding_mask=mask_flatten, attn_mask=attn_mask)
        inter_references = torch.cat([sub_inter_references, inter_references], -1)  # enc_lvl, bs, dnq, 8

        
        # fuse
        kwargs = {}
        verb_embed_padding = torch.zeros(self.verb_embed.weight.shape[0], 1).cuda()
        verb_embed = torch.cat([self.verb_embed.weight, verb_embed_padding], dim=1).unsqueeze(0)  # 1, verb_cls, d_model
        kwargs.update({'verb_embed': verb_embed})
        verb_input = self.fuse_module(hs, sub_hs, **kwargs)  # enc_lvl, bs, dnq, d_model



        interaction_hs = None
        interaction_query_embed = verb_input
        # last layer
        interaction_tgt = interaction_query_embed[-1]
        query_pos = None
        coord = inter_references[-1].clone().detach()  # bs, dnq, 8
        if self.args.vdec_box_type == 'adaptive_shifted_MBR':
            coord = adaptive_shifted_MBR(coord)
        elif self.args.vdec_box_type == 'MBR':
            coord = MBR(coord)
        elif self.args.vdec_box_type == 'shifted_MBR':
            coord = shifted_MBR(coord)
        elif self.args.vdec_box_type == 'subject':
            coord = coord[...,:4]  # sub
        elif self.args.vdec_box_type == 'object':
            coord = coord[...,4:]  # obj
        else:
            raise ValueError('vdec_box_type {} is invalid',
                                'vdec_box_type [MBR, shifted_MBR, adaptive_shifted_MBR, subject, object]'.format(self.args.vdec_box_type))
        interaction_hs, _ = self.vDec(interaction_tgt, coord, memory,
                                        spatial_shapes, level_start_index, valid_ratios,
                                        query_pos=query_pos,
                                        src_padding_mask=mask_flatten, attn_mask=attn_mask)
            
        inter_references_out = inter_references   # dec_lvl, bs, dnq, 8


        return hs, init_reference_out, inter_references_out, interaction_hs, sub_hs, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, train_enc, use_checkpoint=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        for name, parameter in self.named_parameters():
            if not train_enc and 'layers.0' in name:
                parameter.requires_grad_(False)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_lebel, 2]
        """
        output = src
        # bs, sum(hi*wi), 256
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for l_i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                output = checkpoint.checkpoint(layer, output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            else:
                output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, self_attn_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=self_attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, d_model=256, high_dim_query_update=False, no_sine_embed=False,
                 verb_dec=False, arch="", use_checkpoint=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.verb_embed = None
        self.d_model = d_model
        self.verb_dec = verb_dec
        self.hoef_arch = arch
        self.no_sine_embed = no_sine_embed
        self.use_checkpoint = use_checkpoint
        self.query_scale = MLP(d_model, d_model, d_model, 2) if num_layers > 1 else None
        if self.no_sine_embed:
            self.ref_point_head = MLP(4, d_model, d_model, 3)
        else:
            self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)


    def forward(self, tgt, reference_points, src, src_spatial_shapes,       
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, attn_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                        * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] # bs, nq, 4, 4
            else:
                raise NotImplementedError  # don't support
            if self.no_sine_embed:
                raw_query_pos = self.ref_point_head(reference_points_input)
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # bs, nq, 256* 2 
                raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
            pos_scale = self.query_scale(output) if lid != 0 else 1
            query_pos_ref = pos_scale * raw_query_pos
            query_pos_this_layer = query_pos_ref
                
            if self.high_dim_query_update and lid != 0:
                query_pos_this_layer = query_pos_this_layer + self.high_dim_query_proj(output)
            
            output_this_layer = output

            if self.use_checkpoint and self.training:
                output = checkpoint.checkpoint(layer, output_this_layer, query_pos_this_layer, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                            src_padding_mask, attn_mask)
            else:
                output = layer(output_this_layer, query_pos_this_layer, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                            src_padding_mask, self_attn_mask=attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    raise NotImplementedError  # don't support
                reference_points = new_reference_points.detach()  # bs, nq, 4

            if self.return_intermediate:
                intermediate.append(output)
                if self.bbox_embed is not None:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
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


def MBR(reference_points_input):
    lt = reference_points_input[..., [0, 1, 4, 5]] - reference_points_input[..., [2, 3, 6, 7]] * 0.5
    rb = reference_points_input[..., [0, 1, 4, 5]] + reference_points_input[..., [2, 3, 6, 7]] * 0.5
    lt = torch.min(lt[..., :2], lt[..., 2:])
    rb = torch.max(rb[..., :2], rb[..., 2:])
    cxcy = (lt + rb) / 2
    wh = rb - lt
    mbr_cxcy = torch.cat([cxcy, wh], -1)
    return mbr_cxcy


def adaptive_shifted_MBR(reference_points_input):
    reference_points_input_x = (reference_points_input[:, :, 0] + reference_points_input[:, :, 4])/2
    reference_points_input_y = (reference_points_input[:, :, 1] + reference_points_input[:, :, 5])/2
    reference_points_input_w = torch.abs(reference_points_input[:, :, 0] - reference_points_input[:, :, 4]) \
        + (reference_points_input[:, :, 2] + reference_points_input[:, :, 6])/2
    reference_points_input_h = torch.abs(reference_points_input[:, :, 1] - reference_points_input[:, :, 5]) \
        + (reference_points_input[:, :, 3] + reference_points_input[:, :, 7])/2
    reference_points_input = torch.stack([reference_points_input_x, reference_points_input_y, reference_points_input_w, reference_points_input_h],-1)
    return reference_points_input


def shifted_MBR(reference_points_input):
    lt = reference_points_input[..., [0, 1, 4, 5]] - reference_points_input[..., [2, 3, 6, 7]] * 0.5
    rb = reference_points_input[..., [0, 1, 4, 5]] + reference_points_input[..., [2, 3, 6, 7]] * 0.5
    lt = torch.min(lt[..., :2], lt[..., 2:])
    rb = torch.max(rb[..., :2], rb[..., 2:])
    wh = rb - lt
    reference_points_input_x = (reference_points_input[:, :, 0] + reference_points_input[:, :, 4])/2
    reference_points_input_y = (reference_points_input[:, :, 1] + reference_points_input[:, :, 5])/2
    reference_points_input_w = wh[..., 0]
    reference_points_input_h = wh[..., 1]
    reference_points_input = torch.stack([reference_points_input_x, reference_points_input_y, reference_points_input_w, reference_points_input_h],-1)
    return reference_points_input


def build_deforamble_transformer_SOV(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.transformer_activation,
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        args=args)
