# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
import torch
import torch.nn.functional as F
from torch import nn, Tensor

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

# S-O bottom-up
class SOAttn(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, nhead: int, num_layers=6, dropout=0.1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sub_fea: Tensor, obj_fea: Tensor, verb_embed=None) -> Tensor:
        lvl, bs, nq, c = obj_fea.shape
        verb_embed = verb_embed.repeat(bs, 1, 1)
        outputs =[]
        for li in range(self.num_layers):
            fea = (sub_fea[li] + obj_fea[li])/2.0
            output = self.cross_attn(fea.transpose(0, 1), verb_embed.transpose(0, 1), value=verb_embed.transpose(0, 1))[0].transpose(0, 1)
            output = self.norm(fea + self.dropout(output))
            if li > 0:
                output = (outputs[li-1] + output)/ 2.0
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs
