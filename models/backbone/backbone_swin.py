# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from QAHOI (https://github.com/cjw2021/QAHOI)
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet, Bottleneck
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer import SwinTransformer


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, backbone_name: str, num_feature_levels: int, pretrained=""):
        super().__init__()
        if "swin" in backbone_name:
            for name, parameter in backbone.named_parameters():
                if 'absolute_pos_embed' in name or 'relative_position_bias_table' in name or 'norm' in name:
                    parameter.requires_grad_(False) 
            if pretrained != "":
                print("load pretrained model...")
                backbone.init_weights(pretrained)
            self.strides = [8, 16, 32][-num_feature_levels:]
            self.body = backbone
            self.num_channels = [192, 384, 768][-num_feature_levels:]
            if "base" in backbone_name:
                self.num_channels = [256, 512, 1024][-num_feature_levels:]
            if "large" in backbone_name:
                self.num_channels = [384, 768, 1536][-num_feature_levels:]
        else:
            raise NotImplementedError

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    def __init__(self, backbone_name: str, num_feature_levels: int, pretrained=None, use_checkpoint=False, dilation=False):
        if "swin" in backbone_name:
            out_ind = [1, 2, 3][-num_feature_levels:]
            if 'small' in backbone_name:
                backbone = SwinTransformer(depths=[2, 2, 18, 2], out_indices=out_ind, use_checkpoint=use_checkpoint)
            elif 'base' in backbone_name:
                if "384" in backbone_name:
                    backbone = SwinTransformer(pretrain_img_size=384, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32], out_indices=out_ind, window_size=12, use_checkpoint=use_checkpoint)
                else:
                    backbone = SwinTransformer(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32], out_indices=out_ind, use_checkpoint=use_checkpoint)
            elif 'large' in backbone_name:
                if "384" in backbone_name:
                    backbone = SwinTransformer(pretrain_img_size=384, depths=[2, 2, 18, 2], embed_dim=192, num_heads=[6, 12, 24, 48], out_indices=out_ind, window_size=12, use_checkpoint=use_checkpoint)
                else:
                    backbone = SwinTransformer(depths=[2, 2, 18, 2], embed_dim=192, num_heads=[6, 12, 24, 48], out_indices=out_ind, use_checkpoint=use_checkpoint)
            else:
                backbone = SwinTransformer(depths=[2, 2, 6, 2], out_indices=out_ind, use_checkpoint=use_checkpoint)
        else:
            raise NotImplementedError

        super().__init__(backbone, backbone_name, num_feature_levels, pretrained)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    args.pretrained = ""
    backbone = Backbone(
        backbone_name=args.backbone, 
        num_feature_levels=args.num_feature_levels, 
        pretrained=args.swin_pretrained,
        use_checkpoint=args.use_checkpoint,
        dilation=False)
    model = Joiner(backbone, position_embedding)

    return model
