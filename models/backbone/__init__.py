from .backbone_resnet import build_backbone as build_backbone_resnet
from .backbone_swin import build_backbone as build_backbone_swin

def build_backbone(args):
    if 'swin' in args.backbone:
        backbone = build_backbone_swin(args)
    else:
        backbone = build_backbone_resnet(args)
    return backbone

