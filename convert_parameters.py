# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# DN-DETR to SOV-STG

import argparse

import torch
from torch import nn

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load_path', type=str, default='./params/checkpoint0049.pth'
    )
    parser.add_argument(
        '--save_path', type=str, required=True
    )
    parser.add_argument(
        '--dataset', type=str, default='hico',choices=['hico', 'vcoco']
    )
    parser.add_argument(
        '--num_queries', type=int, default=64
    )
    parser.add_argument('--dec_n_points', type=int, default=8, choices=[4, 8, 12, 16])
    parser.add_argument('--enc_n_points', type=int, default=4, choices=[4, 8, 12, 16])
    args = parser.parse_args()

    return args


def main(args):
    ps = torch.load(args.load_path)
    num_queries = args.num_queries

    obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
               82, 84, 85, 86, 87, 88, 89, 90]

    # For no pair
    # obj_ids.append(91)
    
    for k in list(ps['model'].keys()):
        print(k)
        if 'decoder' in k:
            ps['model'][k.replace('decoder', 'sub_decoder')] = ps['model'][k].clone()

    for i in range(6):
        ps['model'][f'class_embed.{i}.weight'] = ps['model'][f'class_embed.{i}.weight'].clone()[obj_ids]
        ps['model'][f'class_embed.{i}.bias'] = ps['model'][f'class_embed.{i}.bias'].clone()[obj_ids]

        if args.dataset == 'vcoco':
            l = nn.Linear(ps['model'][f'class_embed.{i}.weight'].shape[1], 1)
            l.to(ps['model'][f'class_embed.{i}.weight'].device)
            ps['model'][f'class_embed.{i}.weight'] = torch.cat((
                ps['model'][f'class_embed.{i}.weight'], l.weight))
            ps['model'][f'class_embed.{i}.bias'] = torch.cat(
                (ps['model'][f'class_embed.{i}.bias'], l.bias))

        dec_offsets_w = ps['model'][f'transformer.decoder.layers.{i}.cross_attn.sampling_offsets.weight'].clone()
        dec_offsets_b = ps['model'][f'transformer.decoder.layers.{i}.cross_attn.sampling_offsets.bias'].clone()
        dec_attention_weights_w = ps['model'][f'transformer.decoder.layers.{i}.cross_attn.attention_weights.weight'].clone()
        dec_attention_weights_b = ps['model'][f'transformer.decoder.layers.{i}.cross_attn.attention_weights.bias'].clone()

        enc_offsets_w = ps['model'][f'transformer.encoder.layers.{i}.self_attn.sampling_offsets.weight'].clone()
        enc_offsets_b = ps['model'][f'transformer.encoder.layers.{i}.self_attn.sampling_offsets.bias'].clone()
        enc_attention_weights_w = ps['model'][f'transformer.encoder.layers.{i}.self_attn.attention_weights.weight'].clone()
        enc_attention_weights_b = ps['model'][f'transformer.encoder.layers.{i}.self_attn.attention_weights.bias'].clone()

        dec_pattern = args.dec_n_points // 4
        enc_pattern = args.enc_n_points // 4

        ps['model'][f'transformer.decoder.layers.{i}.cross_attn.sampling_offsets.weight'] = torch.cat([dec_offsets_w] * dec_pattern)
        ps['model'][f'transformer.decoder.layers.{i}.cross_attn.sampling_offsets.bias'] = torch.cat([dec_offsets_b] * dec_pattern)
        ps['model'][f'transformer.decoder.layers.{i}.cross_attn.attention_weights.weight'] = torch.cat([dec_attention_weights_w] * dec_pattern)
        ps['model'][f'transformer.decoder.layers.{i}.cross_attn.attention_weights.bias'] = torch.cat([dec_attention_weights_b] * dec_pattern)

        ps['model'][f'transformer.sub_decoder.layers.{i}.cross_attn.sampling_offsets.weight'] = torch.cat([dec_offsets_w] * dec_pattern)
        ps['model'][f'transformer.sub_decoder.layers.{i}.cross_attn.sampling_offsets.bias'] = torch.cat([dec_offsets_b] * dec_pattern)
        ps['model'][f'transformer.sub_decoder.layers.{i}.cross_attn.attention_weights.weight'] = torch.cat([dec_attention_weights_w] * dec_pattern)
        ps['model'][f'transformer.sub_decoder.layers.{i}.cross_attn.attention_weights.bias'] = torch.cat([dec_attention_weights_b] * dec_pattern)

        ps['model'][f'transformer.encoder.layers.{i}.self_attn.sampling_offsets.weight'] = torch.cat([enc_offsets_w] * enc_pattern)
        ps['model'][f'transformer.encoder.layers.{i}.self_attn.sampling_offsets.bias'] = torch.cat([enc_offsets_b] * enc_pattern)
        ps['model'][f'transformer.encoder.layers.{i}.self_attn.attention_weights.weight'] = torch.cat([enc_attention_weights_w] * enc_pattern)
        ps['model'][f'transformer.encoder.layers.{i}.self_attn.attention_weights.bias'] = torch.cat([enc_attention_weights_b] * enc_pattern)

        for j in range(3):
            ps['model'][f'sub_bbox_embed.{i}.layers.{j}.weight'] = ps['model'][f'bbox_embed.{i}.layers.{j}.weight'].clone()
            ps['model'][f'sub_bbox_embed.{i}.layers.{j}.bias'] = ps['model'][f'bbox_embed.{i}.layers.{j}.bias'].clone()


    for ki in ['optimizer', 'lr_scheduler', 'epoch', 'args']:
        if ki in ps:
            ps.pop(ki)

    ps['model'][f'refpoint_obj_embed.weight'] = ps['model'][f'refpoint_embed.weight'].clone()[:num_queries]
    ps['model'][f'refpoint_sub_embed.weight'] = ps['model'][f'refpoint_embed.weight'].clone()[:num_queries]

    ps['model'][f'tgt_embed.weight'] = ps['model'][f'tgt_embed.weight'].clone()[:num_queries]

    ps['model'][f'label_enc.weight'] = ps['model'][f'label_enc.weight'].clone()[obj_ids]
    
    if args.dataset == 'vcoco':
        l = nn.Linear(ps['model']['label_enc.weight'].shape[1], 1)
        l.to(ps['model']['label_enc.weight'].device)
        ps['model']['label_enc.weight'] = torch.cat((
            ps['model']['label_enc.weight'], l.weight))


    ref_point_head_weight = ps['model'][f'transformer.decoder.ref_point_head.layers.0.weight'].clone()
    ref_point_head_bias = ps['model'][f'transformer.decoder.ref_point_head.layers.0.bias'].clone()

    ps['model'][f'transformer.decoder.ref_point_head.layers.0.weight'] = ref_point_head_weight
    ps['model'][f'transformer.decoder.ref_point_head.layers.0.bias'] = ref_point_head_bias
    ps['model'][f'transformer.sub_decoder.ref_point_head.layers.0.weight'] = ref_point_head_weight
    ps['model'][f'transformer.sub_decoder.ref_point_head.layers.0.bias'] = ref_point_head_bias

    torch.save(ps, args.save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
