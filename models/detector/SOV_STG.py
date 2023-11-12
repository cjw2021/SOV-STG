# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np

from scipy.optimize import linear_sum_assignment
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from ..backbone import build_backbone
from ..matcher.matcher import build_matcher
from ..ops.dn_process.dn_components_STG import sigmoid_focal_loss, prepare_for_dn, dn_post_process, compute_dn_loss
from ..encoder_decoder.deformable_transformer_SOV import build_deforamble_transformer_SOV

import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SOV_STG(nn.Module):
    """ This is the DAB-Deformable-DETR for object detection """
    def __init__(self, backbone, transformer, num_classes, num_verb_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=True, 
                 num_patterns=0,
                 random_refpoints_xy=False,
                 args=None
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            num_patterns: number of pattern embeddings
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.num_verb_classes = num_verb_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.verb_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        self.use_same_refpoint = args.use_same_refpoint
        self.dec_layers = args.dec_layers
        
        self.args = args

        # dn label enc
        self.label_enc = nn.Embedding(num_classes, hidden_dim - 1)  # for indicator
        self.verb_enc = nn.Embedding(num_verb_classes, hidden_dim - 1)  # for indicator
        if args.use_dn_weight:
            self.tgt_embed = None
            if self.args.wo_obj and self.args.wo_verb:
                raise NotImplementedError  # don't support
            elif self.args.wo_obj:
                self.tgt_coef = nn.Parameter(torch.zeros((num_queries, num_verb_classes)))
            elif self.args.wo_verb:
                self.tgt_coef = nn.Parameter(torch.zeros((num_queries, num_classes)))
            else:
                self.tgt_coef = nn.Parameter(torch.zeros((num_queries, num_classes + num_verb_classes)))
            nn.init.normal_(self.tgt_coef)
        else:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim - 1)  # for indicator
        #  hack implementation for verb fusion
        self.transformer.verb_embed = self.verb_enc
        
        if self.use_same_refpoint:
            self.refpoint_embed = nn.Embedding(num_queries, 4)
            if random_refpoints_xy:
                self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
                self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                self.refpoint_embed.weight.data[:, :2].requires_grad = False
        else:
            self.refpoint_obj_embed = nn.Embedding(num_queries, 4) 
            self.refpoint_sub_embed = nn.Embedding(num_queries, 4) 
            if random_refpoints_xy:
                self.refpoint_obj_embed.weight.data[:, :2].uniform_(0, 1)
                self.refpoint_obj_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_obj_embed.weight.data[:, :2])
                self.refpoint_obj_embed.weight.data[:, :2].requires_grad = False
                self.refpoint_sub_embed.weight.data[:, :2].uniform_(0, 1)
                self.refpoint_sub_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_sub_embed.weight.data[:, :2])
                self.refpoint_sub_embed.weight.data[:, :2].requires_grad = False
                
        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        #  freezed encoder
        for name, parameter in self.named_parameters():
            if not args.train_enc and 'input_proj.' in name:
                parameter.requires_grad_(False)
                
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.verb_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)


        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.verb_embed = _get_clones(self.verb_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.sub_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
            self.transformer.sub_decoder.bbox_embed = self.sub_bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.verb_embed = nn.ModuleList([self.verb_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
            self.transformer.sub_decoder.bbox_embed = None


    def forward(self, samples: NestedTensor, dn_args=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()  # src: [bs, c, h, w], mask: [bs, h, w]
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.num_patterns == 0:
            if self.args.use_dn_weight:
                if self.args.wo_verb and self.args.wo_obj:
                    raise NotImplementedError  # don't support
                elif self.args.wo_verb:
                    ref_embed = self.label_enc.weight
                elif self.args.wo_obj:
                    ref_embed = self.verb_enc.weight
                else:
                    ref_embed = torch.cat([self.label_enc.weight, self.verb_enc.weight], 0)

                tgt_all_embed = torch.mm(self.tgt_coef, ref_embed)
            else:
                tgt_all_embed = self.tgt_embed.weight  # [nq, 255]

            if self.use_same_refpoint:
                refanchor_obj = self.refpoint_embed.weight
                refanchor_sub = self.refpoint_embed.weight
            else:
                refanchor_obj = self.refpoint_obj_embed.weight   # [nq, 4]
                refanchor_sub = self.refpoint_sub_embed.weight   # [nq, 4]
            refanchor = (refanchor_sub, refanchor_obj)
        else:
            # multi patterns is not used in this version
            assert NotImplementedError

        # prepare for dn
        # input_query_label:         [bs, nq+dn_nq, 255+1]
        # input_query_bbox[0]:       [bs, dnq, 4]
        # attn_mask:                 [dnq, dnq]
        # mask_dict:                 train: {'known_indice', 'batch_idx', 'map_known_indice', 'known_lbs_bboxes', 'know_idx', 'pad_size'}
        #                            test:  {}
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_dn(dn_args, tgt_all_embed, refanchor, src.size(0), self.training, self.num_queries, self.num_classes,
                           self.hidden_dim, self.label_enc, self.verb_enc)
        query_embeds = torch.cat((input_query_label, input_query_bbox[0], input_query_bbox[1]), dim=2)  # [bs, dnq, 264]
       
        # Decoding
        # hs, sub_hs, interaction_hs: [N_layer, bs, dnq, 256]
        # init_reference:             [bs, dnq, 8]
        # initer_references:          [N_l, bs, dnq, 8]
        hs, init_reference, inter_references, interaction_hs, sub_hs, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embeds, attn_mask)

        outputs_classes = []
        outputs_coords = []
        outputs_verbs = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])  # [bs, dnq, 80]
            if interaction_hs is not None:
                outputs_verb = self.verb_embed[lvl](interaction_hs[lvl])  # [bs, dnq, 117]
            else:
                outputs_verb = self.verb_embed[lvl](hs[lvl])
            tmp_obj = self.bbox_embed[lvl](hs[lvl])  # [bs, dnq, 4]
            tmp_sub = self.sub_bbox_embed[lvl](sub_hs[lvl])  # [bs, dnq, 4]
            tmp = torch.cat([tmp_sub, tmp_obj], -1)  # [bs, dnq, 8]
            if reference.shape[-1] == 8:
                tmp += reference
            else:
                raise NotImplementedError  # don't support
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_verbs.append(outputs_verb)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_verb = torch.stack(outputs_verbs)
        
        # dn post process
        # outputs_class: [N_l, bs, nq, 80]
        # outputs_verb:  [N_l, bs, nq, 117]
        # outputs_coord: [N_l, bs, nq, 8]
        outputs_class, outputs_verb, outputs_coord = dn_post_process(outputs_class, outputs_verb, outputs_coord, mask_dict)
        
        outputs_coord_sub, outputs_coord_obj = torch.split(outputs_coord, 4, -1)  # bs, nq, 4

        outputs_class_layers = outputs_class
        outputs_verb_layers = outputs_verb
        outputs_coord_obj_layers = outputs_coord_obj
        outputs_coord_sub_layers = outputs_coord_sub

        out = {'pred_obj_logits': outputs_class_layers[-1], 'pred_obj_boxes': outputs_coord_obj_layers[-1],
               'pred_verb_logits': outputs_verb_layers[-1], 'pred_sub_boxes': outputs_coord_sub_layers[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class_layers, outputs_coord_sub_layers, outputs_coord_obj_layers, outputs_verb_layers)

        return out, mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord_sub, outputs_coord_obj, outputs_verb):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_obj_boxes': c,'pred_sub_boxes':d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_verb[:-1], outputs_coord_obj[:-1], outputs_coord_sub[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_obj_classes, num_verb_classes, matcher, weight_dict, losses, focal_alpha=0.25, pred_all=False, wo_mix_dn=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.pred_all = pred_all
        self.wo_mix_dn = wo_mix_dn

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        """Classification loss (NLL)
        """
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_obj_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_interactions, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        
        loss_verb_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()

        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)

            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes), 
                                                    box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes), 
                                                    box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)


        return losses
    


    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

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

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions, **kwargs))
            
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # dn loss
        if mask_dict:
            aux_num = 0
            if 'aux_outputs' in outputs:
                aux_num = len(outputs['aux_outputs'])
            dn_losses = compute_dn_loss(mask_dict, self.training, aux_num, self.focal_alpha, pred_all=self.pred_all, wo_mix_dn=self.wo_mix_dn)
            losses.update(dn_losses)
        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, subject_category_id, no_obj=False):
        super().__init__()
        self.subject_category_id = subject_category_id
        self.no_obj = no_obj

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        return:
            result: list(dict) 
            len(result): batch_size
            dict: All predicted information in each image
                'labels': Labeling of humans and objects
                'boxes': bboxes of humans and objects
                'verb_scores': human-object pairs' verb scores
                'sub_ids': index of humans in 'labels'
                'obj_ids': index of humans in 'labels'
            size:
                'labels':[tensor, cpu] 2 * nq
                'boxes':[tensor, cpu] 2 * nq, 4
                'verb_scores':[tensor, cpu] nq, cls(HICO-det verb classes: 117)
                'sub_ids':[tensor, cpu] nq,   [0, 1, 2, ..., nq-1]
                'obj_ids':[tensor, cpu] nq,   [nq, nq + 1, ..., 2 * nq -1]

        """
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']



        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        obj_scores = out_obj_logits.sigmoid().max(-1)[0]  # bs, nq
        verb_scores = out_verb_logits.sigmoid()  # bs, nq, cls
        obj_labels = F.softmax(out_obj_logits, -1).max(-1)[1]  #idx: bs, nq

        img_h, img_w = target_sizes.unbind(1)  # bs,
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)  # bs, 4
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)  # bs, nq, 4
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)  # bs, nq, 4
        
        sub_boxes = sub_boxes * scale_fct[:, None, :]  # bs, nq, 4
        obj_boxes = obj_boxes * scale_fct[:, None, :]  # bs, nq, 4

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
        
            sl = torch.full_like(ol, self.subject_category_id) # nq,
            l = torch.cat((sl, ol))  # 2*nq,
            b = torch.cat((sb, ob))  # 2*nq, 4
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)  # nq, cls

            ids = torch.arange(b.shape[0]) # 2*nq, [0, 1, 2, ..., 2 * nq - 1]

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results


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


def build_SOV_STG(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer_SOV(args)
    model = SOV_STG(
        backbone,
        transformer,
        num_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        num_patterns=args.num_patterns,
        random_refpoints_xy=args.random_refpoints_xy,
        args=args
    )

    matcher = build_matcher(args)

    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.iou_loss_coef
    weight_dict['loss_obj_giou'] = args.iou_loss_coef

    # dn loss
    weight_dict['tgt_loss_obj_ce'] = args.obj_loss_coef * args.dn_loss_coef
    weight_dict['tgt_loss_verb_ce'] = args.verb_loss_coef * args.dn_loss_coef
    weight_dict['tgt_loss_sub_bbox'] = args.bbox_loss_coef * args.dn_loss_coef
    weight_dict['tgt_loss_obj_bbox'] = args.bbox_loss_coef * args.dn_loss_coef
    weight_dict['tgt_loss_sub_giou'] = args.iou_loss_coef * args.dn_loss_coef
    weight_dict['tgt_loss_obj_giou'] = args.iou_loss_coef * args.dn_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
    criterion = SetCriterion(args.num_obj_classes, args.num_verb_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, pred_all=args.pred_all, wo_mix_dn=args.wo_mix_dn)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args.subject_category_id)}

    return model, criterion, postprocessors
