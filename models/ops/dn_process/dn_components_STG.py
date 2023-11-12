# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util import box_ops
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes


def prepare_for_dn(dn_args, tgt_weight, embedweight, batch_size, training, num_queries, num_classes, hidden_dim, label_enc, verb_enc):
    """
    The major difference from DN-DAB-DETR is that the author process pattern embedding pattern embedding in its detector
    forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, scalar, label_noise_scale, box_noise_scale, num_patterns
    :param tgt_weight: use learnbal tgt in dab deformable detr
    :param embedweight: positional anchor queries
    :param batch_size: bs
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param num_verbs: number of verbs
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn
    :param verb_enc: encode verb labels in dn
    :return:
    """

    if training and len(dn_args) > 1:
        targets, dy_weight, args, scalar, label_noise_scale, verb_noise_scale, box_noise_scale, num_patterns = dn_args
        try:
            args.verb_noise_prob_l >= 0
            if args.verb_noise_prob_h != -1:
                args.verb_noise_prob_l <= args.verb_noise_prob_h
            elif args.verb_noise_prob_h == -1:
                args.verb_noise_prob_h = args.verb_noise_prob_l
        except:
            raise Exception('verb_noise_prob_l is {} and verb_noise_prob_h is {},verb_noise_prob_h needs to be greater than or equal to verb_noise_prob_l,'
                            'verb_noise_prob_l needs to be greater than or equal to 0'.format(args.verb_noise_prob_l, args.verb_noise_prob_h))
        else:
            verb_noise_prob_l = args.verb_noise_prob_l
            verb_noise_prob_h = args.verb_noise_prob_h

        pred_all = args.pred_all
        wo_mix_dn = args.wo_mix_dn
        wo_obj = args.wo_obj
        wo_verb = args.wo_verb
    else:
        num_patterns = dn_args[-1]

    if num_patterns == 0:
        num_patterns = 1
    indicator0 = torch.zeros([num_queries * num_patterns, 1]).cuda()
    # sometimes the target is empty, add a zero part of label_enc and verb_enc to avoid unused parameters
    tgt = torch.cat([tgt_weight, indicator0], dim=1) + label_enc.weight[0][0] * torch.tensor(0).cuda() + verb_enc.weight[0][0] * torch.tensor(0).cuda()
    refpoint_emb_sub, refpoint_emb_obj = embedweight
    if training and len(dn_args) > 1:
        known = [(torch.ones_like(t['obj_labels'])).cuda() for t in targets]
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]

        if not wo_mix_dn:
            assert scalar % 3 == 0 and scalar > 0, "dn scalar must be divisible by 3 and greater than zero"
            if (scalar > 3) and int(max(known_num)) > 24:  # To avoid OOM
                scalar = scalar - 3
            # obj_dn group, mix_dn group, verb_dn group
            group_size = scalar // 3
        else:
            assert scalar % 2 == 0 and scalar > 0, "dn scalar must be divisible by 2 and greater than zero"

            if (scalar > 2) and int(max(known_num)) > 24:  # To avoid OOM
                scalar = scalar - 2
            # obj_dn group, verb_dn group
            group_size = scalar // 2

        # can be modified to selectively denosie some label or boxes; also known label prediction
        unmask_bbox = unmask_label = torch.cat(known)
        verb_labels = torch.cat([t['verb_labels'] for t in targets]).to(torch.int64)  # sum_{bs}(n_gt), 117
        obj_labels = torch.cat([t['obj_labels'] for t in targets])  # sum_{bs}(n_gt),
        sub_boxes = torch.cat([t['sub_boxes'] for t in targets])  # sum_{bs}(n_gt), 4
        obj_boxes = torch.cat([t['obj_boxes'] for t in targets])  # sum_{bs}(n_gt), 4
        batch_idx = torch.cat([torch.full_like(t['obj_labels'].long(), i) for i, t in enumerate(targets)])
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        num_group_member = known_indice.numel()  # sum_{bs}(n_gt)


        # add noise
        # indice
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # verb
        known_verb_labels = verb_labels.repeat(scalar, 1)  # scalar * sum_{bs}(n_gt), 117
        # labels
        known_obj_labels = obj_labels.repeat(scalar, 1).view(-1)  # scalar * sum_{bs}(n_gt),
        # bboxs
        known_sub_bboxes = sub_boxes.repeat(scalar, 1)
        known_obj_bboxes = obj_boxes.repeat(scalar, 1)
        # clone
        known_verb_labels_expand = known_verb_labels.clone()
        known_obj_labels_expand = known_obj_labels.clone()
        known_sub_bbox_expand = known_sub_bboxes.clone()
        known_obj_bbox_expand = known_obj_bboxes.clone()
        # 
        if not pred_all: 
            if not wo_mix_dn:
                # obj_dn; mix_dn; verb_dn
                known_obj_labels = known_obj_labels[:2 * group_size * num_group_member]  # first two-third group of obj gt
                known_verb_labels = known_verb_labels[group_size * num_group_member:]  # last two-third group of verb gt   
            else:
                # obj_dn; verb_dn
                known_obj_labels = known_obj_labels[:group_size * num_group_member]  # first half group of obj gt
                known_verb_labels = known_verb_labels[group_size * num_group_member:]  # last half group of verb gt

        # noise on the label
        if label_noise_scale > 0:
            # obj_labels
            p = torch.rand_like(known_obj_labels_expand.float())
            chosen_indice = torch.nonzero(p < (label_noise_scale * dy_weight)).view(-1)  # usually half of bbox noise
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_obj_labels_expand.scatter_(0, chosen_indice, new_label)

        # noise on the verb
        if verb_noise_scale > 0 and verb_noise_prob_l > 0:
            # Select the verb that needs DN
            p = torch.rand((known_verb_labels_expand.shape[0],))
            chosen_verbs = torch.nonzero(p < (verb_noise_scale * dy_weight)).view(-1)

            p_chosen = torch.rand((chosen_verbs.shape[0], known_verb_labels_expand.shape[1]))
            replace_verb = known_verb_labels_expand[chosen_verbs].clone()
            v_p = torch.rand_like(p_chosen).uniform_(verb_noise_prob_l, verb_noise_prob_h)
            chosen_verb_element_indice = torch.nonzero(p_chosen < v_p)
            chosen_verb_elem_indice_tuple = chosen_verb_element_indice[:, 0], chosen_verb_element_indice[:, 1]
            new_label = torch.randint_like(chosen_verb_element_indice[:, 0], 1, 2).to('cuda')  # randomly put a new one here
            replace_verb[chosen_verb_elem_indice_tuple] = new_label
            known_verb_labels_expand[chosen_verbs] = replace_verb
 
        # noise on the box
        if box_noise_scale > 0:
            # sub_bboxes
            diff = torch.zeros_like(known_sub_bbox_expand)
            diff[:, :2] = known_sub_bbox_expand[:, 2:] / 2
            diff[:, 2:] = known_sub_bbox_expand[:, 2:]
            known_sub_bbox_expand += torch.mul((torch.rand_like(known_sub_bbox_expand) * 2 - 1.0), diff).cuda() * box_noise_scale
            known_sub_bbox_expand = known_sub_bbox_expand.clamp(min=0.0, max=1.0)

            # obj_bboxes
            diff = torch.zeros_like(known_obj_bbox_expand)
            diff[:, :2] = known_obj_bbox_expand[:, 2:] / 2
            diff[:, 2:] = known_obj_bbox_expand[:, 2:]
            known_obj_bbox_expand += torch.mul((torch.rand_like(known_obj_bbox_expand) * 2 - 1.0), diff).cuda() * box_noise_scale
            known_obj_bbox_expand = known_obj_bbox_expand.clamp(min=0.0, max=1.0)

        # encode labels
        obj_labels_enc = known_obj_labels_expand.long().to('cuda')
        verb_labels_enc = []
        elem_num = known_verb_labels_expand.numel()
        for t in known_verb_labels_expand:
            idx = torch.nonzero(t).view(-1)
            if elem_num > 0:
                verb_labels_enc.append(verb_enc(idx.long().to('cuda')).sum(0, keepdim=True))
               
        if verb_labels_enc:
            verb_labels_enc = torch.cat(verb_labels_enc)
        else:
            verb_labels_enc = verb_enc(torch.tensor([]).long().cuda())

        input_obj_label_enc = label_enc(obj_labels_enc)

        # split obj_dn + dn and dn +verb_dn block
        if not wo_mix_dn:
            # obj_dn + dn block
            input_obj_label_enc_block = input_obj_label_enc[:2 * group_size * num_group_member]
            input_obj_label_enc_zero_padding = torch.zeros_like(input_obj_label_enc[2 * group_size * num_group_member:]).cuda()
            input_obj_label_enc = torch.cat([input_obj_label_enc_block, input_obj_label_enc_zero_padding])
            # dn + verb_dn block
            verb_labels_enc_block = verb_labels_enc[group_size * num_group_member:]
            verb_labels_enc_zero_padding = torch.zeros_like(verb_labels_enc[:group_size * num_group_member]).cuda()
            input_verb_labels_enc = torch.cat([verb_labels_enc_zero_padding, verb_labels_enc_block])
        else:
            # obj_dn block
            input_obj_label_enc_block = input_obj_label_enc[:group_size * num_group_member]
            input_obj_label_enc_zero_padding = torch.zeros_like(input_obj_label_enc[group_size * num_group_member:]).cuda()
            input_obj_label_enc = torch.cat([input_obj_label_enc_block, input_obj_label_enc_zero_padding])
            # verb_dn block
            verb_labels_enc_block = verb_labels_enc[group_size * num_group_member:]
            verb_labels_enc_zero_padding = torch.zeros_like(verb_labels_enc[:group_size * num_group_member]).cuda()
            input_verb_labels_enc = torch.cat([verb_labels_enc_zero_padding, verb_labels_enc_block])

        if wo_obj:
            input_obj_label_enc = torch.zeros_like(input_obj_label_enc).cuda()
        if wo_verb:
            input_verb_labels_enc = torch.zeros_like(input_verb_labels_enc).cuda()

        input_label_embed = input_obj_label_enc + input_verb_labels_enc

        # add dn part indicator
        # labels
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
        # bboxes
        input_sub_bbox_embed = inverse_sigmoid(known_sub_bbox_expand)
        input_obj_bbox_embed = inverse_sigmoid(known_obj_bbox_expand)

        # padding bboxes and labels
        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
        input_query_sub_bbox = torch.cat([padding_bbox, refpoint_emb_sub], dim=0).repeat(batch_size, 1, 1)
        input_query_obj_bbox = torch.cat([padding_bbox, refpoint_emb_obj], dim=0).repeat(batch_size, 1, 1)
        input_query_bbox = (input_query_sub_bbox, input_query_obj_bbox)
        
        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [0,1,2, 0,1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_sub_bbox[(known_bid.long(), map_known_indice)] = input_sub_bbox_embed
            input_query_obj_bbox[(known_bid.long(), map_known_indice)] = input_obj_bbox_embed
            input_query_bbox = (input_query_sub_bbox, input_query_obj_bbox)

        tgt_size = pad_size + num_queries * num_patterns
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_verb_labels, known_obj_labels, known_sub_bboxes, known_obj_bboxes),
            'know_idx': know_idx,
            'pad_size': pad_size,
        }
    else:  # eval mode
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_sub_bbox = refpoint_emb_sub.repeat(batch_size, 1, 1)
        input_query_obj_bbox = refpoint_emb_obj.repeat(batch_size, 1, 1)
        input_query_bbox = (input_query_sub_bbox, input_query_obj_bbox)
        attn_mask = None
        mask_dict = {}

    return input_query_label, input_query_bbox, attn_mask, mask_dict
 

def dn_post_process(outputs_class, outputs_verb, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict

    Returns:
        mask_dict['output_known_lbs_bboxes']:
            outputs_known_class: obj_class embedding [lvl, bs, single_pad * scale, obj_cls]
            outputs_known_verb: verb_class embedding [lvl, bs, single_pad * scale, verb_cls]
            outputs_known_coord: sub and obj bboxes' coordinate  [lvl, bs, single_pad * scale, 8],coordinate(Cx,Cy,W,H):[sub:4,obj:4]
        outputs_class: obj_class embedding [lvl, bs, qn, obj_cls]
        outputs_verb: verb_class embedding [lvl, bs, qn, verb_cls]
        outputs_coord: sub and obj bboxes' coordinate  [lvl, bs, qn, 8],coordinate(Cx,Cy,W,H):[sub:4,obj:4]
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        output_known_verb = outputs_verb[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        outputs_verb = outputs_verb[:, :, mask_dict['pad_size']:, :]
        # if not instance in patch ,mask_dict will without keys:'output_known_lbs_bboxes'
        mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_verb, output_known_coord)
    return outputs_class, outputs_verb, outputs_coord


def prepare_for_loss(mask_dict, pred_all=True, wo_mix_dn=True):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    Returns:

    """
    # outputs
    output_known_class, output_known_verb, output_known_coord = mask_dict['output_known_lbs_bboxes']
    # tgt
    known_verb_labels, known_obj_labels, known_sub_bboxes, known_obj_bboxes = mask_dict['known_lbs_bboxes']
    map_known_indice = mask_dict['map_known_indice']
    known_indice = mask_dict['known_indice']
    batch_idx = mask_dict['batch_idx']
    num_tgt = known_indice.numel()
    bid = batch_idx[known_indice]
    if len(output_known_class) > 0:
        
        # dnq =sum_{all_bs}(bs_tgt_num + bs_padding_num)
        # [N_l, bs, dnq, c] permute [bs, dnq, N_l, c] ->[saclar * bs * bs_tgt_num, N_l, c] premute [N_l, saclar * bs * bs_tgt_num, c]
        if not pred_all:
            if wo_mix_dn:  # [N_l, saclar/2 * bs * bs_tgt_num, c]
                output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)][:num_tgt // 2].permute(1, 0, 2)  # Use the first half group pred to predict obj
                output_known_verb = output_known_verb.permute(1, 2, 0, 3)[(bid, map_known_indice)][num_tgt // 2:].permute(1, 0, 2)  # Use the last half group pred to predict verb
                output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            else:  # [N_l, 2 * saclar/3 * bs * bs_tgt_num, c]
                output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)][:2 * (num_tgt // 3)].permute(1, 0, 2)  # Use the first two-third group pred to predict obj
                output_known_verb = output_known_verb.permute(1, 2, 0, 3)[(bid, map_known_indice)][num_tgt // 3:].permute(1, 0, 2)  # Use the last two-third group pred to predict verb
                output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        else:  # [N_l, saclar * bs * bs_tgt_num, c]
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)  # Use all group pred to predict obj
            output_known_verb = output_known_verb.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)  # Use all group pred to predict verb
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
   
    return known_verb_labels, known_obj_labels, known_sub_bboxes, known_obj_bboxes, output_known_class, output_known_verb, output_known_coord, num_tgt


def tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt, is_sub=False):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    if is_sub:
        key_name = 'sub_'
    else:
        key_name = 'obj_'
    if len(tgt_boxes) == 0:
        return {
            'tgt_loss_' + key_name + 'bbox': torch.as_tensor(0.).to('cuda'),
            'tgt_loss_' + key_name + 'giou': torch.as_tensor(0.).to('cuda'),
        }

    loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

    losses = {}
    losses['tgt_loss_' + key_name + 'bbox'] = loss_bbox.sum() / num_tgt
    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(tgt_boxes)))
    losses['tgt_loss_' + key_name + 'giou'] = loss_giou.sum() / num_tgt
    
    return losses


def tgt_loss_labels(src_logits_, tgt_labels_, num_tgt, focal_alpha):
    """Classification loss 
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_obj_ce': torch.as_tensor(0.).to('cuda'),
            'tgt_obj_class_error': torch.as_tensor(0.).to('cuda'),
        }

    src_logits, tgt_labels = src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:, :, :-1]
    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=focal_alpha, gamma=2) * src_logits.shape[1]

    losses = {'tgt_loss_obj_ce': loss_ce}

    losses['tgt_obj_class_error'] = 100 - accuracy(src_logits_, tgt_labels_)[0]
    return losses


def tgt_loss_verb_labels(src_logits_, tgt_labels_):
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_verb_ce': torch.as_tensor(0.).to('cuda')
        }
    
    src_logits, tgt_labels = src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)

    src_logits = src_logits.sigmoid()
    
    loss_verb_ce = _neg_loss(src_logits, tgt_labels)

    losses = {'tgt_loss_verb_ce': loss_verb_ce}
    return losses


def _neg_loss(pred, gt):
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


def compute_dn_loss(mask_dict, training, aux_num, focal_alpha=0.25, pred_all=True, wo_mix_dn=True):
    """
       compute dn loss in criterion
       Args:
           mask_dict: a dict for dn information
           training: training or inference flag
           aux_num: aux loss number
           focal_alpha:  for focal loss
       """
    losses = {}
    if training and 'output_known_lbs_bboxes' in mask_dict:
        known_verb_labels, known_obj_labels, known_sub_bboxes, known_obj_bboxes, \
            output_known_class, output_known_verb, output_known_coord, num_tgt = prepare_for_loss(mask_dict, pred_all, wo_mix_dn)
        # obj_labels and verb_labels
        if pred_all:
            losses.update(tgt_loss_labels(output_known_class[-1], known_obj_labels, num_tgt, focal_alpha))
        else:
            if wo_mix_dn:
                losses.update(tgt_loss_labels(output_known_class[-1], known_obj_labels, num_tgt // 2, focal_alpha))
            else:
                losses.update(tgt_loss_labels(output_known_class[-1], known_obj_labels, 2 * num_tgt // 3, focal_alpha))

        losses.update(tgt_loss_verb_labels(output_known_verb[-1], known_verb_labels))
        # sub_bboxes and obj_bboxes
        output_known_coord_sub, output_known_coord_obj = torch.split(output_known_coord, 4, -1)
        losses.update(tgt_loss_boxes(output_known_coord_sub[-1], known_sub_bboxes, num_tgt, is_sub=True))
        losses.update(tgt_loss_boxes(output_known_coord_obj[-1], known_obj_bboxes, num_tgt))
        
    else:
        losses['tgt_loss_obj_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_verb_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_sub_bbox'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_obj_bbox'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_sub_giou'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_obj_giou'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_obj_class_error'] = torch.as_tensor(0.).to('cuda')

    # aux dn loss
    if aux_num:
        for i in range(aux_num):
            if training and 'output_known_lbs_bboxes' in mask_dict:

                if pred_all:
                    l_dict = tgt_loss_labels(output_known_class[i], known_obj_labels, num_tgt, focal_alpha)
                else:
                    if wo_mix_dn:
                        l_dict = tgt_loss_labels(output_known_class[i], known_obj_labels, num_tgt // 2, focal_alpha)
                    else:
                        l_dict = tgt_loss_labels(output_known_class[i], known_obj_labels, 2 * num_tgt // 3, focal_alpha)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

                l_dict = tgt_loss_verb_labels(output_known_verb[i], known_verb_labels)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

                l_dict = tgt_loss_boxes(output_known_coord_sub[i], known_sub_bboxes, num_tgt, is_sub=True)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

                l_dict = tgt_loss_boxes(output_known_coord_obj[i], known_obj_bboxes, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_obj_ce' + f'_{i}'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_verb_ce' + f'_{i}'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_sub_bbox' + f'_{i}'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_obj_bbox' + f'_{i}'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_obj_class_error' + f'_{i}'] = torch.as_tensor(0.).to('cuda')
                losses['tgt_loss_sub_giou' + f'_{i}'] = torch.as_tensor(0.).to('cuda')
                losses['tgt_loss_obj_giou' + f'_{i}'] = torch.as_tensor(0.).to('cuda')
                losses.update(l_dict)
    return losses
