# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from QAHOI
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from QPIC (https://github.com/hitachi-rd-cv/qpic)
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcherHOI(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_obj_class: float = 1,
                 cost_verb_class: float = 1,
                 cost_bbox: float = 1,
                 cost_iou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_bbox = cost_bbox
        self.cost_iou_conf = cost_iou
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_iou != 0, 'all costs cant be 0'

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs['pred_obj_logits'].shape[:2]

            out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).sigmoid()
            out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
            out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

            tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
            tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
            tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])


            out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
            tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
            tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
            cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                            (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                            ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_obj_prob**gamma) * (-(1 - out_obj_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_obj_prob)**gamma) * (-(out_obj_prob + 1e-8).log())
            cost_obj_class = pos_cost_class[:,tgt_obj_labels] - neg_cost_class[:,tgt_obj_labels]
            # Compute the L1 cost between boxes
            cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
            cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (
                tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
            if cost_sub_bbox.shape[1] == 0:
                cost_bbox = cost_sub_bbox
            else:
                cost_bbox = torch.stack(
                    (cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]
            # Compute the giou cost betwen boxes
            cost_sub_iou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_sub_bbox),
                box_cxcywh_to_xyxy(tgt_sub_boxes))
            cost_obj_iou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                            cost_sub_iou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
            if cost_sub_iou.shape[1] == 0:
                cost_iou = cost_sub_iou
            else:
                cost_iou = torch.stack(
                    (cost_sub_iou, cost_obj_iou)).max(dim=0)[0]
            # Final cost matrix
            C = self.cost_obj_class * cost_obj_class + self.cost_verb_class * cost_verb_class + \
                self.cost_bbox * cost_bbox + self.cost_iou_conf * cost_iou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v['obj_labels']) for v in targets]
            indices = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(C.split(sizes, -1))
            ]
            return [(torch.as_tensor(i, dtype=torch.int64),
                     torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices]


def build_matcher(args):
    return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class,
                               cost_verb_class=args.set_cost_verb_class,
                               cost_bbox=args.set_cost_bbox,
                               cost_iou=args.set_cost_iou)
