import torch
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
import math
from utils import concat_box_prediction_layers
from boxlist_ops import cat_boxlist
from boxlist_ops import boxlist_iou

INF = 1e8


class BoxCoder(object):

    def __init__(self):
        super().__init__()

    def encode(self, gt_boxes, anchors):
        TO_REMOVE = 1  # TODO remove
        ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
        gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        anchors = anchors.to(preds.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = preds[:, 2::4] / ww
        dh = preds[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)

        return pred_boxes


def token_sigmoid_binary_focal_loss(pred_logits, targets, alpha, gamma, text_mask=None):
    # binary version of focal loss
    # copied from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
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
        Loss tensor with the reduction option applied.
    """
    assert (targets.dim() == 3)
    assert (pred_logits.dim() == 3)  # batch x from x to

    bs, n, _ = pred_logits.shape
    if text_mask is not None:
        assert (text_mask.dim() == 2)
        text_mask = (text_mask > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, pred_logits.size(1), 1)  # copy along the image channel dimension
        pred_logits = torch.masked_select(pred_logits, text_mask)
        targets = torch.masked_select(targets, text_mask)

        # print(pred_logits.shape)
        # print(targets.shape)

    p = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss

class TokenSigmoidFocalLoss(nn.Module):

    def __init__(self, alpha, gamma):
        super(TokenSigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, text_masks=None, version="binary", **kwargs):
        if version == "binary":
            loss_func = token_sigmoid_binary_focal_loss
        else:
            raise NotImplementedError
        loss = loss_func(logits, targets, self.alpha, self.gamma, text_masks)
        return loss.sum()

    def repr(self):
        tmpstr =  "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
        

class GLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_coder = BoxCoder()
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.token_loss = TokenSigmoidFocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, logits, bbox_reg, centerness, dot_product_logits, targets, anchors, captions):
        # Pass only boxes for preparing targets
        labels, reg_targets, token_labels = self.prepare_targets([item['boxes'] for item in targets], anchors, captions)

        N = len(labels)

        box_regression_flatten, box_cls_flatten, _ =  concat_box_prediction_layers(
            bbox_reg,
            logits,
            None,
        )
        
        dot_product_logits = torch.cat(dot_product_logits, dim=1)

        centerness_flatten = [ct.permute(0, 2, 3, 1).reshape(N, -1, 1) for ct in centerness]
        centerness_flatten = torch.cat(centerness_flatten, dim=1).reshape(-1)

        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)
        token_labels_stacked = torch.stack(token_labels, dim=0)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)


        cls_loss = self.focal_loss(box_cls_flatten, labels_flatten.int()) 

        dot_product_token_loss = self.token_loss(dot_product_logits,
                                                                token_labels_stacked, text_masks=text_masks,
                                                                version="binary")
        
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        anchors_flatten = anchors_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
            
        
        centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)
        
        reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                        weight=centerness_targets) 
                                    
        centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets)
        
        
        losses = {
            "loss_cls": cls_loss,
            "loss_reg": reg_loss * 2.0,
            "loss_centerness": centerness_loss,
            "loss_dot_product_token": dot_product_token_loss
        }
        
        return losses

    def prepare_targets(self, targets, anchors, tokenized=None, positive_map=None, proj_tokens=None):
        cls_labels = []
        reg_targets = []
        token_labels = []

        offset = 0
        TOPK =9  # for matching x numbers of anchors to gts

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            # bboxes_per_im = targets_per_im.get_field("boxes")
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            num_gt = len(bboxes_per_im)

            if positive_map is not None:
                token_per_im = positive_map[offset:offset + num_gt, :].to(anchors[0][0].bbox.device)
                offset += num_gt


            anchors_per_im = cat_boxlist(anchors[im_i])

            num_anchors_per_loc = len((1.0,)) 
            num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]
            ious = boxlist_iou(anchors_per_im, targets_per_im)

            gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
            gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

            anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
            anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

            # Selecting candidates based on the center distance between anchor box and object
            candidate_idxs = []
            star_idx = 0
            for level, anchors_per_level in enumerate(anchors[im_i]):
                end_idx = star_idx + num_anchors_per_level[level]
                distances_per_level = distances[star_idx:end_idx, :]
                topk = min(self.cfg.MODEL.ATSS.TOPK * num_anchors_per_loc, num_anchors_per_level[level])
                _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + star_idx)
                star_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            # Limiting the final positive samples’ center to object
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                candidate_idxs[:, ng] += ng * anchor_num
            e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            candidate_idxs = candidate_idxs.view(-1)
            l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
            t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
            r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
            b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
            ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
            index = candidate_idxs.view(-1)[is_pos.view(-1)]
            ious_inf[index] = ious.t().contiguous().view(-1)[index]
            ious_inf = ious_inf.view(num_gt, -1).t()

            anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
            # get positive anchors index from ATSS
            positive_index = [i[0].item() for i in torch.nonzero(anchors_to_gt_indexs)]
            cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            cls_labels_per_im[anchors_to_gt_values == -INF] = 0

            if positive_map is not None:
                token_labels_per_im = token_per_im[anchors_to_gt_indexs]
                unmatched_labels = torch.zeros(token_labels_per_im.shape[1], device=token_labels_per_im.device)
                # TODO: temporarially disable the [NoObj] token logic, and only restrict to binary loss
                unmatched_labels[-1] = 1  # token: none object - > 256
                token_labels_per_im[anchors_to_gt_values == -INF] = unmatched_labels
                # move from cpu to gpu
                token_labels_per_im = token_labels_per_im.to(cls_labels_per_im.device)

                # print(token_labels_per_im[anchors_to_gt_values == -INF].shape)
                # print(cls_labels_per_im[anchors_to_gt_values != -INF][0])
                # print(token_labels_per_im[anchors_to_gt_values != -INF][0].nonzero())

            matched_gts = bboxes_per_im[anchors_to_gt_indexs]

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

            token_labels.append(token_labels_per_im)


    def focal_loss(self, inputs, targets):
        return sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='sum')


    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def centerness_loss(self, pred_centerness, labels, reg_targets, anchors):
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        centerness_targets = self.compute_centerness_targets(reg_targets[pos_inds], anchors[pos_inds])
        pred_centerness_pos = pred_centerness[pos_inds]
        return self.centerness_loss_func(pred_centerness_pos, centerness_targets)

    def compute_centerness_targets(self, reg_targets, anchors):
        gts = self.box_coder.decode(reg_targets, anchors)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * 
                                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        return centerness