# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_detr_encoder
import copy

from models.ops.modules import MSDeformAttn
from .deformable_transformer import DeformableCrossAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DeformableEncoder(nn.Module):
    def __init__(self, backbone, transformer, num_queries, num_feature_levels, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
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
            self.input_proj = nn.ModuleList([nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1)])
        self.backbone = backbone
        self.two_stage = two_stage
        self.hidden_dim = hidden_dim

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)


    def forward(self, samples: NestedTensor):
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
            src, mask = feat.decompose()
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

        # query_embeds = None
        # if not self.two_stage:
        #     query_embeds = self.query_embed.weight
        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        memory, spatial_shapes, valid_ratios, mask_flatten, lvl_pos_embed_flatten = self.transformer(srcs, mask, pos)

        return memory, spatial_shapes, valid_ratios, mask_flatten, lvl_pos_embed_flatten # memory has shape B x (sum of spatial dims) x C

class TransCenter(nn.Module):
    def __init__(self, backbone, transformer, num_queries, num_feature_levels, two_stage=False):
        super().__init__()

        self.encoder = DeformableEncoder(backbone, transformer, num_queries, num_feature_levels, two_stage)
        hidden_dim = self.encoder.hidden_dim

        self.QLN_1 = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)
        # self.QLN_2 = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2)
        # TODO: now only have detection. support tracking later

        self.cross_attn = DeformableCrossAttention(num_layers=1)

        detection_decoder_convs = []
        for _ in range(num_feature_levels):
            in_channels = backbone.num_channels[_]
            detection_decoder_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True)
            ))
        self.detection_decoder_convs = nn.ModuleList(detection_decoder_convs)
        self.detection_decoder_final_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 3, kernel_size=1), # one channel for objectness, two channels for size
            )

    def forward(self, samples: NestedTensor):
        # process by encoder
        memory, spatial_shapes, valid_ratios, mask_flatten, lvl_pos_embed_flatten = self.encoder(samples)

        # do detector branch first
        DQ_t = self.QLN_1(memory.transpose(1, 2)).transpose(1, 2) # B x (sum of spatial dims) x C

        # DQ_t is the query. do cross_attn with the memory
        DF_t = self.cross_attn(DQ_t, memory, spatial_shapes, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # reshape DF_t back into multiple layers
        detection_features = []
        N_, S_, C_ = DF_t.shape
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            DF_flatten_ = DF_t[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, C_)
            DF_flatten_ = DF_flatten_.permute(0, 3, 1, 2) # B x C x H x W

            detection_features.append(DF_flatten_)

            _cur += (H_ * W_)

        # final processing
        # TODO: use deformable conv. now only use normal conv for simplicity
        final_feature = None
        for lvl, detection_feature in enumerate(detection_features[::-1]): # iter the low-res feature first
            if lvl != 0:
                final_feature = self._upsample_add(final_feature, detection_feature)
                final_feature = self.detection_decoder_convs[lvl-1](final_feature) # conv
            else:
                final_feature = detection_feature

        final_feature = F.interpolate(final_feature, scale_factor=2)
        output = self.detection_decoder_final_conv(final_feature)

        return output

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

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


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_detr_encoder(args)
    model = TransCenter(
        backbone, 
        transformer, 
        num_queries=args.num_queries, 
        num_feature_levels=args.num_feature_levels,
        two_stage=args.two_stage)

    return model

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")