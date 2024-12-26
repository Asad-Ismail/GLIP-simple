import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer,BertTokenizerFast
from torchvision.ops import generalized_box_iou_loss
import timm
from typing import OrderedDict
from torchvision.models import swin_b
import math


class GLIPBackbone(nn.Module):
    def __init__(self, 
                 bert_type="bert-base-uncased",
                 max_query_len=256):
        super().__init__()
        
        # Get Swin-B model without classification head
        swin = swin_b(weights='DEFAULT')
        self.features = swin.features  
        
        channels = {
            'p1': 128,   
            'p2': 256,   
            'p3': 512,   
            'p4': 1024   
        }

        # Feature projection to common dimension (256)
        self.feature_proj = nn.ModuleDict({
            level: nn.Conv2d(dim, 256, 1) 
            for level, dim in channels.items()
        })

        # Language backbone
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_type)
        self.bert = BertModel.from_pretrained(bert_type)
        self.max_query_len = max_query_len

    def get_visual_features(self, images):
        """Get proper FPN features (P2-P6) from Swin backbone"""
        tensors, mask = images.decompose()
        valid_mask = ~mask

        input_h, input_w = tensors.shape[2:]
        print(f"\nInput size: {input_h}x{input_w}")
        
        features = {}
        feature_masks = {}
        
        x = tensors
        
        # Process through features layer by layer
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Take features from the second layer of each stage
            if i in [1, 3, 5, 7]:  
                level = f'p{(i//2)+1}' 
                
                # Convert from [B, H, W, C] to [B, C, H, W]
                feat = x.permute(0, 3, 1, 2).contiguous()
                
                # Project to common dimension
                feat = self.feature_proj[level](feat)

                curr_h, curr_w = feat.shape[2:]
                actual_stride_h = input_h // curr_h
                actual_stride_w = input_w // curr_w
                print(f"\n{level} output size: {curr_h}x{curr_w}")
                print(f"{level} stride: {actual_stride_h}x{actual_stride_w} " )
                
                # Get mask for this level
                feat_mask = F.interpolate(valid_mask[None].float(), 
                                        size=feat.shape[-2:],
                                        mode='nearest').bool()[0]
                
                # Store features and mask
                features[level] = feat * feat_mask.unsqueeze(1).float()
                feature_masks[level] = feat_mask
        
        # Add P5
        #p5 = F.max_pool2d(features['p4'], kernel_size=2, stride=2)
        #p5_mask = F.interpolate(feature_masks['p4'][None].float(), 
        #                      size=p5.shape[-2:],
        #                      mode='nearest')[0].bool()
        #features['p5'] = p5 * p5_mask.unsqueeze(1).float()
        #feature_masks['p5'] = p5_mask
        
        return features, feature_masks

    def get_text_features(self, captions, device):
        """Process text through BERT"""
        # Tokenize
        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_query_len,
            return_tensors="pt"
        ).to(device)
        
        # Get BERT features
        text_outputs = self.bert(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            return_dict=True
        )
        
        return {
            'hidden': text_outputs.last_hidden_state,
            'masks': tokens.attention_mask,
            'input_ids': tokens.input_ids
        }

    def forward(self, images, original_sizes, captions):
        """
        Args:
            images: NestedTensor with tensors and mask
            original_sizes: original image sizes before preprocessing
            captions: list of text queries
        """
        device = images.tensors.device
        
        # 1. Get visual features and their masks
        visual_feats, visual_masks = self.get_visual_features(images)
        
        # 2. Process text features
        text_feats = self.get_text_features(captions, device)
        
        # 3. Create output dict with proper structure
        return {
            'visual': visual_feats,
            'visual_masks': visual_masks,  # Add masks for attention
            'language': text_feats,
            'original_sizes': original_sizes
        }

class VLDyHead(nn.Module):
    def __init__(self, num_convs=8, hidden_dim=256, in_channels=256):
        super().__init__()
        
        # Build tower layers
        dyhead_tower = []
        channels = hidden_dim
        # Do processing in this manner Fusion -> Language -> Vision
        
        for i in range(num_convs):
            # 1. Add cross-modality fusion layer
            dyhead_tower.append(VLFuse(hidden_dim=hidden_dim))
            
            # 2. Add language self-attention layer
            dyhead_tower.append(BertEncoderLayer())

            # 3. Add vision path (DyConv)
            curr_in_channels = in_channels if i == 0 else channels
            dyhead_tower.append(
                DyConv(
                    in_channels=curr_in_channels,
                    out_channels=channels
                )
            )
        
        self.dyhead_tower = nn.Sequential(*dyhead_tower)
        # Final projection for language features from 768 to 256
        self.final_lang_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, features):
        """
        Args:
            features: dict containing
                - visual: List[Tensor] of FPN features
                - lang: dict with 'hidden' and 'masks' 
        """
        x=self.dyhead_tower(features)
        x["lang"]["hidden"] = self.final_lang_proj(x["lang"]["hidden"])
        return x

class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim=256, l_dim=768, embed_dim=2048, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Project to common dimension with correct scaling
        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.values_v_proj = nn.Linear(v_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        
        # Project back
        self.out_v_proj = nn.Linear(embed_dim, v_dim)
        self.out_l_proj = nn.Linear(embed_dim, l_dim)
        
        # Initialize projections with small weights for stability
        self._reset_parameters()
        
    def _reset_parameters(self):
        for proj in [self.v_proj, self.l_proj, self.values_v_proj, self.values_l_proj]:
            fan_out = proj.weight.size(0)
            std = (2.0 / fan_out) ** 0.5
            nn.init.normal_(proj.weight, mean=0, std=std)
            nn.init.zeros_(proj.bias)
            
    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        B, Hv, _ = v.shape
        _, Hl, _ = l.shape
        
        # 1. Project to common dimension with scaling
        q = self.v_proj(v) * self.scaling 
        k = self.l_proj(l)
        v_value = self.values_v_proj(v)
        l_value = self.values_l_proj(l)
        
        # 2. Reshape
        q = q.view(B, Hv, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Hl, self.num_heads, self.head_dim).transpose(1, 2)
        v_value = v_value.view(B, Hv, self.num_heads, self.head_dim).transpose(1, 2)
        l_value = l_value.view(B, Hl, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Attention with mask handling
        attn = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply language mask (B, L) -> (B, 1, 1, L)
        if attention_mask_l is not None:
            attn_mask_l = attention_mask_l.bool()
            attn_mask_l = attn_mask_l.unsqueeze(1).unsqueeze(1)
            attn_mask_l = attn_mask_l.expand(B, self.num_heads, Hv, -1)
            attn = attn.masked_fill(~attn_mask_l, -1e4)
        
        # Apply vision mask (B, H, W) -> (B, 1, HW, 1)
        if attention_mask_v is not None:
            attn_mask_v = attention_mask_v.bool()
            attn_mask_v = attn_mask_v.view(B, 1, Hv, 1)
            attn_mask_v = attn_mask_v.expand(B, self.num_heads, -1, Hl)
            attn = attn.masked_fill(~attn_mask_v, -1e4)
            
        # 4. Normalized attention scores
        attn_v = F.softmax(attn, dim=-1)
        attn_l = F.softmax(attn.transpose(-2, -1), dim=-1)
        
        # 5. Compute outputs
        out_v = torch.matmul(attn_v, l_value)
        out_l = torch.matmul(attn_l, v_value)
        
        # 6. Reshape and project back
        out_v = out_v.transpose(1, 2).reshape(B, Hv, -1)
        out_l = out_l.transpose(1, 2).reshape(B, Hl, -1)
        
        return self.out_v_proj(out_v), self.out_l_proj(out_l)

class BiAttentionBlock(nn.Module):
    def __init__(self, v_dim=256, l_dim=768, embed_dim=2048, num_heads=8, dropout=0.1, init_values=1e-4):
        super().__init__()
        
        # Layer norms 
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        
        # Cross attention with proper scaling
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer scale init 
        self.gamma_v = nn.Parameter(torch.ones(v_dim) * 0.125)
        self.gamma_l = nn.Parameter(torch.ones(l_dim) * 0.125)
    
    def forward(self, visual_feats, visual_masks, lang_feats, lang_masks):
        fused_visual = []
        
        for feat, feat_mask in zip(visual_feats, visual_masks):
            # 1. Prepare visual features
            bs, c, h, w = feat.shape
            v = feat.flatten(2).transpose(1, 2)  # [B, HW, 256]
            v_mask = feat_mask.flatten(1)  # [B, HW]
            
            # 2. Pre-norm
            v = self.layer_norm_v(v)
            l = self.layer_norm_l(lang_feats)
            
            # 3. Cross attention with masks
            delta_v, delta_l = self.attn(v, l, 
                                       attention_mask_v=v_mask,
                                       attention_mask_l=lang_masks)
            
            # 4. Layer scale and residual
            v = v + self.gamma_v * delta_v
            l = lang_feats + self.gamma_l * delta_l
            
            # 5. Reshape visual and maintain mask
            v = v.transpose(1, 2).reshape(bs, c, h, w)
            fused_visual.append(v)
            
        return fused_visual, l

class VLFuse(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.bi_attn = BiAttentionBlock(
            v_dim=hidden_dim,
            l_dim=768,
            embed_dim=2048,
            num_heads=8,
            dropout=0.1,
            init_values=1e-4
        )
    
    def forward(self, x):
        visual_feats = x["visual"]
        visual_masks = x["visual_masks"]
        lang_dict = x["lang"]

        # Do cross-modal fusion with masks
        fused_visual, fused_lang = self.bi_attn(
            visual_feats,
            visual_masks,
            lang_dict["hidden"],
            lang_dict["masks"]
        )
        
        return {
            "visual": fused_visual,
            "visual_masks": visual_masks,  # Pass through masks
            "lang": {
                "hidden": fused_lang,
                "masks": lang_dict["masks"]
            }
        }


class BertAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Pre-norm and projections
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights
        self._reset_parameters()
        
    def _reset_parameters(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
            nn.init.zeros_(proj.bias)
            
    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.shape
        
        # Check input for NaN
        #if torch.isnan(x).any():
        #    print("NaN in attention input")
        #    return x
            
        x = self.norm(x)
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape heads
        def reshape_for_attention(x):
            return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            
        q = reshape_for_attention(q)
        k = reshape_for_attention(k)
        v = reshape_for_attention(v)
        
        # Compute scaled attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply attention mask
        if mask is not None:
            attn_weights = attn_weights + mask
            
        # Stable softmax with clipping
        attn_weights = torch.clamp(attn_weights, min=-1e4, max=1e4)
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        context = torch.matmul(attn_probs, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(bsz, seq_len, self.hidden_size)
        
        # Final projection
        output = self.out_proj(context)
        output = self.dropout(output)

        return output
    

class BertEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Pre-norms (more stable than post-norm)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Attention
        self.attn = BertAttention(hidden_size, dropout=dropout)
        
        # FFN with smaller initial weights
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),  # Extra dropout for stability
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Scaled residual connections
        self.gamma1 = nn.Parameter(torch.ones(hidden_size) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(hidden_size) * 0.1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize with smaller weights
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        visual_feats = x["visual"]
        visual_masks = x["visual_masks"]  # Get masks
        lang_dict = x["lang"]
        
        # Get inputs
        hidden = lang_dict["hidden"]

        #if torch.isnan(hidden).any():
        #    print("NaN in attention input")
        #    return x

        mask = None
        if "masks" in lang_dict:
            # Stable mask creation
            mask = lang_dict["masks"].float()
            mask = (1.0 - mask)[:, None, None, :] 
            mask = mask.masked_fill(mask.bool(), -1e4)  # Use smaller value than inf
        
        # Pre-norm + scaled residual for attention
        normed = self.norm1(hidden)
        hidden = hidden + self.gamma1.unsqueeze(0).unsqueeze(0) * self.attn(normed, mask)
        
        # Pre-norm + scaled residual for FFN
        normed = self.norm2(hidden)
        hidden = hidden + self.gamma2.unsqueeze(0).unsqueeze(0) * self.ffn(normed)
        
        return {
            "visual": visual_feats,
            "visual_masks": visual_masks,  # Pass through masks
            "lang": {"hidden": hidden, "masks": lang_dict["masks"]}
        }
        

class DyConv(nn.Module):
    """Vision path with dynamic convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        visual_feats = x["visual"]
        visual_masks = x["visual_masks"]  
        lang_dict = x["lang"]
        
        # Process each FPN level
        processed = []
        for feat in visual_feats:
            out = self.relu(self.bn(self.conv(feat)))
            processed.append(out)
            
        return {
            "visual": processed,
            "visual_masks": visual_masks,  # Pass through masks
            "lang": lang_dict              # Pass through
        }

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
        
    def forward(self, x):
        return x * self.scale
    

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
        loss = loss_func(logits, targets, self.alpha, self.gamma, text_masks, kwargs)
        return loss.sum()

    def repr(self):
        tmpstr =  "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
        
        
class GLIPHead(nn.Module):
    """GLIP detection and grounding head with anchor-free detection"""
    def __init__(self, hidden_dim=256, num_classes=80):
        super().__init__()
        
        # Detection heads
        self.bbox_head = nn.Conv2d(hidden_dim, 4, 3, padding=1)  # Box regression
        self.center_head = nn.Conv2d(hidden_dim, 1, 3, padding=1)  # Centerness
        
        # Grounding heads
        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_text = nn.Linear(hidden_dim, hidden_dim)
        
        # Parameters for stability
        self.log_scale = nn.Parameter(torch.Tensor([1.0]))
        self.bias_lang = nn.Parameter(torch.zeros(hidden_dim))
        self.bias0 = nn.Parameter(torch.Tensor([-2.0]))  # Initial bias for better convergence

        # Loss functions
        self.token_loss = TokenSigmoidFocalLoss(alpha=0.25, gamma=2.0)
        
    def forward(self, features, language_dict_features):
        """
        Args:
            features: List of FPN features after fusion
            language_dict_features: Dict with text embeddings and masks
        """
        batch_size = features[0].shape[0]
        
        # Get language embeddings
        text_embeds = language_dict_features['hidden']
        text_masks = language_dict_features['masks']
        
        # Normalize language embeddings
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(text_embeds / 2.0)
        dot_product_proj_tokens_bias = torch.matmul(text_embeds, self.bias_lang) + self.bias0

        predictions = []
        for feat_level in features:
            # Detection outputs
            boxes = self.bbox_head(feat_level)
            centers = self.center_head(feat_level)
            
            # Grounding outputs
            B, C, H, W = feat_level.shape
            img_embeds = self.dot_product_projection_image(feat_level)
            img_embeds = img_embeds.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
            
            # Compute grounding scores
            img_embeds = F.normalize(img_embeds, p=2, dim=-1)
            grounding_scores = torch.matmul(img_embeds, dot_product_proj_tokens.transpose(-1, -2))
            grounding_scores = grounding_scores / self.log_scale.exp()
            
            # Add bias
            bias = dot_product_proj_tokens_bias.unsqueeze(1).expand(-1, H*W, -1)
            grounding_scores = grounding_scores + bias
            
            # Clamp for stability
            grounding_scores = torch.clamp(grounding_scores, min=-50000, max=50000)
            
            predictions.append({
                'boxes': boxes,
                'centers': centers,
                'grounding': grounding_scores,
            })
            
        return predictions


def compute_losses(predictions, targets, text_masks=None):
    """Compute GLIP losses outside model forward pass"""
    loss_bbox = 0
    loss_centerness = 0 
    loss_grounding = 0
    
    for pred_level, target_level in zip(predictions, targets):
        # Box regression loss using GIoU
        loss_bbox += generalized_box_iou_loss(
            pred_level['boxes'],
            target_level['boxes'],
            reduction='mean'
        )
        
        # Centerness loss
        loss_centerness += F.binary_cross_entropy_with_logits(
            pred_level['centers'],
            target_level['centers'],
            reduction='mean'
        )
        
        # Grounding loss using token focal loss
        loss_grounding += token_sigmoid_binary_focal_loss(
            pred_level['grounding'],
            target_level['positive_map'],
            alpha=0.25,
            gamma=2.0,
            text_mask=text_masks
        )
    
    return {
        'loss_bbox': loss_bbox,
        'loss_centerness': loss_centerness,
        'loss_grounding': loss_grounding
    }


# Example usage:
if __name__ == '__main__':
    backbone = GLIPBackbone()
    
    # Batch with different sizes
    batch_size = 2
    images = torch.randn(batch_size, 3, 800, 1200)  # Resized images
    original_sizes = [(1080, 1920), (720, 1280)]    # Original sizes before resize
    captions = [
        "A person riding a bicycle",
        "A dog playing in the park" 
    ]
    
    outputs = backbone(images, original_sizes, captions)
    
    # Print shapes
    for level, feat in outputs['visual']['features'].items():
        print(f"{level} shape:", feat.shape)
        
    print("Text hidden shape:", outputs['language']['hidden'].shape)
    print("Text mask shape:", outputs['language']['masks'].shape)
