import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torchvision.ops import generalized_box_iou_loss
import timm

class GLIPBackbone(nn.Module):
    """Backbone with Swin from timm and BERT"""
    def __init__(self, 
                 swin_type='swin_base_patch4_window7_224',  # or swin_large
                 bert_type="bert-base-uncased",
                 max_query_len=256):
        super().__init__()
        
        # Visual backbone from timm
        self.swin = timm.create_model(
            swin_type,
            pretrained=True,
            features_only=True,
            out_indices=(0,1, 2, 3) # Features 
        )
        
        # Language backbone
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.bert = BertModel.from_pretrained(bert_type)
        self.max_query_len = max_query_len

    def get_visual_features(self, images, original_sizes):
        """Get feature pyramid from Swin"""
        # Get hierarchical features
        features = self.swin(images)
        
        # Create P6 by pooling P5
        p6 = F.max_pool2d(features[-1], kernel_size=2, stride=2)
        
        # Return ordered dict of features
        feature_dict = {
            'p2': features[0],  # 1/4
            'p3': features[1],  # 1/8
            'p4': features[2],  # 1/16
            'p5': features[3],  # 1/32
            'p6': p6           # 1/64
        }

        return {
            'features': feature_dict,
            'original_sizes': original_sizes
        }

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
        device = images.device
        visual_feats = self.get_visual_features(images, original_sizes)
        text_feats = self.get_text_features(captions, device)
        
        return {
            'visual': visual_feats,
            'language': text_feats
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
            dyhead_tower.append(BertEncoderLayer(hidden_dim))

            # 3. Add vision path (DyConv)
            curr_in_channels = in_channels if i == 0 else channels
            dyhead_tower.append(
                DyConv(
                    in_channels=curr_in_channels,
                    out_channels=channels
                )
            )
        
        self.dyhead_tower = nn.Sequential(*dyhead_tower)
        
    def forward(self, features):
        """
        Args:
            features: dict containing
                - visual: List[Tensor] of FPN features
                - lang: dict with 'hidden' and 'masks' 
        """
        return self.dyhead_tower(features)


class BiMultiHeadAttention(nn.Module):
    """Official GLIP bi-directional attention implementation"""
    def __init__(self, v_dim=256, l_dim=768, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        
        # Projections
        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.values_v_proj = nn.Linear(v_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, v_dim)
        self.out_l_proj = nn.Linear(embed_dim, l_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, _ = v.size()
        src_len = l.size(1)
        
        # Project and reshape
        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)
        
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)
        
        # Compute attention
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        if attention_mask_l is not None:
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        # Bi-directional attention
        attn_weights_v = F.softmax(attn_weights, dim=-1)
        attn_weights_l = F.softmax(attn_weights.transpose(1, 2), dim=-1)
        
        # Apply attention
        attn_output_v = torch.bmm(self.dropout(attn_weights_v), value_l_states)
        attn_output_l = torch.bmm(self.dropout(attn_weights_l), value_v_states)
        
        # Reshape back
        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        
        attn_output_v = attn_output_v.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output_l = attn_output_l.transpose(1, 2).reshape(bsz, src_len, self.embed_dim)
        
        # Final projections
        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)
        
        return attn_output_v, attn_output_l

class BiAttentionBlock(nn.Module):
    def __init__(self, v_dim=256, l_dim=768, embed_dim=256, num_heads=8, dropout=0.1, init_values=1e-4):
        super().__init__()
        
        # Pre-norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        
        # Attention
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer scale parameters
        self.gamma_v = nn.Parameter(init_values * torch.ones(v_dim))
        self.gamma_l = nn.Parameter(init_values * torch.ones(l_dim))
        
    def forward(self, visual_feats, lang_feats, lang_masks):
        # Process each visual level with language
        fused_visual = []
        
        for feat in visual_feats:
            bs, _, h, w = feat.shape
            v = feat.flatten(2).transpose(1, 2)  # [B, HW, C]
            
            # Pre-norm
            v = self.layer_norm_v(v)
            l = self.layer_norm_l(lang_feats)
            
            # Bi-attention
            delta_v, delta_l = self.attn(v, l, attention_mask_l=lang_masks)
            
            # Residual
            v = v + self.gamma_v * delta_v
            l = l + self.gamma_l * delta_l
            
            # Reshape back
            v = v.transpose(1, 2).reshape(bs, -1, h, w)
            fused_visual.append(v)
            
        return fused_visual, l


class VLFuse(nn.Module):
    """Cross-modality fusion module"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.bi_attn = BiAttentionBlock(hidden_dim)
        
    def forward(self, x):
        visual_feats = x["visual"]
        lang_dict = x["lang"]
        
        fused_visual, fused_lang = self.bi_attn(
            visual_feats,
            lang_dict["hidden"],
            lang_dict["masks"]
        )
        
        return {
            "visual": fused_visual,
            "lang": {
                "hidden": fused_lang,
                "masks": lang_dict["masks"]
            }
        }

class BertEncoderLayer(nn.Module):
    """Language self-attention layer"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, 8)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        visual_feats = x["visual"]
        lang_dict = x["lang"]
        
        # Language self-attention
        lang_feats = lang_dict["hidden"]
        lang_masks = lang_dict["masks"]
        
        new_lang = self.self_attn(
            lang_feats, 
            lang_feats,
            lang_feats,
            key_padding_mask=~lang_masks
        )[0]
        new_lang = self.norm(new_lang + lang_feats)  # Add residual
        
        return {
            "visual": visual_feats,  # Pass through
            "lang": {
                "hidden": new_lang,
                "masks": lang_masks
            }
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
        lang_dict = x["lang"]
        
        # Process each FPN level
        processed = []
        for feat in visual_feats:
            out = self.relu(self.bn(self.conv(feat)))
            processed.append(out)
            
        return {
            "visual": processed,
            "lang": lang_dict  # Pass through
        }


class GLIPHead(nn.Module):
    """GLIP detection and grounding head using ATSS/FCOS style detection"""
    def __init__(self, hidden_dim=256, num_classes=80, strides=[8, 16, 32, 64, 128]):
        super().__init__()
        self.strides = strides  # FPN strides
        
        # Shared head convs
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Detection heads
        self.bbox_head = nn.Conv2d(hidden_dim, 4, 3, padding=1)  # Box regression (t,l,b,r)
        self.center_head = nn.Conv2d(hidden_dim, 1, 3, padding=1)  # Centerness
        
        # Grounding heads
        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_text = nn.Linear(hidden_dim, hidden_dim)
        
        # Scale factors for each FPN level
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in strides])

    def forward_single_level(self, x, stride_idx, text_embeds):
        """Forward pass for single FPN level"""
        features = self.head(x)
        B, C, H, W = features.shape
        
        # Get centers for each location
        shifts_x = torch.arange(0, W * stride, step=stride, dtype=torch.float32, device=x.device)
        shifts_y = torch.arange(0, H * stride, step=stride, dtype=torch.float32, device=x.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        locations = torch.stack([shift_x, shift_y], dim=0)  # [2, H, W]
        
        # Box regression relative to center locations
        pred_boxes = self.scales[stride_idx](self.bbox_head(features))  # [B, 4, H, W]
        pred_centers = self.center_head(features)  # [B, 1, H, W]
        
        # Grounding predictions
        img_embeds = self.dot_product_projection_image(features)
        img_embeds = img_embeds.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        img_embeds = F.normalize(img_embeds, p=2, dim=-1)
        
        # Compute grounding scores
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        grounding_scores = torch.matmul(img_embeds, text_embeds.transpose(-1, -2))
        
        return {
            'boxes': pred_boxes,
            'centers': pred_centers,
            'grounding': grounding_scores,
            'locations': locations
        }

    def forward(self, features, language_dict_features):
        """
        Args:
            features: List[Tensor] of FPN features P2-P6
            language_dict_features: Dict with text embeddings and masks
        """
        text_embeds = language_dict_features['hidden']
        predictions = []
        
        for level_idx, (feature, stride) in enumerate(zip(features, self.strides)):
            pred = self.forward_single_level(
                feature, 
                level_idx,
                text_embeds
            )
            predictions.append(pred)
            
        return predictions


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

    def loss(self, predictions, targets):
        """Compute detection and grounding losses"""
        loss_bbox = 0
        loss_center = 0
        loss_grounding = 0
        
        for pred_level, target_level in zip(predictions, targets):
            # GIoU loss - using torchvision's implementation
            loss_bbox += generalized_box_iou_loss(
                pred_level['boxes'],
                target_level['boxes'],
                reduction='mean'
            )
            
            # Center loss
            loss_center += F.binary_cross_entropy_with_logits(
                pred_level['centers'],
                target_level['centers'],
                reduction='mean'
            )
            
            # Grounding loss
            loss_grounding += self.token_loss(
                pred_level['grounding'],
                target_level['grounding'],
                text_masks=target_level.get('text_masks')
            )
            
        return {
            'loss_bbox': loss_bbox,
            'loss_center': loss_center,
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
