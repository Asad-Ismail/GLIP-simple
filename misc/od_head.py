import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from glip import TokenSigmoidFocalLoss
from torchvision.models.detection.image_list import ImageList
#import torchvision.transforms.functional as F
import torch.nn.functional as F 



class GLIPHead(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=80):
        super().__init__()
        
        # RPN setup
        anchor_sizes = ((32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        self.rpn_head = RPNHead(
            hidden_dim,  
            self.anchor_generator.num_anchors_per_location()[0]
        )
        
        self.rpn = RegionProposalNetwork(
            self.anchor_generator, self.rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 1000},
            nms_thresh=0.7
        )
        
        # ROI pooling
        self.roi_pooler = MultiScaleRoIAlign(
            featmap_names=['p2', 'p3', 'p4', 'p5'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Box head and predictor
        self.box_head = TwoMLPHead(hidden_dim * 7 * 7, hidden_dim)
        self.box_predictor = FastRCNNPredictor(hidden_dim, num_classes)
        
        self.roi_heads = RoIHeads(
            box_roi_pool=self.roi_pooler,
            box_head=self.box_head,
            box_predictor=self.box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
        )

        # Grounding head
        self.grounding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Token matching loss
        self.token_loss = TokenSigmoidFocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, x):
        features = x["visual"]  
        language_dict = x["lang"]
        image_sizes = x["original_sizes"]
        targets = x.get("targets", None)
        
        features_dict = {f'p{i+2}': feat for i, feat in enumerate(features)}
        
        # Create ImageList
        batch_size = features[0].shape[0]
        device = features[0].device
        dummy_tensor = torch.zeros(
            batch_size, 3,
            image_sizes[0][0],
            image_sizes[0][1],
            device=device
        )
        image_list = ImageList(dummy_tensor, image_sizes)
        
        # RPN forward pass
        proposals, rpn_losses = self.rpn(image_list, features_dict, targets)
        
        if self.training and targets is not None:
            matched_idxs_list = []
            with torch.no_grad():
                # Get matched indices first
                matched_idxs = self.roi_heads.select_training_samples(proposals, targets)[1]
                matched_idxs_list.extend(matched_idxs)

            # Get detection losses but zero out classification loss
            detector_losses = self.roi_heads(features_dict, proposals, image_sizes, targets)[1]
            
            # Zero out classification loss
            if 'loss_classifier' in detector_losses:
                detector_losses['loss_classifier'] = 0.0 * detector_losses['loss_classifier']
            
            # Get ROI features for grounding
            box_features = self.box_head(
                self.roi_pooler(
                    features_dict, 
                    [t["boxes"] for t in targets],
                    image_sizes
                )
            )

            grounding_features = self.grounding_head(box_features)
            text_features = language_dict['hidden']
            
            # Process each image in batch separately
            batch_losses = []
            start_idx = 0
            for i, matched_idxs in enumerate(matched_idxs_list):
                # Get positive matches for this image
                positive_idxs = (matched_idxs > 0).nonzero().squeeze(1)
                gt_box_indices = matched_idxs[positive_idxs] - 1  # -1 to convert to 0-based indexing
                
                # Get features for positive matches
                curr_grounding_features = grounding_features[start_idx:start_idx + len(targets[i]["boxes"])]
                curr_text_features = text_features[i]
                
                # Get and reorder positive map
                curr_positive_map = targets[i]['positive_map'][gt_box_indices]
                
                # Compute grounding scores
                curr_grounding_scores = torch.matmul(
                    curr_grounding_features,
                    curr_text_features.transpose(0, 1)
                )
                
                # Compute loss
                curr_loss = self.token_loss(
                    curr_grounding_scores.unsqueeze(0),
                    curr_positive_map.unsqueeze(0),
                    text_masks=language_dict.get('masks')[i:i+1] if language_dict.get('masks') is not None else None
                )
                batch_losses.append(curr_loss)
                
                start_idx += len(targets[i]["boxes"])

            grounding_loss = torch.stack(batch_losses).mean()
            # Combine all losses
            losses = {}
            losses.update(rpn_losses)
            losses.update(detector_losses)
            losses['loss_grounding'] = grounding_loss
            
            return losses
            
        else:
            # Inference mode
            detections = self.roi_heads(
                features_dict,
                proposals,
                image_sizes,
                None
            )[0]
            
            # Add grounding scores
            if len(detections[0]['boxes']) > 0:
                det_features = self.roi_pooler(
                    features_dict,
                    [d['boxes'] for d in detections],
                    image_sizes  # Add image_sizes here
                )
                det_features = self.box_head(det_features)
                grounding_features = self.grounding_head(det_features)
                text_features = language_dict['hidden']
                
                grounding_scores = torch.matmul(
                    grounding_features,
                    text_features.transpose(-2, -1)
                )
                
                for det, scores in zip(detections, grounding_scores):
                    det['grounding_scores'] = scores
                    
            return detections
        

class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x
    

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas