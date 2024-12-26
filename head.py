import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class GLIPHead(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=80):
        super().__init__()
        
        # RPN setup
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
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
        

        # ROI heads
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

    def forward(self, x):
        # features is a list
        features = x["visual"]
        language_dict = x["lang"]
        image_sizes = x["original_sizes"]

        features_dict = {str(i): feat for i, feat in enumerate(features)}

        # RPN forward pass
        proposals, _ = self.rpn(
            images=None,
            features=features_dict,
            targets=None,
            image_sizes=image_sizes
        )

        # ROI head forward pass
        detections = self.roi_heads(
            features_dict,
            proposals,
            image_sizes,
            None
        )[0]  # Get detections
        
        # Add grounding scores for detected boxes
        if len(detections[0]['boxes']) > 0:
            det_features = self.roi_pooler(
                features_dict,
                [d['boxes'] for d in detections]
            )
            det_features = self.box_head(det_features)
            grounding_features = self.grounding_head(det_features)
            text_features = language_dict['hidden']
            
            grounding_scores = torch.matmul(
                grounding_features,
                text_features.transpose(-2, -1)
            )
            
            # Add grounding scores to detections
            for det, scores in zip(detections, grounding_scores):
                det['grounding_scores'] = scores
                
        return detections
    

# Helper classes from torchvision
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