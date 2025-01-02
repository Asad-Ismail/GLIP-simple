import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GLIPHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=80):
        super().__init__()
        num_anchors = 1
        self.num_classes = num_classes
        
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(3)])  
        
        # Bert language dimensions
        lang_dim = 768
        self.dot_product_projection_text = nn.Linear(lang_dim, num_anchors * in_channels, bias=True)
        self.dot_product_projection_image = nn.Conv2d(in_channels, num_anchors * in_channels, kernel_size=1)
        
        #Initialization from GLIP paper
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(lang_dim), requires_grad=True)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.cls_logits, self.bbox_pred, self.centerness]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Initialize cls_logits bias
        # Makes the model initially predict a low probability for all classes,
        # which is desirable in object detection where most locations don't contain objects
        prior_pob=0.01
        bias_value = -math.log((1 - prior_pob) / prior_pob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        features = x['visual']
        embedding = x['lang']['hidden']
        
        logits = []
        bbox_reg = []
        centerness = []
        dot_product_logits = []

        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0

        for l, feature in enumerate(features):
            
            logits.append(self.cls_logits(feature))
            bbox_pred = self.scales[l](self.bbox_pred(feature))
            bbox_reg.append(bbox_pred)
            centerness.append(self.centerness(feature))

            B, C, H, W = feature.shape
            dot_product_proj_queries = self.dot_product_projection_image(feature)
            dot_product_proj_queries = dot_product_proj_queries.view(B, -1, C, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, C)
            
            A = dot_product_proj_queries.shape[1]
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)
            dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias
            dot_product_logit = torch.clamp(dot_product_logit, max=50000, min=-50000)
            
            dot_product_logits.append(dot_product_logit)

        return logits, bbox_reg, centerness, dot_product_logits

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale