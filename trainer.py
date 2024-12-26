import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Tuple, Dict, List
from transformers import BertTokenizer
import os
from utils import build_id2posspan_and_caption, create_positive_map_from_span, RandomResize,ToTensor,Normalize,Compose,build_captions_and_token_span
from utils import nested_tensor_from_tensor_list,prepare_batch,convert_od_to_grounding_data
from glip import GLIPBackbone,VLDyHead,compute_losses
from head import GLIPHead

class COCOGLIPDataset(Dataset):
    def __init__(self, 
                 coco_path: str,
                 image_dir: str,
                 tokenizer,
                 transforms=None,
                 num_negatives=2,
                 max_query_len=256,
                 add_task_prompt=False):
        super().__init__()
        self.coco = COCO(coco_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.num_negatives = num_negatives
        self.add_task_prompt = add_task_prompt
        
        # Get category information and mapping
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.ind_to_class = {cat['id']: cat['name'] for cat in self.categories}
        
        # Get image ids
        self.image_ids = list(self.coco.imgs.keys())
        
        # Transform pipeline
        if transforms is None:
            self.transforms = Compose([
                RandomResize([800], max_size=400),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms



    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        h, w = image.shape[0:2]
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Process boxes and categories
        boxes = []
        categories = []
        str_cls_lst = []
        for ann in annotations:
            x, y, w_box, h_box = ann['bbox']
            # Convert to center format [cx,cy,w,h]
            cx = x + w_box/2
            cy = y + h_box/2
            boxes.append([cx, cy, w_box, h_box])
            categories.append(ann['category_id'])
            cat_name = self.coco.loadCats([ann['category_id']])[0]['name']
            str_cls_lst.append(cat_name)
        
        # Create initial target dict
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'size': torch.as_tensor([int(h), int(w)]),
            'labels': torch.tensor(categories, dtype=torch.long),
            'image_id': image_id,
            'str_cls_lst': str_cls_lst,
        }
        
        # Apply transforms to both image and target
        image_tensor, target = self.transforms(image_source, target)
        
        target = convert_od_to_grounding_data(
            target,
            self.tokenizer,
            self.ind_to_class,
            self.max_query_len,
            self.num_negatives,
            self.add_task_prompt,
        )

        return image_tensor, [target]

    def __len__(self):
        return len(self.image_ids)
    

class GLIP(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=None):
        super().__init__()
        # For COCO, num_classes is 80 (91 total with background, but we use 80 for detection)
        if num_classes is None:
            num_classes = 80  # COCO default
        
        self.backbone = GLIPBackbone()
        self.dyhead = VLDyHead(hidden_dim=hidden_dim)
        self.head = GLIPHead(hidden_dim=hidden_dim, num_classes=num_classes)
        
    def forward(self, images, original_sizes, captions):
        """Forward pass without loss computation"""
        # Get backbone features
        features = self.backbone(images, original_sizes, captions)
        
        # Process through dynamic head
        fused_features = self.dyhead({
            'visual': list(features['visual'].values()),
            'visual_masks': list(features['visual_masks'].values()),  # Add masks
            'lang': features['language']
        })
        
        # Get predictions from head
        head_input = {
            'visual': fused_features['visual'],
            'lang': fused_features['lang'],
            'original_sizes': original_sizes
        }
        
        predictions= self.head(head_input)
        # Return predictions and text masks for loss computation
        return predictions, features['language']['masks']


def train_step(model, batch, optimizer, device):
    images, targets, original_sizes, captions = prepare_batch(batch, device)
    
    # Forward pass with separated inputs
    predictions = model(images, original_sizes, captions)
    
    # Compute losses
    losses = compute_losses(predictions, targets)
    
    # Total loss
    total_loss = sum(losses.values())
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return losses


def inference_step(model, batch, device):
    model.eval()
    with torch.no_grad():
        images, targets = prepare_batch(batch, model, device)
        predictions, _ = model(images, targets)
        return predictions


def train_glip():

    classes=81
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GLIP(hidden_dim=256, num_classes=classes)
    model = model.to(device)

    tokenizer = model.backbone.tokenizer

    # Dataset setup
    train_dataset = COCOGLIPDataset(
        coco_path='/home/asad/dev/GLIP/DATASET/coco/annotations/instances_train2017.json',
        image_dir='/home/asad/dev/GLIP/DATASET/coco/train2017',
        tokenizer=tokenizer
    )
    
    val_dataset = COCOGLIPDataset(
        coco_path='/home/asad/dev/GLIP/DATASET/coco/annotations/instances_val2017.json',
        image_dir='/home/asad/dev/GLIP/DATASET/coco/val2017',
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )

    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            total_batch_loss = train_step(model, batch, optimizer, device)
            
            total_loss += total_batch_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {total_batch_loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                boxes = batch['boxes'].to(device)
                categories = batch['categories'].to(device)
                positive_maps = batch['positive_maps'].to(device)
                
                targets = {
                    'boxes': boxes,
                    'labels': categories,
                    'positive_maps': positive_maps
                }
                
                losses = model(
                    images,
                    batch['original_sizes'],
                    batch['captions'],
                    targets
                )
                val_loss += sum(losses.values()).item()
        
        # Print epoch statistics
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Average Val Loss: {avg_val_loss:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_glip()
