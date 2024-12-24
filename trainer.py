import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Tuple, Dict, List
import os
from utils import build_id2posspan_and_caption, create_positive_map_from_span
from glip import GLIPBackbone,GLIPHead,VLDyHead

class COCOGLIPDataset(Dataset):
    def __init__(self, 
                 coco_path: str,
                 image_dir: str,
                 split: str = 'train2017',
                 max_text_len: int = 256):
        super().__init__()
        self.coco = COCO(coco_path)
        self.image_dir = image_dir
        self.max_text_len = max_text_len
        
        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.id2posspan, self.caption = build_id2posspan_and_caption(self.categories)
        
        # Get image ids
        self.image_ids = list(self.coco.imgs.keys())
        
        # Transform pipeline
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_ids)
    
    def load_image(self, image_path: str) -> Tuple[np.array, torch.Tensor]:
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = self.transform(image_source, None)
        return image, image_transformed

    def __getitem__(self, idx: int) -> Dict:
        image_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image, image_tensor = self.load_image(image_path)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Process boxes and categories
        boxes = []
        categories = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            categories.append(ann['category_id'])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        categories = torch.tensor(categories, dtype=torch.long)
        
        # Create positive map for text-box associations
        token_span = [self.id2posspan[cat_id] for cat_id in categories.tolist()]
        tokenized = self.tokenizer(self.caption, return_offsets_mapping=True)
        positive_map = create_positive_map_from_span(
            tokenized,
            token_span,
            max_text_len=self.max_text_len
        )
        
        return {
            'image': image_tensor,
            'boxes': boxes,
            'categories': categories,
            'caption': self.caption,
            'positive_map': positive_map,
            'image_id': image_id,
            'original_size': (img_info['height'], img_info['width'])
        }

def collate_fn(batch):
    """Simple collate function for the dataloader - keeps boxes and categories as lists"""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'captions': [item['caption'] for item in batch],
        'boxes': [item['boxes'] for item in batch],
        'categories': [item['categories'] for item in batch],
        'positive_maps': [item['positive_map'] for item in batch],
        'original_sizes': [item['original_size'] for item in batch]
    }

class GLIP(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=80):
        super().__init__()
        self.backbone = GLIPBackbone()
        self.dyhead = VLDyHead(hidden_dim=hidden_dim)
        self.head = GLIPHead(hidden_dim=hidden_dim, num_classes=num_classes)
        
    def forward(self, images, original_sizes, captions, targets=None):
        # Get backbone features
        features = self.backbone(images, original_sizes, captions)
        
        # Process through dynamic head
        fused_features = self.dyhead({
            'visual': list(features['visual']['features'].values()),
            'lang': features['language']
        })
        
        # Get predictions from head
        predictions = self.head(
            fused_features['visual'],
            fused_features['lang']
        )
        
        if self.training and targets is not None:
            losses = self.head.loss(
                predictions, 
                targets,
                text_masks=features['language']['masks']
            )
            return losses
        return predictions

def train_glip():

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GLIP(hidden_dim=256, num_classes=len(train_dataset.categories))
    model = model.to(device)

    # Dataset setup
    train_dataset = COCOGLIPDataset(
        coco_path='/home/asad/dev/GLIP/DATASET/coco/annotations/instances_train2017.json',
        image_dir='/home/asad/dev/GLIP/DATASET/coco/train2017'
    )
    
    val_dataset = COCOGLIPDataset(
        coco_path='/home/asad/dev/GLIP/DATASET/coco/annotations/instances_val2017.json',
        image_dir='/home/asad/dev/GLIP/DATASET/coco/val2017'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
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
            images = batch['images'].to(device)
            boxes = [b.to(device) for b in batch['boxes']]
            categories = [c.to(device) for c in batch['categories']]
            positive_maps = [p.to(device) for p in batch['positive_maps']]
            
            targets = {
                'boxes': boxes,
                'labels': categories,
                'positive_maps': positive_maps
            }
            
            # Forward pass
            optimizer.zero_grad()
            losses = model(
                images,
                batch['original_sizes'],
                batch['captions'],
                targets
            )
            
            # Compute total loss
            total_batch_loss = sum(losses.values())
            
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()
            
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
