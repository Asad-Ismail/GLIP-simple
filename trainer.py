import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
from utils import Resize,ToTensor,Normalize,Compose
from utils import prepare_batch,convert_od_to_grounding_data
from glip import GLIPBackbone,VLDyHead
from rpn_head import GLIPHead
from anchor_generator import anchor_generator_simple
from bounding_box import BoxList
from glip_loss import GLIPLoss
from image_list import ImageList



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

        self.json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        
        # Get image ids
        self.image_ids = list(self.coco.imgs.keys())
        
        # Transform pipeline
        if transforms is None:
            self.transforms = Compose([
                Resize((480, 560, 640, 720, 800), max_size=1333),
                #Resize((480), max_size=600),
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
            x1, y1, w_box, h_box = ann['bbox']
            # Convert to center format [cx,cy,w,h]
            x2 = x1 + w_box
            y2 = y1 + h_box
            boxes.append([x1, y1, x2, y2])
            categories.append(ann['category_id'])
            cat_name = self.coco.loadCats([ann['category_id']])[0]['name']
            str_cls_lst.append(cat_name)
        
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        boxes = BoxList(boxes, image_source.size, mode="xywh").convert("xyxy")

        # Add classes to boxes directly so we only need to pass boxes as targets
        classes = [self.json_category_id_to_contiguous_id[c] for c in categories]
        classes = torch.tensor(classes)
        boxes.add_field("labels", classes)

        # Create initial target dict
        target = {
            'boxes': boxes,
            'labels': torch.tensor(categories, dtype=torch.long),
            'image_id': image_id,
            # Add required fields for torchvision
            'area': torch.tensor([ann['area'] for ann in annotations]),
            'iscrowd': torch.tensor([ann.get('iscrowd', 0) for ann in annotations]),

            'str_cls_lst': str_cls_lst,
        }
        
        # Apply transforms to both image and target
        image_tensor, boxes = self.transforms(image_source, target["boxes"])
        # Replace boxes in targes with rescaled boxes
        target['boxes']=boxes
        target['size']=torch.as_tensor(image_tensor.shape[-2:])
        
        target = convert_od_to_grounding_data(
            target,
            self.tokenizer,
            self.ind_to_class,
            self.max_query_len,
            self.num_negatives,
            self.add_task_prompt,
        )

        return image_tensor, target

    def __len__(self):
        return len(self.image_ids)
    

class GLIP(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=None):
        super().__init__()
        # For COCO, num_classes is 80 (81 total with background, but we use 80 for detection)
        if num_classes is None:
            num_classes = 80  # COCO default
        
        self.backbone = GLIPBackbone()
        self.dyhead = VLDyHead(hidden_dim=hidden_dim)
        self.anchor_generator=anchor_generator_simple()
        self.head = GLIPHead(in_channels=hidden_dim, num_classes=num_classes)
        self.loss_calculator = GLIPLoss()
        
    def forward(self, images, sizes, captions,targets=None):
        """Forward pass without loss computation"""
        # Get backbone features
        with torch.no_grad():
            features = self.backbone(images, sizes, captions)
        
        # Process through dynamic head
        fused_features = self.dyhead({
            'visual': list(features['visual'].values()),
            'visual_masks': list(features['visual_masks'].values()),  # Add masks
            'lang': features['language']
        })
        
        # Get predictions from head
        head_input = {
            'visual': fused_features['visual'],
            'lang': fused_features['lang']
        }
        
        logits, bbox_reg, centerness, dot_product_logits = self.head(head_input)
        
        if self.training and targets is not None:
            # Only size of these image list is used to create anchors TODO Change it allow anchor generator to also expect only sizes to generator anchors
            images_list=ImageList(images.tensors,sizes)
            anchors = self.anchor_generator(images_list, fused_features['visual'])
            losses = self.loss_calculator(logits, bbox_reg, centerness, dot_product_logits, targets, anchors, captions)
            return losses
        
       

def train_step(model, batch, optimizer, device):
    images, targets, sizes, captions = prepare_batch(batch, device)
    
    # Forward pass with separated inputs
    with torch.autocast(device_type="cuda",dtype=torch.float16):
        losses = model(images, sizes, captions,targets)
    
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
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GLIP(hidden_dim=256)
    model = model.to(device)

    tokenizer = model.backbone.tokenizer

    # Dataset setup
    train_dataset = COCOGLIPDataset(
        coco_path='/home/asad/dev/GLIP/DATASET/coco/annotations/instances_train2017_subset.json',
        image_dir='/home/asad/dev/GLIP/DATASET/coco/train2017',
        tokenizer=tokenizer
    )
    
    val_dataset = COCOGLIPDataset(
        coco_path='/home/asad/dev/GLIP/DATASET/coco/annotations/instances_val2017_subset.json',
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
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) #weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    num_epochs = 100
    log_interval = 5  # Print stats every 100 batches
    #val_interval = 5    # Perform validation every 5 epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device and perform a training step
            batch_loss_dict = train_step(model, batch, optimizer, device)  # train_step returns a dict
            batch_loss = sum(batch_loss_dict.values())  # Sum the losses from the dict
            
            # Accumulate total loss
            total_loss += batch_loss.item()
            
            # Print batch stats every `log_interval` steps
            if batch_idx % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
                loss_details = ", ".join(
                    [f"{k}: {v.item():.4f}" for k, v in batch_loss_dict.items()]
                )
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Batch Loss: {batch_loss.item():.4f}, LR: {current_lr:.6f}, {loss_details}")

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)

        # Print epoch statistics
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Average Train Loss: {avg_train_loss:.4f}')

        # Step the scheduler
        scheduler.step()

        # Save checkpoint every 10 epochs
        '''
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                #'val_loss': avg_val_loss
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
        '''

if __name__ == '__main__':
    train_glip()
