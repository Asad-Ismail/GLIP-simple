import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import numpy as np
import os
from dataset.dataset import COCOGLIPDataset
from model.glip import GLIP
from utils.utils import prepare_batch
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def train_step(model, batch, optimizer, device,scaler):
    model.train()
    images, targets, sizes, captions = prepare_batch(batch, device)

    optimizer.zero_grad()

    # Forward pass with separated inputs
    with torch.autocast(device_type="cuda",dtype=torch.float16):
        losses = model(images, sizes, captions,targets)
        total_loss = sum(losses.values())
    
    # Backward pass
    scaler.scale(total_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    '''
    print("\nParameter Gradient Norms:")
    total_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            print(f"{name}: {param_norm:.4f}")
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm:.4f}")
    '''
    scaler.step(optimizer)
    scaler.update()
    return losses

@torch.no_grad
def val_step(model, batch, device,epoch):
    model.eval()
    images, targets, sizes, captions = prepare_batch(batch, device)
    # Forward pass with separated inputs
    targets[0]["epoch"]=epoch
    with torch.autocast(device_type="cuda",dtype=torch.float16):
        detection = model(images, sizes, captions,targets)
    #print(detection)



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
        num_workers=1,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True
    )


    num_epochs = 300
    log_interval = 50  # Print stats 
    save_interval=50
    val_interval = 25
    max_val_images=200

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,eta_min=1e-7)  
    # Required for mixed training
    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in tqdm(enumerate(train_loader), 
                                desc=f'Epoch {epoch}', 
                                total=len(train_loader),
                                position=1, 
                                leave=False):
            # Move data to device and perform a training step
            batch_loss_dict = train_step(model, batch, optimizer, device,scaler)  # train_step returns a dict
            batch_loss = sum(batch_loss_dict.values())  # Sum the losses from the dict
            
            # Accumulate total loss
            total_loss += batch_loss.item()
            
            # Print batch stats every `log_interval` steps
            if batch_idx % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
                loss_details = ", ".join([f"{k}: {v.item():.4f}" for k, v in batch_loss_dict.items()])
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Batch Loss: {batch_loss.item():.4f}, LR: {current_lr:.6f}, {loss_details}")

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)

        # Print epoch statistics
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Average Train Loss: {avg_train_loss:.4f}')

        # Step the scheduler
        scheduler.step()

        # Save checkpoint 
        if epoch % save_interval == 0:
            checkpoint_dir="weights"
            os.makedirs(checkpoint_dir,exist_ok=True)
            checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                #'val_loss': avg_val_loss
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
        
        # Do validation and visualization
        if epoch%val_interval==0:
            for batch_idx,batch in enumerate(val_loader):
                if batch_idx>max_val_images:
                    break
                val_step(model, batch, device,epoch)

        

if __name__ == '__main__':
    train_glip()
