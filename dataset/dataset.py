import torch
from pycocotools.coco import COCO
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from utils.utils import Resize,ToTensor,Normalize,Compose
from utils.utils import prepare_batch,convert_od_to_grounding_data
from utils.bounding_box import BoxList

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
                #Resize((480, 560, 640, 720, 800), max_size=1333),
                Resize((480), max_size=800),
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
            w = w_box
            h = h_box
            boxes.append([x, y, w, h])
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
    