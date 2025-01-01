import os
import random
from typing import List
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision
from typing import List, Optional
from torch import Tensor
from typing import List, Dict, Tuple
import math
from bounding_box import BoxList
import numpy as np
import cv2
from bounding_box import BoxList
from boxlist_ops import cat_boxlist, boxlist_ml_nms, remove_small_boxes
import supervision as sv

def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def build_captions_and_token_span(cat_list, force_lowercase):
    """
    Return:
        captions: str
        cat2tokenspan: dict
            {
                'dog': [[0, 2]],
                ...
            }
    """

    cat2tokenspan = {}
    captions = ""
    for catname in cat_list:
        class_name = catname
        if force_lowercase:
            class_name = class_name.lower()
        if "/" in class_name:
            class_name_list: List = class_name.strip().split("/")
            class_name_list.append(class_name)
            class_name: str = random.choice(class_name_list)

        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]
        for subname in subnamelist:
            if len(subname) == 0:
                continue
            if len(captions) > 0:
                captions = captions + " "
            strat_idx = len(captions)
            end_idx = strat_idx + len(subname)
            tokens_positive_i.append([strat_idx, end_idx])
            captions = captions + subname

        if len(tokens_positive_i) > 0:
            captions = captions + " ."
            cat2tokenspan[class_name] = tokens_positive_i

    return captions, cat2tokenspan


def build_id2posspan_and_caption(category_dict: dict):
    """Build id2pos_span and caption from category_dict

    Args:
        category_dict (dict): category_dict
    """
    cat_list = [item["name"].lower() for item in category_dict]
    id2catname = {item["id"]: item["name"].lower() for item in category_dict}
    caption, cat2posspan = build_captions_and_token_span(cat_list, force_lowercase=True)
    id2posspan = {catid: cat2posspan[catname] for catid, catname in id2catname.items()}
    return id2posspan, caption



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def matrix_iou(a, b, relative=False):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    if relative:
        ious = area_i / (area_b[:, np.newaxis]+1e-12)
    else:
        ious = area_i / (area_a[:, np.newaxis] + area_b - area_i+1e-12)
    return ious


class RACompose(object):
    def __init__(self, pre_transforms, rand_transforms, post_transforms, concurrent=2):
        self.preprocess = pre_transforms
        self.transforms = post_transforms
        self.rand_transforms = rand_transforms
        self.concurrent = concurrent

    def __call__(self, image, target):
        for t in self.preprocess:
            image, target = t(image, target)
        for t in random.choices(self.rand_transforms, k=self.concurrent):
            image = np.array(image)
            image, target = t(image, target)
        for t in self.transforms:
            image, target = t(image, target)

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\nRandom select {0} from: (".format(self.concurrent)
        for t in self.rand_transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += ")\nThen, apply:"
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        if target is None:
            return image
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size, restrict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.restrict = restrict

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if self.restrict:
            return (size, max_size)
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        if isinstance(image, np.ndarray):
            image_size = self.get_size(image.shape[:2])
            image = cv2.resize(image, image_size)
            new_size = image_size
        else:
            image = F.resize(image, self.get_size(image.size))
            new_size = image.size
        if target is not None:
            target = target.resize(new_size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if isinstance(image, np.ndarray):
                image = np.fliplr(image)
            else:
                image = F.hflip(image)
            if target is not None:
                target = target.transpose(0)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if isinstance(image, np.ndarray):
                image = np.flipud(image)
            else:
                image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, format='rgb'):
        self.mean = mean
        self.std = std
        self.format = format.lower()

    def __call__(self, image, target):
        if 'bgr' in self.format:
            image = image[[2, 1, 0]]
        if '255' in self.format:
            image = image * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=0.0,
                 contrast=0.0,
                 saturation=0.0,
                 hue=0.0,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class RandomCrop(object):
    def __init__(self, prob=0.5, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.prob = prob
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, target):
        if random.random() > self.prob:
            return img, target

        h, w, c = img.shape
        boxes = target.bbox.numpy()
        labels = target.get_field('labels')

        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, target

            min_iou = mode

            new_w = random.uniform(self.min_crop_size * w, w)
            new_h = random.uniform(self.min_crop_size * h, h)

            # h / w in [0.5, 2]
            if new_h / new_w < 0.5 or new_h / new_w > 2:
                continue

            left = random.uniform(0, w - new_w)
            top = random.uniform(0, h - new_h)

            patch = np.array([left, top, left + new_w, top + new_h])
            overlaps = matrix_iou(patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
            if overlaps.min() < min_iou:
                continue

            # center of boxes should inside the crop img
            center = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = (center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * ( center[:, 1] < patch[3])
            if not mask.any():
                continue

            boxes = boxes[mask]
            labels = labels[mask]

            # adjust boxes
            img = img[int(patch[1]):int(patch[3]), int(patch[0]):int(patch[2])]

            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
            boxes -= np.tile(patch[:2], 2)

            new_target = BoxList(boxes, (img.shape[1], img.shape[0]), mode='xyxy')
            new_target.add_field('labels', labels)
            return img, new_target


class RandomAffine(object):
    def __init__(self, prob=0.5, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                 borderValue=(127.5, 127.5, 127.5)):
        self.prob = prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.borderValue = borderValue

    def __call__(self, img, targets=None):
        if random.random() > self.prob:
            return img, targets
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        border = 0  # width of added border (optional)
        #height = max(img.shape[0], img.shape[1]) + border * 2
        height, width, _ = img.shape
        bbox = targets.bbox

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (self.degrees[1] - self.degrees[0]) + self.degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * self.translate[0] * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * self.translate[1] * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (self.shear[1] - self.shear[0]) + self.shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (self.shear[1] - self.shear[0]) + self.shear[0]) * math.pi / 180)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=self.borderValue)  # BGR order borderValue

        # Return warped points also
        if targets:
            n = bbox.shape[0]
            points = bbox[:, 0:4]
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            x1 = np.clip(xy[:,0], 0, width)
            y1 = np.clip(xy[:,1], 0, height)
            x2 = np.clip(xy[:,2], 0, width)
            y2 = np.clip(xy[:,3], 0, height)
            new_bbox = np.concatenate((x1, y1, x2, y2)).reshape(4, n).T
            targets.bbox = torch.as_tensor(new_bbox, dtype=torch.float32)

        return imw, targets


class RandomErasing:
    def __init__(self, prob=0.5, era_l=0.02, era_h=1/3, min_aspect=0.3,
                 mode='const', max_count=1, max_overlap=0.3, max_value=255):
        self.prob = prob
        self.era_l = era_l
        self.era_h = era_h
        self.min_aspect = min_aspect
        self.min_count = 1
        self.max_count = max_count
        self.max_overlap = max_overlap
        self.max_value = max_value
        self.mode = mode.lower()
        assert self.mode in ['const', 'rand', 'pixel'], 'invalid erase mode: %s' % self.mode

    def _get_pixels(self, patch_size):
        if self.mode == 'pixel':
            return np.random.random(patch_size)*self.max_value
        elif self.mode == 'rand':
            return np.random.random((1, 1, patch_size[-1]))*self.max_value
        else:
            return np.zeros((1, 1, patch_size[-1]))

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        ih, iw, ic = image.shape
        ia = ih * iw
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        erase_boxes = []
        for _ in range(count):
            for try_idx in range(10):
                erase_area = random.uniform(self.era_l, self.era_h) * ia / count
                aspect_ratio = math.exp(random.uniform(math.log(self.min_aspect), math.log(1/self.min_aspect)))
                eh = int(round(math.sqrt(erase_area * aspect_ratio)))
                ew = int(round(math.sqrt(erase_area / aspect_ratio)))
                if eh < ih and ew < iw:
                    x = random.randint(0, iw - ew)
                    y = random.randint(0, ih - eh)
                    image[y:y+eh, x:x+ew, :] = self._get_pixels((eh, ew, ic))
                    erase_boxes.append([x,y,x+ew,y+eh])
                break

        if target is not None and len(erase_boxes)>0:
            boxes = target.bbox.numpy()
            labels = target.get_field('labels')
            overlap = matrix_iou(np.array(erase_boxes), boxes, relative=True)
            mask = overlap.max(axis=0)<self.max_overlap
            boxes = boxes[mask]
            labels = labels[mask]
            target.bbox = torch.as_tensor(boxes, dtype=torch.float32)
            target.add_field('labels', labels)

        return image, target

       
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if mask == "auto":
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError(
                    "tensors dim must be 3 or 4 but {}({})".format(
                        self.tensors.dim(), self.tensors.shape
                    )
                )

    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(tensor.dim())
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self):
        """remove the padding and convert to img list

        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {"tensors.shape": self.tensors.shape, "mask.shape": self.mask.shape}
    

@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)

 
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes       

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)       
        

def convert_od_to_grounding_data(target, tokenizer, ind_to_class, max_query_len=256, 
                              num_negatives=2, add_task_prompt=True):
    """
    Convert single object detection target to grounding format using GLIP-style prompts.
    
    Args:
        target: Single target with boxes and labels
        tokenizer: BERT tokenizer
        ind_to_class: Class index to name mapping
        max_query_len: Maximum query length
        num_negatives: Number of negative classes to sample
        add_task_prompt: Whether to add "object detection:" prompt
    """
    # Get positive classes (present in image)
    pos_classes = set(target['labels'].cpu().numpy().tolist())
    
    # Sample negative classes (not present in image) 
    available_negs = [i for i in ind_to_class.keys() 
                     if i not in pos_classes and i != 0]  # exclude background
    if num_negatives > 0:
        neg_classes = random.sample(available_negs, 
                                 min(num_negatives, len(available_negs)))
    else:
        neg_classes = []
        
    # Build caption with both positive and negative classes
    classes = sorted(list(pos_classes)) + neg_classes
    
    # Start caption according to prompt style
    caption = "object detection: " if add_task_prompt else ""
    
    # Track text span positions for each class
    label_to_positions = {}
    
    # Build BERT-friendly caption with period separators
    for i, class_id in enumerate(classes):
        start_pos = len(caption)
        class_name = ind_to_class[class_id].strip()
        caption += class_name
        end_pos = len(caption)
        
        label_to_positions[class_id] = [start_pos, end_pos]
        
        # Add period separator instead of space (better for BERT)
        if i < len(classes) - 1:
            caption += ". "
            
    caption += "."
    
    # Tokenize caption
    tokenized = tokenizer(
        caption,
        return_tensors="pt",
        max_length=max_query_len,
        truncation=True,
        padding='max_length'
    )
    
    # Create positive map tensor
    num_boxes = len(target['boxes'])
    positive_map = torch.zeros((num_boxes, max_query_len), dtype=torch.float)
    
    # Map each box to its class text span
    for box_idx in range(num_boxes):
        class_id = target['labels'][box_idx].item()
        if class_id in label_to_positions:
            char_start, char_end = label_to_positions[class_id]
            
            # Convert char positions to token positions
            token_start = tokenized[0].char_to_token(char_start) 
            token_end = tokenized[0].char_to_token(char_end - 1)
            
            if token_start is not None and token_end is not None:
                positive_map[box_idx, token_start:token_end + 1] = 1.0
    
    # Normalize positive map
    normalizer = positive_map.sum(-1)[:, None] + 1e-6
    positive_map = positive_map / normalizer
    
    # Update target with grounding information
    target.update({
        'positive_map': positive_map,
        'caption': caption,
        'attention_mask': tokenized.attention_mask[0]
    })
    return target

def prepare_batch(batch, device):
    """
    Prepare batch for training by moving tensors to device and handling nested tensors.
    Returns images, targets, original_sizes, and captions separately.
    """
    images, targets = batch  # Now each item in batch is (image, [target])
    
    # Extract captions and sizes
    captions = [t['caption'] for t in targets]
    sizes = [t['size'] for t in targets]
    
    # Convert list of images to NestedTensor 
    if isinstance(images, (list, tuple)):
        images = nested_tensor_from_tensor_list(images)
    images = images.to(device)
    
    # Move target tensors to device
    for target in targets:
        for k, v in target.items():
            if isinstance(v, torch.Tensor):
                target[k] = v.to(device)
            if isinstance(v, BoxList):
                target[k] = v.to(device)
    return images, targets, sizes, captions


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_regression, box_cls=None, token_logits=None):
    box_regression_flattened = []
    box_cls_flattened = []
    token_logit_flattened = []

    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)

    if token_logits is not None:
        for token_logit_per_level in token_logits:
            N, AXT, H, W = token_logit_per_level.shape
            T = AXT // A
            token_logit_per_level = permute_and_flatten(
                token_logit_per_level, N, A, T, H, W
            )
            token_logit_flattened.append(token_logit_per_level)

    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

    token_logits_stacked = None
    if token_logits is not None:
        # stacked
        token_logits_stacked = cat(token_logit_flattened, dim=1)

    return box_regression, box_cls, token_logits_stacked


def aggregate_scores(token_probs,score_agg):
    if score_agg == "MEAN":
        return token_probs.mean(-1)
    elif score_agg == "MAX":
        # torch.max() returns (values, indices)
        return token_probs.max(-1)[0]
    else:
        raise NotImplementedError

class Predictor(torch.nn.Module):
    def __init__(
            self,
            box_coder,
            score_agg='MEAN',
    ):
        super().__init__()
        self.pre_nms_thresh = 0.05
        self.pre_nms_top_n = 1000
        self.nms_thresh = 0.6
        self.fpn_post_nms_top_n = 100
        self.min_size = 0
        self.text_threshold = 0.1
        self.box_coder = box_coder
        self.score_agg = score_agg
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.save_dir="visualizations"


        self.pred_annotator = sv.BoxAnnotator(
            color=sv.Color.red(),
            thickness=8,
            text_scale=0.8,
            text_padding=3
        )
        self.gt_annotator = sv.BoxAnnotator(
            color=sv.Color.green(),
            thickness=2,
            text_scale=0.8,
            text_padding=3
        )

    def denormalize_image(self, image):
        """
        Denormalize image tensor and convert to uint8 numpy array
        Args:
            image: Tensor [C,H,W] normalized with ImageNet stats
        Returns:
            numpy array [H,W,C] with values in [0,255]
        """
        device=image.device
        # Move channels to end: [C,H,W] -> [H,W,C]
        image = image.squeeze(0).permute(1, 2, 0)
        
        # Denormalize
        image = image * self.std.to(device) + self.mean.to(device)
        
        # Clip values to [0,1]
        image = torch.clamp(image, 0, 1)
        
        # Convert to uint8
        image = (image * 255).cpu().numpy().astype(np.uint8)
        
        return image
    
    def visualize_predictions(self, image_tensor, boxlist, target, image_id, epoch,tokenizer,tokenized):
        """
        Visualize both predictions and ground truth
        Args:
            image_tensor: Normalized image tensor [C,H,W]
            boxlist: Predicted BoxList with fields 'boxes', 'scores', 'phrases'
            target: Ground truth with fields 'boxes', 'str_cls_lst'
            image_id: Image identifier
            epoch: Current epoch number
        """
        save_dir = os.path.join(self.save_dir, f'epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Denormalize image
        image = self.denormalize_image(image_tensor)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        # Draw predictions
        if len(boxlist) > 0 and hasattr(boxlist[0], 'bbox') and len(boxlist[0].bbox) > 0:
            boxlist=boxlist[0]
            pred_boxes = boxlist.bbox.cpu().numpy()
            pred_detections = sv.Detections(xyxy=pred_boxes)
            pred_phrases = boxlist.get_field("phrases") if boxlist.has_field("phrases") else None
            image_bgr = self.pred_annotator.annotate(
                scene=image_bgr.copy(),
                detections=pred_detections,
                labels=pred_phrases
            )
        
        # Draw ground truth - extract phrases from target logits
        if target is not None and len(target["boxes"]) > 0:
            gt_boxes = target["boxes"].bbox.cpu().numpy()
            gt_detections = sv.Detections(xyxy=gt_boxes)
            
            # Extract phrases for GT using tokenized input
            if "positive_map" in target:
                gt_logits = target["positive_map"].float()  # Convert to float
                gt_phrases = self.extract_phrases(gt_logits, tokenized, tokenizer)
            else:
                gt_phrases = None
                
            image_bgr = self.gt_annotator.annotate(
                scene=image_bgr,
                detections=gt_detections,
                labels=gt_phrases
            )
        
        cv2.imwrite(os.path.join(save_dir, f'val_img_{image_id}.jpg'), image_bgr)

    def extract_phrases(self, logits, tokenized, tokenizer):
        phrases = []
        for logit in logits:
            # Combine attention mask with confidence threshold
            text_mask = logit > self.text_threshold
            attention_mask = tokenized.attention_mask[0]  # [seq_len]
            combined_mask = text_mask & attention_mask.bool()

            valid_tokens = [
                tid.item() 
                for tid, mask in zip(tokenized.input_ids[0], combined_mask)
                if tid not in [tokenizer.cls_token_id, tokenizer.sep_token_id] and mask
            ]
            
            if valid_tokens:
                phrase = tokenizer.decode(valid_tokens)
                conf = logit.max().item()
                phrases.append(f"{phrase} ({conf:.2f})")
            else:
                phrases.append(f"NoObj")
        return phrases

    def forward_for_single_feature_map(self, box_regression, centerness, anchors,dot_product_logits=None,
                                    tokenizer=None,tokenized=None,):

        N, _, H, W = box_regression.shape
        A = box_regression.size(1) // 4
        #Passing through sigmoid for dor product logits
        dot_product_logits = dot_product_logits.sigmoid()
        scores = aggregate_scores(dot_product_logits,score_agg=self.score_agg)
        print(f"Pred scores max are {scores.max()}")
        box_cls = scores

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        centerness = permute_and_flatten(centerness, N, A, 1, H, W)
        centerness = centerness.reshape(N, -1).sigmoid()

        # multiply the classification scores with centerness scores

        box_cls = box_cls * centerness

        results = []

        for batch_idx, (per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors) \
                in enumerate(zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors)):
            # Add check for empty predictions
            if not per_candidate_inds.any():
                print(f"❌ Level {batch_idx}: No valid boxes found (0/{box_cls.shape[-1]} passed threshold {self.pre_nms_thresh:.3f})")
                # Return empty BoxList with same image size
                empty_boxlist = BoxList(torch.zeros((0, 4)), per_anchors.size, mode="xyxy").to("cpu")
                empty_boxlist.add_field("labels", torch.zeros(0, dtype=torch.long))
                empty_boxlist.add_field("scores", torch.zeros(0))
                empty_boxlist.add_field("phrases", [])  # Add empty phrases field
                results.append(empty_boxlist)
                continue
            
            print(f"✅ Level {batch_idx}: Found {len(per_candidate_inds)}/{box_cls.shape[-1]} valid boxes above threshold {self.pre_nms_thresh:.3f}")
            per_box_cls = per_box_cls[per_candidate_inds]  # Shape: num_valid_boxes
            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            # Modified to handle single class case
            per_candidate_nonzeros = per_candidate_inds.nonzero().squeeze(1)[top_k_indices]
            per_box_loc = per_candidate_nonzeros  
            per_class = torch.ones_like(per_box_loc)  # All boxes belong to same class

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )
            # Move detction back to cpus since prashes are strings and live on cpus and some empty tensors are also on cpus just easier to have everything 
            # in cpu
            boxlist = BoxList(detections, per_anchors.size, mode="xyxy").to("cpu")
            boxlist.add_field("labels", per_class.cpu())
            boxlist.add_field("scores", torch.sqrt(per_box_cls).cpu())
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)

            if tokenized is not None and tokenizer is not None:
                per_phrases_logits = dot_product_logits[batch_idx, per_box_loc]
                phrases = self.extract_phrases(per_phrases_logits, tokenized, tokenizer)
                boxlist.add_field("phrases", phrases)

            results.append(boxlist)

        return results

    def forward(self, box_regression, centerness, anchors, dot_product_logits=None,tokenizer=None, tokenized=None,
                image=None,targets=None,epoch=None):
        sampled_boxes = []
        anchors = list(zip(*anchors))
        for idx, (b, c, a) in enumerate(zip(box_regression, centerness, anchors)):
            d = dot_product_logits[idx]
            sampled_boxes.append(self.forward_for_single_feature_map(b, c, a, d, tokenizer=tokenizer, tokenized=tokenized))

        #print(f"GT box list is {targets['boxes']}")
        boxlists = list(zip(*sampled_boxes))
        #print(f"Pred intial box lists {boxlists}")
        ## Handle Empty boxes
        #print(print(f"Boxlists structure: {[print(b) for b in boxlists]}"))
        # First level is for batch and second level is for freature list
        if any(any(len(box.bbox) > 0 for box in boxlist) for boxlist in boxlists):
            boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
            boxlists = self.select_over_all_levels(boxlists)

        self.visualize_predictions(
                    image_tensor=image,  
                    boxlist=boxlists,
                    target=targets,
                    image_id=targets.get("image_id", idx),
                    epoch=epoch,
                    tokenizer=tokenizer,
                    tokenized=tokenized
                )

        return boxlists
    


    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(cls_scores.cpu().float(),number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            print(f"Output BBOX are {result.bbox}, with labels {boxlist.get_field('phrases')}")
            results.append(result)
        return results