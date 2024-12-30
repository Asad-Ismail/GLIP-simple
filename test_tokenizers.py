import torch 
import random
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional

def convert_od_to_grounding_data(target, tokenizer, ind_to_class, max_query_len=256, 
                              num_negatives=2, add_task_prompt=True):
    """
    Convert single object detection target to grounding format using GLIP-style prompts.
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
            token_start = tokenized.char_to_token(char_start) 
            token_end = tokenized.char_to_token(char_end - 1)
            
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

def create_positive_map_generic(
    tokenized,
    tokens_positive: List[List[Tuple[int, int]]],
    max_len: int = 256
) -> torch.Tensor:
    """
    Create positive map from arbitrary token spans
    """
    num_boxes = len(tokens_positive)
    positive_map = torch.zeros((num_boxes, max_len))
    
    for box_idx, char_spans in enumerate(tokens_positive):
        for (char_start, char_end) in char_spans:
            # Convert char positions to token positions
            token_start = tokenized.char_to_token(char_start) 
            token_end = tokenized.char_to_token(char_end - 1)
            
            if token_start is not None and token_end is not None:
                positive_map[box_idx, token_start:token_end + 1] = 1.0
    
    # Normalize
    normalizer = positive_map.sum(-1)[:, None] + 1e-6
    positive_map = positive_map / normalizer
    
    return positive_map


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Test data
    target = {
        'boxes': torch.tensor([[0,0,10,10], [20,20,30,30]]),
        'labels': torch.tensor([0, 1])
    }
    label_to_class = {0: "person", 1: "car", 2: "dog"}
    
    # Test OD style
    result = convert_od_to_grounding_data(
        target, tokenizer, label_to_class, add_task_prompt=False
    )
    print("OD Caption:", result['caption'])
    print("OD Map shape:", result['positive_map'].shape)
    print("OD Map non-zero:", torch.nonzero(result['positive_map']))
    
    # Test generic style with same data
    encoded = tokenizer(result['caption'], return_tensors="pt")
    person_start = result['caption'].find("person")
    car_start = result['caption'].find("car")
    tokens_positive = [
        [(person_start, person_start + len("person"))],
        [(car_start, car_start + len("car"))]
    ]
    
    pos_map_generic = create_positive_map_generic(
        encoded, tokens_positive
    )
    print("\nGeneric Map shape:", pos_map_generic.shape)
    print("Generic Map non-zero:", torch.nonzero(pos_map_generic))
    
    # Compare
    print("\nMaps equal?", torch.allclose(result['positive_map'], pos_map_generic))