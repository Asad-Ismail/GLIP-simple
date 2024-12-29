import json
from collections import defaultdict

def print_dataset_statistics(coco_data, title="Dataset Statistics"):
    """Print detailed statistics including co-occurrence information."""
    print(f"\n{'='*20} {title} {'='*20}")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total Images: {len(coco_data['images'])}")
    print(f"Total Annotations: {len(coco_data['annotations'])}")
    print(f"Total Categories: {len(coco_data['categories'])}")
    
    # Per category statistics
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    cat_stats = defaultdict(lambda: {'images': set(), 'annotations': 0})
    
    # Track images with multiple annotations
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        cat_name = cat_id_to_name[ann['category_id']]
        cat_stats[cat_name]['images'].add(ann['image_id'])
        cat_stats[cat_name]['annotations'] += 1
        image_annotations[ann['image_id']].append(cat_name)
    
    # Print per-category statistics
    print(f"\nPer Category Statistics:")
    print(f"{'Category':<15} {'Images':<10} {'Annotations':<15} {'Annotations/Image':<20}")
    print('-' * 60)
    for cat_name, stats in cat_stats.items():
        images = len(stats['images'])
        anns = stats['annotations']
        avg = anns / images if images > 0 else 0
        print(f"{cat_name:<15} {images:<10} {anns:<15} {avg:.2f}")
    
    # Multiple annotations statistics
    print(f"\nMultiple Annotations Statistics:")
    annotations_count = defaultdict(int)
    category_pairs = defaultdict(int)
    
    for img_id, categories in image_annotations.items():
        annotations_count[len(categories)] += 1
        
        # Count category co-occurrences
        if len(categories) > 1:
            for i in range(len(categories)):
                for j in range(i + 1, len(categories)):
                    pair = tuple(sorted([categories[i], categories[j]]))
                    category_pairs[pair] += 1
    
    print("\nImages by number of annotations:")
    for count, num_images in sorted(annotations_count.items()):
        print(f"{count} annotations: {num_images} images")
    
    print("\nTop category co-occurrences:")
    top_pairs = sorted(category_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    for (cat1, cat2), count in top_pairs:
        print(f"{cat1} + {cat2}: {count} images")




def create_coco_subset(input_ann_path, output_ann_path, selected_categories, max_images_per_class=100):
    """Create a smaller COCO dataset with selected categories and print statistics."""
    
    print(f"\nProcessing: {input_ann_path}")
    print(f"Target: {max_images_per_class} images per class")
    
    # Load original annotations
    with open(input_ann_path, 'r') as f:
        coco_data = json.load(f)
    
    # Print original statistics
    print_dataset_statistics(coco_data, "Original Dataset")
    
    # Get category IDs for selected categories
    selected_cat_ids = []
    new_categories = []
    for cat in coco_data['categories']:
        if cat['name'] in selected_categories:
            selected_cat_ids.append(cat['id'])
            new_categories.append(cat)
    
    # Get annotations for selected categories
    images_per_category = defaultdict(set)
    selected_anns = []
    
    for ann in coco_data['annotations']:
        if ann['category_id'] in selected_cat_ids:
            cat_name = next(cat['name'] for cat in new_categories if cat['id'] == ann['category_id'])
            if len(images_per_category[cat_name]) < max_images_per_class:
                images_per_category[cat_name].add(ann['image_id'])
                selected_anns.append(ann)
    
    # Get unique image IDs
    selected_image_ids = {ann['image_id'] for ann in selected_anns}
    selected_images = [img for img in coco_data['images'] 
                      if img['id'] in selected_image_ids]
    
    # Create new annotation file
    new_coco_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': new_categories,
        'images': selected_images,
        'annotations': selected_anns
    }
    
    # Print subset statistics
    print_dataset_statistics(new_coco_data, "Subset Dataset")
    
    # Save new annotation file
    with open(output_ann_path, 'w') as f:
        json.dump(new_coco_data, f)
    
    print(f"\nSaved subset to: {output_ann_path}")

# Selected categories
selected_cats = [
    'person', 'car', 'dog', 'cat', 'chair',
    'bottle', 'laptop', 'pizza', 'bird', 'umbrella'
]

# Create train subset
create_coco_subset(
    '/home/asad/dev/GLIP/DATASET/coco/annotations/instances_train2017.json',
    '/home/asad/dev/GLIP/DATASET/coco/annotations/instances_train2017_subset.json',
    selected_cats,
    max_images_per_class=200
)

# Create val subset
create_coco_subset(
    '/home/asad/dev/GLIP/DATASET/coco/annotations/instances_val2017.json',
    '/home/asad/dev/GLIP/DATASET/coco/annotations/instances_val2017_subset.json',
    selected_cats,
    max_images_per_class=50
)