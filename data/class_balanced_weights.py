#!/usr/bin/env python3
import numpy as np
import torch

# 1) Define your object classes
object_classes = [
    'airplane', 'apple', 'backpack', 'banana', 'baseball_bat', 'baseball_glove',
    'bed', 'bench', 'bicycle', 'boat', 'book', 'bottle', 'bowl', 'broccoli',
    'bus', 'cake', 'car', 'carrot', 'cell_phone', 'chair', 'clock', 'couch',
    'cup', 'dining_table', 'donut', 'fire_hydrant', 'fork', 'frisbee',
    'hair_drier', 'handbag', 'hot_dog', 'keyboard', 'kite', 'knife', 'laptop',
    'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking_meter',
    'pizza', 'potted_plant', 'refrigerator', 'remote', 'sandwich', 'scissors',
    'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports_ball',
    'stop_sign', 'suitcase', 'supporting', 'surfboard', 'teddy_bear',
    'tennis_racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic_light',
    'train', 'truck', 'tv', 'umbrella', 'vase', 'wine_glass'
]

# Build a fast lookup from name → index
class_to_idx = {name: i for i, name in enumerate(object_classes)}

# 2) Load the dataset
data = np.load("datasets/Release_Datasets/damon/hot_dca_trainval.npz", allow_pickle=True)
objwise = data['contact_label_objectwise']  # shape: [N_samples, variable-length list of class-names]

# 3) Count frequencies by name
counts = np.zeros(len(object_classes), dtype=np.float64)
for sample in objwise:
    # sample: iterable of class-names (strings)
    for cls_name in sample:
        idx = class_to_idx.get(cls_name)
        if idx is not None:
            counts[idx] += 1
        else:
            # unexpected class-name
            print(f"Warning: unknown class '{cls_name}'")

print("Raw class counts:")
for cls, c in zip(object_classes, counts):
    print(f"  {cls}: {int(c)}")

# 4) Compute inverse-frequency weights
eps = 1e-8
freq = counts / counts.sum()
weights = 1.0 / (freq + eps)

# 5) Normalize weights to mean = 1
weights = weights / weights.mean()

print("\nNormalized inverse-frequency weights (mean=1):")
for cls, w in zip(object_classes, weights):
    print(f"  {cls}: {w:.3f}")

# 6) Save as a torch tensor
weights_tensor = torch.from_numpy(weights.astype(np.float32))
torch.save(weights_tensor, "class_weights_damon.pt")
print("\nSaved class_weights_damon.pt")