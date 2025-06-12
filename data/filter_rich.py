import numpy as np
import os
import re
from models.vlm import TextCache


data = np.load("datasets/Release_Datasets/rich/rich_train_smplx_cropped_bmp.npz", allow_pickle=True)
with open ("rich_paths", "r") as f:
    available_paths = f.readlines()
available_paths = [p.strip() for p in available_paths]
# 3) Compute mask over data['imgname']
all_imgnames = data['imgname']
mask = np.array([img in available_paths for img in all_imgnames])
idxs = np.argsort(all_imgnames)
valid_idxs = np.where(mask)[0]
valid_and_sorted_idxs = np.array([i for i in idxs if i in valid_idxs])
print(f"Keeping {mask.sum()} / {len(mask)} samples")

pat = r'_\d+_\d+_images_refine'

def sort_key(x):
    return re.sub(pat, "", x)

# 4) Filter every array in the .npz by that mask
filtered = { key: data[key][valid_and_sorted_idxs] for key in data.files }
# Filter out the paths in self.data that are not in the paths list
part = np.array(sorted(["rich/seg_cropped/train/" + x for x in os.listdir("rich/seg_cropped/train/")], key=sort_key))
scene = np.array(sorted(["rich/seg_cropped/train/" + x for x in os.listdir("rich/seg_cropped/train/")], key=sort_key))

filtered["scene_seg"] = scene
filtered["part_seg"] = part

np.savez("datasets/Release_Datasets/rich/train-rich.npz", **filtered)
