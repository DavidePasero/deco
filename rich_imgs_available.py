import numpy as np
import os
from pathlib import Path

data = np.load("datasets/Release_Datasets/rich/rich_train_smplx_cropped_bmp.npz", allow_pickle=True)
all_imgnames = data['imgname']

rich_paths_file = "rich_paths"

from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# Check existence of only the listed imgnames in parallel by sharding into 16
all_paths = list(all_imgnames)
num_shards = 16
shard_size = math.ceil(len(all_paths) / num_shards)
shards = [all_paths[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

def check_shard(shard):
    present = []
    missing = []
    for img in shard:
        if os.path.exists(img):
            present.append(img)
        else:
            missing.append(img)
    return present, missing

present_imgs = []
missing_imgs = []
with ThreadPoolExecutor(max_workers=num_shards) as executor:
    futures = [executor.submit(check_shard, shard) for shard in shards]
    for future in as_completed(futures):
        p, m = future.result()
        present_imgs.extend(p)
        missing_imgs.extend(m)

# Cache only the existing image paths
with open(rich_paths_file, "w") as f:
    for img in present_imgs:
        f.write(img + "\n")

print(f"Found {len(missing_imgs)} missing images in {dataset} dataset")

# Filter all arrays in self.data to only those with files on disk
present_set = set(present_imgs)
mask = np.array([img in present_set for img in all_imgnames])

# Rebuild self.data as a dict of filtered arrays
filtered = {k: data[k][mask] for k in data.files}
data = filtered