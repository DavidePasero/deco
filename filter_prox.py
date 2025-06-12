#!/usr/bin/env python3
import numpy as np
import os
import argparse

def convert_prox_path(old_path: str, path_type: str) -> str:
    """
    Convert:
      prox/flipped/SCENE/FNAME
    into:
      RICH/PROX/recordings/SCENE/color/FNAME
    """
    parts = old_path.split("/")
    scene, fname = parts[2], parts[3]
    if path_type == "recordings":
      return f"RICH/PROX/{path_type}/{scene}/Color/{fname}"
    else:
      return f"RICH/PROX/{path_type}/{scene}/{fname}"

def main(npz_in: str, npz_out: str):
    # 1) Load everything into a dict
    orig = np.load(npz_in, allow_pickle=True)
    data = {k: orig[k] for k in orig.files}
    orig.close()

    # 2) Convert all imgname entries
    old_names = data['imgname']
    new_names = [convert_prox_path(p, "recordings") for p in old_names]
    old_seg = data['scene_seg']
    new_seg = [convert_prox_path(p, "segmentation_masks") for p in old_seg]
    old_part = data['part_seg']
    new_part = [convert_prox_path(p, "parts") for p in old_part]
    data['imgname'] = np.array(new_names)
    data['part_seg'] = np.array(new_part)
    data['scene_seg'] = np.array(new_seg)
    print(f"Converted {len(old_names)} imgname entries")

    # 3) Overwrite the original .npz with the updated data
    np.savez(npz_out, **data)
    print(f"Saved updated .npz to {npz_out} (updated imgname paths)")

main("datasets/Release_Datasets/prox/prox_train_smplx_ds4.npz", "datasets/Release_Datasets/prox/train-prox.npz")