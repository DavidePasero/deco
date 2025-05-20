#!/usr/bin/env python
"""
Compute pos_weight (β) for BCEWithLogitsLoss used in MultiClassContactLoss.

β  =  (# vertices with no contact anywhere)
      -------------------------------------
      (# vertices with at least one contact)

The script loads the training split once, so runtime is a few seconds.
"""

import argparse
import numpy as np
from data.base_dataset import BaseDataset
from common import constants      # provides DATASET_FILES mapping
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="dataset key as used in constants.DATASET_FILES "
                             "(e.g. hot, hico, damon)")
    parser.add_argument("--model", default="smpl",
                        choices=["smpl", "smplx"],
                        help="mesh type to decide n_vertices (default: smpl)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load the *train* split only; normalisation not needed for counting
    # ------------------------------------------------------------------ #
    ds = BaseDataset(dataset=args.dataset,
                     mode="train",
                     model_type=args.model,
                     normalize=False)

    V = ds.n_vertices
    total_pos = 0
    total_neg = 0

    print(f"Traversing {len(ds)} training samples…")
    for item in tqdm(ds):
        # item['semantic_contact'] shape: [C, V]
        # If you train only binary contact, you can use item['contact_label_3d']
        sem = item["semantic_contact"].numpy()          # (C, V)
        pos_mask = sem.any(axis=0)                      # [V] bool per vertex
        pos_count = int(pos_mask.sum())
        total_pos += pos_count
        total_neg += V - pos_count

    if total_pos == 0:
        raise RuntimeError("No positive contact vertices found in train split!")

    beta = total_neg / total_pos
    print(f"\nDataset           : {args.dataset} (train)")
    print(f"Model vertices    : {V}")
    print(f"Total pos vertices: {total_pos:,}")
    print(f"Total neg vertices: {total_neg:,}")
    print(f"\npos_weight (β)    : {beta:.3f}")

    # Optionally write to a small JSON for later use
    with open("pos_weight.json", "w") as f:
        import json
        json.dump({"dataset": args.dataset,
                   "model": args.model,
                   "pos_weight": beta}, f, indent=2)
        print("→ Saved β to pos_weight.json")

if __name__ == "__main__":
    main()