#!/usr/bin/env python3
import os
import argparse

def convert_split(raw_root: str, preproc_root: str):
    """
    Walk raw_root/<sequence>/cam_<cam_id>/<frame>_<cam_id>.jpeg
    and create symlinks under
    preproc_root/<sequence>/<frame>/<seq_id>/images_refine/<frame>_<cam_id>.png
    where seq_id is the second token of the sequence folder name.
    """
    for seq in sorted(os.listdir(raw_root)):
        seq_raw = os.path.join(raw_root, seq)
        if not os.path.isdir(seq_raw):
            continue

        # extract sequence ID from e.g. "ParkingLot2_014_takingphotos2"
        parts = seq.split('_')
        if len(parts) < 2:
            print(f"⚠️  skipping malformed seq folder {seq}")
            continue
        seq_id = parts[1]

        for cam_folder in sorted(os.listdir(seq_raw)):
            if not cam_folder.startswith("cam_"):
                continue
            cam_id = cam_folder.split("cam_")[1]  # e.g. "00", "01", ...
            cam_raw = os.path.join(seq_raw, cam_folder)
            if not os.path.isdir(cam_raw):
                continue

            for fname in sorted(os.listdir(cam_raw)):
                name, ext = os.path.splitext(fname)
                if ext.lower() not in (".jpeg", ".jpg", ".png"):
                    continue

                # name should be "<frame>_<cam_id>"
                try:
                    frame_id, cam_idx = name.split("_")
                except ValueError:
                    print(f"  ⚠️  skipping malformed filename {fname}")
                    continue

                # build target directory
                tgt_dir = os.path.join(preproc_root, seq, frame_id, seq_id, "images_refine")
                os.makedirs(tgt_dir, exist_ok=True)

                src = os.path.join(cam_raw, fname)
                # we'll link with a .png extension to match the old naming
                dst = os.path.join(tgt_dir, f"{frame_id}_{cam_idx}.png")

                if os.path.exists(dst):
                    continue
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError as e:
                    print(f"  ❌ could not link {src} → {dst}: {e}")

    print(f"✅ Done. All symlinks created under {preproc_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RICH raw train folder into the expected preprocessed layout."
    )
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Path to your RICH/train folder (contains sequence subfolders)."
    )
    parser.add_argument(
        "--preproc_root",
        required=True,
        help="Path to RICH/preprocessed/train (will be created if needed)."
    )
    args = parser.parse_args()
    convert_split(args.raw_root, args.preproc_root)