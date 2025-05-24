# This script can be used to convert the contact labels from SMPL to SMPL-X format and vice-versa.

import os
import argparse
import pickle as pkl
import torch
import numpy as np
from common import constants

self.object_classes = [
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

def convert_contacts(contact_labels, mapping):
    """
    Converts the contact labels from SMPL to SMPL-X format and vice-versa.

    Args:
        contact_labels: contact labels in SMPL or SMPL-X format
        mapping: mapping from SMPL to SMPL-X vertices or vice-versa

    Returns:
        contact_labels_converted: converted contact labels
    """
    bs = contact_labels.shape[0]
    mapping = mapping[None].expand(bs, -1, -1)
    contact_labels_converted = torch.bmm(mapping, contact_labels[..., None])
    contact_labels_converted = contact_labels_converted.squeeze()
    return contact_labels_converted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contact_npz', type=str, required=True, help='path to contact npz file',
                        default='../datasets/ReleaseDatasets/damon/hot_dca_train.npz')
    parser.add_argument('--input_type', type=str, required=True, help='input type: smpl or smplx',
                        default='smpl')
    args = parser.parse_args()
    
    if args.input_type == 'smpl':
        # load mapping from smpl to smplx vertices 
        mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smpl_to_smplx.pkl")
    elif args.input_type == 'smplx':
        # load mapping from smplx to smpl vertices
        mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smplx_to_smpl.pkl")
    else:
        raise ValueError('input_type must be smpl or smplx')
    
    with open(mapping_pkl, 'rb') as f:
        mapping = pkl.load(f)
        mapping = mapping["matrix"]

    # load contact labels
    contact_data = np.load(args.contact_npz, allow_pickle=True)
    contact_data = dict(contact_data)
    contact_labels = contact_data['contact_label']
    if not isinstance(contact_labels, torch.Tensor):
        contact_labels = torch.from_numpy(contact_labels).float()
    if not isinstance(mapping, torch.Tensor):
        mapping = torch.from_numpy(mapping).float()
    contact_labels_converted = convert_contacts(contact_labels, mapping)
    contact_data['contact_label_smplx'] = contact_labels_converted.numpy()
    # Convert objectwise contact labels ([num_samples, C, V]) using the same mapping
    if 'contact_label_objectwise' in contact_data:
        obj_labels = contact_data['contact_label_objectwise']
        obj_labels_float = obj_labels.astype(np.float32)
        # Ensure tensor of shape [B, C, V]
        # Fill in the semantic contact tensor
        # objectwise_contacts is a dictionary with object names as keys and vertex indices as values
        for obj_name, vertex_indices in obj_labels.items():
            # Map object name to class index
            obj_class_idx = self.object_classes.index(obj_name) if obj_name in self.object_classes else -1
            if obj_class_idx >= 0:  # Valid class index
                semantic_contact[obj_class_idx, vertex_indices] = 1.0
            else:
                raise ValueError(f"Object name '{obj_name}' not found in object classes.")
        obj_tensor = torch.from_numpy(obj_labels).float()
        B, C, V = obj_tensor.shape
        # Flatten to [B*C, V]
        flat = obj_tensor.view(B * C, V)
        # Convert labels via mapping: result shape [B*C, V_converted]
        flat_converted = convert_contacts(flat, mapping)
        V_new = flat_converted.shape[1]
        # Reshape back to [B, C, V_new]
        obj_converted = flat_converted.view(B, C, V_new).numpy()
        # Store under appropriate key
        out_key = 'contact_label_objectwise_smplx' if args.input_type == 'smpl' else 'contact_label_objectwise_smpl'
        contact_data[out_key] = obj_converted
    # save the converted contact labels
    np.savez(args.contact_npz, **contact_data)
