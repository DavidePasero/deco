import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from common import constants

def mask_split(img, num_parts):
    if not len(img.shape) == 2:
        img = img[:, :, 0]
    mask = np.zeros((img.shape[0], img.shape[1], num_parts))
    for i in np.unique(img):
        mask[:, :, i] = np.where(img == i, 1., 0.)
    return np.transpose(mask, (2, 0, 1))

class BaseDataset(Dataset):

    def __init__(self, dataset, mode, model_type='smpl', normalize=False):
        self.dataset = dataset
        self.mode = mode

        print(f'Loading dataset: {constants.DATASET_FILES[mode][dataset]} for mode: {mode}')

        self.data = np.load(constants.DATASET_FILES[mode][dataset], allow_pickle=True)

        self.images = self.data['imgname']

        # get 3d contact labels, if available
        try:
            self.contact_labels_3d = self.data['contact_label']
            # make a has_contact_3d numpy array which contains 1 if contact labels are no empty and 0 otherwise
            self.has_contact_3d = np.array([1 if len(x) > 0 else 0 for x in self.contact_labels_3d])
        except KeyError:
            self.has_contact_3d = np.zeros(len(self.images))

        # get object-wise 3d contact labels, if available (DAMON dataset)
        try:
            self.contact_labels_objectwise = self.data['contact_label_objectwise']
            self.has_semantic_contact = np.ones(len(self.images))
            # Number of COCO object classes (80)
            self.num_object_classes = 80
        except KeyError:
            self.has_semantic_contact = np.zeros(len(self.images))
            self.num_object_classes = 1  # Default to single class for binary contact

        # get 2d polygon contact labels, if available
        try:
            self.polygon_contacts_2d = self.data['polygon_2d_contact']
            self.has_polygon_contact_2d = np.ones(len(self.images))
        except KeyError:
            self.has_polygon_contact_2d = np.zeros(len(self.images))

        # Get camera parameters - only intrinsics for now
        try:
            self.cam_k = self.data['cam_k']
        except KeyError:
            self.cam_k = np.zeros((len(self.images), 3, 3))

        self.sem_masks = self.data['scene_seg']
        self.part_masks = self.data['part_seg']

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(float)
            self.betas = self.data['shape'].astype(float)
            self.transl = self.data['transl'].astype(float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.images))
                self.is_smplx = np.ones(len(self.images)) if model_type == 'smplx' else np.zeros(len(self.images))
        except KeyError:
            self.has_smpl = np.zeros(len(self.images))
            self.is_smplx = np.zeros(len(self.images))

        if model_type == 'smpl':
            self.n_vertices = 6890
        elif model_type == 'smplx':
            self.n_vertices = 10475
        else:
            raise NotImplementedError

        self.normalize = normalize
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    def __getitem__(self, index):
        item = {}

        # Load image
        img_path = self.images[index]
        try:
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1) / 255.0
        except:
            print('Img: ', img_path)

        img_scale_factor = np.array([256 / img_w, 256 / img_h])

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
            transl = self.transl[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)
            transl = np.zeros(3)

        # Load binary contact labels
        if self.has_contact_3d[index]:
            contact_label_3d = self.contact_labels_3d[index]
        else:
            contact_label_3d = np.zeros(self.n_vertices)

        # Load semantic (object-wise) contact labels if available
        if self.has_semantic_contact[index]:
            # Get the object-wise contact labels
            objectwise_contacts = self.contact_labels_objectwise[index]
            
            # Create a tensor of shape [num_object_classes, n_vertices]
            semantic_contact = np.zeros((self.num_object_classes, self.n_vertices))
            
            # Fill in the semantic contact tensor
            # objectwise_contacts is a dictionary with object names as keys and vertex indices as values
            for obj_name, vertex_indices in objectwise_contacts.items():
                # Map object name to COCO class index
                obj_class_idx = self._map_object_to_coco_class(obj_name)
                if obj_class_idx >= 0:  # Valid class index
                    semantic_contact[obj_class_idx, vertex_indices] = 1.0
        else:
            # If no semantic contacts, create a single-class tensor with binary contacts
            semantic_contact = np.zeros((self.num_object_classes, self.n_vertices))
            if self.has_contact_3d[index]:
                # Use the first class for binary contacts
                semantic_contact[0] = contact_label_3d

        sem_mask_path = self.sem_masks[index]
        try:
            sem_mask = cv2.imread(sem_mask_path)
            sem_mask = cv2.resize(sem_mask, (256, 256), cv2.INTER_CUBIC)
            sem_mask = mask_split(sem_mask, 133)
        except:
            print('Scene seg: ', sem_mask_path)

        try:
            part_mask_path = self.part_masks[index]
            part_mask = cv2.imread(part_mask_path)
            part_mask = cv2.resize(part_mask, (256, 256), cv2.INTER_CUBIC)
            part_mask = mask_split(part_mask, 26)
        except:
            print('Part seg: ', part_mask_path)

        try:
            if self.has_polygon_contact_2d[index]:
                polygon_contact_2d_path = self.polygon_contacts_2d[index]
                polygon_contact_2d = cv2.imread(polygon_contact_2d_path)
                polygon_contact_2d = cv2.resize(polygon_contact_2d, (256, 256), cv2.INTER_NEAREST)
                # binarize the part mask
                polygon_contact_2d = np.where(polygon_contact_2d > 0, 1, 0)
            else:
                polygon_contact_2d = np.zeros((256, 256, 3))
        except:
            print('2D polygon contact: ', polygon_contact_2d_path)

        if self.normalize:
            img = torch.tensor(img, dtype=torch.float32)
            item['img'] = self.normalize_img(img)
        else:
            item['img'] = torch.tensor(img, dtype=torch.float32)

        if self.is_smplx[index]:
            # Add 6 zeros to the end of the pose vector to match with smpl
            pose = np.concatenate((pose, np.zeros(6)))

        item['img_path'] = img_path
        item['pose'] = torch.tensor(pose, dtype=torch.float32)
        item['betas'] = torch.tensor(betas, dtype=torch.float32)
        item['transl'] = torch.tensor(transl, dtype=torch.float32)
        item['cam_k'] = self.cam_k[index]
        item['img_scale_factor'] = torch.tensor(img_scale_factor, dtype=torch.float32)
        item['contact_label_3d'] = torch.tensor(contact_label_3d, dtype=torch.float32)
        item['semantic_contact'] = torch.tensor(semantic_contact, dtype=torch.float32)
        item['sem_mask'] = torch.tensor(sem_mask, dtype=torch.float32)
        item['part_mask'] = torch.tensor(part_mask, dtype=torch.float32)
        item['polygon_contact_2d'] = torch.tensor(polygon_contact_2d, dtype=torch.float32)

        item['has_smpl'] = self.has_smpl[index]
        item['is_smplx'] = self.is_smplx[index]
        item['has_contact_3d'] = self.has_contact_3d[index]
        item['has_semantic_contact'] = self.has_semantic_contact[index]
        item['has_polygon_contact_2d'] = self.has_polygon_contact_2d[index]

        return item

    def __len__(self):
        return len(self.images)

    def _map_object_to_coco_class(self, obj_name):
        """
        Map object name from DAMON dataset to COCO class index
        
        Args:
            obj_name: String name of the object from DAMON dataset
        
        Returns:
            int: COCO class index (0-79) or -1 if not found
        """
        # COCO class names (80 classes)
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Normalize object name for matching
        obj_name = obj_name.lower().strip()
        
        # Special case handling for DAMON dataset
        if obj_name == 'supporting':
            # Map to a general class like 'floor' or use a default class
            return 0  # Use 'person' as default for supporting surfaces
        
        # Direct match
        if obj_name in coco_classes:
            return coco_classes.index(obj_name)
        
        # Handle common variations
        if obj_name in ['table', 'desk']:
            return coco_classes.index('dining table')
        if obj_name in ['sofa']:
            return coco_classes.index('couch')
        if obj_name in ['phone', 'smartphone']:
            return coco_classes.index('cell phone')
        if obj_name in ['tv', 'television']:
            return coco_classes.index('tv')
        if obj_name in ['floor', 'ground', 'road', 'sidewalk']:
            # No direct mapping in COCO, use a default class
            return 0  # Use 'person' as default
        
        # If no mapping found, return -1 or a default class
        print(f"Warning: No COCO class mapping found for object '{obj_name}'")
        return -1
