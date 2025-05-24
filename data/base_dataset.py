import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from common import constants
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer, util
import logging

from models.vlm import VLMManager
from models.utils import pad_and_stack


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the sentence transformer model
_sentence_transformer = None

def get_sentence_transformer():
    """
    Lazily load the sentence transformer model to avoid loading it if not needed
    """
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            # Use a lightweight model for efficiency
            model_name = 'all-MiniLM-L6-v2'
            logger.info(f"Loading sentence transformer model: {model_name}")
            _sentence_transformer = SentenceTransformer(model_name)
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer model: {e}")
            logger.warning("Falling back to simple string matching for object mapping")
            _sentence_transformer = False  # Mark as failed to avoid retrying
    return _sentence_transformer

def mask_split(img, num_parts):
    if not len(img.shape) == 2:
        img = img[:, :, 0]
    mask = np.ones((img.shape[0], img.shape[1], num_parts))
    for i in np.unique(img):
        mask[:, :, i] = np.where(img == i, 1., 0.)
    return np.transpose(mask, (2, 0, 1))

class BaseDataset(Dataset):

    def __init__(self, dataset, mode, model_type='smpl', normalize=False, use_vlm: bool = True):
        self.dataset = dataset
        self.mode = mode
        self.use_vlm = use_vlm

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
            # Number of object classes (70 for DAMON dataset)
            self.num_object_classes = 70
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

        # Initialize object class embeddings cache
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

        # Cache for object name mappings to avoid recomputing
        self.object_name_mapping_cache = {}

        # Precompute embeddings for object classes if sentence transformer is available
        self.object_embeddings = None
        model = get_sentence_transformer()
        if model and not isinstance(model, bool):
            # Replace underscores with spaces for better semantic matching
            class_names = [name.replace('_', ' ') for name in self.object_classes]
            self.object_embeddings = model.encode(class_names, convert_to_tensor=True)
        
    def compute_pos_weight (self):
        """
        Compute positive weights for vertices based on effective number of samples
        This is used to balance the loss function during training.
        The weights are computed based on the number of times each vertex appears in the contact labels.
        """
        # Compute effective-number pos weights per vertex (Cui et al., CVPR'19)
        beta = 0.99
        # Initialize positive counts for each vertex
        pos_counts = np.zeros(self.n_vertices, dtype=np.float64)
        # Count appearances of each vertex in contact_labels_3d if available
        if self.has_contact_3d.sum() > 0:
            pos_counts = np.sum (self.contact_labels_3d, axis=0)
        # Compute effective number for each vertex
        eff_num = (1.0 - np.power(beta, pos_counts)) / (1.0 - beta)
        # Compute weights and normalize to mean=1
        pos_weights = 1.0 / (eff_num + 1e-8)
        pos_weights = pos_weights / np.mean(pos_weights) * 6.451
        # Store as a torch tensor for loss usage, scaled by the inverse frequency of positive vertices.
        # (On average there are 6.451 times more negative vertices than positive ones)
        pos_weights = torch.from_numpy(pos_weights.astype(np.float32))
        pos_weights = torch.clamp(
            pos_weights,
            max=torch.mean(pos_weights) + 1.5 * torch.std(pos_weights))
        # Save locally pos_weight for later use
        torch.save(pos_weights, 'pos_weights_damon.pt')
        return pos_weights

        if self.use_vlm:
                self.vlm_manager = VLMManager()
                if not self.vlm_manager.check_cache(self.images):
                    self.vlm_manager.extract_from_paths(self.images)

    def _map_object_to_coco_class(self, obj_name):
        """
        Map object name from DAMON dataset to object class index
        
        Args:
            obj_name: String name of the object from DAMON dataset
        
        Returns:
            int: Object class index (0-69) or -1 if not found
        """
        # Normalize object name for matching
        obj_name = obj_name.lower().strip()
        
        # Check cache first
        if obj_name in self.object_name_mapping_cache:
            return self.object_name_mapping_cache[obj_name]
        
        # Direct match first (with underscore replaced by space)
        obj_name_normalized = obj_name.replace('_', ' ')
        for i, class_name in enumerate(self.object_classes):
            if obj_name_normalized == class_name.replace('_', ' '):
                self.object_name_mapping_cache[obj_name] = i
                return i
        
        # Use sentence transformer for semantic similarity if available
        model = get_sentence_transformer()
        if model and not isinstance(model, bool) and self.object_embeddings is not None:
            try:
                # Encode the query object name
                query_embedding = model.encode(obj_name, convert_to_tensor=True)
                
                # Compute cosine similarity
                cos_scores = util.cos_sim(query_embedding, self.object_embeddings)[0]
                
                # Get the best match
                best_match_idx = torch.argmax(cos_scores).item()
                similarity_score = cos_scores[best_match_idx].item()
                
                # Only use the match if similarity is above threshold
                if similarity_score > 0.6:  # Threshold can be adjusted
                    logger.info(f"Mapped '{obj_name}' to '{self.object_classes[best_match_idx]}' with similarity {similarity_score:.3f}")
                    self.object_name_mapping_cache[obj_name] = best_match_idx
                    return best_match_idx
                else:
                    logger.warning(f"No good match found for '{obj_name}'. Best match was '{self.object_classes[best_match_idx]}' with low similarity {similarity_score:.3f}")
            except Exception as e:
                logger.warning(f"Error using sentence transformer for object mapping: {e}")
        
        # If no mapping found, return -1 or a default class
        logger.warning(f"No object class mapping found for '{obj_name}'")
        self.object_name_mapping_cache[obj_name] = -1
        return -1

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
                # Map object name to class index
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


        if self.use_vlm:
            item["vlm_features"] = self.vlm_manager[self.images[index]]

        return item


    def __len__(self):
        return len(self.images)


class VLMFeatureCollator:
    def __init__(self, pad_and_stack_func=pad_and_stack):
        """
        Custom collator for VLM features.

        Args:
            pad_and_stack_func (callable): Function to pad and stack a list of feature tensors.
        """
        self.pad_and_stack = pad_and_stack_func


    def __call__(self, batch):
        """
        Pads and stacks the 'vlm_features' in a batch of dictionaries.

        Args:
            batch (list of dict): List of dictionaries where one key is 'vlm_features'.

        Returns:
            list of dict: The same list of dictionaries but with padded 'vlm_features'.
        """

        keys = batch[0].keys()

        # Initialize a dictionary to hold the batched tensors
        batched_tensors = {}

        for key in keys:
            # Special handling for vlm_features
            if key == "img_path":
                values = [item[key] for item in batch]
                batched_tensors[key] = values

            elif key == "vlm_features":
                vlm_features = [item["vlm_features"] for item in batch]
                batched_tensors[key] = self.pad_and_stack(vlm_features)
            else:
                # Stack normally for other keys
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    batched_tensors[key] = torch.stack(values)
                elif isinstance(values[0], np.ndarray):
                    batched_tensors[key] = torch.tensor(np.array(values))
                else:
                    batched_tensors[key] = torch.tensor(values)
        return batched_tensors