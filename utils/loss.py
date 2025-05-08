import torch
import torch.nn as nn
from common import constants
from models.smpl import SMPL
from smplx import SMPLX
import pickle as pkl
import numpy as np
from utils.mesh_utils import save_results_mesh
from utils.diff_renderer import Pytorch3D
import os
import cv2
from utils.metrics import det_error_metric

class sem_loss_function(nn.Module):
    def __init__(self):
        super(sem_loss_function, self).__init__()
        self.ce = nn.BCELoss()

    def forward(self, y_true, y_pred):
        loss = self.ce(y_pred, y_true)
        return loss
    
class MultiClassContactLoss(nn.Module):
    """
    Hierarchical loss function for multi-class contact prediction
    
    Applies different penalties for:
    1. Contact/no-contact errors (primary error - high penalty)
    2. Object class errors (secondary error - lower penalty)
    """
    def __init__(self, contact_weight=1.0, class_weight=0.5, dist_weight=0.2):
        super(MultiClassContactLoss, self).__init__()
        self.contact_weight = contact_weight  # Weight for binary contact errors
        self.class_weight = class_weight      # Weight for object class errors
        self.dist_weight = dist_weight        # Weight for distance-based errors
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted contact probabilities [B, C, V] 
                  (batch_size, num_classes, num_vertices)
            target: Ground truth contact labels [B, C, V]
        
        Returns:
            Total loss combining contact detection and class accuracy
        """
        batch_size, num_classes, num_vertices = pred.shape
        
        # 1. Binary Contact Loss - Any contact vs. no contact
        # Collapse class dimension to get binary contact prediction
        pred_any_contact = torch.max(pred, dim=1)[0]  # [B, V]
        target_any_contact = torch.max(target, dim=1)[0]  # [B, V]
        
        binary_loss = self.bce(pred_any_contact, target_any_contact).mean()
        
        # 2. Class-specific Contact Loss - Right object class
        # Only apply to vertices that have contact
        contact_mask = (target_any_contact > 0).unsqueeze(1).expand(-1, num_classes, -1)  # [B, C, V]
        
        if contact_mask.sum() > 0:
            class_loss = self.bce(pred, target)
            class_loss = (class_loss * contact_mask).sum() / (contact_mask.sum() + 1e-8)
        else:
            class_loss = torch.tensor(0.0).to(pred.device)
        
        # 3. Distance-based Loss - Penalize based on geodesic distance
        # Calculate false positive and false negative distances
        fp_dist, fn_dist = det_error_metric(pred_any_contact, target_any_contact)
        dist_loss = (fp_dist.mean() + fn_dist.mean()) / 2.0
        
        # Combine losses with weights
        total_loss = (
            self.contact_weight * binary_loss + 
            self.class_weight * class_loss + 
            self.dist_weight * dist_loss
        )
        
        return total_loss, {
            'binary_loss': binary_loss.item(),
            'class_loss': class_loss.item(),
            'dist_loss': dist_loss.item(),
            'total_loss': total_loss.item()
        }


class class_loss_function(nn.Module):
    def __init__(self):
        super(class_loss_function, self).__init__()
        self.ce_loss = nn.BCELoss()
        # self.ce_loss = nn.MultiLabelSoftMarginLoss()
        # self.ce_loss = nn.MultiLabelMarginLoss()

    def forward(self, y_true, y_pred, valid_mask):
        # Ensure predictions are in [0,1] range using sigmoid
        y_pred = torch.sigmoid(y_pred)
        
        bs = y_true.shape[0]
        if bs != 1:
            # Only select valid samples based on mask
            valid_indices = (valid_mask == 1)
            if valid_indices.sum() > 0:  # Check if there are any valid samples
                y_pred = y_pred[valid_indices]
                y_true = y_true[valid_indices]
            else:
                return torch.tensor(0.0).to(y_pred.device)
        
        # Additional safety check to ensure values are in [0,1]
        y_pred = torch.clamp(y_pred, 0.0, 1.0)
        
        if len(y_pred) > 0:
            return self.ce_loss(y_pred, y_true)
        else:
            return torch.tensor(0.0).to(y_pred.device)


class pixel_anchoring_function(nn.Module):
    def __init__(self, model_type, device='cuda'):
        super(pixel_anchoring_function, self).__init__()

        self.device = device

        self.model_type = model_type

        if self.model_type == 'smplx':
            # load mapping from smpl vertices to smplx vertices
            mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smpl_to_smplx.pkl")
            with open(mapping_pkl, 'rb') as f:
                smpl_to_smplx_mapping = pkl.load(f)
                smpl_to_smplx_mapping = smpl_to_smplx_mapping["matrix"]
            self.smpl_to_smplx_mapping = torch.from_numpy(smpl_to_smplx_mapping).float().to(self.device)


        # Setup the SMPL model
        if self.model_type == 'smpl':
            self.n_vertices = 6890
            self.body_model = SMPL(constants.SMPL_MODEL_DIR).to(self.device)
        if self.model_type == 'smplx':
            self.n_vertices = 10475
            self.body_model = SMPLX(constants.SMPLX_MODEL_DIR,
                                    num_betas=10,
                                    use_pca=False).to(self.device)
        self.body_faces = torch.LongTensor(self.body_model.faces.astype(np.int32)).to(self.device)

        self.ce_loss = nn.BCELoss()

    def get_posed_mesh(self, body_params, debug=False):
        betas = body_params['betas']
        pose = body_params['pose']
        transl = body_params['transl']

        # extra smplx params
        extra_args = {'jaw_pose': torch.zeros((betas.shape[0], 3)).float().to(self.device),
                      'leye_pose': torch.zeros((betas.shape[0], 3)).float().to(self.device),
                      'reye_pose': torch.zeros((betas.shape[0], 3)).float().to(self.device),
                      'expression': torch.zeros((betas.shape[0], 10)).float().to(self.device),
                      'left_hand_pose': torch.zeros((betas.shape[0], 45)).float().to(self.device),
                      'right_hand_pose': torch.zeros((betas.shape[0], 45)).float().to(self.device)}

        smpl_output = self.body_model(betas=betas,
                                      body_pose=pose[:, 3:],
                                      global_orient=pose[:, :3],
                                      pose2rot=True,
                                      transl=transl,
                                      **extra_args)
        smpl_verts = smpl_output.vertices
        smpl_joints = smpl_output.joints

        if debug:
            for mesh_i in range(smpl_verts.shape[0]):
                out_dir = 'temp_meshes'
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f'temp_mesh_{mesh_i:04d}.obj')
                save_results_mesh(smpl_verts[mesh_i], self.body_model.faces, out_file)
        return smpl_verts, smpl_joints


    def render_batch(self, smpl_verts, cam_k, img_scale_factor, vertex_colors=None, face_textures=None, debug=False):

        bs = smpl_verts.shape[0]

        # Incorporate resizing factor into the camera
        img_w = 256 # TODO: Remove hardcoding
        img_h = 256 # TODO: Remove hardcoding
        focal_length_x = cam_k[:, 0, 0] * img_scale_factor[:, 0]
        focal_length_y = cam_k[:, 1, 1] * img_scale_factor[:, 1]
        # convert to float for pytorch3d
        focal_length_x, focal_length_y = focal_length_x.float(), focal_length_y.float()

        # concatenate focal length
        focal_length = torch.stack([focal_length_x, focal_length_y], dim=1)

        # Setup renderer
        renderer = Pytorch3D(img_h=img_h,
                                  img_w=img_w,
                                  focal_length=focal_length,
                                  smpl_faces=self.body_faces,
                                  texture_mode='deco',
                                  vertex_colors=vertex_colors,
                                  face_textures=face_textures,
                                  is_train=True,
                                  is_cam_batch=True)
        front_view = renderer(smpl_verts)
        if debug:
            # visualize the front view as images in a temp_image folder
            for i in range(bs):
                front_view_rgb = front_view[i, :3, :, :].permute(1, 2, 0).detach().cpu()
                front_view_mask = front_view[i, 3, :, :].detach().cpu()
                out_dir = 'temp_images'
                os.makedirs(out_dir, exist_ok=True)
                out_file_rgb = os.path.join(out_dir, f'{i:04d}_rgb.png')
                out_file_mask = os.path.join(out_dir, f'{i:04d}_mask.png')
                cv2.imwrite(out_file_rgb, front_view_rgb.numpy()*255)
                cv2.imwrite(out_file_mask, front_view_mask.numpy()*255)

        return front_view

    def paint_contact(self, pred_contact):
        """
        Paints the contact vertices on the SMPL mesh

        Args:
            pred_contact: prbabilities of contact vertices

        Returns:
            pred_rgb: RGB colors for the contact vertices
        """
        bs = pred_contact.shape[0]

        # initialize black and while colors
        colors = torch.tensor([[0, 0, 0], [1, 1, 1]]).float().to(self.device)
        colors = torch.unsqueeze(colors, 0).expand(bs, -1, -1)

        # add another dimension to the contact probabilities for inverse probabilities
        pred_contact = torch.unsqueeze(pred_contact, 2)
        pred_contact = torch.cat((1 - pred_contact, pred_contact), 2)

        # get pred_rgb colors
        pred_vert_rgb = torch.bmm(pred_contact, colors)
        pred_face_rgb = pred_vert_rgb[:, self.body_faces, :][:, :, 0, :] # take the first vertex color
        pred_face_texture = torch.zeros((bs, self.body_faces.shape[0], 1, 1, 3), dtype=torch.float32).to(self.device)
        pred_face_texture[:, :, 0, 0, :] = pred_face_rgb
        return pred_vert_rgb, pred_face_texture

    def forward(self, pred_contact, body_params, cam_k, img_scale_factor, gt_contact_polygon, valid_mask):
        """
        Takes predicted contact labels (probabilities), transfers them to the posed mesh and
        renders to the image. Loss is computed between the rendered contact and the ground truth
        polygons from HOT.

        Args:
            pred_contact: predicted contact labels (probabilities)
            body_params: SMPL parameters in camera coords
            cam_k: camera intrinsics
            gt_contact_polygon: ground truth polygons from HOT
        """
        # convert pred_contact to smplx
        bs = pred_contact.shape[0]
        if self.model_type == 'smplx':
            smpl_to_smplx_mapping = self.smpl_to_smplx_mapping[None].expand(bs, -1, -1)
            pred_contact = torch.bmm(smpl_to_smplx_mapping, pred_contact[..., None])
            pred_contact = pred_contact.squeeze()

        # get the posed mesh
        smpl_verts, smpl_joints = self.get_posed_mesh(body_params)

        # paint the contact vertices on the mesh
        vertex_colors, face_textures = self.paint_contact(pred_contact)

        # render the mesh
        front_view = self.render_batch(smpl_verts, cam_k, img_scale_factor, vertex_colors, face_textures)
        front_view_rgb = front_view[:, :3, :, :].permute(0, 2, 3, 1)
        front_view_mask = front_view[:, 3, :, :]

        # compute segmentation loss between rendered contact mask and ground truth contact mask
        front_view_rgb = front_view_rgb[valid_mask == 1]
        gt_contact_polygon = gt_contact_polygon[valid_mask == 1]
        loss = self.ce_loss(front_view_rgb, gt_contact_polygon)
        return loss, front_view_rgb, front_view_mask


class SemanticContactLoss(nn.Module):
    """
    Loss function for semantic contact prediction
    
    Only applies to vertices that have contact
    """
    def __init__(self):
        super(SemanticContactLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, pred, target, contact_mask):
        """
        Args:
            pred: Predicted semantic class probabilities [B, C, V] 
                  (batch_size, num_classes, num_vertices)
            target: Ground truth semantic labels [B, C, V]
            contact_mask: Binary mask of contact vertices [B, V]
        
        Returns:
            Semantic classification loss
        """
        batch_size, num_classes, num_vertices = pred.shape
        
        # Only compute loss for vertices with contact
        if contact_mask.sum() > 0:
            # Convert one-hot target to class indices for CrossEntropyLoss
            # Find the index of the 1 in each vertex's one-hot encoding
            target_indices = torch.argmax(target, dim=1)  # [B, V]
            
            # Transpose pred to [B, V, C] for easier masking
            pred_t = pred.transpose(1, 2)  # [B, V, C]
            
            # Apply contact mask
            masked_pred = pred_t[contact_mask]  # [N, C] where N is number of contact vertices
            masked_target = target_indices[contact_mask]  # [N]
            
            # Compute loss
            loss = self.ce_loss(masked_pred, masked_target).mean()
        else:
            loss = torch.tensor(0.0).to(pred.device)
            
        return loss
