# python scripts/datascripts/get_part_seg_mask.py --data_npz data/rich_val_smplx_small.npz --model_type 'smplx'
# python utils/get_part_seg_mask.py --path_file rich_paths --data_npz datasets/Release_Datasets/rich/rich_train_smplx_cropped_bmp.npz --model_type 'smplx'
# python utils/get_part_seg_mask.py --data_npz datasets/Release_Datasets/rich/rich_train_smplx_cropped_bmp.npz --model_type 'smplx'

import os
import sys
sys.path.append('/home/scur0729/deco_davide')
import cv2
import argparse
import numpy as np
import torch
from common import constants
from models.smpl import SMPL
from smplx import SMPLX
from utils.mesh_utils import save_results_mesh
import trimesh
from tqdm import tqdm
from utils.image_utils import get_body_part_texture, generate_part_labels
from utils.diff_renderer import Pytorch3D

class PART_LABELER:
    def __init__(self, body_params, img_w, img_h, model_type, gpu=0, debug=False):
        """
        Get part segmentation masks for images

        Args:
            body_params: SMPL parameters
            img_w: image width
            img_h: image height
            model_type: 'smpl' or 'smplx'
        """
        self.device = torch.device('cuda:{}'.format(gpu)) if torch.cuda.is_available() else torch.device('cpu')
        print("Running on:", self.device)

        self.model_type = model_type

        # Setup the SMPL model
        if self.model_type == 'smpl':
            self.body_model = SMPL(constants.SMPL_MODEL_DIR).to(self.device)
        if self.model_type == 'smplx':
            self.body_model = SMPLX(constants.SMPLX_MODEL_DIR,
                                    num_betas=10,
                                    use_pca=False).to(self.device)

        self.body_part_vertex_colors, self.body_part_texture = get_body_part_texture(self.body_model.faces,
                                                                                     model_type=self.model_type,
                                                                                     non_parametric=False)
        # bins are discrete part labels, add eps to avoid quantization error
        eps = 1e-2
        # self.part_label_bins = (torch.arange(int(constants.N_PARTS)) / float(constants.N_PARTS)) + eps
        self.part_label_bins = torch.linspace(0, constants.N_PARTS-1, constants.N_PARTS) + eps

        ## Run SMPL forward
        self.body_params = body_params

        self.smpl_verts, self.smpl_joints = self.get_posed_mesh(debug)

        # Assumbe same focal lenght for all frames in a seq
        focal_length = self.body_params['cam_k'][0, 0, 0]
        # focal_length = focal_length[0]
        # Setup Pyrender renderer
        # self.renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
        #                          faces=self.smpl_model.faces,
        #                          same_mesh_color=False)

        # Setup Pytorch3D Renderer
        focal_length = torch.FloatTensor([focal_length])
        smpl_faces = torch.from_numpy(self.body_model.faces.astype(np.int32)).to(self.device)
        self.renderer = Pytorch3D(img_h=img_h,
                                  img_w=img_w,
                                  focal_length=focal_length,
                                  smpl_faces=smpl_faces,
                                  texture_mode='partseg',
                                  vertex_colors=self.body_part_vertex_colors,
                                  face_textures=self.body_part_texture,
                                  model_type=self.model_type)
        #--------fleur----------
        self.part_label_bins = self.part_label_bins.to(self.device)
        self.smpl_verts_tensor = torch.from_numpy(self.smpl_verts).float().to(self.device)

    def get_posed_mesh(self, debug=False):
        betas = torch.from_numpy(self.body_params['shape']).float().to(self.device)
        pose = torch.from_numpy(self.body_params['pose']).float().to(self.device)
        transl = torch.from_numpy(self.body_params['transl']).float().to(self.device)

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
        smpl_verts = smpl_output.vertices.detach().cpu().numpy()
        smpl_joints = smpl_output.joints.detach().cpu().numpy()

        if debug:
            for mesh_i in range(smpl_verts.shape[0]):
                out_dir = 'temp_meshes'
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f'temp_mesh_{mesh_i:04d}.obj')
                save_results_mesh(smpl_verts[mesh_i], self.body_model.faces, out_file)
        return smpl_verts, smpl_joints

    def bucketize_part_image(self, color_rgb, mask):
        # make single channel
        body_parts = color_rgb.clone()
        body_parts *= 255.  # multiply it with 255 to make labels distant
        body_parts = body_parts.max(-1)[0]  # reduce to single channel
        body_parts = torch.bucketize(body_parts, self.part_label_bins, right=True)  # np.digitize(body_parts, bins, right=True)
        # add 1 to make background label 0
        body_parts = body_parts.long() + 1
        body_parts = body_parts * mask.detach()
        return body_parts.long()

    def create_part_masks(self, body_parts):
        # extract every pixel as a separate mask
        part_masks = []
        for part_id in range(1, constants.N_PARTS+1): # first one is for background
            part_mask = (body_parts == part_id)
            part_masks.append(part_mask)
        return part_masks
#------------orig---------------
    # def render_part_mask_p3d(self, img_paths, out_dir):
    #     with torch.no_grad():
    #         # os.makedirs(out_dir, exist_ok=True)
    #         for index, img_path in tqdm(enumerate(img_paths), dynamic_ncols=True):
    #             # Load the image
    #             if not os.path.exists(img_path):
    #                 if 'train' in img_path:
    #                     split = 'train'
    #                 elif 'val' in img_path:
    #                     split = 'val'
    #                 else:
    #                     split = 'test'        
    #                 new_img_name = img_path[img_path.index(split)+4:].replace('/', '_')
    #                 new_path = os.path.join('/is/cluster/work/achatterjee/rich/images', split, new_img_name.replace('jpeg', 'bmp'))
    #                 if not os.path.exists(new_path):
    #                     new_path = new_path.replace('bmp', 'png')
    #                 img_path = new_path  

    #             if os.path.exists(out_dir[index]):
    #                 continue 
    #             # img_bgr = cv2.imread(img_path)

    #             chosen_vert_arr = torch.FloatTensor(self.smpl_verts[[index]]).to(self.device)
    #             front_view = self.renderer(chosen_vert_arr)
    #             front_view_rgb = front_view[0, :3, :, :].permute(1,2,0).detach().cpu()
    #             front_view_mask = front_view[0, 3, :, :].detach().cpu()
               
    #             # front_view_depth = front_view[0, 4, :, :].detach().cpu()
             
    #             body_parts = self.bucketize_part_image(front_view_rgb, front_view_mask)
    #             body_parts = body_parts.numpy()
                
    #             front_view_rgb = front_view_rgb.numpy()

    #             # body_part_masks = self.create_part_masks(body_parts)
    #             # display part masks
    #             # for part_id, part_mask in enumerate(body_part_masks):
    #             #     part_mask = part_mask * 255
    #             #     part_dir = os.path.join(out_dir, f'frame_{index:04d}_parts')
    #             #     os.makedirs(part_dir, exist_ok=True)
    #             #     out_file = os.path.join(part_dir, f'part_{part_id:02d}_{index:04d}.png')
    #             #     cv2.imwrite(out_file, part_mask)

    #             # out_file = os.path.join(out_dir, f'front_view_{index:04d}.png')
    #             # cv2.imwrite(out_file, front_view_rgb[: ,:, [2, 1, 0]]*255)
    #             # print(f'wrote front view to {out_file}')
    #             body_parts = cv2.merge((body_parts, body_parts, body_parts))
    #             # out_file = os.path.join(out_dir, f'body_parts_{index:04d}.png')
    #             out_file = out_dir[index]
    #             cv2.imwrite(out_file, body_parts)
    #             # print(f'wrote body part masks to {out_file}')
  
    def render_part_mask_p3d(self, img_paths, out_dir, batch_size=32):
        with torch.no_grad():
            bins = self.part_label_bins
            # os.makedirs(out_dir, exist_ok=True)
            for index, img_path in tqdm(enumerate(img_paths), dynamic_ncols=True):
                # Load the image
                if not os.path.exists(img_path):
                    if 'train' in img_path:
                        split = 'train'
                    elif 'val' in img_path:
                        split = 'val'
                    else:
                        split = 'test'        
                    new_img_name = img_path[img_path.index(split)+4:].replace('/', '_')
                    new_path = os.path.join('/home/scur0729/deco_davide/rich/images', split, new_img_name.replace('jpeg', 'bmp'))
                    if not os.path.exists(new_path):
                        new_path = new_path.replace('bmp', 'png')
                    img_path = new_path  

                if os.path.exists(out_dir[index]):
                    continue 

                verts_batch = self.smpl_verts_tensor[index : index+1]      # (B, V, 3)
                outs = self.renderer(verts_batch)                         # (B, 5, H, W)

                # bucketize all at once on GPU
                colors = (outs[:, :3] * 255).max(dim=1).values            # (B, H, W)
                masks  = outs[:, 3].long()                                # (B, H, W)
                parts  = torch.bucketize(colors, bins, right=True) + 1    # (B, H, W)
                parts  = (parts * masks).byte()                           # (B, H, W)

                body_parts = parts[0].detach().cpu().numpy().astype(np.uint8)

                out_file = out_dir[index]
                cv2.imwrite(out_file, body_parts)


def main(args):
    out_dir = args.out_dir
    data_md = np.load(args.data_npz)

    # get all the jpg files in the folder
    img_paths = data_md['imgname']
    seg_paths = data_md['part_seg']

    path_to_seg = dict(zip(img_paths, seg_paths))

    with open(args.path_file, "r") as f:
        paths = [line.strip() for line in f]

    # filter img_paths to only those in args.path_file, and get their seg_paths
    filtered_img_paths = [img for img in img_paths if img in paths]
    filtered_seg_paths = [path_to_seg[img] for img in filtered_img_paths]


    img_paths, seg_paths = filtered_img_paths, filtered_seg_paths
    print(f'found {len(img_paths)} images')
    # load first image
    img = cv2.imread(img_paths[0])
    img_h, img_w, _ = img.shape

    labeler = PART_LABELER(body_params=data_md, img_w=img_w, img_h=img_h,
                           model_type=args.model_type, gpu=args.gpu, debug=args.debug)
    labeler.render_part_mask_p3d(img_paths=img_paths, out_dir=seg_paths)
    print("all images saved")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_file', type=str, default='rich_paths', help='path_file folder')
    parser.add_argument('--data_npz', type=str, default='.', help='folder with smpl/smpl-x npz')
    parser.add_argument('--model_type', type=str, default='smplx', choices=['smpl', 'smplx'], help='model type')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    args = parser.parse_args()

    main(args)
