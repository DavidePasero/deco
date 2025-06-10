import torch
import os
os.environ["PYOPENGL_PLATFORM"] = "pyglet"
import glob
import argparse
import numpy as np
import cv2
import PIL.Image as pil_img
from loguru import logger
import shutil

import io
from PIL import Image

import trimesh
import pyrender

from models.deco import DECO
from common import constants

import matplotlib.pyplot as plt
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

def initiate_model(args, device):
    deco_model = DECO('hrnet', True, device)

    logger.info(f'Loading weights from {args.model_path}')

    checkpoint = torch.load(args.model_path, map_location=device)
        
    deco_model.load_state_dict(checkpoint['deco'], strict=True)

    deco_model.eval()

    return deco_model

def render_image(mesh, img_res, img=None):
    try:
        # Create a trimesh scene
        scene = trimesh.Scene()
        scene.add_geometry(mesh)

        # Render to PNG (as bytes)
        png = scene.save_image(resolution=(img_res, img_res), visible=False)
        image = pil_img.open(trimesh.util.wrap_as_stream(png)).convert('RGBA')
        color = np.array(image).astype(np.float32) / 255.0

        if img is not None:
            img_resized = cv2.resize(img, (color.shape[1], color.shape[0]))
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img_resized / 255.0 if img.max() > 1 else img_resized
            output_img = (color[:, :, :-1] * valid_mask +
                          (1 - valid_mask) * input_img)
        else:
            output_img = color[:, :, :-1]

        return (output_img * 255).astype(np.uint8)

    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        return (np.ones((img_res, img_res, 3), dtype=np.uint8) * 255)

def create_scene(mesh, img, focal_length=500, camera_center=250, img_res=500):
    # Setup the scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=(0.3, 0.3, 0.3))
    # add mesh for camera
    camera_pose = np.eye(4)
    camera_rotation = np.eye(3, 3)
    camera_translation = np.array([0., 0, 2.5])
    camera_pose[:3, :3] = camera_rotation
    camera_pose[:3, 3] = camera_rotation @ camera_translation
    pyrencamera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center, cy=camera_center)
    scene.add(pyrencamera, pose=camera_pose)
    # create and add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)
    for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
        light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
        # out_mesh.vertices.mean(0) + np.array(lp)
        scene.add(light, pose=light_pose)
    # add body mesh
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh_images = []

    # resize input image to fit the mesh image height
    img_height = img_res
    img_width = int(img_height * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (img_width, img_height))
    mesh_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for angle in [0, 90, 180, 270]:
        out_mesh = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])
        out_mesh.apply_transform(rot)

        # Only pass trimesh object here!
        output_img = render_image(out_mesh, img_res, img)
        output_img = (output_img * 255).astype(np.uint8)
        output_img = cv2.resize(output_img, (img_width, img_height))  # Ensure same shape!
        mesh_images.append(output_img)


    # show upside down view
    for topview_angle in [90, 270]:
        out_mesh = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(
            np.radians(topview_angle), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)
        mesh_pose = np.eye(4)
        scene.add(out_mesh, pose=mesh_pose, name='mesh')
        # output_img = render_image(scene, img_res)
        output_img = render_image(out_mesh, img_res, img)
        output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        output_img = np.asarray(output_img)[:, :, :3]
        mesh_images.append(output_img)
        # delete the previous mesh
        prev_mesh = scene.get_nodes(name='mesh').pop()
        scene.remove_node(prev_mesh)

    # stack images
    IMG = np.hstack(mesh_images)
    IMG = pil_img.fromarray(IMG)
    IMG.thumbnail((3000, 3000))
    # return pil_img.fromarray((np.ones((img_res, img_res, 3), dtype=np.uint8) * 255))
    return IMG    

def main(args):
    if os.path.isdir(args.img_src):
        images = glob.iglob(args.img_src + '/*', recursive=True)
    else:
        images = [args.img_src]
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    deco_model = initiate_model(args)
    
    smpl_path = os.path.join(constants.SMPL_MODEL_DIR, 'smpl_neutral_tpose.ply')
    
    for img_name in images:
        img = cv2.imread(img_name)
        img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
        img = img.transpose(2,0,1)/255.0
        img = img[np.newaxis,:,:,:]
        img = torch.tensor(img, dtype = torch.float32).to(device)

        cont, sem_mask_pred, part_mask_pred = deco_model(img)

        scene_mask = torch.argmax(F.softmax(sem_mask_pred, dim=1), dim=1)
        part_mask = torch.argmax(F.softmax(part_mask_pred, dim=1), dim=1)
        unique_values = torch.unique(part_mask)

        def save_mask(mask_tensor, filename):
            plt.imsave(filename, mask_tensor[0].cpu().numpy(), cmap='jet')

        
        
    # save_mask(contact_mask.float(), "contact_mask.png")

        cont = cont.detach().cpu().numpy().squeeze()
        cont_smpl = []
        for indx, i in enumerate(cont):
            if i >= 0.5:
                cont_smpl.append(indx)
        
        img = img.detach().cpu().numpy()		
        img = np.transpose(img[0], (1, 2, 0))		
        img = img * 255		
        img = img.astype(np.uint8)
        
        contact_smpl = np.zeros((1, 1, 6890))
        contact_smpl[0][0][cont_smpl] = 1

        body_model_smpl = trimesh.load(smpl_path, process=False)
        for vert in range(body_model_smpl.visual.vertex_colors.shape[0]):
            body_model_smpl.visual.vertex_colors[vert] = args.mesh_colour
        body_model_smpl.visual.vertex_colors[cont_smpl] = args.annot_colour

        rend = create_scene(body_model_smpl, img)
        os.makedirs(os.path.join(args.out_dir, 'Renders'), exist_ok=True) 
        rend.save(os.path.join(args.out_dir, 'Renders', os.path.basename(img_name).split('.')[0] + '.png'))
                  
        out_dir = os.path.join(args.out_dir, 'Preds', os.path.basename(img_name).split('.')[0])
        os.makedirs(out_dir, exist_ok=True)          

        logger.info(f'Saving mesh to {out_dir}')
        shutil.copyfile(img_name, os.path.join(out_dir, os.path.basename(img_name)))
        body_model_smpl.export(os.path.join(out_dir, 'pred.obj'))

        save_mask(scene_mask, os.path.join(out_dir, 'scene_mask.png'))
        save_mask(part_mask, os.path.join(out_dir, 'part_mask.png'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_src', help='Source of image(s). Can be file or directory', default='./demo_out', type=str)
    parser.add_argument('--out_dir', help='Where to store images', default='./demo_out', type=str)
    parser.add_argument('--model_path', help='Path to best model weights', default='./checkpoints/Release_Checkpoint/deco_best.pth', type=str)
    parser.add_argument('--mesh_colour', help='Colour of the mesh', nargs='+', type=int, default=[130, 130, 130, 255])
    parser.add_argument('--annot_colour', help='Colour of the mesh', nargs='+', type=int, default=[0, 255, 0, 255])
    args = parser.parse_args()

    main(args)

