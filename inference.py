import torch
import os
import glob
import argparse
import numpy as np
import cv2
import PIL.Image as pil_img
from loguru import logger
import shutil
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont

import trimesh
import pyrender

from models.deco import DECO
from common import constants

os.environ['PYOPENGL_PLATFORM'] = 'egl'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def initiate_model(args):
    deco_model = DECO('hrnet', True, device)

    logger.info(f'Loading weights from {args.model_path}')
    checkpoint = torch.load(args.model_path)
    deco_model.load_state_dict(checkpoint['deco'], strict=True)

    deco_model.eval()

    return deco_model

def render_image(scene, img_res, img=None, viewer=False):
    '''
    Render the given pyrender scene and return the image. Can also overlay the mesh on an image.
    '''
    if viewer:
        pyrender.Viewer(scene, use_raymond_lighting=True)
        return 0
    else:
        r = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        if img is not None:
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img.detach().cpu().numpy()
            output_img = (color[:, :, :-1] * valid_mask +
                          (1 - valid_mask) * input_img)
        else:
            output_img = color
        return output_img

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

    for sideview_angle in [0, 90, 180, 270]:
        out_mesh = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(
            np.radians(sideview_angle), [0, 1, 0])
        out_mesh.apply_transform(rot)
        out_mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)
        mesh_pose = np.eye(4)
        scene.add(out_mesh, pose=mesh_pose, name='mesh')
        output_img = render_image(scene, img_res)
        output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        output_img = np.asarray(output_img)[:, :, :3]
        mesh_images.append(output_img)
        # delete the previous mesh
        prev_mesh = scene.get_nodes(name='mesh').pop()
        scene.remove_node(prev_mesh)

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
        output_img = render_image(scene, img_res)
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
    return IMG    

def main(args):
    if os.path.isdir(args.img_src):
        images = glob.iglob(args.img_src + '/*', recursive=True)
    else:
        images = [args.img_src]

    deco_model = initiate_model(args)
    
    smpl_path = os.path.join(constants.SMPL_MODEL_DIR, 'smpl_neutral_tpose.ply')
    
    for img_name in images:
        img = cv2.imread(img_name)
        img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
        img = img.transpose(2,0,1)/255.0
        img = img[np.newaxis,:,:,:]
        img = torch.tensor(img, dtype = torch.float32).to(device)

        cont, semantic_cont = deco_model(img)
        cont = cont.detach().cpu().numpy().squeeze()
        semantic_cont = semantic_cont.detach().cpu().numpy()  # [batch, num_classes, num_vertices]
        
        # Get contact vertices
        cont_smpl = []
        for indx, i in enumerate(cont):
            if i >= 0.5:
                cont_smpl.append(indx)
        
        img = img.detach().cpu().numpy()		
        img = np.transpose(img[0], (1, 2, 0))		
        img = img * 255		
        img = img.astype(np.uint8)
        
        # Create mesh with default color
        body_model_smpl = trimesh.load(smpl_path, process=False)
        for vert in range(body_model_smpl.visual.vertex_colors.shape[0]):
            body_model_smpl.visual.vertex_colors[vert] = args.mesh_colour
        
        # Find which classes are actually present in the predictions
        if len(cont_smpl) > 0:
            # For each contact vertex, get the predicted class
            vertex_classes = {}
            for vertex_idx in cont_smpl:
                class_idx = np.argmax(semantic_cont[0, :, vertex_idx])
                if class_idx not in vertex_classes:
                    vertex_classes[class_idx] = []
                vertex_classes[class_idx].append(vertex_idx)
            
            # Create colors only for the classes that appear in this image
            present_classes = sorted(vertex_classes.keys())
            num_present_classes = len(present_classes)
            
            # Create a colormap with distinct colors for the present classes
            cmap = plt.cm.get_cmap('tab10' if num_present_classes <= 10 else 'tab20', num_present_classes)
            class_colors = {}
            for i, class_idx in enumerate(present_classes):
                # Convert matplotlib color to RGBA
                r, g, b, a = cmap(i)
                class_colors[class_idx] = np.array([int(r*255), int(g*255), int(b*255), 255])
            
            # Color vertices based on their class
            for class_idx, vertices in vertex_classes.items():
                for vertex_idx in vertices:
                    body_model_smpl.visual.vertex_colors[vertex_idx] = class_colors[class_idx]
            
            # Create a legend for the present classes
            if args.class_names:
                class_names = {idx: args.class_names[idx] for idx in present_classes}
            else:
                class_names = {idx: f"Class {idx}" for idx in present_classes}
            
            legend_img = create_color_legend(class_colors, class_names, img.shape[0])
        else:
            # No contacts detected
            legend_img = create_empty_legend(img.shape[0])
        
        # Render the mesh
        rend = create_scene(body_model_smpl, img)
        
        # Save the rendered image
        os.makedirs(os.path.join(args.out_dir, 'Renders'), exist_ok=True) 
        rend.save(os.path.join(args.out_dir, 'Renders', os.path.basename(img_name).split('.')[0] + '.png'))
        
        # Save the legend
        legend_img.save(os.path.join(args.out_dir, 'Renders', os.path.basename(img_name).split('.')[0] + '_legend.png'))
                  
        out_dir = os.path.join(args.out_dir, 'Preds', os.path.basename(img_name).split('.')[0])
        os.makedirs(out_dir, exist_ok=True)          

        logger.info(f'Saving mesh to {out_dir}')
        shutil.copyfile(img_name, os.path.join(out_dir, os.path.basename(img_name)))
        body_model_smpl.export(os.path.join(out_dir, 'pred.obj'))

def create_color_legend(class_colors, class_names, height=256):
    """Create a legend image showing the color mapping for semantic classes"""
    # Create a PIL image for the legend
    legend_width = 200
    box_height = 30
    legend_height = box_height * len(class_colors)
    legend = pil_img.new('RGB', (legend_width, legend_height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw color boxes and labels
    for i, (class_idx, color) in enumerate(class_colors.items()):
        y = i * box_height
        # Draw colored rectangle
        draw.rectangle([(10, y + 5), (40, y + 25)], fill=tuple(color[:3]))
        # Draw class name
        draw.text((50, y + 10), class_names[class_idx], fill=(0, 0, 0), font=font)
    
    # Resize to match the height of the input image if needed
    if height != legend_height:
        legend = legend.resize((legend_width, height), pil_img.LANCZOS)
    
    return legend

def create_empty_legend(height=256):
    """Create a legend indicating no contacts were detected"""
    legend_width = 200
    legend = pil_img.new('RGB', (legend_width, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text((10, height//2), "No contacts detected", fill=(0, 0, 0), font=font)
    
    return legend

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_src', help='Source of image(s). Can be file or directory', default='./demo_out', type=str)
    parser.add_argument('--out_dir', help='Where to store images', default='./demo_out', type=str)
    parser.add_argument('--model_path', help='Path to best model weights', default='./checkpoints/Release_Checkpoint/deco_best.pth', type=str)
    parser.add_argument('--mesh_colour', help='Colour of the mesh', nargs='+', type=int, default=[130, 130, 130, 255])
    parser.add_argument('--annot_colour', help='Colour of the mesh', nargs='+', type=int, default=[0, 255, 0, 255])
    args = parser.parse_args()

    main(args)
