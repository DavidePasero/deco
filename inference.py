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


from models.deco import DECO, DINOContact
from models.vlm import VLMManager
from common import constants

os.environ['PYOPENGL_PLATFORM'] = 'egl'

object_classes = [
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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def initiate_model(args):
    if args.model_type == 'deco':
        print (args.train_backbone)
        deco_model = DECO(
            args.encoder,
            context=args.context,
            device=args.device,
            classifier_type=args.classifier_type,
            num_encoders=args.num_encoder,
            train_backbone=args.train_backbone,
            train_vlm_text_encoder=args.train_vlm_text_encoder,
            use_vlm=args.use_vlm,
            patch_cross_attention=args.patch_cross_attention,
        )
    elif args.model_type ==  'dinoContact':
        deco_model = DINOContact(args.device)
    else:
        raise ValueError('Model type not supported')

    logger.info(f'Loading weights from {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    key = "deco" if args.model_type == "deco" else "dinocontact"
    deco_model.load_state_dict(checkpoint[key], strict=True)

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
        images = list(glob.iglob(args.img_src + '/*', recursive=True))
    else:
        images = [args.img_src]

    print(images)
    print(".........................")
    print("Out dir: " + args.out_dir)
    deco_model = initiate_model(args)

    if args.use_vlm:
        vlm_manager = VLMManager()
        vlm_manager.generate_texts(images, batch_size=4)
    
    smpl_path = os.path.join(constants.SMPL_MODEL_DIR, 'smpl_neutral_tpose.ply')
    
    for img_name in images:
        img = cv2.imread(img_name)
        img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
        img = img.transpose(2,0,1)/255.0
        img = img[np.newaxis,:,:,:]
        img = torch.tensor(img, dtype = torch.float32).to(device)

        if args.use_vlm:
            text_features = [vlm_manager[img_name]]
        else:
            text_features = None

        if args.context:
            cont, _, _, semantic_logits = deco_model(img, vlm_feats=text_features)
        else:
            cont, semantic_logits = deco_model(img, vlm_feats=text_features)
        cont = cont.detach().cpu()                 # keep as Tensor
        semantic_logits = semantic_logits.detach().cpu()          # keep as Tensor
        
        cont_np = cont.numpy()[0]                  # [V] for later Python loops
        
        # cont            : [B, V]        – probabilities in [0,1]
        # semantic_logits : [B, C, V]     – raw logits (no softmax)

        # 1. Threshold the binary contact map
        contact_mask = (cont >= 0.5)                           # [B, V]   bool

        # 2. Find the winning object class per vertex
        #    (still shape [B, 1, V] so we can scatter)
        winning_idx = torch.argmax(semantic_logits, dim=1, keepdim=True)   # [B, 1, V]  long

        # 3. Build the output tensor
        semantic_cont = torch.zeros_like(semantic_logits, dtype=cont.dtype)  # [B, C, V]

        #    Put a 1 at the winning index for every vertex
        semantic_cont.scatter_(1, winning_idx, 1.0)             # ones at winners, zeros elsewhere

        # 4. Zero–out vertices predicted as “no contact”
        semantic_cont *= contact_mask.unsqueeze(1).float()      # keep only vertices with cont ≥ 0.5

        # Get contact vertices
        cont_smpl = np.where(cont_np >= 0.5)[0].tolist()
        
        img = img.detach().cpu().numpy()		
        img = np.transpose(img[0], (1, 2, 0))		
        img = img * 255		
        img = img.astype(np.uint8)
        
        # Create mesh with default color
        body_model_smpl = trimesh.load(smpl_path, process=False)
        for vert in range(body_model_smpl.visual.vertex_colors.shape[0]):
            body_model_smpl.visual.vertex_colors[vert] = args.mesh_colour
        
        # Find which classes are actually present in the predictions
        # Build per‑vertex → class mapping
        vertex_classes = {}
        for vertex_idx in cont_smpl:                       # cont_smpl from your loop
            class_idx = int(np.argmax(semantic_cont[0, :, vertex_idx]))
            vertex_classes.setdefault(class_idx, []).append(vertex_idx)

        # Create a palette for the classes that really appear
        present_classes = sorted(vertex_classes.keys())
        num_present      = len(present_classes)
        cmap_name        = 'tab10' if num_present <= 10 else 'tab20'
        cmap             = plt.cm.get_cmap(cmap_name, num_present)

        class_colors = {
            class_idx: np.array([*(np.asarray(cmap(i)[:3]) * 255).astype(int), 255])
            for i, class_idx in enumerate(present_classes)
        }

        # Apply colors to the mesh
        for class_idx, vertices in vertex_classes.items():
            body_model_smpl.visual.vertex_colors[vertices] = class_colors[class_idx]

        # ---------- NEW: build class‑name dictionary here ----------
        # Build class‑name dictionary using the global `object_classes` list
        class_names = {
            idx: object_classes[idx] if idx < len(object_classes) else f"Class {idx}"
            for idx in present_classes
        }

        # Create the colour legend
        legend_img = create_color_legend(class_colors, class_names, img.shape[0])
        
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D Mesh Rendering and Semantic Contact Visualization")
    parser.add_argument('--img_src', help='Source of image(s). Can be file or directory',
                        default='./example_images', type=str)
    parser.add_argument('--out_dir', help='Where to store images',
                        default='./demo_out_help', type=str)
    parser.add_argument('--model_path', help='Path to best model weights',
                        default='./checkpoints/Other_Checkpoints/deco_dino_no_vlm_rich_pca_best.pth', type=str)
    parser.add_argument('--mesh_colour', help='Colour of the mesh', nargs='+',
                        type=int, default=[130, 130, 130, 255])
    parser.add_argument('--annot_colour', help='Colour of the annotation', nargs='+',
                        type=int, default=[0, 255, 0, 255])
    parser.add_argument('--model_type', help='Type of the model to load (deco or dinoContact)',
                    default='deco', type=str)
    parser.add_argument('--encoder', help='Flag to train the encoder',
                        type=str, default="dinov2-giant")
    parser.add_argument('--num_encoder', help='Number of encodersr',
                        type=int, default=2)
    parser.add_argument('--classifier_type', help='Classifier type for the model',
                        default='shared', type=str)
    parser.add_argument('--device', help='Device to use (cuda or cpu)',
                        default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--train-backbone', action='store_true')
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--use-vlm', action='store_true')
    parser.add_argument('--train-vlm-text-encoder', action='store_true')
    parser.add_argument('--patch-cross-attention', action='store_true')

    args = parser.parse_args()
    main(args)
