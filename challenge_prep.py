import pickle

import torch
import os
import glob
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from loguru import logger
import zipfile


from models.deco import DECO, DINOContact
from models.vlm import VLMManager
from common import constants

os.environ['PYOPENGL_PLATFORM'] = 'egl'

object_classes = [
    'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove',
    'bed', 'bench', 'bicycle', 'boat', 'book', 'bottle', 'bowl', 'broccoli',
    'bus', 'cake', 'car', 'carrot', 'cell_phone', 'chair', 'clock', 'couch',
    'cup', 'dining table', 'donut', 'fire_hydrant', 'fork', 'frisbee',
    'hair_drier', 'handbag', 'hot_dog', 'keyboard', 'kite', 'knife', 'laptop',
    'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking_meter',
    'pizza', 'potted_plant', 'refrigerator', 'remote', 'sandwich', 'scissors',
    'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball',
    'stop_sign', 'suitcase', 'supporting', 'surfboard', 'teddy_bear',
    'tennis_racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic_light',
    'train', 'truck', 'tv', 'umbrella', 'vase', 'wine_glass'
]

pred_obj_classes = [
    'motorcycle', 'bicycle', 'boat', 'car', 'truck', 'bus', 'train', 'backpack', 'tie', 'handbag',
    'baseball glove', 'bench', 'chair', 'couch', 'bed', 'toilet',
    'dining table', 'book', 'umbrella', 'cell phone', 'laptop', 'kite', 'suitcase', 'bottle', 'remote', 'toothbrush', 'teddy bear', 'scissors', 'clock', 'frisbee', 'sports ball', 'tennis racket', 'baseball bat', 'skateboard', 'snowboard', 'skis', 'surfboard', 'banana', 'cake', 'apple', 'carrot', 'pizza', 'donut', 'hot dog', 'sandwich', 'broccoli', 'knife', 'spoon', 'cup', 'wine glass', 'fork'
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


def main(args):
    if os.path.isdir(args.img_src):
        images = list(glob.iglob(args.img_src + '/*', recursive=True))
    else:
        images = [args.img_src]

    object_classes = [
        'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove',
        'bed', 'bench', 'bicycle', 'boat', 'book', 'bottle', 'bowl', 'broccoli',
        'bus', 'cake', 'car', 'carrot', 'cell phone', 'chair', 'clock', 'couch',
        'cup', 'dining table', 'donut', 'fire hydrant', 'fork', 'frisbee',
        'hair drier', 'handbag', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop',
        'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter',
        'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors',
        'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball',
        'stop sign', 'suitcase', 'supporting', 'surfboard', 'teddy bear',
        'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
        'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass'
    ]

    obj_mapping = {i: v for i,v in enumerate(object_classes)}

    rev_obj_mapping = {v: k for k,v in obj_mapping.items()}

    if args.use_vlm:
        vlm_manager = VLMManager()
        vlm_manager.generate_texts(images, batch_size=4)

    deco_model = initiate_model(args)
    challenge_data = {}

    for img_name in tqdm(images, desc="Prepping images..."):
        img = cv2.imread(img_name)
        if args.use_vlm:
            text_features = vlm_manager[img_name]
        else:
            text_features = None
        img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
        img = img.transpose(2, 0, 1) / 255.0
        img = img[np.newaxis, :, :, :]
        img = torch.tensor(img, dtype=torch.float32).to(device)

        if args.context:
            cont, _, _, sem_cont = deco_model(img, vlm_feats=[text_features])
        else:
            cont, sem_cont = deco_model(img, vlm_feats=[text_features])
        cont = cont.detach().cpu().numpy().squeeze()

        # Get contact vertices
        sem_cont_per_vid = torch.argmax(sem_cont, dim=1).squeeze()

        cont_smpl = []
        for indx, i in enumerate(cont):
            if i >= 0.5:
                cont_smpl.append(1)
            else:
                cont_smpl.append(0)

        vertex_mask = [cont > 0.5]
        cont_sem_chl = {}
        for obj_name in pred_obj_classes:
            obj_idx = rev_obj_mapping[obj_name]
            object_mask = sem_cont_per_vid == obj_idx
            cont_sem_chl[obj_name] = torch.where((object_mask & torch.tensor(vertex_mask).cuda()))[1].tolist()


        challenge_data[img_name.split("./")[1]] = {"gen_contact_vids": cont, "sem_contact_vids": cont_sem_chl}

    os.remove("results.pkl") if os.path.exists( "results.pkl") else []

    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, "results.pkl"), "wb") as f:
        pickle.dump(challenge_data, f)

    with open("results.pkl", "wb") as f:
        pickle.dump(challenge_data, f)

    with zipfile.ZipFile(os.path.join(args.out_path, "submission.zip"), 'w') as zipf:
        zipf.write("results.pkl")

    os.remove("results.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D Mesh Rendering and Semantic Contact Visualization")
    parser.add_argument('--img_src', help='Source of image(s). Can be file or directory',
                        default='./datasets/HOT-Annotated/test', type=str)
    parser.add_argument("--out_path", default="challenges/pred", type=str)
    parser.add_argument('--model_path', help='Path to best model weights',
                        default='/home/lukas/Projects/deco/checkpoints/Other_Checkpoints/deco-pathc-ca_best.pth'
                        , type=str)
    parser.add_argument('--model_type', help='Type of the model to load (deco or dinoContact)',
                        default='deco', type=str)
    parser.add_argument('--encoder', help='Flag to train the encoder',
                        type=str, default="dinov2-large")
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

