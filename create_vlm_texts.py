from models.vlm import VLMManager
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='path to image directory')
    parser.add_argument('--out_dir', type=str, default="./cache/vlm_texts_cache", help='path to output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    args = parser.parse_args()
    vlm_manager = VLMManager(device="cpu")
    imgs = [os.path.join(args.img_dir, x) for x in os.listdir(args.img_dir)]
    vlm_manager.generate_texts(imgs, args.batch_size)
