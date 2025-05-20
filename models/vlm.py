import torchvision.transforms as tt
import torch
import numpy as np
from typing import List, Union
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm
import hashlib
from pathlib import Path
from torch.utils.data import DataLoader


class VLMFeatureCache:
    def __init__(self, cache_dir: str = "./cache/vlm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, image_path: str) -> str:
        return hashlib.md5(image_path.encode()).hexdigest()

    def exists(self, image_path: str) -> bool:
        file_path = self.cache_dir / f"{self._generate_id(image_path)}.pt"
        return file_path.exists()

    def load(self, image_path: str) -> torch.Tensor:
        file_path = self.cache_dir / f"{self._generate_id(image_path)}.pt"
        return torch.load(file_path, map_location="cpu")

    def save(self, image_path: str, features: torch.Tensor):
        assert len(features.shape) == 3, "Features are not three-dimensional"
        file_path = self.cache_dir / f"{self._generate_id(image_path)}.pt"
        torch.save(features, file_path)


class VLMManager:
    def __init__(self,
                 vlm_id: str = "HuggingFaceTB/SmolVLM-Instruct", device: str = "cuda",
                 cache_dir: str = "./cache/vlm_cache"):
        self.vlm_id = vlm_id
        self.device = device
        self.cache = VLMFeatureCache(cache_dir)

        self._cache_check = False

    def _load_vlm(self):
        self.device = self.device
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_id)
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            self.vlm_id, torch_dtype=torch.float16
        ).to(self.device)

    def _close_vlm(self):
        del self.vlm_processor
        del self.vlm_model

    def _with_vlm(self, func, *args, **kwargs):
        """
        Internal context manager for loading and unloading the VLM model.
        """
        try:
            print("Loading VLM model into memory...")
            self._load_vlm()
            result = func(*args, **kwargs)
            return result

        finally:
            print("Unloading VLM model from memory...")
            self._close_vlm()
            torch.cuda.empty_cache()

    def _register_hook(self):
        extracted_features = []

        def hook_fn(module, input, output):
            extracted_features.append(output.clone())

        handle = self.vlm_model.model.text_model.norm.register_forward_hook(hook_fn)
        return handle, extracted_features

    def _prepare_inputs(self, images, prompt):
        return self.vlm_processor(text=prompt, images=images, return_tensors="pt", padding=True).to(self.device)

    def _process_image(self, image, debug=False):
        trans = tt.ToPILImage()
        pil_image = trans(image) if isinstance(image, torch.Tensor) else image
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Describe exactly which objects the human is in contact with, what action is being performed and with what body part"}
            ]
        }]
        prompt = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._prepare_inputs([pil_image], prompt)
        return inputs

    def extract_features(self, images, debug=False):
        return self._with_vlm(self._extract_features, images, debug)

    def _extract_features(self, images: Union[List[str], torch.Tensor, List[Image.Image]], batch_size: int = 8,
                         debug=False):
        if isinstance(images, list) and all(isinstance(img, str) for img in images):
            return self._extract_from_paths(images, batch_size, debug)
        elif isinstance(images, torch.Tensor):
            return self._extract_from_tensor(images, debug)
        elif isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            return self._extract_from_pil(images, debug)

    def extract_from_tensor(self, images, debug=False):
        return self._with_vlm(self._extract_from_tensor, images, debug)

    def _extract_from_tensor(self, img_tensor, debug=False):
        batch_size = img_tensor.shape[0]
        return [self._extract_single_image(img_tensor[i], debug) for i in range(batch_size)]

    def extract_from_pil(self, images, debug=False):
        return self._with_vlm(self._extract_from_pil, images, debug)

    def _extract_from_pil(self, pil_images: List[Image.Image], debug=False):
        return [self._extract_single_image(img, debug) for img in pil_images]

    def extract_from_paths_batched(self, images, debug=False):
        return self._with_vlm(self._extract_from_paths_batched, images, debug)

    def _extract_from_paths_batched(self, img_paths: List[str], batch_size: int = 8, debug=False):
        features = []
        to_compute = [path for path in img_paths if not self.cache.exists(path)]
        cached = [self.cache.load(path) for path in img_paths if self.cache.exists(path)]

        if to_compute:
            dataloader = DataLoader(to_compute, batch_size=batch_size, shuffle=False)
            for batch in tqdm(dataloader, desc="Processing VLM features in batches"):
                batch_images = [Image.open(path) for path in batch]
                batch_features = self.batch_process(batch_images, debug)
                for i, img_path in enumerate(batch):
                    self.cache.save(img_path, batch_features[i])
                    cached.append(batch_features[i])

        return cached

    def extract_from_paths(self, images, debug=False):
        return self._with_vlm(self._extract_from_paths, images, debug)

    def _extract_from_paths(self, img_paths: List[str], debug=False):
        features = []

        for img_path in tqdm(img_paths, desc="Precomputing VLM image features"):
            if self.cache.exists(img_path):
                # print(f"Loading cached features for: {img_path}")
                features.append(self.cache.load(img_path))
            else:
                print(f"Computing VLM features for: {img_path}")
                img = Image.open(img_path)
                img_features = self._extract_single_image(img)
                # Save to cache
                self.cache.save(img_path, img_features)
                features.append(img_features)

        return features

    def batch_process(self, images: List[Image.Image], debug=False):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Describe exactly which objects the human is in contact with, what action is being performed and with what body part"}
            ]
        }]
        prompt = [self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True)] * len(images)
        inputs = self._prepare_inputs(images, prompt)
        handle, extracted_features = self._register_hook()

        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=60)

        if debug:
            generated_texts = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(generated_texts)

        handle.remove()
        return [torch.cat(extracted_features[1:]).to(self.device) for _ in images]

    def _extract_single_image(self, image, debug=False):
        inputs = self._process_image(image, debug)
        handle, extracted_features = self._register_hook()

        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=60)

        if debug:
            generated_texts = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(generated_texts)

        handle.remove()
        return torch.cat(extracted_features[1:]).to(self.device)

    def check_cache(self, img_paths: List[str]):
        self._cache_check = all([self.cache.exists(x) for  x in img_paths])
        return self._cache_check

    def __getitem__(self, img_path: str):
        return self.cache.load(img_path)

    def generate_from_features(self, features: List[torch.Tensor], max_new_tokens: int = 60) -> List[str]:
        """
        Generates text descriptions from stored features by passing them through the lm_head.

        Args:
            features (List[torch.Tensor]): List of pre-extracted features.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            List[str]: List of generated text descriptions.
        """
        self.vlm_model.eval()
        generated_texts = []

        for feature in features:
            with torch.no_grad():
                # Add an extra dimension to simulate batch processing
                feature = feature.unsqueeze(0).to(self.device)
                output_ids = self.vlm_model.model.text_model.lm_head(feature)
                decoded_text = self.vlm_processor.batch_decode(output_ids, skip_special_tokens=True)
                generated_texts.append(decoded_text[0])

        return generated_texts


def apply_vlm_on_tensor(img, vlm_processor, vlm_model, device, debug=False):
    """
    Apply VLM model to each image in the batch and extract text features.

    Parameters:
    - img: Tensor of shape (batch_size, img_dim), where img_dim is the flattened image.
    - vlm_processor: The processor for the VLM model.
    - vlm_model: The VLM model instance.
    - device: Device to perform the computation on (e.g., "cuda" or "cpu").

    Returns:
    - text_features: Tensor of concatenated text features from all images.
    """
    trans = tt.ToPILImage()
    batch_size = img.shape[0]
    extracted_features_per_image = []

    for i in range(batch_size):
        # Extract individual image
        single_img = img[i]

        # Prepare input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "Describe exactly which objects the human is in contact with, what action is being performed and with what body part"}
                ]
            },
        ]

        # Prepare inputs
        prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = vlm_processor(text=prompt, images=[trans(single_img)],
                               return_tensors="pt")
        inputs = inputs.to(device)

        # Collect the features for the current image
        extracted_features_per_token = []

        # Define the hook function
        def hook_fn(module, input, output):
            extracted_features_per_token.append(output.clone())

        # Register the hook to the last layer of the text_model (before lm_head)
        handle = vlm_model.model.text_model.norm.register_forward_hook(hook_fn)

        # Generate outputs
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=60)

        if debug:
            generated_texts = vlm_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
        )
            print(generated_texts)

        # Collect features for the current image
        text_features = torch.cat(extracted_features_per_token[1:]).to(device)
        extracted_features_per_image.append(text_features)

        # Remove the hook after the forward pass to avoid side effects
        handle.remove()

    # Concatenate all the features along the batch dimension
    return extracted_features_per_image


def apply_vlm_on_pil(imgs: List[Image.Image], vlm_processor, vlm_model, device, debug=False):
    """
    Apply VLM model to each image in the batch and extract text features.

    Parameters:
    - img: List of PIL images
    - vlm_processor: The processor for the VLM model.
    - vlm_model: The VLM model instance.
    - device: Device to perform the computation on (e.g., "cuda" or "cpu").

    Returns:
    - text_features: Tensor of concatenated text features from all images.
    """
    extracted_features_per_image = []

    for single_img in imgs:

        # Prepare input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "Describe exactly which objects the human is in contact with, what action is being performed and with what body part"}
                ]
            },
        ]

        # Prepare inputs
        prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = vlm_processor(text=prompt, images=[single_img],
                               return_tensors="pt")
        inputs = inputs.to(device)

        # Collect the features for the current image
        extracted_features_per_token = []

        # Define the hook function
        def hook_fn(module, input, output):
            extracted_features_per_token.append(output.clone())

        # Register the hook to the last layer of the text_model (before lm_head)
        handle = vlm_model.model.text_model.norm.register_forward_hook(hook_fn)

        # Generate outputs
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=60)

        if debug:
            generated_texts = vlm_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
        )
            print(generated_texts)

        # Collect features for the current image
        text_features = torch.cat(extracted_features_per_token[1:]).to(device)
        extracted_features_per_image.append(text_features)

        # Remove the hook after the forward pass to avoid side effects
        handle.remove()

    # Concatenate all the features along the batch dimension
    return extracted_features_per_image


def apply_vlm_on_pil_batch(imgs, vlm_processor, vlm_model, device, cache, debug=False,
                           batch_size: int = 8):
    """
    Apply VLM model to a batch of images and extract text features.

    Parameters:
    - imgs: List of image paths
    - vlm_processor: Processor for the VLM model.
    - vlm_model: The VLM model instance.
    - device: Device to perform the computation on (e.g., "cuda" or "cpu").
    - cache: An instance of VLMFeatureCache to manage caching.
    - debug: Boolean flag to display generated text for debugging.

    Returns:
    - text_features: List of tensors containing text features for all images.
    """
    dataloader = DataLoader(imgs, batch_size=batch_size, shuffle=False)
    all_features = []

    for batch in tqdm(dataloader, desc="Processing VLM features in batches..."):
        # Check which images are already cached
        to_compute = []
        cached_features = []

        for img_path in batch:
            if cache.exists(img_path):
                cached_features.append(cache.load(img_path))
            else:
                to_compute.append(img_path)

        if len(to_compute) > 0:
            # Open images and prepare messages
            pil_images = [Image.open(path) for path in to_compute]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": "Describe exactly which objects the human is in contact with, what action is being performed and with what body part"}
                    ]
                }
            ]

            # Prepare inputs
            prompts = [vlm_processor.apply_chat_template(messages, add_generation_prompt=True)] * len(pil_images)
            inputs = vlm_processor(text=prompts, images=pil_images, return_tensors="pt", padding=True)
            inputs = inputs.to(device)

            # Collect features for the current batch
            extracted_features_per_token = []

            # Define the hook function
            def hook_fn(module, input, output):
                extracted_features_per_token.append(output.clone())

            # Register the hook to the last layer of the text_model (before lm_head)
            handle = vlm_model.model.text_model.norm.register_forward_hook(hook_fn)

            # Generate outputs
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=60)

            if debug:
                generated_texts = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)
                print(generated_texts)

            # Collect features for the current batch
            for i, path in enumerate(to_compute):
                features = extracted_features_per_token[i]
                cache.save(path, features)
                cached_features.append(features)

            # Remove the hook after the forward pass to avoid side effects
            handle.remove()

        # Append to the full list of features
        all_features.extend(cached_features)

    return all_features

def precompute_vlm_features(imgs: Union[List[str], torch.Tensor],
                            vlm_id: str = "HuggingFaceTB/SmolVLM-Instruct",
                            device: str = "cuda",
                            cache_dir: str = "./cache/vlm_cache"):

    vlm_cache = VLMFeatureCache(cache_dir=cache_dir)
    vlm_processor = AutoProcessor.from_pretrained(vlm_id)
    vlm_model = AutoModelForVision2Seq.from_pretrained(
        vlm_id,
        torch_dtype=torch.float16,
    ).to(device)

    features = []

    for img_path in tqdm(imgs, desc="Precomputing VLM image features"):
        if vlm_cache.exists(img_path):
            #print(f"Loading cached features for: {img_path}")
            features.append(vlm_cache.load(img_path))
        else:
            print(f"Computing VLM features for: {img_path}")
            img = Image.open(img_path)
            img_features = apply_vlm_on_pil([img], vlm_processor, vlm_model, device)
            # Save to cache
            vlm_cache.save(img_path, img_features[0])
            features.append(img_features[0])


    return features


def precompute_vlm_features_batched(imgs: Union[List[str], torch.Tensor],
                            vlm_id: str = "HuggingFaceTB/SmolVLM-Instruct",
                            device: str = "cuda",
                            cache_dir: str = "./cache/vlm_cache",
                            batch_size: int = 8,
                            debug: bool = False):
    vlm_cache = VLMFeatureCache(cache_dir=cache_dir)
    vlm_processor = AutoProcessor.from_pretrained(vlm_id)
    vlm_model = AutoModelForVision2Seq.from_pretrained(
        vlm_id,
        torch_dtype=torch.float16,
    ).to(device)

    features = []
    to_compute = []
    cached_features = []

    # Split into cached and non-cached
    for img_path in imgs:
        if vlm_cache.exists(img_path):
            cached_features.append(vlm_cache.load(img_path))
        else:
            to_compute.append(img_path)

    # If there are images to compute, batch them
    if to_compute:
        print(f"Computing VLM features for {len(to_compute)} images in batches of {batch_size}")
        dataloader = DataLoader(to_compute, batch_size=batch_size, shuffle=False)

        for batch in tqdm(dataloader, desc="VLM Forward Pass"):
            pil_images = [Image.open(path) for path in batch]

            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": "Describe exactly which objects the human is in contact with, what action is being performed and with what body part"}
                    ]
                }
            ]
            prompts = [vlm_processor.apply_chat_template(messages, add_generation_prompt=False)] * len(pil_images)
            inputs = vlm_processor(text=prompts, images=pil_images, return_tensors="pt", padding=True,
                                   padding_side="left")
            inputs = inputs.to(device)

            # Collect features for the current batch
            extracted_features_per_token = []

            # Hook to extract hidden features
            def hook_fn(module, input, output):
                extracted_features_per_token.append(output.clone())

            handle = vlm_model.model.text_model.norm.register_forward_hook(hook_fn)

            # Generate outputs
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=200)

            if debug:
                generated_texts = vlm_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                print(generated_texts)

            # Unregister hook
            handle.remove()

            # Save features and append to results
            for i, img_path in enumerate(batch):
                features.append(extracted_features_per_token[i])
                vlm_cache.save(img_path, extracted_features_per_token[i])

    features = []

    # Reload them to have them in the same order as the input imgs
    for img_path in imgs:
        features.append(vlm_cache.load(img_path))

    features