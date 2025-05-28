import torchvision.transforms as tt
import torch
from typing import List, Union, Tuple
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm
import hashlib
from pathlib import Path
from torch.utils.data import DataLoader


class VLMFeatureCache:
    """
    Caches VLM features extracted from images.
    """
    def __init__(self, cache_dir: str = "./cache/vlm_features_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, image_path: str) -> str:
        """Generates a unique ID for an image path using MD5 hash."""
        return hashlib.md5(image_path.encode()).hexdigest()

    def exists(self, image_path: str) -> bool:
        """Checks if features for a given image path exist in the cache."""
        return (self.cache_dir / f"{self._generate_id(image_path)}.pt").exists()

    def load(self, image_path: str) -> torch.Tensor:
        """Loads features for a given image path from the cache."""
        return torch.load(self.cache_dir / f"{self._generate_id(image_path)}.pt", map_location="cpu")

    def save(self, image_path: str, features: torch.Tensor):
        """Saves features for a given image path to the cache."""
        (self.cache_dir / f"{self._generate_id(image_path)}.pt").save(features)


class TextCache:
    """
    Caches generated text descriptions from images.
    """
    def __init__(self, cache_dir: str = "./cache/vlm_texts_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, image_path: str) -> str:
        """Generates a unique ID for an image path using MD5 hash."""
        return hashlib.md5(image_path.encode()).hexdigest()

    def exists(self, image_path: str) -> bool:
        """Checks if a text description for a given image path exists in the cache."""
        return (self.cache_dir / f"{self._generate_id(image_path)}.txt").exists()

    def load(self, image_path: str) -> str:
        """Loads a text description for a given image path from the cache."""
        with open(self.cache_dir / f"{self._generate_id(image_path)}.txt", 'r', encoding='utf-8') as f:
            return f.read()

    def save(self, image_path: str, text: str):
        """Saves a text description for a given image path to the cache."""
        with open(self.cache_dir / f"{self._generate_id(image_path)}.txt", 'w', encoding='utf-8') as f:
            f.write(text)


class VLMManager:
    """
    Manages VLM model, feature extraction, and text generation.
    """
    def __init__(self,
                 vlm_id: str = "HuggingFaceTB/SmolVLM-Instruct",
                 device: str = "cuda",
                 feature_cache_dir: str = "./cache/vlm_features_cache",
                 text_cache_dir: str = "./cache/vlm_texts_cache"):
        self.vlm_id = vlm_id
        self.device = device
        self.feature_cache = VLMFeatureCache(feature_cache_dir)
        self.text_cache = TextCache(text_cache_dir)
        self.vlm_processor = None
        self.vlm_model = None

    def _load_vlm(self):
        """Loads the VLM model and processor."""
        if self.vlm_model is None or self.vlm_processor is None:
            print("Loading VLM model and processor...")
            self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_id)
            self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                self.vlm_id, torch_dtype=torch.float16
            ).to(self.device)

    def _close_vlm(self):
        """Unloads the VLM model and processor."""
        if self.vlm_model is not None:
            print("Unloading VLM model from memory...")
            del self.vlm_processor
            del self.vlm_model
            self.vlm_processor = None
            self.vlm_model = None
            torch.cuda.empty_cache()

    def _with_vlm(self, func, *args, **kwargs):
        """Context manager for loading/unloading VLM model."""
        try:
            self._load_vlm()
            return func(*args, **kwargs)
        finally:
            self._close_vlm()

    def _prepare_inputs(self, images: List[Image.Image], prompt: Union[str, List[str]]):
        """Prepares inputs for the VLM model."""
        return self.vlm_processor(text=prompt, images=images, return_tensors="pt", padding=True, padding_side="left").to(self.device)

    def _extract_single_image_features(self, image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Extracts features for a single image.
        """

        def hook_fn(module, input, output):
            nonlocal extracted_features
            extracted_features.append(output.clone().detach())

        extracted_features = []
        handle, extracted_features = self._register_hook(hook_fn)
        pil_image = tt.ToPILImage()(image) if isinstance(image, torch.Tensor) else image
        prompt = self._get_fixed_prompt()
        inputs = self._prepare_inputs([pil_image], prompt)
        self.vlm_model.generate(**inputs, max_new_tokens=0)  # No text generation here
        handle.remove()
        return extracted_features[0].to(self.device)

    def _batch_process_images_features(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Processes a batch of PIL images to extract features.
        """

        def hook_fn(module, input, output):
            nonlocal extracted_features
            extracted_features.append(output.clone().detach())

        extracted_features = []
        handle, extracted_features = self._register_hook(hook_fn)
        prompt = self._get_fixed_prompt()
        prompts = [prompt] * len(images)
        inputs = self._prepare_inputs(images, prompts)
        self.vlm_model.generate(**inputs, max_new_tokens=0)  # No text generation here
        handle.remove()
        return [extracted_features[0][i].to(self.device) for i in range(len(images))]

    def _generate_text_single(self, image: Union[torch.Tensor, Image.Image]) -> str:
        """Generates text for a single image."""
        pil_image = tt.ToPILImage()(image) if isinstance(image, torch.Tensor) else image
        prompt = self._get_fixed_prompt()
        inputs = self._prepare_inputs([pil_image], prompt)
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=60)
        return self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _generate_text_batch(self, images: List[Image.Image]) -> List[str]:
        """Generates text for a batch of images."""
        prompt = self._get_fixed_prompt()
        prompts = [prompt] * len(images)
        inputs = self._prepare_inputs(images, prompts)
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=60)
        return self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)

    def extract_features(self, images: Union[List[str], torch.Tensor, List[Image.Image]], batch_size: int = 8) -> List[torch.Tensor]:
        """
        Extracts VLM features from images, with caching.
        """
        return self._with_vlm(self._extract_features_internal, images, batch_size)

    def _extract_features_internal(self, images, batch_size) -> List[torch.Tensor]:
        if isinstance(images, list) and all(isinstance(img, str) for img in images):
            return self._extract_features_from_paths_batched(images, batch_size)
        elif isinstance(images, torch.Tensor):
            pil_images = [tt.ToPILImage()(img) for img in images]
            return self._extract_features_from_pil(pil_images)
        elif isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            return self._extract_features_from_pil(images)
        else:
            raise ValueError("Input must be a list of image paths, a torch.Tensor, or a list of PIL Images.")

    def _extract_features_from_pil(self, pil_images: List[Image.Image]) -> List[torch.Tensor]:
        """Extracts features from a list of PIL images."""
        return [self._extract_single_image_features(img) for img in pil_images]

    def _extract_features_from_paths_batched(self, img_paths: List[str], batch_size: int) -> List[torch.Tensor]:
        """Extracts features from a list of image paths using batched processing."""
        features = []
        to_compute_paths = [path for path in img_paths if not self.feature_cache.exists(path)]
        cached_features = [self.feature_cache.load(path) for path in img_paths if self.feature_cache.exists(path)]

        if to_compute_paths:
            dataloader = DataLoader(to_compute_paths, batch_size=batch_size, shuffle=False)
            for batch_paths in tqdm(dataloader, desc="Processing VLM features in batches"):
                batch_images = [Image.open(path).convert("RGB") for path in batch_paths]  # Ensure RGB
                batch_features = self._batch_process_images_features(batch_images)
                for i, path in enumerate(batch_paths):
                    self.feature_cache.save(path, batch_features[i])
                    cached_features.insert(img_paths.index(path), batch_features[i])  # Keep original order
        return cached_features

    def _extract_features_from_paths(self, img_paths: List[str]) -> List[torch.Tensor]:
        """Extracts features from a list of image paths."""
        features = []
        for img_path in tqdm(img_paths, desc="Precomputing VLM image features"):
            if self.feature_cache.exists(img_path):
                features.append(self.feature_cache.load(img_path))
            else:
                img = Image.open(img_path).convert("RGB")  # Ensure RGB
                img_features = self._extract_single_image_features(img)
                self.feature_cache.save(img_path, img_features)
                features.append(img_features)
        return features

    def generate_texts(self, images: Union[List[str], torch.Tensor, List[Image.Image]], batch_size: int = 8) -> List[str]:
        """
        Generates text descriptions for images, with caching.
        """
        return self._with_vlm(self._generate_texts_internal, images, batch_size)

    def _generate_texts_internal(self, images, batch_size):
        if not isinstance(images, list):
            images = list(images)
        if all(isinstance(img, str) for img in images):
            return self._generate_texts_from_paths_batched(images, batch_size)
        elif isinstance(images, torch.Tensor):
            pil_images = [tt.ToPILImage()(img) for img in images]
            return self._generate_texts_from_pil(pil_images)
        elif all(isinstance(img, Image.Image) for img in images):
            return self._generate_texts_from_pil(images)
        else:
            raise ValueError("Input must be a list of image paths, a torch.Tensor, or a list of PIL Images.")

    def _generate_texts_from_pil(self, pil_images: List[Image.Image]) -> List[str]:
        """Generates texts from a list of PIL images."""
        return [self._generate_text_single(img) for img in pil_images]

    def _generate_texts_from_paths_batched(self, img_paths: List[str], batch_size: int) -> List[str]:
        """Generates texts from a list of image paths using batched processing."""
        texts = []
        to_compute_paths = [path for path in img_paths if not self.text_cache.exists(path)]
        cached_texts = [self.text_cache.load(path) for path in img_paths if self.text_cache.exists(path)]

        if to_compute_paths:
            dataloader = DataLoader(to_compute_paths, batch_size=batch_size, shuffle=False)
            for batch_paths in tqdm(dataloader, desc="Generating VLM texts in batches"):
                batch_images = [Image.open(path).convert("RGB") for path in batch_paths]  # Ensure RGB
                batch_texts = self._generate_text_batch(batch_images)
                batch_texts = [self._postprocess_text(text) for text in batch_texts]
                for i, path in enumerate(batch_paths):
                    self.text_cache.save(path, batch_texts[i])
                    cached_texts.insert(img_paths.index(path), batch_texts[i])  # Keep original order
        return cached_texts

    def _postprocess_text(self, text: str) -> str:
        return text.split("Assistant: ")[1]

    def _generate_texts_from_paths(self, img_paths: List[str]) -> List[str]:
        """Generates texts from a list of image paths."""
        texts = []
        for img_path in tqdm(img_paths, desc="Generating VLM texts"):
            if self.text_cache.exists(img_path):
                texts.append(self.text_cache.load(img_path))
            else:
                img = Image.open(img_path).convert("RGB")  # Ensure RGB
                text = self._generate_text_single(img)
                self.text_cache.save(img_path, text)
                texts.append(text)
        return texts

    def _get_fixed_prompt(self):
        """Gets the fixed prompt for VLM descriptions."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Describe exactly which objects the human is in contact with, what action is being performed and with what body part"}
            ]
        }]
        return self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True)

    def check_feature_cache(self, img_paths: List[str]) -> bool:
        """Checks if features for all given image paths exist in the feature cache."""
        return all(self.feature_cache.exists(x) for x in img_paths)

    def __getitem__(self, img_path: str) -> torch.Tensor:
        """Allows accessing cached features using dictionary-like syntax."""
        return self.text_cache.load(img_path)


# --- Global utility functions refactored to use VLMManager ---

def precompute_vlm_features_and_texts(imgs: List[str],
                                      vlm_id: str = "HuggingFaceTB/SmolVLM-Instruct",
                                      device: str = "cuda",
                                      feature_cache_dir: str = "./cache/vlm_features_cache",
                                      text_cache_dir: str = "./cache/vlm_texts_cache",
                                      batch_size: int = 8,
                                      ) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Precomputes VLM image features and generates text descriptions for a list of image paths,
    leveraging caching for both. Uses batched processing for efficiency.

    Args:
        imgs (List[str]): List of image file paths.
        vlm_id (str): HuggingFace VLM model ID.
        device (str): Device to run the VLM on ("cuda" or "cpu").
        feature_cache_dir (str): Directory for caching image features.
        text_cache_dir (str): Directory for caching text descriptions.
        batch_size (int): Number of images to process in each batch.

    Returns:
        Tuple[List[torch.Tensor], List[str]]: A tuple containing a list of feature tensors
        and a list of corresponding text descriptions.
    """
    vlm_manager = VLMManager(vlm_id, device, feature_cache_dir, text_cache_dir)
    features = vlm_manager.extract_features(imgs, batch_size)
    texts = vlm_manager.generate_texts(imgs, batch_size)
    return features, texts



def precompute_vlm_features(imgs: List[str],
                            vlm_id: str = "HuggingFaceTB/SmolVLM-Instruct",
                            device: str = "cuda",
                            feature_cache_dir: str = "./cache/vlm_features_cache",
                            batch_size: int = 8) -> List[torch.Tensor]:
    """
    Precomputes VLM image features for a list of image paths, with caching.

    Args:
        imgs (List[str]): List of image file paths.
        vlm_id (str): HuggingFace VLM model ID.
        device (str): Device to run the VLM on ("cuda" or "cpu").
        feature_cache_dir (str): Directory for caching image features.
        batch_size (int): Number of images to process in each batch.

    Returns:
        List[torch.Tensor]: A list of feature tensors.
    """
    vlm_manager = VLMManager(vlm_id, device, feature_cache_dir)
    return vlm_manager.extract_features(imgs, batch_size)



def precompute_vlm_texts(imgs: List[str],
                         vlm_id: str = "HuggingFaceTB/SmolVLM-Instruct",
                         device: str = "cuda",
                         text_cache_dir: str = "./cache/vlm_texts_cache",
                         batch_size: int = 8) -> List[str]:
    """
    Generates and caches text descriptions for a list of image paths.

    Args:
        imgs (List[str]): List of image file paths.
        vlm_id (str): HuggingFace VLM model ID.
        device (str): Device to run the VLM on ("cuda" or "cpu").
        text_cache_dir (str): Directory for caching text descriptions.
        batch_size (int): Number of images to process in each batch.

    Returns:
        List[str]: A list of text descriptions.
    """
    vlm_manager = VLMManager(vlm_id, device, text_cache_dir=text_cache_dir)
    return vlm_manager.generate_texts(imgs, batch_size)



