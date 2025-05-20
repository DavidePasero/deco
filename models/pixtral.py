import glob
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
# ----------------------------------------------------------------------
# 1.  Model / processor
# ----------------------------------------------------------------------
model_id = "mistral-community/pixtral-12b"
model      = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype="float16").to("cuda")
processor  = AutoProcessor.from_pretrained(model_id, torch_dtype="float16")

# ----------------------------------------------------------------------
# 2.  Load local images
#     (adjust the pattern if you use .png or nested folders)
# ----------------------------------------------------------------------
IMG_DIR   = Path("example_images")
img_paths = sorted(glob.glob(str(IMG_DIR / "*.*")))[:4]      # first 4 images
images    = [Image.open(p).convert("RGB") for p in img_paths]

# ----------------------------------------------------------------------
# 3.  Build prompt: one [IMG] token per image
# ----------------------------------------------------------------------
img_tokens = "[IMG]" * len(images)
PROMPT = (
    "<s>[INST]Identify every 3-D contact point between the human body and "
    "surrounding objects in each image.  "
    "For every contact, output (1) the body part and (2) the object it touches.\n"
    f"{img_tokens}[/INST]"
)

# ----------------------------------------------------------------------
# 4.  Run Pixtral
# ----------------------------------------------------------------------
# Check for dtype debug this
inputs = processor(text=PROMPT, images=images, return_tensors="pt").to("cuda")
print (inputs)
inputs = {k: v.half() if isinstance (v, torch.Tensor) and v.dtype == torch.float32 else v for k, v in inputs.items()}  # half-precision
gen_ids = model.generate(**inputs, max_new_tokens=256)
output  = processor.batch_decode(gen_ids, skip_special_tokens=True, 
                                 clean_up_tokenization_spaces=False)[0]

print(output)
"""
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from pathlib import Path
import os
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image_dir = "example_images"
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  print(preds)
  return preds


for image_path in os.listdir(image_dir):
  image_path = os.path.join(image_dir, image_path)
  if os.path.isfile(image_path):
    print(f"Predicting {image_path}")
    predict_step([image_path])
  else:
    print(f"Skipping {image_path} as it is not a file")
"""