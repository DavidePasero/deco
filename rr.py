import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image1 = Image.open("/home/lukas/Projects/deco/example_images/pexels-photo-15732209.jpeg")
image2 = Image.open("/home/lukas/Projects/deco/example_images/213.jpg")
# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe exactly which objects the human is in contact with, what action is being performed and with what body part."
                                    }
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image2], return_tensors="pt")
inputs = inputs.to(DEVICE)

extracted_features_per_token = []

# Define the hook function
def hook_fn(module, input, output):
    # We clone the output to avoid any modifications
    extracted_features_per_token.append(output.clone())

# Register the hook to the last layer of the text_model (before lm_head)
model.model.text_model.norm.register_forward_hook(hook_fn)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])

