"""
from PIL import Image
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration
model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
PROMPT = "<s>[INST]Describe the person on the images. What and how they are in contact with.\n[IMG][IMG][/INST]"
inputs = processor(text=PROMPT, images=imgs[:2], return_tensors="pt").to("cuda")
generate_ids = model.generate(**inputs, max_new_tokens=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"""

from vllm import LLM
from vllm.sampling_params import SamplingParams
import os
from PIL import Image

model_name = "mistralai/Pixtral-12B-2409"

sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(model=model_name, tokenizer_mode="mistral")

prompt = "Describe this image in one sentence."
image_url = "https://picsum.photos/id/237/200/300"

img_dir = "example_imgs"
imgs = []

for img_f in os.listdir(img_dir):
    imgs.append(Image.open(os.path.join(img_dir, img_f)))

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt},  {"type": "image", "image": imgs[0]}]
    },
]

outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
