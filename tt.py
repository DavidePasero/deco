from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
model = AutoModel.from_pretrained('facebook/dinov2-giant').cuda()

inputs = processor(images=image, return_tensors="pt", device="cuda")
inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
