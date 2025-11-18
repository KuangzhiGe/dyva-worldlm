import torch
import time
from PIL import Image
from prismatic import load
import os

os.environ['HF_HUB_OFFLINE'] = '1'

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = "YOUR_HF_KEY" # Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)
model_id = "dyva_siglip_qwen+7b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

# Open an image from a local path
image_paths = ["./assets/rocket.png"]
images = []
for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    images.append(image)

# Build prompt
user_prompt = "What is in the image and what is going on?"
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
start_time = time.time()
generated_text = vlm.generate(
    image, # can also be a list of images for multi-image inputs: images
    prompt_text,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=1024,
    min_length=1,
)
end_time = time.time()
print(f"Inference Speed: {end_time - start_time}s")
print(f"Generated Text: {generated_text}\n")