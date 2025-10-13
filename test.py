import requests
import torch
import time
from PIL import Image
from pathlib import Path
from prismatic import load, load_wm_vlm
import os

os.environ['HF_HUB_OFFLINE'] = '1'

# For gated LMs like Llama-2, make sure to request official access, and generate an access token

hf_token = "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" # Path(".hf_token").read_text().strip()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)

model_id = "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/svd_siglip/prism-svd_dual_siglip_ft_proj_noalign_qwen" # "prism-svd+7b" "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/align"

baseline_model_id = "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/original_weight/reproduction-llava-v15+7b"

vlm = load(model_id, hf_token=hf_token)

vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt

# image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"

# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# Open an image from a local path

image_paths = ["/mnt/world_foundational_model/gkz/prismatic-vlms/assets/rocket.png"] #, "/mnt/world_foundational_model/gkz/prismatic-vlms/xiaowei1.png", "/mnt/world_foundational_model/gkz/prismatic-vlms/xiaowei2.png", "/mnt/world_foundational_model/gkz/prismatic-vlms/xiaowei3.png", "/mnt/world_foundational_model/gkz/prismatic-vlms/xiaowei4.png"]


for image_path in image_paths:
    image1 = Image.open(image_path).convert("RGB")
    image2 = Image.open(image_path).convert("RGB")
    image = [image1, image2]

user_prompt = "What is in the image and what is going on?"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
start_time = time.time()
generated_text = vlm.generate(
    image1,
    prompt_text,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=1024,
    min_length=1,
)
end_time = time.time()

print(f"Inference Speed: {end_time - start_time}s")
print(f"Generated Text: {generated_text}\n")