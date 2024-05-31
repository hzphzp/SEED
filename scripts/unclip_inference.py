from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image
import torch


import hydra
from omegaconf import OmegaConf
from PIL import Image
import pyrootutils
import os


image_dir = 'images/'
save_dir = 'recon_images_pure_unclip/'
# save_path = os.path.join(save_dir, os.path.basename(image_path))

os.makedirs(save_dir, exist_ok=True)

device = 'cuda'

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
pipe = pipe.to("cuda")

# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"
# init_image = load_image(url)

# images = pipe(init_image).images
# images[0].save("variation_image.png")

for image_path in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_path)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    img = load_image(image_path)
    images = pipe(img).images
    images[0].save(save_path)