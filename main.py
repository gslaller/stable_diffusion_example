import torch
from diffusers import StableDiffusionPipeline
import argparse
from dotenv import load_dotenv
import os

parser = argparse.ArgumentParser(
    description=
        'Generating Images from Text. GPU/Cuda is needed'
)
parser.add_argument('prompt', 
metavar='N', type=str, nargs='+',
                    help='Please enter a text.')

args = parser.parse_args()
prompt = " ".join(args.prompt)

load_dotenv()
hf_access_token = os.getenv("HF_TOKEN")

pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=hf_access_token
)

pipe = pipeline.to("cuda")

def generation(prompt: str):
    pipe.enable_attention_slicing()
    with torch.autocast('cuda'):
        return pipe(prompt).images[0]

image = generation(prompt)
image.save("./images/"+prompt+".png")