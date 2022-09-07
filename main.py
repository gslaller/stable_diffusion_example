import torch
from diffusers import StableDiffusionPipeline
import argparse
from dotenv import load_dotenv
import os

parser = argparse.ArgumentParser(
    description=
        'Generating Images from Text. You can use GPU/CPU'
)
parser.add_argument('prompt', 
metavar='N', type=str, nargs='+',
                    help='Please enter a text.')

args = parser.parse_args()
prompt = " ".join(args.prompt)

load_dotenv()
hf_access_token = os.getenv("HF_TOKEN")

cuda_available = torch.cuda.is_available()

pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=hf_access_token
)

print("before casting")

if cuda_available:
    pipe = pipeline.to("cuda")
else:
    pipe = pipeline

print("after casting")

def generation(prompt: str):
    print("before attention slicing")
    pipe.enable_attention_slicing()
    print("After attention slicing")
    if cuda_available:
        with torch.autocast('cuda'):
            return pipe(prompt).images[0]
    with torch.autocast('cpu'):
        return pipe(prompt).images[0]

    pass


image = generation(prompt)
image.save("./images/"+prompt+".png")