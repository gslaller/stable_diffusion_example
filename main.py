import torch
from diffusers import StableDiffusionPipeline

import argparse

parser = argparse.ArgumentParser(
    description='Generating Images from Text. You can use GPU/CPU'
)
parser.add_argument('prompt', metavar='N', type=str, nargs='+',
                    help='Please enter a text.')

parser.add_argument('--no_cuda', 
    default=True, 
    action="store_false",
    help="If Cuda should be used.")

args = parser.parse_args()

prompt = " ".join(args.prompt)
cuda = not args.no_cuda

access_token = "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" # This is invalid
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=access_token
)

pipe = pipeline.to("cuda")

def generation(prompt: str):
    pipe.enable_attention_slicing()
    with torch.autocast('cuda'):
        return pipe(prompt).images[0]


