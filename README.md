# stable_diffusion_example
This is a very simple example of stable diffusion

## How to run on google Colab

Please enable cuda/GPU 

```
!pip install git+https://github.com/huggingface/diffusers.git
!pip install transformers

import torch
from diffusers import StableDiffusionPipeline

access_token = "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" # Man muss sich bei huggingface.co registerien und die Zugangsberechtigung für das Repo anfordern.
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=access_token
)

pipe = pipeline.to("cuda");

def generation(prompt: str):
  pipe.enable_attention_slicing()
  with torch.autocast('cuda'):
      return pipe(prompt).images[0]

generation("A man fighting the east indian company in india, 4k, detailed, trending in artstation")
```

## Note. 

make a pull request for the images, you find something nice.
