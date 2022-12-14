# Stable Diffusion Collection
This is a very simple example of stable diffusion and collection of example.\
CUDA is needed.

![Screenshot](pub/main.png)

## Note.
Upload interesting images as a push request.

## How to run on [Google Colab](https://colab.research.google.com/notebooks/empty.ipynb)

Please enable cuda/GPU. Runtime > Change runtime type > Hardware Accelerator change to GPU.\
You also need to register on [huggingface.co](https://huggingface.co) & ask for access permission [for this repo](https://huggingface.co/CompVis/stable-diffusion-v1-4) and replace the token access_token. The token in this code is INVALID.

```
!pip install git+https://github.com/huggingface/diffusers.git
!pip install transformers

import torch
from diffusers import StableDiffusionPipeline

access_token = "hf_zQqhpJCCHmpqCLSOmHlKpQucwPKujrber" # PLEASE INSERT YOUR OWN TOKEN
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=access_token
)

pipe = pipeline.to("cuda"); # Please enable runtime

def generation(prompt: str):
  pipe.enable_attention_slicing()
  with torch.autocast('cuda'):
      return pipe(prompt).images[0]

generation("A man fighting the east indian company in india, 4k, detailed, trending in artstation")
```

## How to run from CMD

You environment should have cuda/gpu.  
And there should be a token in your .env file with the name of "HF_TOKEN"

```bash
pip install -r requirements.txt
python3 main.py A man fighting the east indian company in india, 4k, detailed, trending in artstation
```
