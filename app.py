from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import logging
import torch
import base64
import rembg
import numpy as np
from io import BytesIO
from PIL import Image
from diffusers import (
    DiffusionPipeline, 
    EulerDiscreteScheduler, 
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler
)
# import spaces
app = FastAPI()

# Define a Pydantic model to validate request data
class LoRARequest(BaseModel):
    prompt: str
    negative_prompt: str# "low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    cfg_scale: float
    steps: int
    scheduler: str
    seed: int
    width: int
    height: int
    lora_scale: float


base_model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16, local_files_only=True)
pipe.to("cuda")

# Load LoRA weights
# pipe.load_lora_weights("/root/.cache/huggingface/hub/Abdullah-Habib/logolora", scale=1.0)

def image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # You can change the format as needed (e.g., "JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def remove_bg(image: Image):
    input_array_bg = np.array(image)
    # Apply background removal using rembg
    output_array_bg = rembg.remove(input_array_bg)
    # Create a PIL Image from the output array
    img = Image.fromarray(output_array_bg)

    mask = img.convert('L')  # Convert to grayscale
    mask_array = np.array(mask)

    # Create a binary mask (non-background areas are 255, background areas are 0)
    binary_mask = mask_array > 0

    # Find the bounding box of the non-background areas
    coords = np.argwhere(binary_mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    # Crop the output image using the bounding box
    cropped_output_image = img.crop((y0, x0, y1, x1))

    # Resize the cropped image to 1024x1024
    upscaled_image = cropped_output_image.resize((1024, 1024), Image.LANCZOS)
    return upscaled_image

@app.get("/")
def hello():
    return {"response":"hello world"}

@app.post("/generate")
async def run_lora(request: LoRARequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model pipeline not initialized")
    try:
        prompt, negative_prompt, cfg_scale, steps, scheduler, seed, width, height, lora_scale = request.prompt, request.negative_prompt, request.cfg_scale, request.steps, request.scheduler, request.seed, request.width, request.height, request.lora_scale
    except:
        raise HTTPException(status_code=400, detail="Invalid request")
    # Load LoRA weights
    pipe.load_lora_weights("/root/.cache/huggingface/hub/Abdullah-Habib/logolora", scale=lora_scale)
    prompt = f"rounded square, logo, logoredmaf, {prompt}, icons"
    print("prompt:",prompt)
    print("neg_prompt:",negative_prompt)
    # Set scheduler
    scheduler_config = pipe.scheduler.config
    if scheduler == "DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
    elif scheduler == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True)
    elif scheduler == "DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config, algorithm_type="sde-dpmsolver++")
    elif scheduler == "DPM++ 2M SDE Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif scheduler == "DPM++ SDE":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(scheduler_config)
    elif scheduler == "DPM++ SDE Karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=True)
    elif scheduler == "DPM2":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(scheduler_config)
    elif scheduler == "DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(scheduler_config, use_karras_sigmas=True)
    elif scheduler == "DPM2 a":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(scheduler_config)
    elif scheduler == "DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(scheduler_config, use_karras_sigmas=True)
    elif scheduler == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
    elif scheduler == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
    elif scheduler == "Heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(scheduler_config)
    elif scheduler == "LMS":
        pipe.scheduler = LMSDiscreteScheduler.from_config(scheduler_config)
    elif scheduler == "LMS Karras":
        pipe.scheduler = LMSDiscreteScheduler.from_config(scheduler_config, use_karras_sigmas=True)
    elif scheduler == "DEIS":
        pipe.scheduler = DEISMultistepScheduler.from_config(scheduler_config)
    elif scheduler == "UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(scheduler_config)

    # Set random seed for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # Generate image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        width=width,
        height=height,
        generator=generator,
        # cross_attention_kwargs={"scale": lora_scale},
    ).images[0]

    # Unload LoRA weights
    pipe.unload_lora_weights()
    image_without_bg = remove_bg(image)
    image_base64 = image_to_base64(image_without_bg)
    return {"image": image_base64}

