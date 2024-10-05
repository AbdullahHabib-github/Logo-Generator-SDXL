import runpod
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


# Initialize the base model
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
pipe.to("cuda")

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



def handler(event):
    # Extract text from the event
    input_data = event.get("input", {})

    # Fetching values from input_data using .get() with the provided default values
    prompt = input_data.get(
        "prompt",
        "#008080, #00BFFF, #007B5F, #F0F8FF, logo for an app named Portifolio, featuring a modern and sleek design that symbolizes online presence and digital connectivity, using teal and blue colors, visually appealing and easy to recognize, with an emphasis on professionalism and creativity"
    )
    negative_prompt = input_data.get(
        "negative_prompt",
        "low quality, bad anatomy, bad shapes, text, error, blurry, distorted, unprofessional, cluttered, amateur design, worst quality, low resolution, poorly aligned"
    )
    cfg_scale = input_data.get("cfg_scale", 7.5)
    steps = input_data.get("steps", 30)
    scheduler = input_data.get("scheduler", "DPM++ 2M SDE Karras")
    seed = input_data.get("seed", 3110589682)
    width = input_data.get("width", 1024)
    height = input_data.get("height", 1024)
    lora_scale = input_data.get("lora_scale", 1)
    # Load LoRA weights
    pipe.load_lora_weights("Abdullah-Habib/logolora",scale = lora_scale)
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

    # Return result
    return {"image": image_base64}
runpod.serverless.start({"handler": handler})