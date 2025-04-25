
import os
import uuid
import traceback
import time
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

sd_pipe = None

def get_sd_pipeline():
    """Loads and returns the Stable Diffusion v1.5 pipeline instance, loading if necessary."""
    global sd_pipe
    if sd_pipe is None:
        print("Loading Stable Diffusion v1.5 model...")
        try:
            sd_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            ).to("cuda")
            sd_pipe.enable_model_cpu_offload()
            print("Stable Diffusion v1.5 model loaded successfully.")
        except Exception as e:
            print(f"ERROR loading Stable Diffusion v1.5 model: {e}")
            traceback.print_exc()
            return None
    return sd_pipe

def generate_image(prompt: str,
                   width: int,
                   height: int,
                   save_folder_path: str,
                   seed: int = 0,
                   **pipeline_kwargs): 
    """
    Generates an image based on the prompt using the Stable Diffusion v1.5 model,
    allowing pass-through of additional pipeline arguments.

    Args:
        prompt (str): The text prompt for image generation.
        width (int): The desired width of the image.
        height (int): The desired height of the image.
        save_folder_path (str): The absolute path to the folder where the image should be saved.
        seed (int, optional): Random seed for generation. Defaults to 0.
        **pipeline_kwargs: Additional keyword arguments to pass directly to the
                           StableDiffusionPipeline call (e.g., guidance_scale,
                           num_inference_steps, negative_prompt, etc.).

    Returns:
        str: The filename of the generated image if successful, otherwise None.
    """
    # Log received parameters, including extra ones
    print(f"Generating SD image with prompt: '{prompt}'")
    print(f"Base Params: Width={width}, Height={height}, Seed={seed}")
    if pipeline_kwargs:
        print(f"Additional Pipeline Params: {pipeline_kwargs}")
    else:
        print("Additional Pipeline Params: None")

    try:
        pipe = get_sd_pipeline()
        if pipe is None:
            print("ERROR: Stable Diffusion pipeline not available for image generation.")
            return None

        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.no_grad():
            image_result = pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                generator=generator,
                **pipeline_kwargs
            )
            image = image_result.images[0]

        if image.mode != 'RGB':
            image = image.convert('RGB')

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4()
        filename = f"sd_img_{timestamp}_{unique_id}.png"
        save_path = os.path.join(save_folder_path, filename)

        os.makedirs(save_folder_path, exist_ok=True)
        image.save(save_path)
        print(f"SD Image saved to: {save_path}")
        return filename

    except Exception as e:
        print(f"ERROR during Stable Diffusion image generation: {e}")
        print("--- SD Image Generation Traceback ---")
        traceback.print_exc()
        print("--- End SD Image Generation Traceback ---")
        return None