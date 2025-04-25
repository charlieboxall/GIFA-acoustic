
import os
import uuid
import traceback
import time
import torch
from diffusers import Lumina2Pipeline
from PIL import Image

#  Global Model Loading 
lumina_pipe = None

def get_lumina_pipeline():
    """Loads and returns the Lumina-2 pipeline instance, loading if necessary."""
    global lumina_pipe
    if lumina_pipe is None:
        print("Loading Lumina-2 model...")
        try:
            lumina_pipe = Lumina2Pipeline.from_pretrained(
                "Alpha-VLLM/Lumina-Image-2.0",
                torch_dtype=torch.float16
            ).to("cuda")
            lumina_pipe.enable_model_cpu_offload()
            print("Lumina-2 model loaded.")
        except Exception as e:
            print(f"ERROR loading Lumina-2 model: {e}")
            traceback.print_exc()
            return None
    return lumina_pipe

def generate_image(prompt: str,
                   width: int,
                   height: int,
                   save_folder_path: str,
                   seed: int = 0,
                   **pipeline_kwargs):
    """
    Generates an image based on the prompt using the Lumina-2 model,
    allowing pass-through of additional pipeline arguments.

    Args:
        prompt (str): The text prompt for image generation.
        width (int): The desired width of the image.
        height (int): The desired height of the image.
        save_folder_path (str): The absolute path to the folder where the image should be saved.
        seed (int, optional): Random seed for generation. Defaults to 0.
        **pipeline_kwargs: Additional keyword arguments to pass directly to the
                           Lumina2Pipeline call (e.g., guidance_scale,
                           num_inference_steps, cfg_trunc_ratio, cfg_normalization).

    Returns:
        str: The filename of the generated image if successful, otherwise None.
    """
    # Log received parameters, including extra ones
    print(f"Generating Lumina image with prompt: '{prompt}'")
    print(f"Base Params: Width={width}, Height={height}, Seed={seed}")
    if pipeline_kwargs:
        print(f"Additional Pipeline Params: {pipeline_kwargs}")
    else:
        print("Additional Pipeline Params: None")

    try:
        pipe = get_lumina_pipeline()
        if pipe is None:
            print("ERROR: Lumina pipeline not available for image generation.")
            return None

        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                generator=generator,
                **pipeline_kwargs
            ).images[0]

        if image.mode != 'RGB':
            image = image.convert('RGB')

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4()
        filename = f"img_{timestamp}_{unique_id}.png"
        save_path = os.path.join(save_folder_path, filename)

        os.makedirs(save_folder_path, exist_ok=True)
        image.save(save_path)
        print(f"Image saved to: {save_path}")
        return filename

    except Exception as e:
        print(f"ERROR during image generation: {e}")
        print(" Image Generation Traceback ")
        traceback.print_exc()
        print(" End Image Generation Traceback ")
        return None