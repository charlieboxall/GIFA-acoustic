# image_generator.py
import os
import uuid
import traceback
import time
import torch
from diffusers import Lumina2Pipeline
from PIL import Image

# --- Global Model Loading ---
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
            # Return None if loading fails, allowing the calling function to handle it
            return None
    return lumina_pipe

def generate_image(prompt, width, height, cfg_trunc_ratio, save_folder_path, seed=0):
    """
    Generates an image based on the prompt using the Lumina-2 model.

    Args:
        prompt (str): The text prompt for image generation.
        width (int): The desired width of the image.
        height (int): The desired height of the image.
        cfg_trunc_ratio (float): Configuration truncation ratio for the model.
        save_folder_path (str): The absolute path to the folder where the image should be saved.
        seed (int, optional): Random seed for generation. Defaults to 0.

    Returns:
        str: The filename of the generated image if successful, otherwise None.
    """
    print(f"Generating image with prompt: '{prompt}'")
    print(f"Parameters: Width={width}, Height={height}, CFG Trunc Ratio={cfg_trunc_ratio:.2f}, Seed={seed}")
    try:
        pipe = get_lumina_pipeline()
        if pipe is None:
            # Handle model load failure gracefully during generation attempt
            print("ERROR: Lumina pipeline not available for image generation.")
            return None # Return None if pipeline failed to load earlier

        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                guidance_scale=4.0,
                num_inference_steps=25,
                cfg_trunc_ratio=float(cfg_trunc_ratio),
                cfg_normalization=True,
                generator=generator
            ).images[0]

        if image.mode != 'RGB':
            image = image.convert('RGB')

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4()
        filename = f"img_{timestamp}_{unique_id}.png"
        save_path = os.path.join(save_folder_path, filename)

        # Ensure the save directory exists (it should from run.py, but double-check)
        os.makedirs(save_folder_path, exist_ok=True)

        image.save(save_path)
        print(f"Image saved to: {save_path}")
        return filename

    except Exception as e:
        print(f"ERROR during image generation: {e}")
        print("--- Image Generation Traceback ---")
        traceback.print_exc()
        print("--- End Image Generation Traceback ---")
        return None