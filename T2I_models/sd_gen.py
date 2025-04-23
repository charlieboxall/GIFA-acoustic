import os
import uuid
import traceback
import time
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Global Model Loading
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
            # Enable CPU offloading to save VRAM when the model is idle
            sd_pipe.enable_model_cpu_offload()
            print("Stable Diffusion v1.5 model loaded successfully.")
        except Exception as e:
            print(f"ERROR loading Stable Diffusion v1.5 model: {e}")
            traceback.print_exc()
            # Return None if loading fails, allowing the calling function to handle it
            return None
    return sd_pipe

# Renamed from generate_sd_image
def generate_image(prompt, width, height, guidance_scale, num_inference_steps, save_folder_path, seed=0):
    """
    Generates an image based on the prompt using the Stable Diffusion v1.5 model.

    Args:
        prompt (str): The text prompt for image generation.
        width (int): The desired width of the image (typically 512 for SD v1.5).
        height (int): The desired height of the image (typically 512 for SD v1.5).
        guidance_scale (float): Controls how much the prompt influences the generation.
        num_inference_steps (int): The number of denoising steps.
        save_folder_path (str): The absolute path to the folder where the image should be saved.
        seed (int, optional): Random seed for generation. Defaults to 0.

    Returns:
        str: The filename of the generated image if successful, otherwise None.
    """
    print(f"Generating SD image with prompt: '{prompt}'")
    print(f"Parameters: Width={width}, Height={height}, Guidance Scale={guidance_scale:.1f}, Steps={num_inference_steps}, Seed={seed}")
    try:
        pipe = get_sd_pipeline()
        if pipe is None:
            print("ERROR: Stable Diffusion pipeline not available for image generation.")
            return None # Return None if pipeline failed to load earlier

        # Create a generator for reproducibility
        generator = torch.Generator("cuda").manual_seed(seed)

        # Generate the image within a no_grad context to save memory
        with torch.no_grad():

            # Call the diffusion pipeline
            image_result = pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                generator=generator
            )
            # Extract the first image from the result
            image = image_result.images[0]

        # Ensure the image is in RGB format for saving as PNG/JPG
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create a unique filename using timestamp and UUID
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4()
        # Add an 'sd_' prefix to distinguish from other potential image types
        filename = f"sd_img_{timestamp}_{unique_id}.png"
        # Construct the full save path
        save_path = os.path.join(save_folder_path, filename)

        # Ensure the save directory exists; create it if it doesn't
        os.makedirs(save_folder_path, exist_ok=True)

        # Save the generated image
        image.save(save_path)
        print(f"SD Image saved to: {save_path}")
        return filename

    except Exception as e:
        print(f"ERROR during Stable Diffusion image generation: {e}")
        print("--- SD Image Generation Traceback ---")
        traceback.print_exc()
        print("--- End SD Image Generation Traceback ---")
        return None
