# example_use.py
import os

# 1. Import the GIFA class from your library file
from gifa_library import GIFA # Assumes gifa_library.py is in the same folder or Python path

# --- Configuration ---
audio_file_to_process = r"C:/GIFA-acoustic/test_audios/test3.mp3" # Use raw string for paths
output_folder_base = "generated_images_examples" # Base folder for outputs

# --- Ensure the base output directory exists ---
os.makedirs(output_folder_base, exist_ok=True)

# 2. Create an instance of the GIFA handler
#    'none' avoids preloading, models load on first use.
#    Use 'lumina', 'sd', or 'both' to preload if preferred.
print("Initializing GIFA handler (preload='none')...")
gifa = GIFA(preload_image_model='none')
print("Initialization complete.")


# --- Example 1: Generate using Lumina (Default) ---
print("\n--- Generating with Lumina ---")
lumina_output_folder = os.path.join(output_folder_base, "lumina")
lumina_result = gifa.pipe(
    audio_path=audio_file_to_process,
    image_model_type='lumina', # Explicitly specify or rely on default
    audio_model='ftwhispertiny', # Or 'ftcanvers'
    prefixes=["ethereal", "glowing particles"],
    output_dir=lumina_output_folder,
    width=1024, # Example Lumina dimensions
    height=1024,
    seed=123,
    save_image=True,
    return_image_object=False
    # cfg_trunc_ratio=1.0 # Default, can be specified
)

if lumina_result:
    print(f"Lumina Success! Output path: {lumina_result}")
else:
    print("Lumina Pipeline failed. Check console/logs for errors (ensure Lumina is available).")


# --- Example 2: Generate using Stable Diffusion ---
print("\n--- Generating with Stable Diffusion ---")
sd_output_folder = os.path.join(output_folder_base, "sd")
sd_result = gifa.pipe(
    audio_path=audio_file_to_process,
    image_model_type='sd',       # Specify Stable Diffusion
    audio_model='ftcanvers',     # Try the other audio model
    prefixes=["oil painting", "textured canvas"],
    output_dir=sd_output_folder,
    width=512,                   # Standard SD dimensions
    height=512,
    seed=456,
    guidance_scale=7.5,          # SD specific parameter (default)
    num_inference_steps=25,      # SD specific parameter (default)
    save_image=True,
    return_image_object=False
)

if sd_result:
    print(f"Stable Diffusion Success! Output path: {sd_result}")
else:
    print("Stable Diffusion Pipeline failed. Check console/logs for errors (ensure SD is available).")

print("\n--- Example Script Finished ---")
