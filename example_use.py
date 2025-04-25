
import os

# 1. Import the GIFA class from your library file
from gifa_library import GIFA 

#  Configuration 
audio_file_to_process = r"C:/GIFA-acoustic/test_audios/test10.mp3" 
output_folder_base = "generated_images_examples_v2" 
os.makedirs(output_folder_base, exist_ok=True)


print("Initializing GIFA handler (preload='none')...")
gifa = GIFA(preload_image_model='none')
print("Initialization complete.")


#  Example 1: Generate using Lumina (Default) 
print("\n Generating with Lumina ")
lumina_output_folder = os.path.join(output_folder_base, "lumina")
lumina_result = gifa.pipe(
    audio_path=audio_file_to_process,
    image_model_type='lumina',
    audio_model='ftwhispertiny',
    prefixes=["ethereal", "glowing particles", "high resolution"],
    output_dir=lumina_output_folder,
    width=512,
    height=512,
    seed=123,
    save_image=True,
    return_image_object=False,
    #  Lumina Specific Pipeline Arguments 
    guidance_scale=4.5,           
    num_inference_steps=30,      
    cfg_trunc_ratio=0.9,          
    cfg_normalization=True,       
)

if lumina_result:
    print(f"Lumina Success! Output path/object: {lumina_result}")
else:
    print("Lumina Pipeline failed. Check console/logs for errors (ensure Lumina is available).")


#  Example 2: Generate using Stable Diffusion 
print("\n Generating with Stable Diffusion ")
sd_output_folder = os.path.join(output_folder_base, "sd")
sd_result = gifa.pipe(
    audio_path=audio_file_to_process,
    image_model_type='sd',          
    audio_model='ftcanvers',        
    prefixes=["oil painting", "textured canvas", "masterpiece"],
    output_dir=sd_output_folder,
    width=512,         
    height=768,               
    seed=456,
    save_image=True,
    return_image_object=False,
    #  Stable Diffusion Specific Pipeline Arguments 
    guidance_scale=8.0,            
    num_inference_steps=40,       
    #negative_prompt="ugly, deformed, blurry, low quality, text, words", 
)

if sd_result:
    print(f"Stable Diffusion Success! Output path/object: {sd_result}")
else:
    print("Stable Diffusion Pipeline failed. Check console/logs for errors (ensure SD is available).")
