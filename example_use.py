# my_project.py

# 1. Import the GIFA class from your library file
from gifa_library import GIFA

audio_file_to_process = """C:/GIFA-acoustic/test_audios/test3.mp3"""

# Optional: Define any prefixes or other parameters
"""
--------------------- EXAMPLE STYLE PREFIXES: ---------------------
["Cartoon", "Pixel Art", 
"Sketch", "Line Art", 
"Vector Art", "Comic Book Art", 
"Graphic Novel Art", "Minimalist Poster", 
"Geometric Design", "Children's Book Illustration", 
"Concept Art", "Low Poly", 
"Isometric Art", "Glitch Art", 
"Vaporwave Aesthetic", "Synthwave Aesthetic", 
"ASCII Art", "Digital Painting", 
"Matte Painting", "Watercolour", 
"Charcoal Sketch", "Chalk Drawing", 
"Woodcut Print Style", "Engraving Style", 
"Stained Glass", "Psychedelic Art", 
"Fantasy Art", "Sci-Fi Art", 
"Cyberpunk Art", "Steampunk Art", 
"Grunge Aesthetic", "Art Deco Poster", 
"Art Nouveau Poster"]
"""
style_prefixes_to_add = ["cinematic", "wide angle"]
output_folder = "generated_images"
image_width = 1024
image_height = 768
generation_seed = 42


# 2. Create an instance of the GIFA handler
# (This might take a moment as it can preload models)
print("Initializing GIFA handler...")
gifa = GIFA()
print("Initialization complete.")


# 3. Call the pipe method with your desired arguments
print(f"Processing '{audio_file_to_process}'...")
result = gifa.pipe(
    audio_path=audio_file_to_process,
    prefixes=style_prefixes_to_add,
    output_dir=output_folder,
    width=image_width,
    height=image_height,
    seed=generation_seed,
    save_image=True,            # Keep True to save the image
    return_image_object=False   # Keep False to get the file path back
)


# 4. Print the result (which will be the image path or None)
if result:
    print(f"\nSuccess! Output path: {result}")
else:
    print("\nPipeline failed. Check console/logs for errors.")

    