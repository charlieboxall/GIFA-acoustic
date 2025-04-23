# gifa_library.py

import os
import traceback
from typing import List, Optional, Union

try:
    # Assuming predict.py's main function returns the caption string
    from ac_models.predict.predict import main as generate_audio_caption
except ImportError as e:
    print(f"Warning: Could not import audio captioning module: {e}")
    print("Audio captioning functionality will be unavailable.")
    generate_audio_caption = None
except AttributeError:
    print("Warning: Could not find 'main' function in the imported predict module.")
    print("Audio captioning functionality will be unavailable.")
    generate_audio_caption = None

try:
    # Import both the generation function and the pipeline loader
    from T2I_models.lumina_gen import generate_image, get_lumina_pipeline
    from PIL import Image # Import PIL here as we might return Image objects
except ImportError as e:
    print(f"Warning: Could not import image generation module or PIL: {e}")
    print("Image generation functionality will be unavailable.")
    generate_image = None
    get_lumina_pipeline = None
    Image = None # type: ignore
except AttributeError:
    print("Warning: Could not find 'generate_image' or 'get_lumina_pipeline' in 'T2I_models.lumina_gen'.")
    print("Image generation functionality will be unavailable.")
    generate_image = None
    get_lumina_pipeline = None


class GIFA:
    """
    A class to generate images from audio files using audio captioning
    and text-to-image models.
    """

    # --- Default Configuration ---
    AUDIO_MODEL_BASE_PATH: str = "ac_models/finetuned_models"
    DEFAULT_OUTPUT_DIR: str = "output_images"
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512
    DEFAULT_CFG_TRUNC_RATIO: float = 1.0
    DEFAULT_SEED: int = 0
    FIXED_PROMPT_PREFIX: str = "Album Cover art inspired by: "

    def __init__(self, preload_image_model: bool = True):
        """
        Initializes the GIFA wrapper.

        Args:
            preload_image_model (bool): If True, attempts to load the
                                        image generation model immediately.
                                        Defaults to True.
        """
        print("Initializing GIFA...")
        self._check_dependencies()
        self.lumina_pipe = None

        if preload_image_model and get_lumina_pipeline:
            print("Preloading image generation model...")
            try:
                # Attempt to load the pipeline for potentially faster first use
                self.lumina_pipe = get_lumina_pipeline()
                if self.lumina_pipe is None:
                    print("Warning: Lumina pipeline failed to load during initialization.")
                else:
                    print("GIFA Initialized: Lumina pipeline preloaded.")
            except Exception as e:
                print(f"Warning: Exception during Lumina pipeline preloading: {e}")
                traceback.print_exc()
        elif preload_image_model:
             print("Warning: Cannot preload image model - 'get_lumina_pipeline' not available.")
        else:
            print("Image model preloading skipped.")


    def _check_dependencies(self):
        """Checks if necessary functions were imported."""
        if not generate_audio_caption:
             print("Critical Warning: Audio captioning function ('generate_audio_caption') is missing.")
        if not generate_image:
             print("Critical Warning: Image generation function ('generate_image') is missing.")
        if not get_lumina_pipeline:
             print("Critical Warning: Image pipeline loader ('get_lumina_pipeline') is missing.")
        if not Image:
             print("Critical Warning: PIL/Pillow library ('Image') is missing.")


    def pipe(self,
             audio_path: str,
             model: str = 'ftwhispertiny',
             prefixes: Optional[List[str]] = None,
             output_dir: str = DEFAULT_OUTPUT_DIR,
             width: int = DEFAULT_WIDTH,
             height: int = DEFAULT_HEIGHT,
             cfg_trunc_ratio: float = DEFAULT_CFG_TRUNC_RATIO,
             seed: int = DEFAULT_SEED,
             save_image: bool = True,
             return_image_object: bool = False) -> Union[str, 'Image.Image', None]:
        """
        Processes an audio file to generate an image based on its caption.

        Args:
            audio_path (str): Path to the input audio file (e.g., sound.wav).
            model (str): Name of the audio captioning model to use
                         ('ftwhispertiny' or 'ftcanvers'). Defaults to 'ftwhispertiny'.
            prefixes (Optional[List[str]]): A list of strings to prepend to the
                                            prompt (e.g., ["Pixel art", "8-bit"]).
                                            Defaults to None (no prefixes).
            output_dir (str): Directory where the generated image will be saved
                              if save_image is True. Defaults to 'output_images'.
            width (int): Desired width of the generated image. Defaults to 512.
            height (int): Desired height of the generated image. Defaults to 512.
            cfg_trunc_ratio (float): CFG Truncation Ratio for the image model.
                                     Defaults to 1.0.
            seed (int): Random seed for image generation. Defaults to 0.
            save_image (bool): If True, saves the generated image to the output_dir.
                               Defaults to True.
            return_image_object (bool): If True, returns the generated image as a
                                        PIL Image object. Defaults to False.

        Returns:
            Union[str, Image.Image, None]:
            - If save_image is True and return_image_object is False:
              Returns the full path (str) to the saved image file.
            - If return_image_object is True:
              Returns the PIL Image object. (The image is still saved to disk
              if save_image is True).
            - Returns None if any critical step fails (e.g., file not found,
              model error, dependencies missing).
        """
        if prefixes is None:
            prefixes = [] # Default to empty list if None is passed

        print(f"\n--- Starting GIFA.pipe ---")
        print(f"Processing: {audio_path}")

        # --- Dependency Check ---
        if not generate_audio_caption or not generate_image or not Image:
            print("Error: Critical dependency missing (check initialization warnings). Cannot proceed.")
            return None

        # --- Input Validation ---
        if not os.path.isfile(audio_path):
            print(f"Error: Audio file not found at '{audio_path}'")
            return None

        # --- Step 1: Audio Captioning ---
        caption: Optional[str] = None
        try:
            audio_checkpoint_name = model
            # Construct path to the directory containing the model files
            if audio_checkpoint_name == "ftwhispertiny":
                audio_checkpoint_path = "ac_models/finetuned_models/ftwhispertiny"
            elif audio_checkpoint_name == "ftcanvers":
                audio_checkpoint_path = "ac_models/finetuned_models/ftcanvers"
            print(f"1. Generating caption using '{model}' model (path: {audio_checkpoint_path})...")

            # Ensure the expected checkpoint path exists (optional but good practice)
            if not os.path.isdir(audio_checkpoint_path):
                 print(f"Warning: Audio checkpoint directory not found at '{audio_checkpoint_path}'")
                 # generate_audio_caption might handle this internally, or fail.

            caption = generate_audio_caption(checkpoint=audio_checkpoint_path, audio_path=audio_path)

            if not caption or not isinstance(caption, str):
                print("Error: Audio captioning failed or returned an invalid result.")
                return None
            print(f"   Generated Raw Caption: '{caption}'")

        except Exception as e:
            print(f"Error during audio captioning step: {e}")
            traceback.print_exc()
            return None

        # --- Step 2: Construct Prompt ---
        prefix_str = ""
        if prefixes:
            prefix_str = ", ".join(prefixes) + " "
        final_prompt = prefix_str + self.FIXED_PROMPT_PREFIX + caption
        print(f"2. Constructed Prompt: '{final_prompt}'")

        # --- Step 3: Image Generation ---
        image_filename: Optional[str] = None
        generated_image_path: Optional[str] = None
        try:
            print(f"3. Generating image ({width}x{height}, seed={seed})...")
            # Ensure output directory exists if we intend to save
            if save_image:
                os.makedirs(output_dir, exist_ok=True)

            # generate_image expects the *folder* path where it will save the image
            image_filename = generate_image(
                prompt=final_prompt,
                width=width,
                height=height,
                cfg_trunc_ratio=cfg_trunc_ratio,
                save_folder_path=output_dir, # Pass the directory
                seed=seed
            )

            if image_filename:
                generated_image_path = os.path.join(output_dir, image_filename)
                print(f"   Image generated: {generated_image_path}")
            else:
                print("Error: Image generation function did not return a filename.")
                return None # Failed

        except Exception as e:
            print(f"Error during image generation step: {e}")
            traceback.print_exc()
            return None # Failed

        # --- Step 4: Handle Output ---
        result: Union[str, 'Image.Image', None] = None
        img_object: Optional['Image.Image'] = None

        if return_image_object:
            if generated_image_path and os.path.exists(generated_image_path):
                try:
                    print(f"4. Loading generated image into PIL object...")
                    img_object = Image.open(generated_image_path)
                    img_object.load() # Load data into memory before returning
                    result = img_object
                    print(f"   Returning PIL Image object.")
                except Exception as e:
                    print(f"Error loading generated image into PIL object: {e}")
                    # Fallback if loading fails but saving was intended
                    if save_image:
                        result = generated_image_path
                        print(f"   Error loading image, returning path instead: {result}")
                    else:
                        result = None # Failed to return object, wasn't saving path
            else:
                 print("Error: Cannot return image object - file not found or not generated.")
                 result = None # Failed

        # If not returning object, or if returning object failed but saving was intended
        if result is None and save_image and generated_image_path:
             result = generated_image_path
             print(f"4. Returning saved image path: {result}")

        # If saving was explicitly False and returning object was False or failed
        if not save_image and not isinstance(result, Image.Image) and generated_image_path and os.path.exists(generated_image_path):
             try:
                 print(f"   Cleaning up unsaved file: {generated_image_path}")
                 os.remove(generated_image_path)
             except OSError as e:
                 print(f"   Warning: Could not remove unsaved file {generated_image_path}: {e}")

        print(f"--- GIFA.pipe Finished ---")
        return result