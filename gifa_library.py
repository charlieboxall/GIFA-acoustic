import os
import traceback
from typing import List, Optional, Union, Literal

# --- Import Audio Captioning ---
# NOTE: It's assumed 'generate_audio_caption' can handle Hugging Face model IDs.
# If it strictly requires a local path, you would need to modify
# 'generate_audio_caption' or add code here to download/cache the model first.
try:
    # Assuming this function can now accept a Hugging Face model ID string
    from ac_models.predict.predict import main as generate_audio_caption
except ImportError as e:
    print(f"Warning: Could not import audio captioning module: {e}")
    print("Audio captioning functionality will be unavailable.")
    generate_audio_caption = None
except AttributeError:
    print("Warning: Could not find 'main' function in the imported predict module.")
    print("Audio captioning functionality will be unavailable.")
    generate_audio_caption = None

# --- Import Image Generation (Lumina) ---
try:
    from T2I_models.lumina_gen import generate_image as generate_lumina_image, get_lumina_pipeline
    LUMINA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Lumina image generation module (lumina_gen): {e}")
    generate_lumina_image = None
    get_lumina_pipeline = None
    LUMINA_AVAILABLE = False
except AttributeError:
    print("Warning: Could not find expected functions ('generate_image', 'get_lumina_pipeline') in 'T2I_models.lumina_gen'.")
    generate_lumina_image = None
    get_lumina_pipeline = None
    LUMINA_AVAILABLE = False

# --- Import Image Generation (Stable Diffusion) ---
try:
    from T2I_models.sd_gen import generate_image as generate_sd_image, get_sd_pipeline
    SD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Stable Diffusion image generation module (sd_gen): {e}")
    generate_sd_image = None
    get_sd_pipeline = None
    SD_AVAILABLE = False
except AttributeError:
    print("Warning: Could not find expected functions ('generate_image', 'get_sd_pipeline') in 'T2I_models.sd_gen'.")
    generate_sd_image = None
    get_sd_pipeline = None
    SD_AVAILABLE = False

# --- Import PIL ---
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("Warning: PIL/Pillow library not found. Returning image objects is disabled.")
    Image = None
    PIL_AVAILABLE = False


class GIFA:
    """
    A class to generate images from audio files using audio captioning
    (from Hugging Face Hub) and text-to-image models (Lumina or Stable Diffusion).
    """

    # Removed AUDIO_MODEL_BASE_PATH as it's no longer needed for HF models
    DEFAULT_OUTPUT_DIR: str = "output_images"
    DEFAULT_SEED: int = 0
    FIXED_PROMPT_PREFIX: str = "Album Cover art inspired by: "

    # Hugging Face Model Identifiers for Audio Captioning
    HF_WHISPER_TINY_FINETUNE: str = "boxallcharlie/whisper-tiny-AAC-acoustic-music-finetune"
    HF_CANVERS_FINETUNE: str = "boxallcharlie/canvers-AAC-acoustic-music-finetune"

    DEFAULT_LUMINA_WIDTH: int = 512
    DEFAULT_LUMINA_HEIGHT: int = 512
    DEFAULT_SD_WIDTH: int = 512
    DEFAULT_SD_HEIGHT: int = 512


    def __init__(self, preload_image_model: Optional[Literal['lumina', 'sd', 'both', 'none']] = 'lumina'):
        """
        Initializes the GIFA wrapper.

        Args:
            preload_image_model (Optional[Literal['lumina', 'sd', 'both', 'none']]):
                Specifies which image generation model(s) to preload.
                'lumina': Preload only Lumina (if available).
                'sd': Preload only Stable Diffusion (if available).
                'both': Preload both Lumina and Stable Diffusion (if available).
                'none': Do not preload any image models.
                Defaults to 'lumina'.
        """
        print("Initializing GIFA...")
        self._check_dependencies()
        self.lumina_pipe = None
        self.sd_pipe = None

        # --- Image Model Preloading (Unchanged) ---
        if preload_image_model in ['lumina', 'both']:
            if LUMINA_AVAILABLE and get_lumina_pipeline:
                print("Preloading Lumina image generation model...")
                try:
                    self.lumina_pipe = get_lumina_pipeline()
                    if self.lumina_pipe is None:
                        print("Warning: Lumina pipeline failed to load during initialization.")
                    else:
                        print("GIFA Initialized: Lumina pipeline preloaded.")
                except Exception as e:
                    print(f"Warning: Exception during Lumina pipeline preloading: {e}")
                    traceback.print_exc()
            elif preload_image_model == 'lumina':
                print("Warning: Cannot preload Lumina model - it's not available or loader is missing.")

        if preload_image_model in ['sd', 'both']:
            if SD_AVAILABLE and get_sd_pipeline:
                print("Preloading Stable Diffusion image generation model...")
                try:
                    self.sd_pipe = get_sd_pipeline()
                    if self.sd_pipe is None:
                        print("Warning: Stable Diffusion pipeline failed to load during initialization.")
                    else:
                        print("GIFA Initialized: Stable Diffusion pipeline preloaded.")
                except Exception as e:
                    print(f"Warning: Exception during Stable Diffusion pipeline preloading: {e}")
                    traceback.print_exc()
            elif preload_image_model == 'sd':
                print("Warning: Cannot preload Stable Diffusion model - it's not available or loader is missing.")

        if preload_image_model == 'none':
            print("Image model preloading skipped.")


    def _check_dependencies(self):
        """Checks if necessary functions were imported."""
        if not generate_audio_caption:
            print("Critical Warning: Audio captioning function ('generate_audio_caption') is missing.")
        # The rest of the checks remain the same
        if not LUMINA_AVAILABLE:
            print("Warning: Lumina generation functions (from lumina_gen) are missing.")
        if not SD_AVAILABLE:
            print("Warning: Stable Diffusion generation functions (from sd_gen) are missing.")
        if not LUMINA_AVAILABLE and not SD_AVAILABLE:
            print("Critical Warning: NO image generation models (Lumina or SD) are available.")
        if not PIL_AVAILABLE:
            print("Warning: PIL/Pillow library ('Image') is missing. Cannot return Image objects.")


    def pipe(self,
             audio_path: str,
             image_model_type: Literal['lumina', 'sd'] = 'lumina',
             audio_model: str = 'ftwhispertiny', # Keep friendly names
             prefixes: Optional[List[str]] = None,
             output_dir: str = DEFAULT_OUTPUT_DIR,
             seed: int = DEFAULT_SEED,
             save_image: bool = True,
             return_image_object: bool = False,
             width: Optional[int] = None,
             height: Optional[int] = None,
             **model_specific_params
             ) -> Union[str, 'Image.Image', None]:
        """
        Processes an audio file to generate an image based on its caption,
        using either Lumina or Stable Diffusion, allowing pass-through of
        model-specific parameters. Audio captioning model is loaded from Hugging Face Hub.

        Args:
            audio_path (str): Path to the input audio file (e.g., sound.wav).
            image_model_type (Literal['lumina', 'sd']): The image generation model to use. Defaults to 'lumina'.
            audio_model (str): Friendly name of the audio captioning model
                               ('ftwhispertiny' or 'ftcanvers'). Defaults to 'ftwhispertiny'.
            prefixes (Optional[List[str]]): List of strings to prepend to the prompt.
            output_dir (str): Directory to save the image if save_image is True.
            seed (int): Random seed for image generation.
            save_image (bool): If True, saves the image to output_dir.
            return_image_object (bool): If True and PIL is available, returns the PIL Image object.
            width (Optional[int]): Desired image width. Defaults to model standard.
            height (Optional[int]): Desired image height. Defaults to model standard.
            **model_specific_params: Additional keyword arguments to pass directly to the
                                     chosen image generation pipeline (e.g., guidance_scale,
                                     num_inference_steps, cfg_trunc_ratio, negative_prompt).

        Returns:
            Union[str, Image.Image, None]:
            - If save_image is True and return_image_object is False (or PIL unavailable):
              Returns the full path (str) to the saved image file.
            - If return_image_object is True and PIL is available:
              Returns the PIL Image object. (Image is still saved if save_image is True).
            - Returns None if any critical step fails.
        """
        if prefixes is None:
            prefixes = []

        print(f"\n--- Starting GIFA.pipe ({image_model_type.upper()})")
        print(f"Processing: {audio_path}")

        # --- Dependency Checks (Unchanged) ---
        if not generate_audio_caption:
            print("Error: Audio captioning dependency missing. Cannot proceed.")
            return None
        if image_model_type == 'lumina' and not LUMINA_AVAILABLE:
            print(f"Error: Lumina model selected, but it's not available (check lumina_gen). Cannot proceed.")
            return None
        if image_model_type == 'sd' and not SD_AVAILABLE:
            print(f"Error: Stable Diffusion model selected, but it's not available (check sd_gen). Cannot proceed.")
            return None
        if return_image_object and not PIL_AVAILABLE:
            print("Warning: 'return_image_object' is True, but PIL is not available. Will return path instead.")
            return_image_object = False

        # --- Check Audio File (Unchanged) ---
        if not os.path.isfile(audio_path):
            print(f"Error: Audio file not found at '{audio_path}'")
            return None

        # --- Step 1: Generate Audio Caption using Hugging Face Model ID ---
        caption: Optional[str] = None
        audio_model_id: Optional[str] = None # Use a specific variable for HF ID
        try:
            if audio_model == "ftwhispertiny":
                audio_model_id = self.HF_WHISPER_TINY_FINETUNE
            elif audio_model == "ftcanvers":
                audio_model_id = self.HF_CANVERS_FINETUNE
            else:
                print(f"Error: Invalid audio_model specified: '{audio_model}'. Use 'ftwhispertiny' or 'ftcanvers'.")
                return None

            print(f"1. Generating caption using '{audio_model}' model from Hugging Face ({audio_model_id})...")

            # IMPORTANT ASSUMPTION:
            # Assumes 'generate_audio_caption' can accept a Hugging Face model ID string.
            # If it needs a local path, you'd need extra steps here or modify that function.
            caption = generate_audio_caption(checkpoint=audio_model_id, audio_path=audio_path)

            if not caption or not isinstance(caption, str):
                print("Error: Audio captioning failed or returned an invalid result.")
                return None
            print(f"   Generated Raw Caption: '{caption}'")

        except Exception as e:
            print(f"Error during audio captioning step (Model: {audio_model_id}): {e}")
            traceback.print_exc()
            return None

        # --- Step 2: Construct Prompt (Unchanged) ---
        prefix_str = ""
        if prefixes:
            prefix_str = ", ".join(prefixes) + " "
        final_prompt = prefix_str + self.FIXED_PROMPT_PREFIX + caption
        print(f"2. Constructed Prompt: '{final_prompt}'")

        # --- Step 3: Generate Image (Logic Unchanged, uses Lumina/SD functions) ---
        image_filename: Optional[str] = None
        generated_image_path: Optional[str] = None

        # Set default width/height if not provided
        if width is None:
            width = self.DEFAULT_LUMINA_WIDTH if image_model_type == 'lumina' else self.DEFAULT_SD_WIDTH
        if height is None:
            height = self.DEFAULT_LUMINA_HEIGHT if image_model_type == 'lumina' else self.DEFAULT_SD_HEIGHT

        try:
            print(f"3. Generating image using {image_model_type.upper()} ({width}x{height}, seed={seed})...")
            if model_specific_params:
                 print(f"   Passing additional args: {model_specific_params}")
            if save_image:
                os.makedirs(output_dir, exist_ok=True)

            if image_model_type == 'lumina':
                if not generate_lumina_image:
                    raise RuntimeError("Lumina generation function (from lumina_gen) is unavailable.")

                image_filename = generate_lumina_image(
                    prompt=final_prompt,
                    width=width,
                    height=height,
                    save_folder_path=output_dir,
                    seed=seed,
                    **model_specific_params
                )
            elif image_model_type == 'sd':
                if not generate_sd_image:
                    raise RuntimeError("Stable Diffusion generation function (from sd_gen) is unavailable.")

                image_filename = generate_sd_image(
                    prompt=final_prompt,
                    width=width,
                    height=height,
                    save_folder_path=output_dir,
                    seed=seed,
                    **model_specific_params
                )
            else:
                print(f"Error: Unsupported image_model_type '{image_model_type}'")
                return None

            # Check result from image generation function
            if image_filename and isinstance(image_filename, str):
                generated_image_path = os.path.join(output_dir, image_filename)
                print(f"   Image generated: {generated_image_path}")
            else:
                print(f"Error: {image_model_type.upper()} generation function did not return a valid filename.")
                return None

        except Exception as e:
            print(f"Error during {image_model_type.upper()} image generation step: {e}")
            traceback.print_exc()
            return None

        # --- Step 4: Handle Output (Return Path or PIL Object) (Unchanged) ---
        result: Union[str, 'Image.Image', None] = None
        img_object: Optional['Image.Image'] = None

        if return_image_object and PIL_AVAILABLE and generated_image_path and os.path.exists(generated_image_path):
            try:
                print(f"4. Loading generated image into PIL object...")
                img_object = Image.open(generated_image_path)
                img_object.load() # Ensure image data is loaded
                result = img_object
                print(f"   Returning PIL Image object.")
            except Exception as e:
                print(f"Error loading generated image into PIL object: {e}")
                if save_image:
                    result = generated_image_path
                    print(f"   Error loading image, returning path instead: {result}")
                else:
                    result = None # Failed to load and not saving, return None

        elif save_image and generated_image_path:
            result = generated_image_path
            print(f"4. Returning saved image path: {result}")

        elif not save_image and not return_image_object:
            print("4. Image generated but neither saved nor returned as object.")
            pass # result remains None implicitly

        # --- Cleanup Unsaved File (Logic Unchanged) ---
        # Only delete if we are not saving AND we didn't successfully return a PIL object
        should_delete = not save_image and not isinstance(result, Image.Image if PIL_AVAILABLE else type(None))

        if should_delete and generated_image_path and os.path.exists(generated_image_path):
            try:
                print(f"   Cleaning up unsaved file: {generated_image_path}")
                os.remove(generated_image_path)
            except OSError as e:
                print(f"   Warning: Could not remove unsaved file {generated_image_path}: {e}")

        print(f"--- GIFA.pipe Finished ---")
        return result