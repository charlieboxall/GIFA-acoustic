# ... (imports remain the same) ...
import os
import sys
import uuid
import traceback
import time
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from ac_models.predict import main as run_model_prediction

# --- Image Generation Imports ---
import torch
from diffusers import Lumina2Pipeline
from PIL import Image

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELPERS_DIR = os.path.join(BASE_DIR, "helpers") # Adjust if run.py is inside helpers
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_audio_uploads_direct")
SAVE_IMAGE_FOLDER_RELATIVE = os.path.join("helpers", "static", "generated_images")
ABS_GENERATED_IMAGE_FOLDER = os.path.join(BASE_DIR, SAVE_IMAGE_FOLDER_RELATIVE)
SERVE_IMAGE_DIR = ABS_GENERATED_IMAGE_FOLDER
ALLOWED_EXTENSIONS = {"mp3", "wav"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# --- Global Model Loading ---
lumina_pipe = None
def get_lumina_pipeline():
    # ... (model loading code remains the same) ...
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
            raise
    return lumina_pipe


# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ABS_GENERATED_IMAGE_FOLDER, exist_ok=True)
print(f"Ensured UPLOAD_FOLDER exists: {UPLOAD_FOLDER}")
print(f"Ensured IMAGE_FOLDER exists: {ABS_GENERATED_IMAGE_FOLDER}")


def allowed_file(filename):
    # ... (allowed_file remains the same) ...
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Updated generate_image function ---
def generate_image(prompt, width, height, cfg_trunc_ratio, seed=0):
    """Generates an image using Lumina-2 with fixed guidance and variable cfg_trunc_ratio."""
    print(f"Generating image with prompt: '{prompt}'")
    # Note: guidance_scale is now fixed at 4.0
    print(f"Parameters: Width={width}, Height={height}, CFG Trunc Ratio={cfg_trunc_ratio:.2f}, Seed={seed}")
    try:
        pipe = get_lumina_pipeline()
        if pipe is None:
            raise RuntimeError("Lumina pipeline could not be loaded.")

        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.no_grad():
             image = pipe(
                 prompt=prompt,
                 height=int(height),
                 width=int(width),
                 guidance_scale=4.0, # *** Fixed guidance scale ***
                 num_inference_steps=25,
                 cfg_trunc_ratio=float(cfg_trunc_ratio), # *** Use slider value ***
                 cfg_normalization=True,
                 generator=generator
             ).images[0]

        if image.mode != 'RGB':
             image = image.convert('RGB')

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4()
        filename = f"img_{timestamp}_{unique_id}.png"
        save_path = os.path.join(ABS_GENERATED_IMAGE_FOLDER, filename)
        image.save(save_path)
        print(f"Image saved to: {save_path}")
        return filename

    except Exception as e:
        print(f"ERROR during image generation: {e}")
        print("--- Image Generation Traceback ---")
        traceback.print_exc()
        print("--- End Image Generation Traceback ---")
        return None

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        # --- 1. Validate Audio Input ---
        # ... (audio validation remains the same) ...
        if "file" not in request.files:
            flash("No file part provided.")
            return redirect(request.url)
        file = request.files["file"]
        model_checkpoint = request.form.get("model_checkpoint")
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)
        if not model_checkpoint:
             flash("No audio captioning model checkpoint selected.")
             return redirect(request.url)
        if not allowed_file(file.filename):
            flash(f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")
            return redirect(request.url)


        # --- Get Image Generation Parameters ---
        style_tuning = request.form.getlist('style_tuning')
        # *** Get the CFG Trunc Ratio slider value (0.1 - 1.0) ***
        cfg_trunc_slider = request.form.get('cfg_trunc_slider', default=1.0, type=float)
        width_slider = request.form.get('width_slider', default=512, type=int)
        height_slider = request.form.get('height_slider', default=512, type=int)

        # --- 2. Save Audio File Temporarily ---
        # ... (audio saving remains the same) ...
        _, ext = os.path.splitext(file.filename)
        temp_filename = secure_filename(f"{uuid.uuid4()}{ext}")
        temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
        absolute_audio_filepath = temp_filepath

        caption = "Error during processing."
        generated_image_filename = None

        try:
            file.save(temp_filepath)
            print(f"Temporary audio file saved to: {temp_filepath}")

            # --- 3. Run Audio Captioning ---
            # ... (audio captioning call remains the same) ...
            print(f"Running audio captioning with checkpoint: {model_checkpoint}")
            print(f"Input audio path: {absolute_audio_filepath}")
            caption_result = run_model_prediction(
                checkpoint=model_checkpoint,
                audio_path=absolute_audio_filepath
            )

            if caption_result is None:
                # ... (handling caption failure remains the same) ...
                caption = "Audio captioning function did not return a result."
                print(caption)
                flash(caption)
                return render_template("result.html", caption=caption, image_filename=None)

            else:
                caption = str(caption_result).strip()
                print(f"Audio captioning successful. Caption: {caption}")

                # --- 4. Construct Image Prompt ---
                # ... (prompt construction remains the same) ...
                prompt_prefix = [s for s in style_tuning if s in ["Abstract", "Cartoon", "Photorealistic", "Oil Painting", "Sketch", "Pixel Art", "Vintage", "Surrealism"]]
                prompt_prefix.append("Album cover art inspired:")
                final_image_prompt = " ".join(prompt_prefix) + " " + caption


                # --- 5. Run Image Generation ---
                # *** Pass cfg_trunc_slider to generate_image ***
                generated_image_filename = generate_image(
                    prompt=final_image_prompt,
                    width=width_slider,
                    height=height_slider,
                    cfg_trunc_ratio=cfg_trunc_slider # Pass the slider value here
                )
                if generated_image_filename is None:
                    flash("Image generation failed. Displaying caption only.")

        except Exception as e:
            # ... (general error handling remains the same) ...
            print(f"ERROR during processing: {e}")
            print("--- Overall Traceback ---")
            traceback.print_exc()
            print("--- End Overall Traceback ---")
            flash(f"An error occurred: {str(e)}")
            caption = f"An error occurred during processing: {str(e)}"


        finally:
            # --- 6. Cleanup Audio File ---
            # ... (audio cleanup remains the same) ...
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                    print(f"Removed temporary audio file: {temp_filepath}")
                except OSError as e:
                    print(f"Error removing temporary audio file {temp_filepath}: {e}")


        # --- 7. Render Result Page ---
        # ... (rendering result remains the same) ...
        return render_template("result.html",
                               caption=caption,
                               image_filename=generated_image_filename)

    # --- GET Request ---
    return render_template("upload.html")

# --- Route to serve generated images ---
@app.route('/generated_images/<path:filename>')
def serve_generated_image(filename):
    # ... (serving route remains the same, using absolute path) ...
    print(f"Serving request for '{filename}' from directory '{SERVE_IMAGE_DIR}'")
    try:
        return send_from_directory(SERVE_IMAGE_DIR, filename)
    except FileNotFoundError:
         print(f"File not found error: {filename} in {SERVE_IMAGE_DIR}")
         flash(f"Error: Could not find generated image {filename}", "error")
         return f"Error: Image {filename} not found.", 404

if __name__ == "__main__":
    # ... (startup messages remain the same) ...
    print("Starting Flask development server...")
    print(f"Base directory (BASE_DIR): {BASE_DIR}")
    print(f"Absolute image save/serve path (SERVE_IMAGE_DIR): {SERVE_IMAGE_DIR}")
    if not os.path.isdir(SERVE_IMAGE_DIR):
         print(f"WARNING: The image directory does not seem to exist: {SERVE_IMAGE_DIR}")

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, Lumina model might fail or be very slow.")

    app.run(debug=True, host='0.0.0.0', port=5000)