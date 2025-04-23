import os
import sys
import uuid
import traceback
import time
import shutil
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ac_models.predict import main as run_model_prediction
import re
from T2I_models.lumina_gen import *

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELPERS_DIR = os.path.join(BASE_DIR, "helpers")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_audio_uploads_direct")
SAVE_IMAGE_FOLDER_RELATIVE = os.path.join("helpers", "static", "generated_images")
ABS_GENERATED_IMAGE_FOLDER = os.path.join(BASE_DIR, SAVE_IMAGE_FOLDER_RELATIVE) # Keep this path
SERVE_IMAGE_DIR = ABS_GENERATED_IMAGE_FOLDER # Keep this path
SERVE_AUDIO_FOLDER_RELATIVE = os.path.join("helpers", "static", "served_audio")
ABS_SERVED_AUDIO_FOLDER = os.path.join(BASE_DIR, SERVE_AUDIO_FOLDER_RELATIVE)
SERVE_AUDIO_DIR = ABS_SERVED_AUDIO_FOLDER

ALLOWED_EXTENSIONS = {"mp3", "wav"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

#  In-memory storage for background job details 
jobs = {}

# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ABS_GENERATED_IMAGE_FOLDER, exist_ok=True) # Keep this
os.makedirs(ABS_SERVED_AUDIO_FOLDER, exist_ok=True)
print(f"Ensured UPLOAD_FOLDER exists: {UPLOAD_FOLDER}")
print(f"Ensured IMAGE_FOLDER exists: {ABS_GENERATED_IMAGE_FOLDER}")
print(f"Ensured SERVED_AUDIO_FOLDER exists: {ABS_SERVED_AUDIO_FOLDER}")


def allowed_file(filename):
    # allowed_file remains the same
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#  Main Route 
@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    form_data_repopulate = {}
    style_tuning_selected = []
    job_id = None # ID for the background processing job
    served_audio_filename = None # Filename for immediate audio playback

    if request.method == "POST":
        form_data_repopulate = request.form
        style_tuning_selected = request.form.getlist('style_tuning')

        #  1. Validate Input 
        if "file" not in request.files:
            flash("No file part provided.")
            return render_template("upload.html", form_data=form_data_repopulate, style_tuning_selected=style_tuning_selected)
        file = request.files["file"]
        model_checkpoint = request.form.get("model_checkpoint")
        if file.filename == "" or not model_checkpoint or not allowed_file(file.filename):
            # Flash messages are set inside the checks now
            if file.filename == "": flash("No file selected.")
            if not model_checkpoint: flash("No audio captioning model checkpoint selected.")
            if file.filename != "" and not allowed_file(file.filename): flash(f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")
            return render_template("upload.html", form_data=form_data_repopulate, style_tuning_selected=style_tuning_selected)

        #  2. Prepare for Processing 
        job_id = str(uuid.uuid4()) # Use a new UUID for the job ID
        _, ext = os.path.splitext(file.filename)
        # Use job_id in filename to easily link stored job data and files
        temp_filename_base = secure_filename(f"job_{job_id}")
        temp_filename = f"{temp_filename_base}{ext}"
        temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)

        # Store parameters needed for the background job
        job_params = {
            "audio_path": temp_filepath, # Path to the temporary audio file
            "model_checkpoint": model_checkpoint,
            "style_tuning": style_tuning_selected,
            "cfg_trunc_slider": request.form.get('cfg_trunc_slider', default=1.0, type=float),
            "width_slider": request.form.get('width_slider', default=512, type=int),
            "height_slider": request.form.get('height_slider', default=512, type=int),
            "remove_figure": request.form.get('remove_figure') == 'on',
            "status": "pending",
            "timestamp": time.time()
        }

        try:
            #  3. Save & Copy Audio Immediately 
            file.save(temp_filepath)
            print(f"Temporary audio file saved for job {job_id}: {temp_filepath}")

            served_audio_filepath = os.path.join(ABS_SERVED_AUDIO_FOLDER, temp_filename)
            shutil.copy2(temp_filepath, served_audio_filepath)
            served_audio_filename = temp_filename
            print(f"Copied audio for immediate playback for job {job_id}: {served_audio_filepath}")

            #  4. Store Job & Render Page 
            jobs[job_id] = job_params
            print(f"Stored job {job_id} parameters.")

            return render_template("upload.html",
                                   job_id=job_id,
                                   audio_filename=served_audio_filename,
                                   form_data=form_data_repopulate,
                                   style_tuning_selected=style_tuning_selected)

        except Exception as e:
            print(f"ERROR during initial file handling for job {job_id}: {e}")
            traceback.print_exc()
            flash(f"An error occurred during file preparation: {str(e)}")
            # Clean up if file was saved but copy failed etc.
            if os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError: pass
            if job_id in jobs: # Remove job if setup failed
                del jobs[job_id]
            return render_template("upload.html", form_data=form_data_repopulate, style_tuning_selected=style_tuning_selected)

    #  GET Request 
    return render_template("upload.html", form_data={}, style_tuning_selected=[])


@app.route("/process_job/<job_id>", methods=["POST"])
def process_job(job_id):
    print(f"Received request to process job: {job_id}")
    job_info = jobs.get(job_id)

    if not job_info:
        print(f"Error: Job ID {job_id} not found.")
        return jsonify({"error": "Job not found or expired."}), 404

    if job_info["status"] != "pending":
        print(f"Warning: Job {job_id} already processed or in progress (Status: {job_info['status']}).")
        return jsonify({"error": f"Job already {job_info['status']}."}), 400

    job_info["status"] = "processing"
    temp_audio_filepath = job_info["audio_path"]

    try:
        #  1. Run Audio Captioning 
        print(f"Job {job_id}: Running audio captioning...")
        caption_result = run_model_prediction(
            checkpoint=job_info["model_checkpoint"],
            audio_path=temp_audio_filepath
        )

        display_caption = "Captioning failed."
        image_prompt_caption = None
        final_image_prompt = None
        prompt_prefix_list = []
        generated_image_filename = None

        if caption_result is not None:
            #  2. Clean Caption 
            raw_caption = str(caption_result).strip()
            display_caption = raw_caption.replace("clotho > caption: ", "").replace("audioset > keywords: ", "").strip()
            print(f"Job {job_id}: Caption generated: '{display_caption}'")

            #  3. Prepare Caption for Image Prompt (Potentially Modify) 
            image_prompt_caption = display_caption # Start with the cleaned caption

            #  Apply modifications ONLY if "Remove Figure" is checked 
            if job_info["remove_figure"]:
                original_caption_for_modif_step = image_prompt_caption # For logging comparison

                # First check: Remove "Someone plays" prefix
                if image_prompt_caption.lower().startswith("someone plays"):
                    image_prompt_caption = image_prompt_caption[len("someone plays"):].lstrip(' ,.')
                    print(f"Job {job_id}: [Remove Figure] Applied 'Someone plays' removal. Result: '{image_prompt_caption}'")

                # Second check: Replace "portrays in" with "portrays" (case-insensitive)
                pattern = r'portrays\s+in\b'
                replacement = 'portrays'
                image_prompt_caption, num_subs = re.subn(pattern, replacement, image_prompt_caption, flags=re.IGNORECASE)

                if num_subs > 0:
                        print(f"Job {job_id}: [Remove Figure] Applied 'portrays in' -> 'portrays' replacement ({num_subs} found). Result: '{image_prompt_caption}'")

                # Log final result if any modification happened in this block
                if image_prompt_caption != original_caption_for_modif_step:
                        print(f"Job {job_id}: [Remove Figure] Final modified caption for image prompt: '{image_prompt_caption}'")
                else:
                        print(f"Job {job_id}: [Remove Figure] No modifications applied to caption for image prompt.")

            #  End of "Remove Figure" modifications 

            #  4. Construct Image Prompt & Prefixes 
            prompt_prefix_list = [s for s in job_info["style_tuning"] if s in ["Cartoon", "Pixel Art", "Sketch", "Line Art", "Vector Art", "Comic Book Art", "Graphic Novel Art", "Minimalist Poster", "Geometric Design", "Children's Book Illustration", "Concept Art", "Low Poly", "Isometric Art", "Glitch Art", "Vaporwave Aesthetic", "Synthwave Aesthetic", "ASCII Art", "Digital Painting", "Matte Painting", "Watercolour", "Charcoal Sketch", "Chalk Drawing", "Woodcut Print Style", "Engraving Style", "Stained Glass", "Psychedelic Art", "Fantasy Art", "Sci-Fi Art", "Cyberpunk Art", "Steampunk Art", "Grunge Aesthetic", "Art Deco Poster", "Art Nouveau Poster"]]
            prompt_prefix_list.append("Album cover art inspired by:")
            final_image_prompt = " ".join(prompt_prefix_list) + " " + image_prompt_caption
            print(f"Job {job_id}: Final Image Prompt: {final_image_prompt}")

            #  5. Run Image Generation (Using imported function) 
            print(f"Job {job_id}: Starting image generation...")
            generated_image_filename = generate_image(
                prompt=final_image_prompt,
                width=job_info["width_slider"],
                height=job_info["height_slider"],
                cfg_trunc_ratio=job_info["cfg_trunc_slider"],
                save_folder_path=ABS_GENERATED_IMAGE_FOLDER
            )
            if generated_image_filename:
                print(f"Job {job_id}: Image generated: {generated_image_filename}")
            else:
                print(f"Job {job_id}: Image generation failed.")

        else:
            print(f"Job {job_id}: Audio captioning returned None.")
            display_caption = "Audio captioning failed to produce a result."

        #  6. Update Job Status & Prepare Result 
        job_info["status"] = "completed"
        result_data = {
            "caption": display_caption,
            "image_filename": generated_image_filename,
            "prefixes": prompt_prefix_list,
            "error": None
        }
        print(f"Job {job_id}: Processing complete.")
        return jsonify(result_data)

    except Exception as e:
        job_info["status"] = "failed"
        print(f"ERROR processing job {job_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}", "caption": None, "image_filename": None, "prefixes": []}), 500

    finally:
        #  7. Cleanup 
        if temp_audio_filepath and os.path.exists(temp_audio_filepath):
            try:
                os.remove(temp_audio_filepath)
                print(f"Job {job_id}: Removed temporary audio file: {temp_audio_filepath}")
            except OSError as e:
                print(f"Job {job_id}: Error removing temporary audio file {temp_audio_filepath}: {e}")
        if job_id in jobs and jobs[job_id]["status"] in ["completed", "failed"]:
            print(f"Job {job_id}: Removing job entry from memory.")
            try:
                del jobs[job_id]
            except KeyError:
                pass
 
@app.route('/generated_images/<path:filename>')
def serve_generated_image(filename):
    print(f"Serving image request for '{filename}' from directory '{SERVE_IMAGE_DIR}'")
    try:
        return send_from_directory(SERVE_IMAGE_DIR, filename)
    except FileNotFoundError:
        print(f"File not found error: {filename} in {SERVE_IMAGE_DIR}")
        return f"Error: Image {filename} not found.", 404

@app.route('/served_audio/<path:filename>')
def serve_uploaded_audio(filename):
    print(f"Serving audio request for '{filename}' from directory '{SERVE_AUDIO_DIR}'")
    try:
        return send_from_directory(SERVE_AUDIO_DIR, filename, as_attachment=False)
    except FileNotFoundError:
        print(f"File not found error: {filename} in {SERVE_AUDIO_DIR}")
        return f"Error: Audio file {filename} not found.", 404


if __name__ == "__main__":
    print("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5000)