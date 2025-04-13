import os
import sys
import uuid # For unique temporary filenames
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import traceback # For detailed error logging
from ac_models.predict import main as run_model_prediction



app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = "temp_audio_uploads_direct" # Folder for temporary uploads
ALLOWED_EXTENSIONS = {"mp3", "wav"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# SECRET_KEY is needed for flashing messages
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        # --- 1. Validate Input ---
        if "file" not in request.files:
            flash("No file part provided.")
            return redirect(request.url)

        file = request.files["file"]
        model_checkpoint = request.form.get("model_checkpoint") # This is the path now

        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        if not model_checkpoint:
             flash("No model checkpoint selected.")
             return redirect(request.url)

        if not allowed_file(file.filename):
            flash(f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")
            return redirect(request.url)

        # --- 2. Save File Temporarily ---
        # Use a unique filename to prevent overwrites during concurrent requests
        _, ext = os.path.splitext(file.filename)
        temp_filename = secure_filename(f"{uuid.uuid4()}{ext}")
        temp_filepath = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)
        absolute_filepath = os.path.abspath(temp_filepath) # Use absolute path

        try:
            file.save(temp_filepath)
            print(f"Temporary file saved to: {temp_filepath}")

            # --- 3. Run Prediction ---
            print(f"Running prediction with checkpoint: {model_checkpoint}")
            print(f"Input audio path: {absolute_filepath}")

            # Call your imported prediction function directly
            # Ensure this function handles its own model loading, inference, and returns a string
            caption_result = run_model_prediction(
                checkpoint=model_checkpoint,
                audio_path=absolute_filepath
            )

            # --- 4. Process Result ---
            if caption_result is None:
                # Handle case where function might return None on failure
                caption = "Prediction function did not return a result."
                print(caption)
            else:
                caption = str(caption_result).strip() # Ensure string and remove leading/trailing whitespace
                print(f"Prediction successful. Caption: {caption}")

            # Render result page with the caption or status message
            return render_template("result.html", caption=caption)

        except Exception as e:
            # Catch any exception from the prediction function or file handling
            print(f"ERROR during prediction or file handling: {e}")
            print("--- Traceback ---")
            traceback.print_exc() # Print detailed traceback to console
            print("--- End Traceback ---")
            flash(f"An error occurred during prediction: {str(e)}")
            # Redirect back to upload page on error after saving
            return redirect(request.url)

        finally:
            # --- 5. Cleanup ---
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                    print(f"Removed temporary file: {temp_filepath}")
                except OSError as e:
                    print(f"Error removing temporary file {temp_filepath}: {e}")

    # --- GET Request ---
    # Display the upload form
    return render_template("upload.html")


# --- HTML Templates ---
# (These remain the same as the previous 'subprocess' version, but ensure
# the <select> options have the correct checkpoint paths as values)

if __name__ == "__main__":
    print("Starting Flask development server...")
    # Ensure model checkpoints are accessible from this location
    # Ensure all dependencies for ac_models.predict are installed
    app.run(debug=True, host='0.0.0.0') # Use debug=False for production