=========================================
 GIFA: Generate Image From Audio Project
=========================================

Overview
--------
This project implements a pipeline to generate images based on the content of audio files. It works in two main stages:
1. Audio Captioning: Generates a descriptive text caption from an input audio file using a fine-tuned model (either Whisper Tiny or CANVers based).
2. Text-to-Image Generation: Uses the generated caption as a prompt for a text-to-image model (currently Lumina-2, via diffusers) to create a relevant image, often styled as album cover art.

The primary interface for using the full pipeline is the `GIFA` class found in `gifa_library.py`.

Dataset
-------
The audio captioning models were fine-tuned primarily on the Acoustic Music Scenes dataset.

The dataset is publicly available on Hugging Face:
https://huggingface.co/datasets/boxallcharlie/acoustic-music-scenes

A local version of the dataset might be used for development and testing purposes. Ensure the dataset is correctly placed and configured if running the fine-tuning scripts locally.

Models
------
Audio Captioning:
* This project uses fine-tuned audio captioning models based on Whisper Tiny (`ftwhispertiny`) and CANVers (`ftcanvers`).
* The fine-tuned model checkpoints are expected to be located within the project structure, typically under `ac_models/finetuned_models/`.

Image Generation:
* The pipeline uses the `Alpha-VLLM/Lumina-Image-2.0` model via the Hugging Face `diffusers` library for text-to-image generation. This model is loaded automatically when the pipeline is initialized or run.

Usage
-----

### 1. Using the Full Pipeline (Recommended)

The most convenient way to use the complete audio-to-image pipeline is via the `GIFA` class.

* **Class Definition:** `gifa_library.py`
* **Example Usage:** `example_use.py`

Refer to `example_use.py` for a clear demonstration of how to:
    - Import the `GIFA` class.
    - Instantiate the pipeline.
    - Call the `.pipe()` method with parameters like `audio_path`, `prefixes`, `width`, `height`, `seed`, etc.
    - Handle the results (image file path or PIL Image object).

### 2. Fine-tuning Audio Models

If you need to retrain or fine-tune the audio captioning models using the dataset:

* **Fine-tune Whisper Tiny:**
    ```bash
    python -m ac_models.finetune.whispertiny
    ```

* **Fine-tune CANVers:**
    ```bash
    python -m ac_models.finetune.canvers
    ```
    (Note: Ensure the dataset is correctly set up and accessible by these training scripts.)

### 3. Generating Captions Only (Standalone Scripts)

To run only the audio captioning part using the dedicated prediction modules:

* **Predict using Fine-tuned Whisper Tiny:**
    ```bash
    python -m ac_models.predict.ftwhispertiny
    ```

* **Predict using Fine-tuned CANVers:**
    ```bash
    python -m ac_models.predict.ftcanvers
    ```
    (Note: These commands execute the prediction modules directly. They likely require command-line arguments (e.g., for the audio file path) not shown here, or they might operate on default files. For most integrated uses, prefer the `GIFA` pipeline class.)


Notes
-----
* Ensure all required dependencies are installed (check `requirements.txt` if available, otherwise install `torch`, `diffusers`, `transformers`, `librosa`, `Pillow`, etc.).
* The fine-tuned Whisper Tiny model checkpoint (144MB safetensors) is currently managed locally due to size constraints for direct upload in some environments.