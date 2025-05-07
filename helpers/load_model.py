from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperProcessor,
    AutoProcessor,             # Added
    AutoModelForSpeechSeq2Seq  # Added
)
# model, tokenizer, feature_extractor, processor

def load_ac_model(checkpoint, TRAINING=False):
    """
    Loads an audio captioning model and associated components based on the checkpoint identifier.

    Args:
        checkpoint (str): The identifier for the model checkpoint. This can be a local path
                          or a Hugging Face Hub model ID.
        TRAINING (bool): Flag indicating if the model is being loaded for training
                         (affects loading for "MU-NLPC/whisper-tiny-audio-captioning").

    Returns:
        tuple: A tuple containing (model, tokenizer, feature_extractor, processor).
               Some elements might be None depending on the model type and checkpoint.
               Returns (None, None, None, None) if loading fails.
    """
    try:
        print(f"Attempting to load audio captioning model: *{checkpoint}*")

        # --- Local Finetuned Whisper Tiny ---
        if checkpoint == "ac_models/finetuned_models/ftwhispertiny":
            print("Loading local ftwhispertiny...")
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
            feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
            print("Local ftwhispertiny loaded successfully.")
            return model, tokenizer, feature_extractor, None

        # --- Local Finetuned Whisper Tiny PLUS (Example) ---
        elif checkpoint == "ac_models/finetuned_models/ftwhispertiny-PLUS":
            print("Loading local ftwhispertiny-PLUS...")
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
            feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
            print("Local ftwhispertiny-PLUS loaded successfully.")
            return model, tokenizer, feature_extractor, None

        # --- Hugging Face MU-NLPC Whisper Tiny ---
        elif checkpoint == "MU-NLPC/whisper-tiny-audio-captioning":
            print(f"Loading Hugging Face model: {checkpoint}")
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            if TRAINING:
                tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
                feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
                print("Model, tokenizer, feature extractor loaded successfully (for training).")
                return model, tokenizer, feature_extractor, None
            else:
                processor = WhisperProcessor.from_pretrained(checkpoint)
                print("Model and processor loaded successfully (for inference).")
                return model, None, None, processor

        # --- Hugging Face Circulus Canvers ---
        elif checkpoint == "circulus/canvers-audio-caption-v1":
            print(f"Loading Hugging Face model: {checkpoint}")
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            processor = WhisperProcessor.from_pretrained(checkpoint, language="en", task="transcribe")
            print("Model and processor loaded successfully.")
            return model, None, None, processor

        # --- Local Finetuned Canvers ---
        elif checkpoint == "ac_models/finetuned_models/ftcanvers":
            print("Loading local ftcanvers...")
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            processor = WhisperProcessor.from_pretrained(checkpoint, language="en", task="transcribe")
            print("Local ftcanvers loaded successfully.")
            return model, None, None, processor

        # --- Local Finetuned Canvers PLUS (Example) ---
        elif checkpoint == "ac_models/finetuned_models/ftcanvers-PLUS":
            print("Loading local ftcanvers-PLUS...")
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            processor = WhisperProcessor.from_pretrained(checkpoint, language="en", task="transcribe")
            print("Local ftcanvers-PLUS loaded successfully.")
            return model, None, None, processor

        # --- NEW: Hugging Face boxallcharlie Canvers Finetune ---
        elif checkpoint == "boxallcharlie/canvers-AAC-acoustic-music-finetune":
            print(f"Loading Hugging Face model: {checkpoint}")
            # Use Auto classes as the base model might differ from standard Whisper
            processor = AutoProcessor.from_pretrained(checkpoint)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(checkpoint)
            print("Model and processor loaded successfully.")
            # Return format consistent with other processor-based models
            return model, None, None, processor

        # --- NEW: Hugging Face boxallcharlie Whisper Tiny Finetune ---
        elif checkpoint == "boxallcharlie/whisper-tiny-AAC-acoustic-music-finetune":
            print(f"Loading Hugging Face model: {checkpoint}")
            # Use Auto classes
            processor = AutoProcessor.from_pretrained(checkpoint)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(checkpoint)
            print("Model and processor loaded successfully.")
            # Return format consistent with other processor-based models
            return model, None, None, processor

        # --- Checkpoint Not Found ---
        else:
             print(f"Error: Checkpoint '{checkpoint}' not recognized in load_ac_model function.")
             return None, None, None, None

    except Exception as e:
        print(f"Error loading model/components for checkpoint '{checkpoint}': \n{e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None, None, None, None