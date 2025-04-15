from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor
# model, tokenizer, feature_extractor, processor

def load_ac_model(checkpoint, TRAINING=False):
    try:
        print(f"Loading *{checkpoint}")
        
        if checkpoint == "ac_models/finetuned_models/ftwhispertiny":
            print("IN")
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
            feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
            print("Model loaded succesfully.")
            return model, tokenizer, feature_extractor, None
        
        elif checkpoint == "ac_models/finetuned_models/ftwhispertiny-PLUS":
            print("IN")
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
            feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
            print("Model loaded succesfully.")
            return model, tokenizer, feature_extractor, None
        
        elif checkpoint == "MU-NLPC/whisper-tiny-audio-captioning":
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            if TRAINING:
                tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
                feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
                print("Model loaded succesfully.")
                return model, tokenizer, feature_extractor, None
            else:
                processor = WhisperProcessor.from_pretrained(checkpoint)
                print("Model loaded succesfully.")
                return model, None, None, processor

        elif checkpoint == "circulus/canvers-audio-caption-v1":
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            if TRAINING:
                processor = WhisperProcessor.from_pretrained(checkpoint, language="en", task="transcribe")
                return model, None, None, processor
        
        elif checkpoint == "ac_models/finetuned_models/ftcanvers":
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            processor = WhisperProcessor.from_pretrained(checkpoint, language="en", task="transcribe")
            return model, None, None, processor
        
        elif checkpoint == "ac_models/finetuned_models/ftcanvers-PLUS":
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
            processor = WhisperProcessor.from_pretrained(checkpoint, language="en", task="transcribe")
            return model, None, None, processor
        
    except Exception as e:
        print(f"Error loading: \n{e}")