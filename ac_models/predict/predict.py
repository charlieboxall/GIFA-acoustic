import os
import torch
import librosa
from helpers.load_model import load_ac_model

def generate_caption(model, tokenizer, feature_extractor, audio_path, device):
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    features = feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features

    # Ensure 3D shape: [batch_size, num_features, sequence_length]
    if features.dim() == 2:
        features = features.unsqueeze(0)  # Add batch dimension
    elif features.dim() == 3 and features.shape[0] != 1:
        raise ValueError(f"Expected shape [1, 80, N], but got: {features.shape}")
    
    features = features.to(device)

    # Clear forced decoder settings
    if hasattr(model, "config"):
        model.config.forced_decoder_ids = None
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_features=features,
            max_length=100,
            return_timestamps=False
        )

    # Decode output
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    else:
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main(checkpoint, audio_path):
    model, tokenizer, feature_extractor, processor = load_ac_model(checkpoint)

    # Handle models that use a combined processor
    if processor is not None:
        tokenizer = processor.tokenizer
        feature_extractor = processor.feature_extractor

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # Generate and print caption
    caption = generate_caption(model, tokenizer, feature_extractor, audio_path, device)
    #print(caption)
    return caption
