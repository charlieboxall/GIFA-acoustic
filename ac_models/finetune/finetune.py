import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from helpers.load_model import load_ac_model

class AudioCaptioningDataset(Dataset):
    def __init__(self, csv_file, feature_extractor, tokenizer, max_length=100):
        self.data = pd.read_csv(csv_file)  # Load CSV file
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]["file_path"]  # File path is already complete
        caption = self.data.iloc[idx]["caption"]

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.feature_extractor.sampling_rate)

        # Extract features
        features = self.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features.squeeze(0)

        # Tokenize caption
        labels = self.tokenizer(
            caption, padding="max_length", truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        ).input_ids.squeeze(0)

        return {"input_features": features, "labels": labels}
    
def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved successfully!")

def train(num_epochs, model, dataloader, device, optimizer):
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())
    print("Training complete.")

def main(checkpoint, dataset_path, save_path, batch_size, num_epochs):
    # Load model, tokenizer, and feature_extractor
    model, tokenizer, feature_extractor, processor = load_ac_model(checkpoint, TRAINING=True)

    # If processor is available, use it; otherwise, use separate tokenizer and feature_extractor
    if processor:
        dataset = AudioCaptioningDataset(dataset_path, processor.feature_extractor, processor.tokenizer)
    else:
        dataset = AudioCaptioningDataset(dataset_path, feature_extractor, tokenizer)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # Freeze encoder layers to prevent catastrophic forgetting
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # Optimizer setup
    optimizer = AdamW(model.parameters(), lr=5e-6)

    # Training the model
    train(num_epochs=num_epochs, model=model, dataloader=dataloader, device=device, optimizer=optimizer)

    # Save the fine-tuned model
    if processor:
        save_model(model=model, processor=processor, save_path=save_path)
    else:
        save_model(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor, save_path=save_path)
