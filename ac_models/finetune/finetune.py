import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib as plt
from helpers.load_model import load_ac_model

class AudioCaptioningDataset(Dataset):
    """Dataset for loading audio files and captions (Simplified Error Handling)."""
    def __init__(self, csv_file, feature_extractor, tokenizer, max_length=100, sampling_rate=16000):
        self.data = pd.read_csv(csv_file)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sampling_rate = getattr(feature_extractor, 'sampling_rate', sampling_rate)
        print(f"Dataset initialized with {len(self.data)} samples from {csv_file}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row["file_path"]
        caption = row["caption"]

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sampling_rate)

        # Extract features
        features = self.feature_extractor(
            audio, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_features.squeeze(0)

        # Tokenize caption
        labels = self.tokenizer(
            str(caption), # Ensure caption is string
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return {"input_features": features, "labels": labels}

def save_model(model, save_path, tokenizer=None, feature_extractor=None, processor=None):
    """Saves the model, tokenizer, and optionally feature extractor or processor."""
    # Ensure directory exists, create if not
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    if processor:
        processor.save_pretrained(save_path)
        print(f"Processor saved to {save_path}")
    else:
        if tokenizer:
            tokenizer.save_pretrained(save_path)
            print(f"Tokenizer saved to {save_path}")
        if feature_extractor:
             if hasattr(feature_extractor, 'save_pretrained'):
                 feature_extractor.save_pretrained(save_path)
                 print(f"Feature extractor saved to {save_path}")
             else:
                 # Cannot save feature extractor if method doesn't exist
                 print("Note: Feature extractor cannot be saved using save_pretrained.")

def train(num_epochs, model, dataloader, device, optimizer):
    """Trains the model and returns the list of average losses per epoch (Simplified)."""
    epoch_losses = []
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for batch in progress_bar:
            # Move batch to device
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

        # Calculate and store average loss for the epoch
        if num_batches > 0:
             avg_epoch_loss = total_loss / num_batches
             epoch_losses.append(avg_epoch_loss)
             print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        else:
             print(f"Epoch {epoch+1} completed with no batches processed.")
             epoch_losses.append(float('nan'))

    print("Training complete.")
    return epoch_losses

def plot_loss(epoch_losses, num_epochs, save_path="training_loss.png"):
    """Plots the training loss and saves it to a file"""
    # Filter out potential NaN values
    valid_epochs = [i + 1 for i, loss in enumerate(epoch_losses) if not pd.isna(loss)]
    valid_losses = [loss for loss in epoch_losses if not pd.isna(loss)]

    if not valid_losses:
        print("No valid loss data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(valid_epochs, valid_losses, marker='o', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.xticks(range(1, num_epochs + 1)) if num_epochs > 0 else plt.xticks([1])
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    # Ensure the directory for the plot exists
    plot_dir = os.path.dirname(save_path)
    if plot_dir: # Only create if path includes a directory
         os.makedirs(plot_dir, exist_ok=True)

    # Save the figure - Error if fails
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")

    plt.show() # Display the plot

def main(checkpoint, dataset_path, save_path, batch_size, num_epochs):
    """Main function (Simplified Error Handling)."""
    print("Starting main training process...")
    print(f"Configuration:\n Checkpoint: {checkpoint}\n Dataset: {dataset_path}\n Save Path: {save_path}\n Batch Size: {batch_size}\n Epochs: {num_epochs}")

    # Load model components
    model, tokenizer, feature_extractor, processor = load_ac_model(checkpoint, TRAINING=True)
    print(f"Successfully loaded model components from {checkpoint}.")

    # Determine components
    current_tokenizer = processor.tokenizer if processor else tokenizer
    current_feature_extractor = processor.feature_extractor if processor else feature_extractor

    # Create Dataset
    dataset = AudioCaptioningDataset(dataset_path, current_feature_extractor, current_tokenizer)
    if len(dataset) == 0:
        print("Warning: Dataset loaded but is empty.")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_collate,
        num_workers=0,
        pin_memory=False
        )
    print("DataLoader created.")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder_frozen = False
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        encoder_frozen = True
    elif hasattr(model, 'encoder'):
         for param in model.encoder.parameters():
             param.requires_grad = False
         encoder_frozen = True

    if encoder_frozen:
         print("Encoder layers frozen.")
    else:
         print("Encoder layers not found or not frozen. Training all layers.")

    # Optimizer setup
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable}")

    if num_trainable == 0:
        print("Error: No trainable parameters found. Check model or freezing logic.")
        return # Exit if nothing to train

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    # Training
    print("Starting training...")
    epoch_losses = train(
        num_epochs=num_epochs,
        model=model,
        dataloader=dataloader,
        device=device,
        optimizer=optimizer
    )

    # Saving Model
    print("Saving model...")
    model_save_directory = save_path
    save_model(
        model=model,
        save_path=model_save_directory,
        tokenizer=current_tokenizer,
        feature_extractor=current_feature_extractor,
        processor=processor
    )

    # Plotting Loss
    print("Plotting training loss...")
    if epoch_losses:
        plot_save_path = os.path.join(model_save_directory, "training_loss_plot.png")
        plot_loss(epoch_losses, num_epochs, save_path=plot_save_path)
    else:
        print("No loss data recorded to plot.")

    print("Finetuning process finished.")
