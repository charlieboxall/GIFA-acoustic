import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.optim import AdamW
from tqdm import tqdm
# Removed: import matplotlib as plt
from helpers.load_model import load_ac_model
import datetime # Added for timestamp in report

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
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return None or handle appropriately (e.g., return a dummy sample)
            # For simplicity, we might let it crash or return None and filter later in collate_fn
            # Returning a dummy might be safer for dataloader robustness
            # For this example, we'll raise it to be explicit about the error source
            raise IOError(f"Could not load audio file: {audio_path}") from e


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
             epoch_losses.append(float('nan')) # Use NaN for epochs with no data

    print("Training complete.")
    return epoch_losses

# Removed plot_loss function

def write_training_summary(summary_path, checkpoint, dataset_path, num_audio_files, num_epochs, batch_size, learning_rate, epoch_losses):
    """Writes the training summary to a text file."""
    # Ensure the directory exists
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    try:
        with open(summary_path, 'w') as f:
            f.write("--- Training Summary ---\n\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Checkpoint: {checkpoint}\n")
            f.write(f"Dataset Path: {dataset_path}\n")
            f.write(f"Number of Audio Files Trained: {num_audio_files}\n")
            f.write(f"Number of Epochs: {num_epochs}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Learning Rate: {learning_rate}\n\n")
            f.write("Average Loss per Epoch:\n")
            if epoch_losses:
                for i, loss in enumerate(epoch_losses):
                    # Handle potential NaN values from training function
                    loss_str = f"{loss:.4f}" if not pd.isna(loss) else "NaN"
                    f.write(f"  Epoch {i+1}: {loss_str}\n")
            else:
                f.write("  No loss data recorded.\n")
            f.write("\n--- End of Summary ---\n")
        print(f"Training summary saved to {summary_path}")
    except IOError as e:
        print(f"Error: Could not write training summary to {summary_path}. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing the summary: {e}")


def main(checkpoint, dataset_path, save_path, batch_size, num_epochs, learning_rate):
    """Main function (Simplified Error Handling)."""
    print("Starting main training process...")
    print(f"Configuration:\n Checkpoint: {checkpoint}\n Dataset: {dataset_path}\n Save Path: {save_path}\n Batch Size: {batch_size}\n Epochs: {num_epochs}")

    # --- Learning Rate ---
    print(f" Learning Rate: {learning_rate}")
    # ---------------------

    # Load model components
    try:
        model, tokenizer, feature_extractor, processor = load_ac_model(checkpoint, TRAINING=True)
        print(f"Successfully loaded model components from {checkpoint}.")
    except Exception as e:
        print(f"Error loading model components from {checkpoint}: {e}")
        return # Exit if model loading fails

    # Determine components
    current_tokenizer = processor.tokenizer if processor else tokenizer
    current_feature_extractor = processor.feature_extractor if processor else feature_extractor

    # Create Dataset
    try:
        dataset = AudioCaptioningDataset(dataset_path, current_feature_extractor, current_tokenizer)
        num_audio_files = len(dataset) # Get number of files for reporting
        if num_audio_files == 0:
            print("Warning: Dataset loaded but is empty. Check CSV file and paths.")
            # Decide whether to exit or continue (might error in dataloader)
            # return # Optional: Exit if dataset is empty
    except FileNotFoundError:
        print(f"Error: Dataset CSV file not found at {dataset_path}")
        return
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_collate, # Using default collate - assumes __getitem__ returns tensors
        num_workers=0, # Set to 0 for simplicity, adjust based on system
        pin_memory=False # Set to False, can be True if using GPU and helps performance
        )
    print("DataLoader created.")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Freeze encoder layers (Example: freezing the audio encoder)
    encoder_frozen = False
    # Adjust attribute names based on the actual model structure (e.g., 'audio_encoder', 'encoder', 'model.encoder')
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'): # Common structure for EncoderDecoderModel
        print("Attempting to freeze model.encoder parameters...")
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        encoder_frozen = True
        # Verify freezing
        frozen_params = sum(p.numel() for p in model.model.encoder.parameters() if not p.requires_grad)
        print(f"Verified {frozen_params} parameters in model.encoder are frozen.")
    elif hasattr(model, 'encoder'): # Simpler structure
        print("Attempting to freeze encoder parameters...")
        for param in model.encoder.parameters():
            param.requires_grad = False
        encoder_frozen = True
         # Verify freezing
        frozen_params = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)
        print(f"Verified {frozen_params} parameters in encoder are frozen.")
    # Add other potential attribute checks if needed for different model types

    if encoder_frozen:
         print("Encoder layers frozen successfully.")
    else:
         print("Encoder layers not found or not frozen using checked attributes. Training all layers.")


    # Optimizer setup
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable}")

    if num_trainable == 0:
       print("Error: No trainable parameters found. Check model loading or freezing logic.")
       return # Exit if nothing to train

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

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
        processor=processor # Pass processor if it exists
    )

    # Writing Training Summary
    print("Writing training summary report...")
    summary_save_path = os.path.join(model_save_directory, "training_summary.txt")
    write_training_summary(
        summary_path=summary_save_path,
        checkpoint=checkpoint,
        dataset_path=dataset_path,
        num_audio_files=num_audio_files,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epoch_losses=epoch_losses
    )

    # Removed Plotting section

    print("Finetuning process finished.")
