import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import datetime
import evaluate
import nltk
nltk.download('punkt')
import numpy as np
from sklearn.model_selection import train_test_split # Added for splitting

# Ensure nltk punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)

from helpers.load_model import load_ac_model

# --- Dataset Class (Modified to accept DataFrame) ---
class AudioCaptioningDataset(Dataset):
    """Dataset for loading audio files and captions from CSV file or DataFrame."""
    def __init__(self, data_source, feature_extractor, tokenizer, max_length=100, sampling_rate=16000):
        # Load data from path or use passed DataFrame
        if isinstance(data_source, str):
            print(f"Loading dataset from CSV file: {data_source}")
            try:
                self.data = pd.read_csv(data_source)
            except FileNotFoundError:
                raise FileNotFoundError(f"Error: Dataset CSV file not found at {data_source}")
            except Exception as e:
                raise RuntimeError(f"Error reading or processing CSV {data_source}: {e}")
        elif isinstance(data_source, pd.DataFrame):
            print(f"Loading dataset from provided DataFrame.")
            self.data = data_source.copy() # Use a copy to avoid modifying original DataFrame
        else:
            raise TypeError("data_source must be a file path (str) or a pandas DataFrame")

        # Basic validation of the data
        if "caption" not in self.data.columns or "file_path" not in self.data.columns:
            raise ValueError("Dataset must contain 'file_path' and 'caption' columns.")
        self.data = self.data.dropna(subset=['caption', 'file_path']) # Remove rows with NaN captions or paths
        self.data['caption'] = self.data['caption'].astype(str) # Ensure captions are strings

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sampling_rate = getattr(feature_extractor, 'sampling_rate', sampling_rate)
        print(f"Dataset initialized with {len(self.data)} valid samples.")
        if len(self.data) == 0:
             print("Warning: Dataset contains zero valid samples after loading/cleaning.")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ... (rest of __getitem__ remains the same) ...
        if idx >= len(self.data):
             raise IndexError("Index out of bounds")
        row = self.data.iloc[idx]
        audio_path = row["file_path"]
        caption = row["caption"]

        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            if len(audio) == 0:
                print(f"Warning: Loaded empty audio file {audio_path}")
                raise ValueError(f"Loaded empty audio file: {audio_path}")
        except Exception as e:
            print(f"Error loading or processing audio file {audio_path}: {e}")
            raise IOError(f"Could not load or process audio file: {audio_path}") from e

        features = self.feature_extractor(
            audio, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_features.squeeze(0)

        labels = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return {"input_features": features, "labels": labels, "caption": caption}

# --- Custom Collate Function (Unchanged) ---
def custom_collate_fn(batch):
    # ... (collate function remains the same) ...
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    input_features = [item['input_features'] for item in batch]
    labels = [item['labels'] for item in batch]
    captions = [item['caption'] for item in batch]

    labels_padded = torch.stack(labels)

    try:
        features_padded = torch.stack(input_features)
    except RuntimeError as e:
        print(f"Error stacking input features: {e}. Features might have variable lengths. Check feature extractor padding or implement manual padding in collate_fn.")
        max_len = max(f.shape[1] for f in input_features)
        padded_features = []
        for f in input_features:
            pad_len = max_len - f.shape[1]
            padded_f = torch.nn.functional.pad(f, (0, pad_len))
            padded_features.append(padded_f)
        features_padded = torch.stack(padded_features)

    return {
        "input_features": features_padded,
        "labels": labels_padded,
        "captions": captions
    }


# --- Save Model Function (Unchanged) ---
def save_model(model, save_path, tokenizer=None, feature_extractor=None, processor=None):
    # ... (save_model function remains the same) ...
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
                print("Note: Feature extractor cannot be saved using save_pretrained.")

# --- Evaluate Function (Unchanged) ---
def run_evaluation(model, dataloader, tokenizer, device, metrics_dict, generation_kwargs):
    # ... (evaluate function remains the same) ...
    model.eval()
    all_predictions = []
    all_references = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
             if batch is None:
                 continue
             input_features = batch["input_features"].to(device)
             references = batch["captions"]

             output_ids = model.generate(input_features, **generation_kwargs)
             predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

             all_predictions.extend([p.strip() for p in predictions])
             all_references.extend([[r.strip()] for r in references])

    results = {}
    print("\nComputing evaluation metrics...")
    for name, metric in metrics_dict.items():
        try:
            metric_result = metric.compute(predictions=all_predictions, references=all_references)
            if name == 'bleu':
                results[name] = metric_result.get('bleu', 0.0) * 100
            elif name == 'rouge':
                 results['rougeL_f1'] = metric_result.get('rougeL', 0.0) * 100
            elif name == 'meteor':
                results[name] = metric_result.get('meteor', 0.0) * 100
            else:
                 if isinstance(metric_result, dict):
                      results.update({f"{name}_{k}": v for k, v in metric_result.items()})
                 else:
                      results[name] = metric_result
        except Exception as e:
            print(f"Failed to compute metric {name}: {e}")
            results[name] = 0.0

    print(f"Evaluation Results: {results}")
    return results

# --- Train One Epoch Function (Unchanged) ---
def train_one_epoch(model, dataloader, device, optimizer):
    # ... (train_one_epoch function remains the same) ...
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Training Epoch", leave=True)

    for batch in progress_bar:
        if batch is None:
            continue
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1
        progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

    if num_batches > 0:
        avg_epoch_loss = total_loss / num_batches
    else:
        print("Epoch completed with no batches processed.")
        avg_epoch_loss = float('nan')

    return avg_epoch_loss

# --- Write Training Summary (Modified for split info) ---
def write_training_summary(summary_path, checkpoint, input_csv_path, validation_split_ratio, num_total_files, num_train_files, num_val_files, num_epochs_run, best_epoch, batch_size, learning_rate, train_losses, val_metrics_history, best_val_metric_score, metric_for_best_model):
    """Writes the training summary including validation results and split info."""
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    try:
        with open(summary_path, 'w') as f:
            f.write("--- Training Summary ---\n\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Checkpoint: {checkpoint}\n")
            f.write(f"Input Dataset: {input_csv_path}\n")
            f.write(f"Total Usable Files in Input: {num_total_files}\n")
            if validation_split_ratio > 0:
                 f.write(f"Validation Split Ratio: {validation_split_ratio:.2f}\n")
                 f.write(f"Training Files: {num_train_files}\n")
                 f.write(f"Validation Files: {num_val_files}\n")
            else:
                 f.write("Validation Split: Not performed\n")
                 f.write(f"Training Files: {num_train_files}\n")

            f.write(f"Target Max Epochs: {num_epochs_run['target']}\n")
            f.write(f"Actual Epochs Run: {num_epochs_run['actual']}\n")
            if validation_split_ratio > 0 and best_epoch > 0:
                f.write(f"Best Epoch (based on validation {metric_for_best_model}): {best_epoch}\n")
                f.write(f"Best Validation Score ({metric_for_best_model}): {best_val_metric_score:.4f}\n")
            elif validation_split_ratio > 0:
                 f.write("Best Epoch: N/A (No improvement observed on validation set)\n")

            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Learning Rate: {learning_rate}\n\n")

            f.write("--- Training Loss per Epoch ---\n")
            # ... (rest of reporting remains similar) ...
            if train_losses:
                for i, loss in enumerate(train_losses):
                    loss_str = f"{loss:.4f}" if not pd.isna(loss) else "NaN"
                    f.write(f"  Epoch {i+1}: {loss_str}\n")
            else:
                f.write("  No training loss data recorded.\n")

            f.write("\n--- Validation Metrics per Epoch ---\n")
            if val_metrics_history:
                all_keys = sorted(list(set(key for epoch_metrics in val_metrics_history for key in epoch_metrics)))
                header = "Epoch | " + " | ".join(all_keys)
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                for i, epoch_metrics in enumerate(val_metrics_history):
                    metric_strs = [f"{epoch_metrics.get(key, float('nan')):.4f}" for key in all_keys]
                    f.write(f"{i+1:<5} | " + " | ".join(metric_strs) + "\n")
            elif validation_split_ratio > 0:
                 f.write("  No validation metrics recorded (Check evaluation step).\n")
            else:
                 f.write("  Validation not performed.\n")

            f.write("\n--- End of Summary ---\n")
        print(f"Training summary saved to {summary_path}")
    except IOError as e:
        print(f"Error: Could not write training summary to {summary_path}. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing the summary: {e}")


# --- Main Function (Modified for Auto-Split) ---
def main(
    checkpoint,
    input_csv_path, # Changed: Single input CSV
    validation_split_ratio, # Added: Ratio for validation split (e.g., 0.15)
    random_seed, # Added: For reproducible splits
    save_path,
    batch_size,
    eval_batch_size,
    num_epochs,
    learning_rate,
    early_stopping_patience,
    metric_for_best_model,
    generation_max_length=100
    ):
    """Main function with automatic data splitting, validation and early stopping."""
    print("Starting main training process...")
    print(f"Configuration:\n Checkpoint: {checkpoint}\n Input Dataset: {input_csv_path}")
    print(f" Validation Split Ratio: {validation_split_ratio if validation_split_ratio > 0 else 'N/A'}")
    print(f" Save Path: {save_path}\n Batch Size (Train): {batch_size}\n Batch Size (Eval): {eval_batch_size}")
    print(f" Epochs: {num_epochs}\n Learning Rate: {learning_rate}")
    if validation_split_ratio > 0:
        print(f" Early Stopping Patience: {early_stopping_patience}\n Metric for Best Model: {metric_for_best_model}\n Random Seed for Split: {random_seed}")

    # --- Load model components ---
    try:
        model, tokenizer, feature_extractor, processor = load_ac_model(checkpoint, TRAINING=True)
        print(f"Successfully loaded model components from {checkpoint}.")
    except Exception as e:
        print(f"Error loading model components from {checkpoint}: {e}")
        return

    current_tokenizer = processor.tokenizer if processor else tokenizer
    current_feature_extractor = processor.feature_extractor if processor else feature_extractor

    # --- Load Data and Perform Split ---
    try:
        print(f"Loading data from {input_csv_path}...")
        df_full = pd.read_csv(input_csv_path)
        # Initial cleaning matching dataset class logic
        if "caption" not in df_full.columns or "file_path" not in df_full.columns:
             raise ValueError("Input CSV must contain 'file_path' and 'caption' columns.")
        df_full = df_full.dropna(subset=['caption', 'file_path'])
        df_full['caption'] = df_full['caption'].astype(str)
        num_total_files = len(df_full)
        print(f"Found {num_total_files} usable entries in the CSV.")
        if num_total_files == 0:
             print("Error: No usable data found in the input CSV after cleaning.")
             return

        train_df = None
        val_df = None
        perform_validation = validation_split_ratio is not None and 0 < validation_split_ratio < 1

        if perform_validation:
            print(f"Splitting data into train/validation sets with ratio {validation_split_ratio:.2f}...")
            train_df, val_df = train_test_split(
                df_full,
                test_size=validation_split_ratio,
                random_state=random_seed,
                shuffle=True # Shuffle before splitting
            )
            print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
        else:
            print("Using full dataset for training. Validation will be skipped.")
            train_df = df_full
            val_df = None # Ensure val_df is None if no split

        num_train_files = len(train_df)
        num_val_files = len(val_df) if val_df is not None else 0


    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error loading or splitting data: {e}")
        return

    # --- Create Datasets ---
    try:
        # Pass DataFrames directly to the Dataset constructor
        train_dataset = AudioCaptioningDataset(train_df, current_feature_extractor, current_tokenizer, max_length=generation_max_length)
        if len(train_dataset) == 0:
            print("Error: Training dataset is empty after initialization.")
            return

        val_dataset = None
        if val_df is not None and len(val_df) > 0:
             val_dataset = AudioCaptioningDataset(val_df, current_feature_extractor, current_tokenizer, max_length=generation_max_length)
             if len(val_dataset) == 0:
                  print("Warning: Validation dataset is empty after initialization. Disabling validation.")
                  perform_validation = False # Disable validation if set is empty
                  val_dataset = None
        elif perform_validation:
             print("Warning: Validation DataFrame was empty. Disabling validation.")
             perform_validation = False # Disable validation if val_df was empty

    except Exception as e:
        print(f"Error creating dataset(s): {e}")
        return

    # --- Create DataLoaders ---
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=custom_collate_fn, num_workers=2, pin_memory=True
    )
    print("Train DataLoader created.")

    val_dataloader = None
    if perform_validation and val_dataset:
        val_dataloader = DataLoader(
            val_dataset, batch_size=eval_batch_size, shuffle=False,
            collate_fn=custom_collate_fn, num_workers=2, pin_memory=True
        )
        print("Validation DataLoader created.")
    elif perform_validation:
         print("Validation DataLoader not created because validation dataset was empty.")
         perform_validation = False # Ensure flag is consistent

    # --- Set up device, Freeze layers, Optimizer --- (Unchanged sections)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Optional Freezing logic...
    # ... (freezing logic remains the same) ...
    encoder_frozen = False
    # Adjust attribute names based on the actual model structure (e.g., 'audio_encoder', 'encoder', 'model.encoder')
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'): # Common structure for EncoderDecoderModel
        print("Attempting to freeze model.encoder parameters...")
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        encoder_frozen = True
        frozen_params = sum(p.numel() for p in model.model.encoder.parameters() if not p.requires_grad)
        print(f"Verified {frozen_params} parameters in model.encoder are frozen.")
    elif hasattr(model, 'encoder'): # Simpler structure
        print("Attempting to freeze encoder parameters...")
        for param in model.encoder.parameters():
            param.requires_grad = False
        encoder_frozen = True
        frozen_params = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)
        print(f"Verified {frozen_params} parameters in encoder are frozen.")

    if encoder_frozen:
        print("Encoder layers frozen successfully.")
    else:
        print("Encoder layers not found or not frozen using checked attributes. Training all layers.")


    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable}")
    if num_trainable == 0:
        print("Error: No trainable parameters found.")
        return
    optimizer = AdamW(trainable_params, lr=learning_rate)

    # --- Load Metrics --- (Only if validation is performed)
    metrics_dict = {}
    if perform_validation:
        print("Loading evaluation metrics...")
        try:
             metrics_dict['bleu'] = evaluate.load('bleu')
             metrics_dict['rouge'] = evaluate.load('rouge')
             metrics_dict['meteor'] = evaluate.load('meteor')
             print(f"Loaded metrics: {list(metrics_dict.keys())}")
        except Exception as e:
             print(f"Warning: Failed to load some metrics: {e}. Proceeding without them.")

    # --- Training Loop --- (Mostly unchanged logic, checks perform_validation flag)
    print("Starting training loop...")
    all_train_losses = []
    all_val_metrics = []
    best_metric_value = -float('inf') if metric_for_best_model != 'val_loss' else float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    actual_epochs_run = 0

    generation_kwargs = {
        "max_length": generation_max_length, "num_beams": 4, "early_stopping": True,
    }
    if hasattr(model.config, "forced_decoder_ids"):
        generation_kwargs["forced_decoder_ids"] = model.config.forced_decoder_ids
    if hasattr(model.config, "suppress_tokens"):
         generation_kwargs["suppress_tokens"] = model.config.suppress_tokens

    for epoch in range(num_epochs):
        actual_epochs_run = epoch + 1
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        avg_train_loss = train_one_epoch(model, train_dataloader, device, optimizer)
        all_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

        # Validation (if enabled and possible)
        current_val_metrics = {}
        if perform_validation and val_dataloader and metrics_dict:
            current_val_metrics = run_evaluation( # Renamed function call
                model=model,
                dataloader=val_dataloader,
                tokenizer=current_tokenizer,
                device=device,
                metrics_dict=metrics_dict,
                generation_kwargs=generation_kwargs
            )
            all_val_metrics.append(current_val_metrics)

            metric_to_check = current_val_metrics.get(metric_for_best_model, None)
            if metric_to_check is None and metric_for_best_model != 'val_loss':
                 print(f"Warning: Metric '{metric_for_best_model}' not found. Skipping early stopping.")
            elif metric_to_check is not None:
                 print(f"Validation metric ({metric_for_best_model}): {metric_to_check:.4f}")
                 improved = False
                 if metric_for_best_model == 'val_loss':
                      if metric_to_check < best_metric_value: improved = True
                 else:
                      if metric_to_check > best_metric_value: improved = True

                 if improved:
                      print(f"Validation metric improved from {best_metric_value:.4f} to {metric_to_check:.4f}")
                      best_metric_value = metric_to_check
                      epochs_no_improve = 0
                      best_epoch = epoch + 1
                      best_model_save_path = os.path.join(save_path, "best_model")
                      print(f"Saving best model to {best_model_save_path}...")
                      save_model(model, best_model_save_path, current_tokenizer, current_feature_extractor, processor)
                 else:
                      epochs_no_improve += 1
                      print(f"Validation metric did not improve for {epochs_no_improve} epoch(s). Best: {best_metric_value:.4f} at Epoch {best_epoch}")

                 if epochs_no_improve >= early_stopping_patience:
                      print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                      break
        elif perform_validation:
             print("Skipping validation for this epoch (dataloader or metrics unavailable).")


    print("\n--- Training Loop Finished ---")

    # --- Writing Training Summary ---
    print("Writing final training summary report...")
    summary_save_path = os.path.join(save_path, "training_summary_auto_split.txt") # New name
    num_epochs_info = {"target": num_epochs, "actual": actual_epochs_run}
    write_training_summary(
        summary_path=summary_save_path,
        checkpoint=checkpoint,
        input_csv_path=input_csv_path,
        validation_split_ratio=validation_split_ratio if perform_validation else 0,
        num_total_files=num_total_files,
        num_train_files=num_train_files,
        num_val_files=num_val_files,
        num_epochs_run=num_epochs_info,
        best_epoch=best_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_losses=all_train_losses,
        val_metrics_history=all_val_metrics,
        best_val_metric_score=best_metric_value if best_epoch > 0 else float('nan'),
        metric_for_best_model=metric_for_best_model if perform_validation else 'N/A'
    )

    print("Finetuning process finished.")


# --- Main Execution Block (Modified for Auto-Split) ---
if __name__ == "__main__":
    # --- Configuration ---
    input_csv_path = "dataset/datasetV4.csv"  # CHANGE: Path to your SINGLE input CSV
    save_path = "ac_models/finetuned_models/ftcanversV4_auto_split" # Changed save path
    checkpoint = "circulus/canvers-audio-caption-v1"

    # Data Split Settings
    validation_split_ratio = 0.15  # Use 15% of data for validation, set to 0 or None to disable validation
    random_seed = 42               # Seed for reproducibility of the split

    # Hyperparameters
    batch_size = 4
    eval_batch_size = 8
    num_epochs = 30
    learning_rate = 2e-5

    # Evaluation & Early Stopping Settings (only used if validation_split_ratio > 0)
    early_stopping_patience = 5
    metric_for_best_model = 'rougeL_f1' # e.g., 'bleu', 'rougeL_f1', 'meteor'
    generation_max_length = 128

    # --- Run the main training process ---
    if not os.path.exists(input_csv_path):
         print(f"ERROR: Input CSV not found at {input_csv_path}. Please check the path.")
    else:
         main(
              checkpoint=checkpoint,
              input_csv_path=input_csv_path, # Pass single CSV
              validation_split_ratio=validation_split_ratio, # Pass ratio
              random_seed=random_seed, # Pass seed
              save_path=save_path,
              batch_size=batch_size,
              eval_batch_size=eval_batch_size,
              num_epochs=num_epochs,
              learning_rate=learning_rate,
              early_stopping_patience=early_stopping_patience,
              metric_for_best_model=metric_for_best_model,
              generation_max_length=generation_max_length
         )