from .finetune import *

# python -m ac_models.finetune.whispertiny
if __name__ == "__main__":
    dataset_path = "dataset/dataset.csv"
    save_path = "ac_models/finetuned_models/ftwhispertiny"
    checkpoint = "MU-NLPC/whisper-tiny-audio-captioning"
    batch_size = 4
    num_epochs = 5

    # Run the main training process
    main(checkpoint=checkpoint, dataset_path=dataset_path, save_path=save_path, batch_size=batch_size, num_epochs=num_epochs)