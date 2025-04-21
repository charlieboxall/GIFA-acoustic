from .finetune import *

# python -m ac_models.finetune.whispertiny
if __name__ == "__main__":
    dataset_path = "dataset/datasetV4.csv"
    save_path = "ac_models/finetuned_models/ftwhispertinyV4"
    checkpoint = "MU-NLPC/whisper-tiny-audio-captioning"
    batch_size = 4
    num_epochs = 10
    learning_rate = 5e-5

    # Run the main training process
    main(checkpoint=checkpoint, dataset_path=dataset_path, save_path=save_path, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)