from .finetune import *

# python -m ac_models.finetune.canvers
if __name__ == "__main__":
    dataset_path = "dataset/dataset.csv"
    save_path = "ac_models/finetuned_models/ftcanvers"
    checkpoint = "circulus/canvers-audio-caption-v1"
    batch_size = 8
    num_epochs = 3

    # Run the main training process
    main(checkpoint=checkpoint, dataset_path=dataset_path, save_path=save_path, batch_size=batch_size, num_epochs=num_epochs)