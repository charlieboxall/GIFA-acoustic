finetuned whisper safetensors (144mb) too big to upload
dataset on local device for testing, also uploaded to huggingface @ https://huggingface.co/datasets/boxallcharlie/acoustic-music-scenes | will adjust later on to use huggingface


To train model:
    python -m ac_models.finetune.whispertiny
    OR
    python -m ac_models.finetune.canvers

To predict captions:
    python -m ac_models.predict.ftwhispertiny
    OR
    python -m ac_models.predict.ftcanvers

