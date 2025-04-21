from .predict import *

# python -m ac_models.predict.ftwhispertiny
if __name__ == "__main__":
    audio_path = "stomp-and-claps-logo-116097.mp3"
    checkpoint = "ac_models/finetuned_models/ftwhispertiny"

    main(checkpoint=checkpoint, audio_path=audio_path)

