from .predict import *

# python -m ac_models.predict.ftcanvers
if __name__ == "__main__":
    audio_path = "stomp-and-claps-logo-116097.mp3"
    checkpoint = "ac_models/finetuned_models/ftcanvers"

    main(checkpoint=checkpoint, audio_path=audio_path)

