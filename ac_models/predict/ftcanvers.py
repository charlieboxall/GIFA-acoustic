from .predict import *

# python -m ac_models.predict.ftcanvers
if __name__ == "__main__":
    audio_path = "MGlai7_m_760.wav"
    checkpoint = "ac_models/finetuned_models/ftcanvers"

    main(checkpoint=checkpoint, audio_path=audio_path)

