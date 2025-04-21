from .predict import *

# python -m ac_models.predict.ftwhispertiny
if __name__ == "__main__":
    audio_paths = ["test1.mp3","test2.mp3","test3.mp3","test4.mp3","test5.mp3","test6.mp3","test7.mp3","test8.mp3","test9.mp3","test10.mp3", ]
    checkpoint = "circulus/canvers-audio-caption-v1"
    captions = []

    for audio_path in audio_paths:
        c = main(checkpoint=checkpoint, audio_path=audio_path)
        captions.append((audio_path, c))

    for c in captions:
        print(c)