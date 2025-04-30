from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("boxallcharlie/whisper-tiny-AAC-acoustic-music-finetune")
model = AutoModelForSpeechSeq2Seq.from_pretrained("boxallcharlie/whisper-tiny-AAC-acoustic-music-finetune")