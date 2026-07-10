import torch
import soundfile as sf
from pyannote.audio import Pipeline

print("Carregando")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="HFAKmGMz8Y0x2ammb9lToV8Bb1ggGNi"
)

print("Carregando audio...")

waveform, sample_rate = sf.read("audio.wav", dtype="float32", always_2d=True)
waveform = torch.tensor(waveform.T)

audio = {
    "waveform": waveform,
    "sample_rate": sample_rate
}

print("Analisando audio...")

diarization = pipeline(audio)

print("\nResultado:")
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s → {turn.end:.1f}s : {speaker}")