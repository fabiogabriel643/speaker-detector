import torch
import soundfile as sf
from pyannote.audio import Pipeline
from ultralytics import YOLO
import cv2

# ── audio ──────────────────────────────────────────
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="HFAKmGMz8Y0x2ammb9lToV8Bb1ggGNi"
)

waveform, sample_rate = sf.read("audio.wav", dtype="float32", always_2d=True)
waveform = torch.tensor(waveform.T)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# transforma em lista de {inicio, fim, speaker}
segmentos = []
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    segmentos.append({
        "inicio": turn.start,
        "fim": turn.end,
        "speaker": speaker
    })

# ── video ──────────────────────────────────────────
model = YOLO("yolov8n-face.pt")
video = cv2.VideoCapture("testePalestra.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frame_num = 0

print("\nResultado final:")
print("-" * 45)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    if frame_num % int(fps) == 0:
        tempo = frame_num / fps
        results = model.track(frame, persist=True, verbose=False)

        face_ids = []
        if results[0].boxes.id is not None:
            face_ids = results[0].boxes.id.int().tolist()

        # descobre quem está falando nesse momento
        speaker_ativo = None
        for seg in segmentos:
            if seg["inicio"] <= tempo <= seg["fim"]:
                speaker_ativo = seg["speaker"]
                break

        if face_ids and speaker_ativo:
            print(f"t={tempo:.0f}s → faces {face_ids} → {speaker_ativo}")

    frame_num += 1

video.release()