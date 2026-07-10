import torch
import soundfile as sf
from pyannote.audio import Pipeline
from ultralytics import YOLO
import cv2 

HF_TOKEN = ""
VIDEO = "conferencia_curta.mp4"
AUDIO = "audio.wav"


video = cv2.VideoCapture(VIDEO)
fps = video.get(cv2.CAP_PROP_FPS)
model = YOLO("yolov8n.pt")
'''extracao de dados'''
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

waveform, sample_rate = sf.read(AUDIO, dtype="float32",always_2d=True)
waveform = torch.tensor(waveform.T)
audio = {"waveform": waveform, "sample_rate": sample_rate}

'''quem falou e quando'''

diarizacao = pipeline(audio) 


frame_num = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    if frame_num % int(fps) == 0:
        tempo = frame_num / fps
        faceId = model.track(frame, persist=True, verbose=False)
        face_ids = []
        if faceId[0].boxes.id is not None:
            face_ids = faceId[0].boxes.id.int().tolist()
    frame_num += 1
    