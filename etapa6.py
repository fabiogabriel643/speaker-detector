import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-face.pt")

video = cv2.VideoCapture("testePalestra.mp4")
fps = video.get(cv2.CAP_PROP_FPS)

rastreamento = []
frame_num = 0

print("Processando vídeo...")

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

        rastreamento.append({
            "tempo": round(tempo, 1),
            "faces": face_ids
        })

        print(f"t={tempo:.0f}s → faces detectadas: {face_ids}")

    frame_num += 1

video.release()
print("\nConcluído!")