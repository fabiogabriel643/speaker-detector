import cv2

video = cv2.VideoCapture("testePalestra.mp4")

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
duracao = total_frames / fps

print(f"FPS: {fps}")
print(f"Total de frames: {total_frames}")
print(f"Duração: {duracao:.1f} segundos")

video.release()