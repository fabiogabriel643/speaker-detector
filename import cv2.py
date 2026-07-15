import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))  # ← muda o det_size

img = cv2.imread("face_1_SPEAKER_03.jpg")

# recorta só onde está o rosto (removendo o fundo preto)
img_recortado = img[30:130, 30:130]  # ajusta se necessário
img_grande = cv2.resize(img_recortado, (320, 320))

faces = app.get(img_grande)
print(f"Rostos detectados: {len(faces)}")