import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
VIDEO = "palestra.mp4"

video = cv2.VideoCapture(VIDEO)
fps = int(video.get(cv2.CAP_PROP_FPS))

frame_num = 0
face_embeddings = []
face_imagens = []
face_score = []
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))


if not video.isOpened():
    print("Erro ao abrir video, verifique a grafia e caminho")
    exit()

os.makedirs("rostos salvos", exist_ok=True)

print("Pressione 'q' para sair")

while True:
    ret, frame = video.read()
    if not ret:
        break
    if frame_num % fps == 0:
        print(f"processando frame {frame_num}")
        faces = app.get(frame)
        for face in faces:
            if face.det_score < 0.80:
                continue
            
            face_embeddings.append(face.embedding)
            face_score.append(face.det_score)
            
            bbox = face.bbox.astype(int)
            
            padding = 0.30  # 20% maior
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            x1 = max(0, bbox[0] - pad_x)
            y1 = max(0, bbox[1] - pad_y)
            x2 = min(frame.shape[1], bbox[2] + pad_x)
            y2 = min(frame.shape[0], bbox[3] + pad_y)
            croped_face = frame[y1:y2, x1:x2].copy()
                
                # VERIFICAR SE A IMAGEM NÃO ESTÁ VAZIA
            if croped_face.size > 0:
                face_imagens.append(croped_face)
            
                idx = len(face_embeddings) - 1
                cv2.imwrite(f"rostos salvos/rostos_{idx}_score_{face.det_score:.2f}.jpg", croped_face)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Score: {face.det_score:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Deteccao facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_num += 1
video.release()
cv2.destroyAllWindows()
print(f"processamento de video finalizado! {len(face_embeddings)} total de extraido")
  
if len(face_embeddings) > 0:
    clustering = DBSCAN(
    metric='cosine',
    eps=0.5,
    min_samples=2
) 
    labels = clustering.fit_predict(face_embeddings)
    unique, counts = np.unique(labels, return_counts=True)
    print("resultado final")
    os.makedirs("rostos", exist_ok=True)
    pessoas_reais = [p for p in unique if p != -1]
    for person_id, count in zip(unique, counts):
        print(f"Pessoa {person_id}: {count} aparições")
        if person_id == -1:
            continue
        best_index = -1
        max_score = -1.0
        for i in range(len(labels)):
            if labels[i] == person_id:
                if face_score[i] > max_score:
                    max_score = face_score[i]
                    best_index = i
        reference_photo = face_imagens[best_index]
        filename = f"rostos/pessoa_{person_id}.jpg"
        cv2.imwrite(filename, reference_photo)
        print(f"Foto de referência salva em: {filename}")
        
else:
    print(f"\n detectamos apenas {len(face_embeddings)} rostos, menos que o numero de palestrantes ({NUM_SPEAKERS}).")