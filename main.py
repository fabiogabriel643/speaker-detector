import torch
import soundfile as sf
from pyannote.audio import Pipeline
from ultralytics import YOLO
import cv2
from collections import Counter
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import glob

HF_TOKEN = "HFAKmGMz8Y0x2ammb9lToV8Bb1ggGNi"

# ── diarização ─────────────────────────────────────
print("Carregando modelo de audio")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

waveform, sample_rate = sf.read("audio.wav", dtype="float32", always_2d=True)
waveform = torch.tensor(waveform.T)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

segmentos = []
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    segmentos.append({
        "inicio": turn.start,
        "fim": turn.end,
        "speaker": speaker
    })

def get_speaker(tempo):
    for seg in segmentos:
        if seg["inicio"] <= tempo <= seg["fim"]:
            return seg["speaker"]
    return None


print("Processando vídeo...")
model = YOLO("model.pt")
video = cv2.VideoCapture("testePalestra.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frame_num = 0

# face_id → lista de speakers vistos juntos
associacoes = {}

faces_salvas = set()


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

        speaker_ativo = get_speaker(tempo)

        if face_ids and speaker_ativo:
            for fid in face_ids:
                if fid not in associacoes:
                    associacoes[fid] = []
                associacoes[fid].append(speaker_ativo)
                if fid not in faces_salvas:
                    faces_salvas.add(fid)
                    #"""cv2.imwrite(f"face_{fid}_{speaker_ativo}.jpg",frame)"""
                    box = results[0].boxes[face_ids.index(fid)].xyxy[0].int().tolist()
                    x1, y1, x2, y2 = box
                    rosto = frame[y1:y2, x1:x2]
                    if rosto.size > 0:
                        cv2.imwrite(f"face_{fid}_{speaker_ativo}.jpg", rosto)
                    
    frame_num += 1

video.release()

"""#
print("Agrupamento de rosto")
imagens = glob.glob("face_*.jpg")
grupos = {}
pessoa_num = 0

for img in imagens:
    fid = int(img.split("_")[1])
    encontra_grupo = False
    
    for pessoa, membros in grupos.items():
        imagem_referencia = f"face_{membros[0]}_*.jpg"
        referencia = glob.glob(imagem_referencia)[0]
        
        try:
            resultado = DeepFace.verify(img, referencia, enforce_detection=False)
            if resultado["verificado"]:
                grupos[pessoa].append(fid)
                encontra_grupo = True
                break
        except:
            pass
    if not encontra_grupo:
        grupos[f"Pessoa_{pessoa_num}"] = [fid]
        pessoa_num += 1
        
    
"""
print("Carregando reconhecimento facial")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

imagens = glob.glob("face_*.jpg")

embeddings = {}
for img_path in imagens:
    fid = int(img_path.split("_")[1])
    img = cv2.imread(img_path)
    faces = app.get(img)
    if faces:
        embeddings[fid] = faces[0].embedding
        print(f"Embeddings gerados: {len(embeddings)} de {len(imagens)} imagens")

'tentativa para assimilação'
grupos = {}
pessoa_num = 0

for fid, emb in embeddings.items():
    encontrou = False
    for pessoa, dados in grupos.items():
        similaridade = np.dot(emb, dados["emb"]) / (np.linalg.norm(emb) * np.linalg.norm(dados["emb"]))
        if similaridade > 0.8:
            grupos[pessoa]["ids"].append(fid)
            encontrou = True
            break
    if not encontrou:
        grupos[f"Pessoa_{pessoa_num}"] = {"emb": emb, "ids": [fid]}
        pessoa_num += 1

print("\nGRUPOS DE PESSOAS:")
for pessoa, dados in grupos.items():
    print(f"{pessoa} → faces {dados['ids']}")


# ── resultado final ────────────────────────────────
print("\n" + "=" * 45)
print("ASSOCIAÇÃO FACE → SPEAKER")
print("=" * 45)

face_para_speaker = {}
for fid, speakers in associacoes.items():
    speaker_mais_comum = Counter(speakers).most_common(1)[0][0]
    face_para_speaker[fid] = speaker_mais_comum
    print(f"Face {fid:>3} → {speaker_mais_comum}")

# ── salva em arquivo ───────────────────────────────
with open("resultado2.txt", "w", encoding="utf-8") as f:
    f.write("RESULTADO DA ANÁLISE\n")
    f.write("=" * 45 + "\n\n")
    for seg in segmentos:
        inicio = f"{int(seg['inicio']//60):02d}:{int(seg['inicio']%60):02d}"
        fim = f"{int(seg['fim']//60):02d}:{int(seg['fim']%60):02d}"
        f.write(f"{inicio} → {fim}  :  {seg['speaker']}\n")

print("\nArquivo 'resultado2.txt' salvo!")