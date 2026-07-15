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
from sklearn.cluster import AgglomerativeClustering

HF_TOKEN = "HFAKmGMz8Y0x2ammb9lToV8Bb1ggGNi"

# ── 1. DIARIZAÇÃO DE ÁUDIO ─────────────────────────────────────
print("Carregando modelo de áudio...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

waveform, sample_rate = sf.read("audio.wav", dtype="float32", always_2d=True)
waveform = torch.tensor(waveform.T)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=3)

segmentos = []
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    segmentos.append({
        "inicio": turn.start,
        "fim": turn.end,
        "speaker": speaker
    })

def get_speaker(tempo, tolerancia=0.15):
    speakers_encontrados = []
    for seg in segmentos:
        if (seg["inicio"] - tolerancia) <= tempo <= (seg["fim"] + tolerancia):
            speakers_encontrados.append(seg["speaker"])
    if not speakers_encontrados:
        return None
    return Counter(speakers_encontrados).most_common(1)[0][0]


# ── 2. PROCESSAMENTO DE VÍDEO E DETECÇÃO DE FACES
print("Processando vídeo...")
model = YOLO("model.pt") #
video = cv2.VideoCapture("testePalestra.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
fps_arredondado = round(fps)
frame_num = 0

associacoes = {}
faces_salvas = set()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    if frame_num % fps_arredondado == 0:
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
                    box = results[0].boxes[face_ids.index(fid)].xyxy[0].int().tolist()
                    x1, y1, x2, y2 = box
                    rosto = frame[y1:y2, x1:x2]
                    
                    if rosto.size > 0:
                        # Salva em 224x224 para manter a qualidade ideal para o InsightFace
                        rosto_grande = cv2.resize(rosto, (224, 224))
                        cv2.imwrite(f"face_{fid}_{speaker_ativo}.jpg", rosto_grande)              
    frame_num += 1

video.release()


# ── 3. RECONHECIMENTO FACIAL E EMBEDDINGS ──────────────────────
print("Carregando reconhecimento facial (InsightFace)...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(224, 224))

imagens = glob.glob("face_*.jpg")
embeddings = {}

for img_path in imagens:
    fid = int(img_path.split("_")[1])
    img = cv2.imread(img_path)
    
    # CORREÇÃO: Redimensionar para 224x224 (mesmo tamanho do salvamento), não 320x320
    img_processada = cv2.resize(img, (224, 224)) 
    
    faces = app.get(img_processada)
    if faces:
        embeddings[fid] = faces[0].embedding

print(f"Total embeddings gerados: {len(embeddings)} de {len(imagens)} imagens")


# ── 4. AGRUPAMENTO HIERÁRQUICO (SKLEARN) ───────────────────────
print("Agrupando rostos com algoritmo hierárquico...")
fid_list = list(embeddings.keys()) # CORREÇÃO: era .key()
emb_list = list(embeddings.values())

clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.55, # Ponto ideal para agrupar a mesma pessoa
    metric='cosine', 
    linkage='average' 
)
labels = clustering.fit_predict(emb_list)

grupos = {}
for i, label in enumerate(labels):
    fid = fid_list[i]
    nome_grupo = f"Pessoa_{label}"
    
    if nome_grupo not in grupos:
        grupos[nome_grupo] = {"ids": [], "emb": emb_list[i]}
    
    grupos[nome_grupo]["ids"].append(fid)

print("\nGRUPOS DE PESSOAS CONSOLIDADOS:")
for pessoa, dados in sorted(grupos.items()): # CORREÇÃO: era .items
    print(f"{pessoa} → faces {dados['ids']} (Total: {len(dados['ids'])} detecções)")


# ── 5. RESULTADO FINAL E VOTAÇÃO ───────────────────────────────
print("\n" + "=" * 50)
print("ASSOCIAÇÃO FACE → SPEAKER")
print("=" * 50)

MIN_VOTOS_CONF = 2
face_para_speaker = {}

for fid, speakers in associacoes.items():
    if not speakers:
        continue
    
    contagem = Counter(speakers)
    speaker_mais_comum, votos = contagem.most_common(1)[0]
    total_votos = sum(contagem.values())
    confianca = (votos / total_votos) * 100

    if votos >= MIN_VOTOS_CONF and confianca > 70.0:
        face_para_speaker[fid] = speaker_mais_comum
        print(f"Face {fid:>3} -> {speaker_mais_comum:<10} (Votos: {votos}/{total_votos} | {confianca:.1f}%)")
    else:
        print(f"Face {fid:>3} → INDEFINIDO (Votos dispersos: {dict(contagem)})")


# ── 6. SALVAR EM ARQUIVO ───────────────────────────────────────
with open("resultado2.txt", "w", encoding="utf-8") as f:
    f.write("RESULTADO DA ANÁLISE\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. DIARIZAÇÃO DE ÁUDIO:\n")
    for seg in segmentos:
        inicio = f"{int(seg['inicio']//60):02d}:{int(seg['inicio']%60):02d}"
        fim = f"{int(seg['fim']//60):02d}:{int(seg['fim']%60):02d}"
        f.write(f"{inicio} → {fim}  :  {seg['speaker']}\n")
    
    f.write("\n" + "=" * 50 + "\n")
    f.write("2. MAPEAMENTO FACE → SPEAKER (Filtrado por confiança):\n")
    for fid, speaker in face_para_speaker.items():
        f.write(f"Face ID {fid} → {speaker}\n")

print("\n✅ Arquivo 'resultado2.txt' salvo com sucesso!")