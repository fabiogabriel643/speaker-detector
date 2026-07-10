import subprocess

subprocess.run([
    "ffmpeg",
    "-i", "testePalestra.mp4",
    "-ac", "1",           # mono
    "-ar", "16000",       # 16kHz (padrão para modelos de áudio)
    "-y",                 # sobrescreve se já existir
    "audio.wav"
])

print("audio extraído")