import os

from detector import Detector

# Cria a pasta de resultados caso ela não exista
os.makedirs("resultados", exist_ok=True)

# Cria uma instância do detector
detector = Detector()

# Define o caminho do vídeo original
caminho_video = "videos/video_penalti_teste.mp4"

# Define o caminho do vídeo processado
caminho_saida = "resultados/video_analisado.mp4"

# Executa a detecção e o rastreamento
resultado = detector.processar_video(
    caminho_video,
    caminho_saida
)

# Exibe as informações da análise
print(resultado)