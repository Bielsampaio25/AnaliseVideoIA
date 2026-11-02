# Importa a biblioteca OpenCV para manipulação de vídeos e imagens
import cv2

# Importa a classe YOLO da biblioteca Ultralytics
from ultralytics import YOLO


# Classe responsável pela detecção e rastreamento dos objetos
class Detector:

    # Construtor da classe: executado quando criamos um objeto Detector
    def __init__(self):

        # Carrega o modelo YOLO pré-treinado
        # O arquivo yolo26n.pt contém os pesos da inteligência artificial
        self.model = YOLO("yolo26n.pt")


    # Método responsável por processar o vídeo
    # caminho_video: localização do vídeo original
    # caminho_saida: localização onde será salvo o vídeo processado
    def processar_video(self, caminho_video, caminho_saida):

        # Abre o vídeo utilizando o OpenCV
        video = cv2.VideoCapture(caminho_video)

        # Verifica se o vídeo foi aberto corretamente
        # Se não foi possível abrir, interrompe a execução com um erro
        if not video.isOpened():
            raise Exception("Não foi possível abrir o vídeo.")

        # Obtém a quantidade de quadros por segundo do vídeo
        fps = video.get(cv2.CAP_PROP_FPS)

        # Obtém a largura do vídeo em pixels
        largura = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Obtém a altura do vídeo em pixels
        altura = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define 30 FPS como valor padrão caso o vídeo não informe um FPS válido
        if fps <= 0:
            fps = 30

        # Define o codec utilizado para codificar o vídeo de saída
        # mp4v é um codec utilizado para vídeos MPEG-4
        codec = cv2.VideoWriter_fourcc(*"mp4v")

        # Cria o objeto responsável por gravar o novo vídeo
        saida = cv2.VideoWriter(
            caminho_saida,   # Caminho onde o vídeo será salvo
            codec,           # Codec de codificação
            fps,              # Quadros por segundo
            (largura, altura) # Resolução do vídeo
        )

        # Cria um conjunto para armazenar os IDs únicos dos objetos rastreados
        # O set não permite valores duplicados
        objetos = set()

        # Contador que armazenará a quantidade de quadros processados
        total_quadros = 0

        # Inicia um loop para processar o vídeo quadro por quadro
        while True:

            # Lê o próximo quadro do vídeo
            # sucesso indica se a leitura ocorreu corretamente
            # frame armazena a imagem lida
            sucesso, frame = video.read()

            # Se não conseguiu ler outro quadro, chegou ao final do vídeo
            # O break encerra o loop
            if not sucesso:
                break

            # Envia o quadro para o YOLO realizar a detecção e o rastreamento
            resultados = self.model.track(
                frame,                 # Imagem que será analisada
                persist=True,          # Mantém o estado do rastreador entre os quadros
                tracker="bytetrack.yaml", # Algoritmo utilizado para rastrear os objetos
                verbose=False,         # Desativa mensagens detalhadas no terminal
                classes=[0, 32]        # Detecta apenas pessoas (0) e bolas esportivas (32)
            )

            # Obtém o resultado correspondente ao quadro analisado
            resultado = resultados[0]

            # Verifica se existem caixas de detecção no resultado
            if resultado.boxes is not None:

                # Verifica se as detecções possuem IDs de rastreamento
                if resultado.boxes.id is not None:

                    # Obtém os IDs dos objetos
                    # cpu() transfere os dados para a memória da CPU
                    # tolist() converte os valores em uma lista Python
                    ids = resultado.boxes.id.cpu().tolist()

                    # Percorre todos os IDs encontrados no quadro atual
                    for identificador in ids:

                        # Converte o ID para inteiro e adiciona ao conjunto
                        # IDs repetidos não serão armazenados novamente
                        objetos.add(int(identificador))

            # Desenha as caixas, classes e IDs sobre o quadro analisado
            frame_anotado = resultado.plot()

            # Adiciona o quadro processado ao vídeo de saída
            saida.write(frame_anotado)

            # Incrementa o contador de quadros processados
            total_quadros += 1

        # Libera o vídeo original após finalizar o processamento
        video.release()

        # Finaliza a gravação e libera o arquivo de saída
        saida.release()

        # Retorna um dicionário com as informações da análise
        return {
            # Quantidade de IDs únicos observados durante o vídeo
            "total_objetos": len(objetos),

            # Quantidade total de quadros processados
            "total_quadros": total_quadros,

            # FPS original do vídeo
            "fps": fps,

            # Caminho do vídeo processado
            "caminho_resultado": caminho_saida
        }