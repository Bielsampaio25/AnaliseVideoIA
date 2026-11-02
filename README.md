# AnaliseVideoIA — versão modular

## Instalação (uma vez por ambiente)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Executar
Coloque o vídeo em `videos/` e rode na raiz do projeto:
```powershell
python metricas_atleta.py --video "videos/video_penalti_teste.mp4"
```

O primeiro uso pode baixar os pesos `yolo26n.pt`. Depois eles ficam no cache do Ultralytics.
Saídas em `resultados/`: vídeo anotado, CSV de detecções e JSON de análise.

## Importante: limites reais desta versão
- O pipeline processa vídeos em diferentes resoluções e dimensões sem coordenadas fixas de gol.
- O YOLO COCO padrão detecta pessoas e bola; não detecta automaticamente a estrutura do gol com confiabilidade.
- Portanto, gols, assistências, finalizações, dribles e passes continuam `null` até existir evidência suficiente. O módulo registra apenas candidatos de eventos, não os promove a eventos confirmados.
- A detecção da bola pode falhar por tamanho pequeno, desfoque, oclusão, compressão ou ângulo extremo.
- IDs ByteTrack são IDs temporários, não identidade real do atleta.
- Distância e velocidade reais exigem calibração geométrica do campo. Um único valor metros/px não generaliza para perspectiva variável; para produção, implementar homografia usando linhas/pontos do campo.
- "Qualquer ângulo" não pode ser garantido sem dados de treino representativos, validação e possivelmente modelos especializados/treinados. Este projeto é uma base executável e modular, não um sistema de análise profissional completo.

## Próximas etapas para elevar a precisão
1. Treinar/avaliar detector de bola e gol em dataset multiângulo.
2. Detectar linhas e pontos de referência do campo e estimar homografia por frame/câmera.
3. Usar rastreador especializado para bola rápida e reidentificação de jogadores.
4. Treinar modelo temporal de ações com clipes rotulados (passe, chute, drible, gol).
5. Validar cada evento contra ground truth e manter `null` quando a confiança for insuficiente.
