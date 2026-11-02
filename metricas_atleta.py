
"""
Módulo responsável pelo cálculo das métricas dos atletas.

O objetivo é calcular estatísticas somente quando existem condições
para avaliar determinada ação durante uma partida.

IMPORTANTE:
- YOLO e ByteTrack detectam e rastreiam objetos.
- Uma camada adicional precisa identificar posse de bola e eventos.
- None significa que a métrica não pôde ser avaliada.
- Zero significa que a métrica foi avaliada, mas não aconteceu.
"""

# Dataclass facilita a criação de classes utilizadas para armazenar dados.
from dataclasses import dataclass

# Iterable permite receber listas e outros tipos de coleções.
# Optional permite indicar que um valor também pode ser None.
from typing import Iterable, Optional

# Math disponibiliza operações matemáticas, como cálculo de distância.
import math


# ============================================================
# 1. CONTEXTO DA JOGADA
# ============================================================

# Essa classe armazena as condições observadas durante uma jogada.
# Ela permite verificar se existe contexto para calcular cada métrica.

@dataclass
class ContextoJogada:

    # Indica se o atleta está visível no vídeo.
    atleta_visivel: bool = True

    # Indica se a bola foi identificada no vídeo.
    bola_visivel: bool = False

    # Indica se o atleta está com a posse da bola.
    atleta_com_posse: bool = False

    # Indica se existe um adversário próximo.
    adversario_proximo: bool = False

    # Indica se existe um companheiro visível.
    companheiro_visivel: bool = False

    # Indica se um companheiro recebeu o passe.
    companheiro_recebeu_passe: bool = False

    # Indica se um chute foi identificado.
    chute_confirmado: bool = False

    # Indica se a bola entrou no gol.
    bola_entrou_no_gol: bool = False


# ============================================================
# 2. REGRAS PARA AVALIAR AS JOGADAS
# ============================================================

# Verifica se existe contexto para avaliar um drible.
# O atleta precisa ter a bola e existir um adversário próximo.

def pode_avaliar_drible(contexto: ContextoJogada) -> bool:

    return (
        contexto.atleta_visivel
        and contexto.bola_visivel
        and contexto.atleta_com_posse
        and contexto.adversario_proximo
    )


# Verifica se existe contexto para avaliar um passe.
# É necessário ter a posse da bola e um companheiro visível.

def pode_avaliar_passe(contexto: ContextoJogada) -> bool:

    return (
        contexto.atleta_visivel
        and contexto.bola_visivel
        and contexto.atleta_com_posse
        and contexto.companheiro_visivel
    )


# Verifica se existe contexto para avaliar uma finalização.
# O atleta precisa ter a posse e um chute precisa ser identificado.

def pode_avaliar_finalizacao(contexto: ContextoJogada) -> bool:

    return (
        contexto.atleta_visivel
        and contexto.bola_visivel
        and contexto.atleta_com_posse
        and contexto.chute_confirmado
    )


# Verifica se a bola entrou no gol.
# A confirmação da entrada da bola deve vir da análise do vídeo.

def pode_confirmar_gol(contexto: ContextoJogada) -> bool:

    return (
        contexto.bola_visivel
        and contexto.bola_entrou_no_gol
    )


# Verifica se existe contexto para atribuir uma assistência.
# É necessário identificar o passe para um companheiro e a conclusão em gol.

def pode_avaliar_assistencia(contexto: ContextoJogada) -> bool:

    return (
        contexto.atleta_visivel
        and contexto.bola_visivel
        and contexto.atleta_com_posse
        and contexto.companheiro_recebeu_passe
        and contexto.bola_entrou_no_gol
    )


# ============================================================
# 3. ESTATÍSTICAS DE UMA PARTIDA
# ============================================================

# Essa classe representa os dados de um atleta em uma partida.
# Os campos podem ser preenchidos durante ou depois da análise do vídeo.

@dataclass
class EstatisticasPartida:

    # Identificação do atleta.
    jogador_id: str

    # Temporada em que a partida aconteceu.
    temporada: str

    # Identificação da partida.
    jogo_id: str

    # Quantidade de minutos que o atleta participou da partida.
    minutos_jogados: float = 0.0

    # Estatísticas que podem não ter sido avaliadas.
    gols: Optional[int] = None
    assistencias: Optional[int] = None
    finalizacoes: Optional[int] = None
    dribles: Optional[int] = None

    # Quantidade de passes completos e tentados.
    passes_completos: Optional[int] = None
    passes_tentados: Optional[int] = None

    # Métricas físicas do atleta.
    distancia_m: Optional[float] = None
    velocidade_media_kmh: Optional[float] = None
    velocidade_maxima_kmh: Optional[float] = None

    # Nota atribuída ao desempenho na partida.
    nota: Optional[float] = None

    # Método responsável por validar os dados da partida.
    def validar(self):

        # Lista os campos que não podem receber valores negativos.
        campos = (
            "minutos_jogados",
            "gols",
            "assistencias",
            "finalizacoes",
            "dribles",
            "passes_completos",
            "passes_tentados",
            "distancia_m",
            "velocidade_media_kmh",
            "velocidade_maxima_kmh",
        )

        # Percorre cada campo e verifica se o valor é válido.
        for campo in campos:

            valor = getattr(self, campo)

            # None é permitido porque significa que não houve avaliação.
            if valor is not None and valor < 0:
                raise ValueError(f"{campo} não pode ser negativo.")

        # Verifica se a quantidade de passes completos é válida.
        if (
            self.passes_completos is not None
            and self.passes_tentados is not None
            and self.passes_completos > self.passes_tentados
        ):
            raise ValueError(
                "Passes completos não pode ser maior que passes tentados."
            )

        # Verifica se a nota está entre zero e dez.
        if self.nota is not None and not 0 <= self.nota <= 10:
            raise ValueError("A nota deve estar entre 0 e 10.")


# ============================================================
# 4. CÁLCULO DE VELOCIDADE
# ============================================================

# Converte a distância percorrida em pixels para metros.
# Depois calcula a velocidade em metros por segundo e km/h.

def calcular_velocidade(
    distancia_pixels: float,
    intervalo_segundos: float,
    metros_por_pixel: float
) -> dict:

    # Impede que sejam utilizados valores inválidos.
    if distancia_pixels < 0:
        raise ValueError("A distância em pixels não pode ser negativa.")

    if intervalo_segundos <= 0 or metros_por_pixel <= 0:
        raise ValueError("Tempo e escala devem ser maiores que zero.")

    # Converte a distância em pixels para metros.
    distancia_m = distancia_pixels * metros_por_pixel

    # Calcula a velocidade em metros por segundo.
    velocidade_ms = distancia_m / intervalo_segundos

    # Converte metros por segundo para quilômetros por hora.
    velocidade_kmh = velocidade_ms * 3.6

    # Retorna os resultados em um dicionário.
    return {
        "distancia_m": distancia_m,
        "velocidade_ms": velocidade_ms,
        "velocidade_kmh": velocidade_kmh
    }


# Calcula a velocidade utilizando as posições identificadas pelo rastreador.
# Cada posição contém: número do frame, coordenada X e coordenada Y.

def calcular_velocidades_rastreamento(
    posicoes: list[tuple[int, float, float]],
    fps: float,
    metros_por_pixel: float
) -> list[dict]:

    # Verifica se os parâmetros são válidos.
    if fps <= 0 or metros_por_pixel <= 0:
        raise ValueError("FPS e escala devem ser maiores que zero.")

    # Lista que armazenará as velocidades calculadas.
    velocidades = []

    # Percorre duas posições consecutivas do atleta.
    for anterior, atual in zip(posicoes, posicoes[1:]):

        # Extrai o número do frame e as coordenadas da posição anterior.
        frame_1, x1, y1 = anterior

        # Extrai o número do frame e as coordenadas da posição atual.
        frame_2, x2, y2 = atual

        # Calcula quantos frames existem entre as duas posições.
        diferenca_frames = frame_2 - frame_1

        # Ignora posições repetidas ou fora de ordem.
        if diferenca_frames <= 0:
            continue

        # Calcula a distância entre as duas posições em pixels.
        distancia_pixels = math.hypot(x2 - x1, y2 - y1)

        # Converte a diferença de frames em segundos.
        intervalo = diferenca_frames / fps

        # Calcula a velocidade e adiciona o resultado à lista.
        velocidades.append({
            "frame_inicial": frame_1,
            "frame_final": frame_2,
            **calcular_velocidade(
                distancia_pixels,
                intervalo,
                metros_por_pixel
            )
        })

    return velocidades


# ============================================================
# 5. REGISTRO DE EVENTOS
# ============================================================

# Registra gols, dribles, passes, assistências e finalizações.
# O evento só é contabilizado quando:
# 1. Existe contexto para avaliar a ação.
# 2. O evento foi confirmado pela análise do vídeo.

def registrar_evento(
    estatisticas: EstatisticasPartida,
    evento: str,
    contexto: ContextoJogada,
    confirmado: bool
) -> bool:

    # Relaciona cada evento à sua regra de avaliação e ao campo correspondente.
    regras = {

        "drible": (
            pode_avaliar_drible,
            "dribles"
        ),

        "passe": (
            pode_avaliar_passe,
            "passes_tentados"
        ),

        "finalizacao": (
            pode_avaliar_finalizacao,
            "finalizacoes"
        ),

        "gol": (
            pode_confirmar_gol,
            "gols"
        ),

        "assistencia": (
            pode_avaliar_assistencia,
            "assistencias"
        )
    }

    # Verifica se o evento solicitado existe.
    if evento not in regras:
        raise ValueError(f"Evento desconhecido: {evento}")

    # Obtém a regra e o nome do atributo que será atualizado.
    regra, campo = regras[evento]

    # Se não houver contexto ou confirmação, não registra o evento.
    if not regra(contexto) or not confirmado:
        return False

    # Obtém o valor atual da estatística.
    valor_atual = getattr(estatisticas, campo)

    # Se ainda não foi avaliada, começa a contagem em zero.
    if valor_atual is None:
        valor_atual = 0

    # Incrementa a quantidade do evento.
    setattr(estatisticas, campo, valor_atual + 1)

    # Informa que o evento foi registrado.
    return True


# Atualiza a quantidade de passes tentados e completos.
# Deve ser chamada depois que um passe elegível for analisado.

def registrar_resultado_passe(
    estatisticas: EstatisticasPartida,
    completo: bool
):

    # Inicializa os contadores caso ainda não tenham sido avaliados.
    if estatisticas.passes_tentados is None:
        estatisticas.passes_tentados = 0

    if estatisticas.passes_completos is None:
        estatisticas.passes_completos = 0

    # Registra mais uma tentativa de passe.
    estatisticas.passes_tentados += 1

    # Se o companheiro recebeu a bola, registra o passe como completo.
    if completo:
        estatisticas.passes_completos += 1


# ============================================================
# 6. PRECISÃO DE PASSES
# ============================================================

# Calcula a porcentagem de passes que chegaram corretamente ao companheiro.

def calcular_precisao_passes(
    passes_completos: Optional[int],
    passes_tentados: Optional[int]
) -> Optional[float]:

    # Se os passes não foram avaliados, não existe resultado.
    if passes_completos is None or passes_tentados is None:
        return None

    # Verifica se os valores são válidos.
    if passes_completos < 0 or passes_tentados < 0:
        raise ValueError("A quantidade de passes não pode ser negativa.")

    if passes_completos > passes_tentados:
        raise ValueError("Passes completos não pode superar tentativas.")

    # Se não houve tentativas, a precisão é zero.
    if passes_tentados == 0:
        return 0.0

    # Calcula a porcentagem de passes completos.
    return (passes_completos / passes_tentados) * 100


# ============================================================
# 7. RESISTÊNCIA ESTIMADA
# ============================================================

# Compara a velocidade média do início com a velocidade média do final.
# O resultado indica a manutenção da velocidade em porcentagem.
# Não representa uma medição fisiológica de resistência.

def calcular_resistencia_estimada(
    velocidades_inicio_kmh: Iterable[float],
    velocidades_fim_kmh: Iterable[float]
) -> Optional[float]:

    # Transforma os valores recebidos em listas.
    inicio = list(velocidades_inicio_kmh)
    fim = list(velocidades_fim_kmh)

    # Sem dados nos dois períodos, não é possível calcular.
    if not inicio or not fim:
        return None

    # Impede velocidades negativas.
    if any(v < 0 for v in inicio + fim):
        raise ValueError("As velocidades não podem ser negativas.")

    # Calcula a velocidade média no início da partida.
    media_inicio = sum(inicio) / len(inicio)

    # Calcula a velocidade média no final da partida.
    media_fim = sum(fim) / len(fim)

    # Evita divisão por zero.
    if media_inicio == 0:
        return None

    # Calcula a porcentagem de manutenção da velocidade.
    return (media_fim / media_inicio) * 100


# ============================================================
# 8. NOTA MÉDIA
# ============================================================

# Calcula a média das notas das partidas.
# Partidas sem avaliação são ignoradas.

def calcular_nota_media(
    notas: Iterable[Optional[float]]
) -> Optional[float]:

    # Mantém somente as notas que foram preenchidas.
    validas = [
        nota for nota in notas
        if nota is not None
    ]

    # Sem notas válidas, retorna None.
    if not validas:
        return None

    # Verifica se todas as notas estão entre zero e dez.
    if any(not 0 <= nota <= 10 for nota in validas):
        raise ValueError("As notas devem estar entre 0 e 10.")

    # Retorna a média das notas.
    return sum(validas) / len(validas)


# ============================================================
# 9. SOMA DE MÉTRICAS AVALIADAS
# ============================================================

# Soma os valores de uma métrica em várias partidas.
# Ignora os valores None, pois indicam que não houve avaliação.

def somar_valores_avaliados(
    valores: list[Optional[int]]
) -> Optional[int]:

    # Mantém apenas os valores que foram avaliados.
    validos = [
        valor for valor in valores
        if valor is not None
    ]

    # Se nenhum valor foi avaliado, retorna None.
    if not validos:
        return None

    # Soma os valores válidos.
    return sum(validos)


# ============================================================
# 10. ESTATÍSTICAS POR TEMPORADA
# ============================================================

# Consolida as estatísticas de todas as partidas de um atleta.
# Filtra os dados pelo ID do atleta e pela temporada desejada.

def calcular_estatisticas_temporada(
    partidas: list[EstatisticasPartida],
    jogador_id: str,
    temporada: str
) -> dict:

    # Seleciona somente as partidas do atleta na temporada informada.
    selecionadas = [
        p for p in partidas
        if p.jogador_id == jogador_id
        and p.temporada == temporada
    ]

    # Se não houver partidas, retorna estatísticas vazias.
    if not selecionadas:

        return {
            "jogador_id": jogador_id,
            "temporada": temporada,
            "jogos": 0,
            "minutos_jogados": 0,
            "gols": None,
            "assistencias": None,
            "finalizacoes": None,
            "dribles": None,
            "passes_completos": None,
            "passes_tentados": None,
            "precisao_passes_percentual": None,
            "distancia_total_km": None,
            "velocidade_media_kmh": None,
            "velocidade_maxima_kmh": None,
            "nota_media": None
        }

    # Valida os dados de cada partida selecionada.
    for partida in selecionadas:
        partida.validar()

    # Soma os passes completos que foram avaliados.
    passes_ok = somar_valores_avaliados(
        [p.passes_completos for p in selecionadas]
    )

    # Soma os passes tentados que foram avaliados.
    passes_total = somar_valores_avaliados(
        [p.passes_tentados for p in selecionadas]
    )

    # Seleciona as partidas que possuem velocidade média calculada.
    partidas_com_velocidade = [
        p for p in selecionadas
        if p.velocidade_media_kmh is not None
        and p.minutos_jogados > 0
    ]

    # Soma os minutos das partidas com dados de velocidade.
    minutos_com_velocidade = sum(
        p.minutos_jogados
        for p in partidas_com_velocidade
    )

    # Inicialmente, não existe velocidade média calculada.
    velocidade_media = None

    # Se existem dados, calcula uma média ponderada pelos minutos.
    if minutos_com_velocidade:

        velocidade_media = sum(
            p.velocidade_media_kmh * p.minutos_jogados
            for p in partidas_com_velocidade
        ) / minutos_com_velocidade

    # Separa as distâncias que foram calculadas.
    distancias = [
        p.distancia_m
        for p in selecionadas
        if p.distancia_m is not None
    ]

    # Separa as velocidades máximas que foram calculadas.
    velocidades_maximas = [
        p.velocidade_maxima_kmh
        for p in selecionadas
        if p.velocidade_maxima_kmh is not None
    ]

    # Separa as notas disponíveis.
    notas = [
        p.nota
        for p in selecionadas
        if p.nota is not None
    ]

    # Monta o resumo final da temporada.
    return {

        # Identificação do atleta e temporada.
        "jogador_id": jogador_id,
        "temporada": temporada,

        # Conta partidas em que o atleta jogou pelo menos um minuto.
        "jogos": sum(
            1 for p in selecionadas
            if p.minutos_jogados > 0
        ),

        # Soma os minutos jogados.
        "minutos_jogados": round(
            sum(p.minutos_jogados for p in selecionadas), 2
        ),

        # Soma os gols avaliados.
        "gols": somar_valores_avaliados(
            [p.gols for p in selecionadas]
        ),

        # Soma as assistências avaliadas.
        "assistencias": somar_valores_avaliados(
            [p.assistencias for p in selecionadas]
        ),

        # Soma as finalizações avaliadas.
        "finalizacoes": somar_valores_avaliados(
            [p.finalizacoes for p in selecionadas]
        ),

        # Soma os dribles avaliados.
        "dribles": somar_valores_avaliados(
            [p.dribles for p in selecionadas]
        ),

        # Informa os passes completos e tentados.
        "passes_completos": passes_ok,
        "passes_tentados": passes_total,

        # Calcula a precisão dos passes.
        "precisao_passes_percentual": calcular_precisao_passes(
            passes_ok,
            passes_total
        ),

        # Converte a distância total de metros para quilômetros.
        "distancia_total_km": (
            round(sum(distancias) / 1000, 2)
            if distancias else None
        ),

        # Retorna a velocidade média da temporada.
        "velocidade_media_kmh": (
            round(velocidade_media, 2)
            if velocidade_media is not None else None
        ),

        # Retorna a maior velocidade registrada.
        "velocidade_maxima_kmh": (
            round(max(velocidades_maximas), 2)
            if velocidades_maximas else None
        ),

        # Calcula a nota média das partidas.
        "nota_media": calcular_nota_media(notas)
    }


# ============================================================
# 11. TESTE DO MÓDULO
# ============================================================

# Esse bloco só é executado quando o arquivo é executado diretamente.
# Ele serve para testar as regras sem precisar processar um vídeo.

if __name__ == "__main__":

    # Cria um contexto em que o atleta está sozinho com a bola.
    contexto_isolado = ContextoJogada(

        atleta_visivel=True,
        bola_visivel=True,
        atleta_com_posse=True,

        # Não existe adversário próximo.
        adversario_proximo=False,

        # Não existe companheiro visível.
        companheiro_visivel=False
    )

    # Cria o registro de uma partida fictícia.
    partida = EstatisticasPartida(

        jogador_id="ATLETA_01",
        temporada="2026",
        jogo_id="JOGO_001",

        minutos_jogados=15
    )

    # Tenta registrar um drible.
    # Não será registrado porque não há adversário próximo.
    drible_registrado = registrar_evento(
        partida,
        "drible",
        contexto_isolado,
        confirmado=True
    )

    # Tenta registrar um passe.
    # Não será registrado porque não há companheiro visível.
    passe_registrado = registrar_evento(
        partida,
        "passe",
        contexto_isolado,
        confirmado=True
    )

    # Exibe o resultado dos testes.
    print("Drible registrado?", drible_registrado)
    print("Passe registrado?", passe_registrado)

    # Calcula e exibe as estatísticas da temporada.
    resumo = calcular_estatisticas_temporada(
        [partida],
        "ATLETA_01",
        "2026"
    )

    print("\nEstatísticas da temporada:")

    for indicador, valor in resumo.items():
        print(f"{indicador}: {valor}")