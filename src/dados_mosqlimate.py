"""
dados_mosqlimate.py
Integração com a API do Mosqlimate — InfoDengue DF.
Usa urllib (sem requests/pandas/numpy) — compatível com Python 3.13.
"""

import json
import os
import time
import math
import urllib.request
from datetime import datetime, date
from pathlib import Path

# Chave da API lida da variável de ambiente MOSQLIMATE_API_KEY (nunca versionar segredos).
# Ex.: export MOSQLIMATE_API_KEY="usuario:uuid"  (Linux/macOS)
#      $env:MOSQLIMATE_API_KEY="usuario:uuid"    (PowerShell)
API_KEY    = os.environ.get("MOSQLIMATE_API_KEY", "")
GEOCODE_DF = 5300108
CACHE_JSON = Path(__file__).parent.parent / "data" / "dados_dengue_df_real.json"

NIVEL_LABEL = {1: "verde", 2: "amarelo", 3: "laranja", 4: "vermelho"}


def _fetch_page(url: str, tentativas: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"X-UID-Key": API_KEY})
    for t in range(1, tentativas + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            if t == tentativas:
                raise
            time.sleep(3 * t)
    return {}


def atualizar_cache(start: str = "2022-01-01", end: str = None) -> list:
    """Baixa todos os dados do InfoDengue DF e atualiza o cache local."""
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    BASE = "https://api.mosqlimate.org/api/datastore/infodengue/"
    todos = []
    page  = 1

    while True:
        url  = f"{BASE}?disease=dengue&geocode={GEOCODE_DF}&start={start}&end={end}&page={page}&per_page=5"
        data = _fetch_page(url)
        items = data.get("items", [])
        if not items:
            break
        todos.extend(items)
        total = data["pagination"]["total_pages"]
        if page >= total:
            break
        page += 1
        time.sleep(1.5)

    todos.sort(key=lambda x: x["data_iniSE"])
    CACHE_JSON.write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")
    return todos


def carregar_dados(usar_cache: bool = True) -> list:
    """
    Retorna lista de semanas epidemiológicas do DF, ordenada por data.
    Cada item é um dict com todos os campos do HistoricoAlertaSchema.
    Usa cache local se disponível e usar_cache=True.
    """
    if usar_cache and CACHE_JSON.exists():
        with open(CACHE_JSON, encoding="utf-8") as f:
            dados = json.load(f)
        if dados:
            return dados
    return atualizar_cache()


def preparar_serie(dados: list) -> dict:
    """
    Transforma a lista bruta em estrutura pronta para o dashboard:
    dicts de listas paralelas, ordenadas cronologicamente.
    """
    datas       = []
    casos       = []
    casos_est   = []
    casos_min   = []
    casos_max   = []
    rt_vals     = []
    p_rt1_vals  = []
    nivel_api   = []
    nivel_label = []
    tempmed     = []
    umidmed     = []
    receptivo   = []
    transmissao = []
    p_inc100k   = []

    for item in dados:
        try:
            dt = datetime.strptime(item["data_iniSE"], "%Y-%m-%d")
        except Exception:
            continue
        datas.append(dt)
        casos.append(item.get("casos") or 0)
        casos_est.append(item.get("casos_est") or item.get("casos") or 0)
        casos_min.append(item.get("casos_est_min") or item.get("casos") or 0)
        casos_max.append(item.get("casos_est_max") or item.get("casos") or 0)
        rt_vals.append(item.get("Rt") or 1.0)
        p_rt1_vals.append(item.get("p_rt1") or 0.5)
        nv = item.get("nivel") or 1
        nivel_api.append(nv)
        nivel_label.append(NIVEL_LABEL.get(nv, "verde"))
        tempmed.append(item.get("tempmed") or 0.0)
        umidmed.append(item.get("umidmed") or 0.0)
        receptivo.append(item.get("receptivo") or 0)
        transmissao.append(item.get("transmissao") or 0)
        p_inc100k.append(item.get("p_inc100k") or 0.0)

    return {
        "datas":       datas,
        "casos":       casos,
        "casos_est":   casos_est,
        "casos_min":   casos_min,
        "casos_max":   casos_max,
        "rt":          rt_vals,
        "p_rt1":       p_rt1_vals,
        "nivel_api":   nivel_api,
        "nivel_label": nivel_label,
        "tempmed":     tempmed,
        "umidmed":     umidmed,
        "receptivo":   receptivo,
        "transmissao": transmissao,
        "p_inc100k":   p_inc100k,
    }


def detectar_anomalia_rt(serie: dict) -> dict:
    """
    Detector de anomalia baseado em Rt (número reprodutivo).
    Mais robusto que o Z-score — usa critérios epidemiológicos reais.
    """
    if not serie["rt"]:
        return {"detectada": False, "tipo": "sem dados", "rt_atual": 1.0, "p_rt1_atual": 0.5}

    rt_atual   = serie["rt"][-1]
    p_rt1_atu  = serie["p_rt1"][-1]
    trans_atu  = serie["transmissao"][-1]
    recep_atu  = serie["receptivo"][-1]

    # Critérios epidemiológicos do InfoDengue
    surto_ativo   = rt_atual > 1.0 and p_rt1_atu > 0.95
    surto_provav  = rt_atual > 1.0 and p_rt1_atu > 0.75
    em_declinio   = rt_atual < 1.0 and p_rt1_atu < 0.5

    if surto_ativo:
        tipo = "SURTO ATIVO"
        detectada = True
    elif surto_provav:
        tipo = "CRESCIMENTO PROVÁVEL"
        detectada = True
    elif trans_atu == 1 and recep_atu == 1:
        tipo = "TRANSMISSÃO SUSTENTADA"
        detectada = True
    else:
        tipo = "Normal"
        detectada = False

    # Tendência do Rt nas últimas 4 semanas
    rt_4 = serie["rt"][-4:]
    tend_rt = rt_4[-1] - rt_4[0] if len(rt_4) >= 2 else 0

    return {
        "detectada":   detectada,
        "tipo":        tipo,
        "rt_atual":    round(rt_atual, 4),
        "p_rt1_atual": round(p_rt1_atu, 3),
        "tend_rt":     round(tend_rt, 4),
        "transmissao": trans_atu,
        "receptivo":   recep_atu,
    }


def score_climatico_real(serie: dict) -> dict:
    """
    Score de risco climático com dados reais do InfoDengue.
    """
    if not serie["tempmed"]:
        return {"score": 0, "temp": 0, "umid": 0, "receptivo": 0}

    temp   = serie["tempmed"][-1]
    umid   = serie["umidmed"][-1]
    recep  = serie["receptivo"][-1]

    score = min(100, int(
        (max(0, temp - 20) / 12 * 40) +
        (max(0, umid - 50) / 40 * 35) +
        (recep * 25)
    ))
    return {
        "score":     score,
        "temp":      round(temp, 1),
        "umid":      round(umid, 1),
        "receptivo": recep,
    }


def comparar_com_anos_anteriores(serie: dict) -> dict:
    """
    Separa a série por ano para comparação visual.
    Retorna dict {ano: {datas: [...], casos: [...]}}
    """
    anos = {}
    for dt, c in zip(serie["datas"], serie["casos"]):
        ano = str(dt.year)
        if ano not in anos:
            anos[ano] = {"datas": [], "casos": []}
        anos[ano]["datas"].append(dt)
        anos[ano]["casos"].append(c)
    return anos


def ultima_semana(serie: dict) -> dict:
    """Retorna os dados da semana mais recente."""
    if not serie["datas"]:
        return {}
    return {
        "data":        serie["datas"][-1],
        "casos":       serie["casos"][-1],
        "casos_est":   serie["casos_est"][-1],
        "rt":          serie["rt"][-1],
        "p_rt1":       serie["p_rt1"][-1],
        "nivel_label": serie["nivel_label"][-1],
        "nivel_api":   serie["nivel_api"][-1],
        "tempmed":     serie["tempmed"][-1],
        "umidmed":     serie["umidmed"][-1],
        "transmissao": serie["transmissao"][-1],
        "receptivo":   serie["receptivo"][-1],
        "p_inc100k":   serie["p_inc100k"][-1],
    }


if __name__ == "__main__":
    print("Carregando dados do cache...")
    dados = carregar_dados()
    serie = preparar_serie(dados)
    print(f"Semanas carregadas: {len(serie['datas'])}")
    print(f"Periodo: {serie['datas'][0].date()} ate {serie['datas'][-1].date()}")
    print(f"Casos max: {max(serie['casos']):,}")
    print(f"Rt atual:  {serie['rt'][-1]:.4f}")

    ult = ultima_semana(serie)
    print(f"\nUltima semana ({ult['data'].strftime('%d/%m/%Y')}):")
    print(f"  Casos:       {ult['casos']:,}")
    print(f"  Rt:          {ult['rt']:.4f}")
    print(f"  P(Rt>1):     {ult['p_rt1']:.3f}")
    print(f"  Nivel API:   {ult['nivel_label'].upper()}")
    print(f"  Temp media:  {ult['tempmed']:.1f} C")
    print(f"  Umid media:  {ult['umidmed']:.1f} %")

    anom = detectar_anomalia_rt(serie)
    print(f"\nAnomalia: {anom['tipo']} | Detectada: {anom['detectada']}")

    clima = score_climatico_real(serie)
    print(f"Score climatico: {clima['score']}/100 (Temp={clima['temp']}C, Umid={clima['umid']}%)")
