"""
feature_engineering.py
Construção de features para o VQR com dados reais do Mosqlimate.
Atualização Fase 3: adiciona Rt, p_rt1, receptivo, transmissao.
Compatível com Python 3.13 (sem numpy/pandas em runtime — usa apenas math/json).
"""

import json
import math
from datetime import datetime
from pathlib import Path

CACHE_JSON = Path(__file__).parent.parent / "data" / "dados_dengue_df_real.json"


def _zscore_normalize(values):
    """Z-score normalization. Retorna (normalizado, media, std)."""
    n = len(values)
    mu  = sum(values) / n
    std = math.sqrt(sum((v - mu)**2 for v in values) / (n - 1)) if n > 1 else 1.0
    std = std if std > 1e-8 else 1.0
    return [(v - mu) / std for v in values], mu, std


def _sin_cos_week(se_num):
    """Codificação sazonal da SE (1–52) em seno/cosseno."""
    angle = 2 * math.pi * se_num / 52
    return math.sin(angle), math.cos(angle)


def construir_features(dados_brutos: list, n_lags: int = 4) -> dict:
    """
    Constrói matriz de features a partir dos dados brutos do Mosqlimate.

    Retorna:
        {
            "X": [[feature_vec], ...],  # uma linha por semana válida
            "y": [target, ...],          # casos_est da semana alvo
            "datas": [datetime, ...],
            "feature_names": [str, ...],
            "stats": {campo: (mu, std)}  # para desnormalizar previsões
        }

    Features (ordem fixa — deve ser preservada nos notebooks):
        0-3:  casos_est_lag1 a lag4   (z-score)
        4-5:  Rt_lag1, Rt_lag2        (z-score)
        6:    p_rt1_lag1              (já em [0,1] — não normalizar)
        7:    receptivo_lag1          (binário 0/1)
        8:    transmissao_lag1        (binário 0/1)
        9:    tempmed_lag1            (z-score)
        10:   umidmed_lag1            (z-score)
        11:   SE_sin                  (sazonal)
        12:   SE_cos                  (sazonal)
    """
    # Ordena por data
    dados = sorted(dados_brutos, key=lambda x: x["data_iniSE"])

    # Extrai séries
    datas       = [datetime.strptime(d["data_iniSE"], "%Y-%m-%d") for d in dados]
    casos_est   = [float(d.get("casos_est") or d.get("casos") or 0) for d in dados]
    rt_vals     = [float(d.get("Rt")          or 1.0)  for d in dados]
    p_rt1_vals  = [float(d.get("p_rt1")       or 0.5)  for d in dados]
    receptivo   = [int(d.get("receptivo")     or 0)    for d in dados]
    transmissao = [int(d.get("transmissao")   or 0)    for d in dados]
    tempmed     = [float(d.get("tempmed")     or 25.0) for d in dados]
    umidmed     = [float(d.get("umidmed")     or 75.0) for d in dados]
    se_nums     = [int(str(d.get("SE", 0))[-2:]) for d in dados]

    # Normalização (guarda estatísticas para desnormalizar depois)
    casos_norm, mu_casos, std_casos = _zscore_normalize(casos_est)
    rt_norm,    mu_rt,    std_rt    = _zscore_normalize(rt_vals)
    temp_norm,  mu_temp,  std_temp  = _zscore_normalize(tempmed)
    umid_norm,  mu_umid,  std_umid  = _zscore_normalize(umidmed)

    stats = {
        "casos_est": (mu_casos, std_casos),
        "Rt":        (mu_rt,    std_rt),
        "tempmed":   (mu_temp,  std_temp),
        "umidmed":   (mu_umid,  std_umid),
    }

    # Monta X e y com lag
    X, y, datas_out = [], [], []
    for i in range(n_lags, len(dados)):
        # Lags de casos_est
        lags_casos = [casos_norm[i - lag] for lag in range(1, n_lags + 1)]
        # Lags de Rt (2 lags)
        lags_rt    = [rt_norm[i - 1], rt_norm[i - 2]]
        # p_rt1, receptivo, transmissao (1 lag)
        p_rt1_1    = p_rt1_vals[i - 1]
        recep_1    = float(receptivo[i - 1])
        trans_1    = float(transmissao[i - 1])
        # Clima (1 lag)
        temp_1     = temp_norm[i - 1]
        umid_1     = umid_norm[i - 1]
        # Sazonalidade
        se_sin, se_cos = _sin_cos_week(se_nums[i])

        features = (lags_casos + lags_rt +
                    [p_rt1_1, recep_1, trans_1, temp_1, umid_1, se_sin, se_cos])
        X.append(features)
        y.append(casos_est[i])
        datas_out.append(datas[i])

    feature_names = (
        [f"casos_est_lag{k}" for k in range(1, n_lags + 1)] +
        ["Rt_lag1", "Rt_lag2", "p_rt1_lag1", "receptivo_lag1",
         "transmissao_lag1", "tempmed_lag1", "umidmed_lag1",
         "SE_sin", "SE_cos"]
    )

    return {
        "X":             X,
        "y":             y,
        "datas":         datas_out,
        "feature_names": feature_names,
        "stats":         stats,
        "casos_brutos":  casos_est,
        "datas_completas": datas,
    }


def split_treino_teste(dataset: dict,
                       data_corte_treino: str = "2023-12-31",
                       data_inicio_teste: str = "2024-01-01") -> tuple:
    """
    Split temporal cronológico estrito (sem data leakage).
    data_corte_treino: último dia incluído no treino
    data_inicio_teste: primeiro dia do conjunto de teste
    """
    corte  = datetime.strptime(data_corte_treino, "%Y-%m-%d")
    inicio = datetime.strptime(data_inicio_teste, "%Y-%m-%d")

    X_tr, y_tr, dt_tr = [], [], []
    X_te, y_te, dt_te = [], [], []

    for x_row, y_val, dt in zip(dataset["X"], dataset["y"], dataset["datas"]):
        if dt <= corte:
            X_tr.append(x_row); y_tr.append(y_val); dt_tr.append(dt)
        elif dt >= inicio:
            X_te.append(x_row); y_te.append(y_val); dt_te.append(dt)

    return (
        {"X": X_tr, "y": y_tr, "datas": dt_tr},
        {"X": X_te, "y": y_te, "datas": dt_te},
    )


def splits_validacao(dataset: dict) -> list:
    """
    Retorna 4 splits de validação retrospectiva no padrão IMDC.
    Cada split: (treino, teste) com períodos alinhados às SE.
    """
    splits_config = [
        ("2022-01-02", "2022-10-02", "2022-10-03", "2023-10-01"),  # V1
        ("2022-01-02", "2023-10-01", "2023-10-02", "2024-10-06"),  # V2
        ("2022-01-02", "2024-06-22", "2024-06-23", "2025-06-21"),  # V3
        ("2022-01-02", "2024-10-06", "2024-10-07", "2025-10-05"),  # V4
    ]
    resultado = []
    for ini_tr, fim_tr, ini_te, fim_te in splits_config:
        tr, te = split_treino_teste(dataset, fim_tr, ini_te)
        resultado.append({
            "treino": tr, "teste": te,
            "periodo_treino": f"{ini_tr} a {fim_tr}",
            "periodo_teste":  f"{ini_te} a {fim_te}",
        })
    return resultado


def normalizar_features_para_circuito(X_row: list, n_qubits: int = 4) -> list:
    """
    Mapeia features para o intervalo [0, π] exigido pelo AngleEmbedding (RY gates).
    Toma os primeiros n_qubits features normalizados por sigmoide.
    """
    def sigmoid_to_pi(v):
        return math.pi / (1 + math.exp(-v))
    return [sigmoid_to_pi(v) for v in X_row[:n_qubits]]


if __name__ == "__main__":
    with open(CACHE_JSON, encoding="utf-8") as f:
        dados_brutos = json.load(f)

    dataset = construir_features(dados_brutos, n_lags=4)

    print(f"Total de amostras: {len(dataset['X'])}")
    print(f"Número de features: {len(dataset['feature_names'])}")
    print(f"Features: {dataset['feature_names']}")
    print(f"Período: {dataset['datas'][0].date()} a {dataset['datas'][-1].date()}")

    treino, teste = split_treino_teste(dataset)
    print(f"\nTreino: {len(treino['X'])} amostras "
          f"({treino['datas'][0].date()} a {treino['datas'][-1].date()})")
    print(f"Teste:  {len(teste['X'])} amostras "
          f"({teste['datas'][0].date()} a {teste['datas'][-1].date()})")

    splits = splits_validacao(dataset)
    print(f"\nSplits de validação IMDC:")
    for i, s in enumerate(splits, 1):
        print(f"  V{i}: treino={s['periodo_treino']} | "
              f"teste={s['periodo_teste']} "
              f"({len(s['teste']['X'])} semanas)")
