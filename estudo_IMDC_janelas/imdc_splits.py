# -*- coding: utf-8 -*-
"""
Splits de validação alinhados às janelas dos Testes 1/2/3 do 2º IMDC (DF).
Cada teste é uma temporada de 52 semanas; o treino é expansivo (todo o histórico
anterior ao início do teste, a partir de jan/2022).

Janelas (DF, extraídas de preds_2nd_sprint_update.csv do repositório oficial):
  T1: 2022-10-09 -> 2023-10-01   (temporada típica)
  T2: 2023-10-08 -> 2024-09-29   (surto recorde de 2024)
  T3: 2024-10-06 -> 2025-09-28   (pós-surto)
"""
import os, sys, json
import numpy as np

REPO = r"C:/Users/julia/OneDrive/Área de Trabalho/PIBIT/Experimentos/Relatorio_Final_IC"
AQUI = os.path.join(REPO, "estudo_IMDC_janelas")
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, REPO)
from feature_engineering import construir_features

JANELAS = {
    "T1": ("2022-10-09", "2023-10-01"),
    "T2": ("2023-10-08", "2024-09-29"),
    "T3": ("2024-10-06", "2025-09-28"),
}
DESCRICAO = {
    "T1": "IMDC Teste 1 — temporada típica  (out/2022 – out/2023)",
    "T2": "IMDC Teste 2 — surto recorde 2024  (out/2023 – set/2024)",
    "T3": "IMDC Teste 3 — pós-surto  (out/2024 – set/2025)",
}
DATA_JSON = os.path.join(AQUI, "data", "dados_dengue_df_2022_2025out.json")


def carregar_dataset(n_lags=4):
    dados = json.load(open(DATA_JSON, encoding="utf-8"))
    return construir_features(dados, n_lags=n_lags)


def montar_cenarios(dataset):
    """Retorna dict {T1,T2,T3: {X_train,y_train,X_test,y_test, datas_test}} com treino expansivo."""
    X = np.asarray(dataset["X"]); y = np.asarray(dataset["y"])
    datas = np.array([str(d)[:10] for d in dataset["datas"]])
    CEN = {}
    for nome, (ini, fim) in JANELAS.items():
        m_test = (datas >= ini) & (datas <= fim)
        m_train = datas < ini
        CEN[nome] = {
            "X_train": X[m_train], "y_train": y[m_train],
            "X_test": X[m_test],   "y_test": y[m_test],
            "datas_test": datas[m_test],
        }
    return CEN


if __name__ == "__main__":
    ds = carregar_dataset()
    CEN = montar_cenarios(ds)
    print("features:", len(ds["feature_names"]))
    for k, d in CEN.items():
        print(f"{k}: treino={len(d['y_train'])} sem | teste={len(d['y_test'])} sem "
              f"({d['datas_test'][0]} -> {d['datas_test'][-1]}) | pico_teste={int(d['y_test'].max())}")
