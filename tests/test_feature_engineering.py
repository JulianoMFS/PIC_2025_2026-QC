"""Testes do contrato de src/feature_engineering.py.

Protegem o que os notebooks assumem: ordem fixa das 13 features, split
cronológico sem vazamento e mapeamento para o intervalo do AngleEmbedding.
"""
import math

from feature_engineering import (
    construir_features,
    normalizar_features_para_circuito,
    split_treino_teste,
    splits_validacao,
)

FEATURES_ESPERADAS = [
    "casos_est_lag1", "casos_est_lag2", "casos_est_lag3", "casos_est_lag4",
    "Rt_lag1", "Rt_lag2", "p_rt1_lag1", "receptivo_lag1",
    "transmissao_lag1", "tempmed_lag1", "umidmed_lag1", "SE_sin", "SE_cos",
]


def test_ordem_das_features_e_fixa(dataset):
    assert dataset["feature_names"] == FEATURES_ESPERADAS


def test_dimensoes_consistentes(dataset, dados_brutos):
    n = len(dados_brutos) - 4
    assert len(dataset["X"]) == len(dataset["y"]) == len(dataset["datas"]) == n
    assert all(len(linha) == 13 for linha in dataset["X"])


def test_alvo_e_casos_da_semana_e_nao_normalizado(dataset):
    # y[i] corresponde a casos_brutos[i + n_lags]
    assert dataset["y"][0] == dataset["casos_brutos"][4]


def test_lag1_aponta_para_semana_anterior(dataset):
    mu, std = dataset["stats"]["casos_est"]
    lag1_desnorm = dataset["X"][1][0] * std + mu
    assert math.isclose(lag1_desnorm, dataset["y"][0], rel_tol=1e-9)


def test_entrada_desordenada_gera_mesmo_resultado(dados_brutos):
    a = construir_features(dados_brutos)
    b = construir_features(list(reversed(dados_brutos)))
    assert a["X"] == b["X"] and a["y"] == b["y"]


def test_split_cronologico_sem_vazamento(dataset):
    treino, teste = split_treino_teste(dataset, "2023-12-31", "2024-01-01")
    assert treino["datas"] and teste["datas"]
    assert max(treino["datas"]) < min(teste["datas"])


def test_quatro_splits_imdc(dataset):
    splits = splits_validacao(dataset)
    assert len(splits) == 4
    for s in splits:
        if s["treino"]["datas"] and s["teste"]["datas"]:
            assert max(s["treino"]["datas"]) < min(s["teste"]["datas"])


def test_normalizacao_para_circuito_no_intervalo_0_pi():
    valores = normalizar_features_para_circuito([-50.0, -1.0, 0.0, 1.0, 50.0], n_qubits=4)
    assert len(valores) == 4
    assert all(0.0 <= v <= math.pi for v in valores)
    assert math.isclose(valores[2], math.pi / 2)
