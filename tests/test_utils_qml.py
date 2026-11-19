"""Testes das métricas de utils_qml.py (WIS, R², RMSE, MAE)."""
import numpy as np
import pytest

import utils_qml
from utils_qml import calcular_wis, metricas


def test_autoteste_embutido_passa():
    utils_qml._testar()


def test_predicao_perfeita_tem_wis_zero():
    y = np.array([10.0, 250.0, 3000.0])
    assert calcular_wis(y, np.tile(y, (20, 1))) == pytest.approx(0.0, abs=1e-9)


def test_wis_nunca_negativo():
    rng = np.random.default_rng(42)
    for _ in range(200):
        n = int(rng.integers(3, 25))
        y = rng.uniform(0, 5000, n)
        preds = rng.uniform(0, 5000, (int(rng.integers(2, 10)), n))
        assert calcular_wis(y, preds) >= 0


def test_wis_cresce_com_o_erro():
    y = np.full(10, 100.0)
    perto = np.tile(y + 5, (20, 1))
    longe = np.tile(y + 50, (20, 1))
    assert calcular_wis(y, perto) < calcular_wis(y, longe)


def test_wis_norm_e_wis_sobre_media():
    rng = np.random.default_rng(1)
    y = rng.uniform(10, 1000, 15)
    preds = np.maximum(y + rng.normal(0, 50, (8, 15)), 0)
    m = metricas(y, np.median(preds, axis=0), preds)
    assert m["WIS_norm"] == pytest.approx(m["WIS"] / np.mean(y), rel=1e-6)


def test_sem_matriz_de_previsoes_nao_ha_wis():
    y = np.array([1.0, 2.0, 3.0])
    assert "WIS" not in metricas(y, y)
