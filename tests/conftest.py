"""Configuração comum dos testes: põe a raiz do repositório e src/ no sys.path."""
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))


@pytest.fixture(scope="session")
def dados_brutos():
    """Série semanal agregada do DF (sem dado individual de paciente)."""
    with open(RAIZ / "data" / "dados_dengue_df_real.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def dataset(dados_brutos):
    from feature_engineering import construir_features
    return construir_features(dados_brutos, n_lags=4)
