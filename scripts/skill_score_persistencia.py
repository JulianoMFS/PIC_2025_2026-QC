# -*- coding: utf-8 -*-
"""
skill_score_persistencia.py

Reproduz o Quadro da Seção 4.6 do relatório: compara o WIS de todos os modelos
com o de um baseline de persistência ingênua, sob intervalos PADRONIZADOS
(block bootstrap dos resíduos one-step, mesmo B e mesmo bloco para todos),
e calcula o skill score = WIS_modelo / WIS_persistência (<1 = supera o baseline).

Entrada: os sidecars *_plotdata.npz (predição mediana + y observado por cenário),
já versionados no repositório. Não depende de re-treinar nenhum modelo.

Uso:  python scripts/skill_score_persistencia.py
"""
import os
import numpy as np

# Raiz do repositório (este script está em scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT)
import utils_qml as U  # calcular_wis

MODELOS = {
    "RF":         "notebooks/fase0a_RF_pred_vs_obs_plotdata.npz",
    "XGBoost":    "notebooks/fase0b_xgb_pred_vs_obs_plotdata.npz",
    "LSTM":       "notebooks/fase4c_lstm_pred_vs_obs_plotdata.npz",
    "SARIMA":     "ensembles/sarima_pred_vs_obs_plotdata.npz",
    "Ens RF+XGB": "ensembles/ensemble_rfxgb_pred_vs_obs_plotdata.npz",
    "Ens Regime": "ensembles/ensemble_regime_pred_vs_obs_plotdata.npz",
    "QRC":        "notebooks/fase4a_qrc_pred_vs_obs_plotdata.npz",
    "VQR-Angle":  "notebooks/fase1_vqr-angleembedding-pred_vs_obs_plotdata.npz",
    "VQR-ReUp":   "notebooks/fase2_vqr-pred_vs_obs_plotdata.npz",
    "QSVM":       "notebooks/fase3_qsvm_pred_vs_obs_plotdata.npz",
    "QLSTM":      "notebooks/fase4b_qlstm_pred_vs_obs_plotdata.npz",
}
CENARIOS = ["C1", "C2", "C3"]
B = 1000       # nº de amostras (igual para todos)
BLOCO = 8      # comprimento do bloco (preserva dependência temporal)
rng = np.random.default_rng(42)


def block_bootstrap_resid(resid, n, B, L):
    """Gera matriz (B, n) de resíduos reamostrados em blocos."""
    m = len(resid)
    nblocks = int(np.ceil(n / L))
    out = np.zeros((B, n))
    for b in range(B):
        starts = rng.integers(0, max(m - L, 1), size=nblocks)
        serie = np.concatenate([resid[s:s + L] for s in starts])[:n]
        if len(serie) < n:
            serie = np.resize(serie, n)
        out[b] = serie
    return out


def wis_padronizado(y, med):
    """WIS com intervalos reconstruídos por block bootstrap dos resíduos."""
    resid = y - med
    boot = block_bootstrap_resid(resid, len(y), B, BLOCO)
    samples = np.maximum(med[None, :] + boot, 0.0)
    return U.calcular_wis(y, samples)


def persistencia(y):
    med = np.concatenate([[y[0]], y[:-1]])  # ŷ_t = y_{t-1}
    return wis_padronizado(y, med)


def main():
    persist = {}
    z0 = np.load(os.path.join(ROOT, MODELOS["RF"]))
    for cen in CENARIOS:
        persist[cen] = persistencia(z0[f"{cen}_ytest"])

    resultados = {}
    for nome, arq in MODELOS.items():
        z = np.load(os.path.join(ROOT, arq))
        resultados[nome] = {cen: wis_padronizado(z[f"{cen}_ytest"], z[f"{cen}_mediana"])
                            for cen in CENARIOS}

    print(f"Intervalos padronizados: block bootstrap L={BLOCO}, B={B}\n")
    print(f"{'Modelo':14}{'WIS C1':>9}{'WIS C2':>9}{'WIS C3':>9}   skill (WIS/persist)")
    print("-" * 78)
    print(f"{'PERSISTÊNCIA':14}{persist['C1']:>9.0f}{persist['C2']:>9.0f}{persist['C3']:>9.0f}   (referência)")
    print("-" * 78)
    for nome in sorted(resultados, key=lambda m: resultados[m]["C2"]):
        r = resultados[nome]
        sk = [r[c] / persist[c] for c in CENARIOS]
        print(f"{nome:14}{r['C1']:>9.0f}{r['C2']:>9.0f}{r['C3']:>9.0f}   "
              f"{sk[0]:.2f} / {sk[1]:.2f} / {sk[2]:.2f}")

    print("\nModelos que superam a persistência (skill < 1):")
    for cen in CENARIOS:
        vence = [m for m in resultados if resultados[m][cen] < persist[cen]]
        print(f"  {cen}: {len(vence)}/{len(resultados)}  {vence if vence else '(nenhum)'}")


if __name__ == "__main__":
    main()
