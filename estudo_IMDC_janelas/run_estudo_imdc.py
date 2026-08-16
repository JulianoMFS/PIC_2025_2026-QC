# -*- coding: utf-8 -*-
"""Runner do estudo com janelas do 2º IMDC. Uso: python run_estudo_imdc.py [grupo]
   grupo: 'rapidos' (RF,XGB,LSTM,SARIMA,ens,QRC) | 'lentos' (QSVM,regime,VQR-ReUp,VQR-Angle,QLSTM) | 'todos'
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, time
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, AQUI)
import imdc_splits as S
import modelos_imdc as M

FIG = os.path.join(AQUI, "figuras"); RES = os.path.join(AQUI, "resultados")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

ds = S.carregar_dataset(); CEN = S.montar_cenarios(ds); NF = len(ds["feature_names"])

def salvar(nome, R, titulo):
    M.plot_imdc(R, titulo, os.path.join(FIG, f"{nome}_pred_vs_obs.png"))
    out = {"modelo": nome, "janelas": S.JANELAS}
    for cen, r in R.items():
        out[cen] = {k: (float(r[k]) if isinstance(r.get(k), (int, float, np.floating)) else r[k])
                    for k in ("R2", "RMSE", "MAE", "WIS", "WIS_norm", "acc_regime") if k in r}
    json.dump(out, open(os.path.join(RES, f"{nome}_resultados.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    linha = " | ".join(f"{c}: R2={R[c]['R2']:.3f} WIS={R[c]['WIS']:.0f}" for c in R)
    print(f"[OK] {nome:14s} {linha}", flush=True)

RAPIDOS = [
    ("RF",        lambda: M.run_rf(CEN),                 "RF — janelas IMDC"),
    ("XGBoost",   lambda: M.run_xgb(CEN),                "XGBoost — janelas IMDC"),
    ("LSTM",      lambda: M.run_lstm(CEN, NF),           "LSTM clássico — janelas IMDC"),
    ("SARIMA",    lambda: M.run_sarima(CEN),             "SARIMA — janelas IMDC"),
    ("EnsRFXGB",  lambda: M.run_ensemble_rfxgb(CEN),     "Ensemble RF+XGBoost — janelas IMDC"),
    ("QRC",       lambda: M.run_qrc(CEN),                "QRC — janelas IMDC"),
]
LENTOS = [
    ("QSVM",      lambda: M.run_qsvm(CEN),               "QSVM — janelas IMDC"),
    ("EnsRegime", lambda: M.run_regime(CEN),             "Ensemble regime QSVM→RF — janelas IMDC"),
    ("VQR_ReUp",  lambda: M.run_vqr_reup(CEN),           "VQR Re-uploading — janelas IMDC"),
    ("VQR_Angle", lambda: M.run_vqr_angle(CEN),          "VQR AngleEmbedding — janelas IMDC"),
    ("QLSTM",     lambda: M.run_qlstm(CEN, NF),          "QLSTM — janelas IMDC"),
]

def rodar(lista):
    for nome, fn, tit in lista:
        t = time.time()
        try:
            R = fn(); salvar(nome, R, tit); print(f"     ({round(time.time()-t)}s)", flush=True)
        except Exception as e:
            import traceback; print(f"[ERRO] {nome}: {e!r}"); traceback.print_exc()

if __name__ == "__main__":
    grupo = sys.argv[1] if len(sys.argv) > 1 else "todos"
    print(f"=== Estudo IMDC | grupo={grupo} | features={NF} ===", flush=True)
    for k, d in CEN.items():
        print(f"  {k}: treino={len(d['y_train'])} teste={len(d['y_test'])} pico={int(d['y_test'].max())}", flush=True)
    if grupo in ("rapidos", "todos"): rodar(RAPIDOS)
    if grupo in ("lentos", "todos"):  rodar(LENTOS)
    print("=== FIM ===", flush=True)
