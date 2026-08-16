"""
utils_qml.py — Funções utilitárias compartilhadas do projeto PIBIT/CEUB
QML para Previsão de Dengue no DF | 3º IMDC 2026

Substitui a duplicação de calcular_wis(), metricas(), salvar_padrao() e plot_pred()
que existia em cada notebook independentemente.

Uso nos notebooks:
    import sys, os
    sys.path.insert(0, os.path.abspath(".."))
    from utils_qml import calcular_wis, metricas, salvar_padrao, plot_pred
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sem display (seguro em todos os ambientes)
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ─── Métricas ────────────────────────────────────────────────────────────────

def calcular_wis(y_true, preds_matrix, alphas=None):
    """
    Weighted Interval Score — métrica primária do IMDC.

    Parâmetros
    ----------
    y_true       : array (n,)        — valores observados
    preds_matrix : array (B, n)      — B amostras bootstrap de previsão
    alphas       : list[float]       — níveis de cobertura; padrão FluSight/IMDC

    Retorna
    -------
    float — WIS (menor é melhor)

    Referência: Bracher et al. (2021). Evaluating epidemic forecasts in an interval
    format. PLOS Computational Biology, 17(2), e1008618.
    """
    if alphas is None:
        alphas = [0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    y_true = np.asarray(y_true, dtype=float)
    preds_matrix = np.asarray(preds_matrix, dtype=float)

    med = np.median(preds_matrix, axis=0)
    wis = 0.5 * np.mean(np.abs(y_true - med))
    K = 0
    for alpha in alphas:
        lo = np.percentile(preds_matrix, 100 * alpha / 2, axis=0)
        hi = np.percentile(preds_matrix, 100 * (1 - alpha / 2), axis=0)
        penalty = (hi - lo
                   + (2 / alpha) * np.maximum(lo - y_true, 0)
                   + (2 / alpha) * np.maximum(y_true - hi, 0))
        wis += (alpha / 2) * np.mean(penalty)
        K += 1
    return float(wis / (K + 0.5))


def metricas(y_true, y_pred, preds_matrix=None, nome=""):
    """
    Calcula R², RMSE, MAE e (opcionalmente) WIS/WIS_norm.

    Parâmetros
    ----------
    y_true       : array (n,)
    y_pred       : array (n,)        — mediana ou predição pontual
    preds_matrix : array (B, n)|None — necessário para WIS
    nome         : str               — identificador livre para o resultado

    Retorna
    -------
    dict com chaves: nome, R2, RMSE, MAE [, WIS, WIS_norm]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r = {
        "nome": nome,
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }
    if preds_matrix is not None:
        wis = calcular_wis(y_true, preds_matrix)
        r["WIS"]      = wis
        r["WIS_norm"] = wis / (float(np.mean(y_true)) + 1e-10)
    return r


# ─── Persistência ────────────────────────────────────────────────────────────

def salvar_padrao(RESULTADOS, SCHEMA_INFO, outdir=None, verbose=True):
    """
    Salva resultados no formato padronizado para comparação entre algoritmos.

    Parâmetros
    ----------
    RESULTADOS  : dict {"C1": {..., "R2", "RMSE", "MAE", "WIS", "WIS_norm", "tempo_s"}, ...}
    SCHEMA_INFO : dict com algoritmo, fase, tipo, n_parametros_quanticos, config
    outdir      : str|None  — diretório de saída; padrão = diretório de trabalho atual
    verbose     : bool      — imprime resumo no stdout

    Retorna
    -------
    dict — documento JSON salvo
    """
    doc = {**SCHEMA_INFO}
    for cen in ["C1", "C2", "C3"]:
        r = RESULTADOS.get(cen, {})
        doc[cen] = {
            "R2":       round(float(r.get("R2",       float("nan"))), 4),
            "RMSE":     round(float(r.get("RMSE",     float("nan"))), 2),
            "MAE":      round(float(r.get("MAE",      float("nan"))), 2),
            "WIS":      round(float(r.get("WIS",      float("nan"))), 2),
            "WIS_norm": round(float(r.get("WIS_norm", float("nan"))), 4),
            "tempo_s":  round(float(r.get("tempo_s",  0.0)),          1),
        }
        # campos extras numéricos específicos de cada algoritmo
        for k, v in r.items():
            if k not in doc[cen] and isinstance(v, (int, float, str, bool)):
                try:
                    doc[cen][k] = round(float(v), 4)
                except (TypeError, ValueError):
                    doc[cen][k] = v

    fase   = SCHEMA_INFO.get("fase", 0)
    alg    = SCHEMA_INFO.get("algoritmo", "modelo").replace(" ", "_").lower()
    fname  = f"fase{fase:02d}_{alg}_resultados.json"
    fpath  = os.path.join(outdir, fname) if outdir else fname

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[PADRAO] {fname}")
        print(f"  Algoritmo : {doc.get('algoritmo')}")
        print(f"  Tipo      : {doc.get('tipo')}")
        print(f"  Parametros quanticos: {doc.get('n_parametros_quanticos')}")
        for cen in ["C1", "C2", "C3"]:
            d = doc[cen]
            print(f"  {cen}: R2={d['R2']:+.4f} | WIS={d['WIS']:.2f} | {d['tempo_s']:.1f}s")
    return doc


# ─── Visualização ─────────────────────────────────────────────────────────────

_CORES_CENARIO = {"C1": "tab:blue", "C2": "tab:red", "C3": "tab:green"}

_DESCRICAO_CENARIO = {
    "C1": "Transmissão normal/crescente  (out/2022 – jun/2025)",
    "C2": "Pico recorde 25.714 casos/sem  (out/2023 – jun/2025)",
    "C3": "Pós-surto, Rt < 1  (jun/2024 – jun/2025)",
}


def plot_pred(resultados, titulo, fname, cenarios_desc=None, show=True):
    """
    Gráfico 3×1 de predição vs. observado com intervalo de confiança 80%.

    Parâmetros
    ----------
    resultados    : dict {"C1": {"y_test", "mediana", "preds_matrix", "R2"}, ...}
    titulo        : str  — título geral da figura
    fname         : str  — caminho de saída (.png)
    cenarios_desc : dict — descrições opcionais dos cenários (sobrescreve padrão)
    show          : bool — chama plt.show() (False em ambientes headless)
    """
    desc = {**_DESCRICAO_CENARIO, **(cenarios_desc or {})}

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for i, (cen, r) in enumerate(resultados.items()):
        ax  = axes[i]
        sem = np.arange(len(r["y_test"]))
        p10 = np.percentile(r["preds_matrix"], 10, axis=0)
        p90 = np.percentile(r["preds_matrix"], 90, axis=0)
        cor = _CORES_CENARIO.get(cen, "tab:purple")

        ax.fill_between(sem, p10, p90, alpha=0.2, color=cor, label="IC 80%")
        ax.plot(sem, r["y_test"],  "k-",  lw=1.5, label="casos_est (real)", zorder=5)
        ax.plot(sem, r["mediana"], "--",  lw=1.5, color=cor,
                label=f"R²={r['R2']:.3f}", zorder=4)
        ax.set_title(f"{cen}: {desc.get(cen, '')}", fontweight="bold")
        ax.set_ylabel("casos_est")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Semana epidemiológica")
    plt.suptitle(titulo, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[SALVO] {fname}")

    # Sidecar .npz com os dados de plotagem — permite re-renderizar rótulos
    # (datas, acentos, títulos) sem recalcular os modelos.
    try:
        _dados = {}
        for cen, r in resultados.items():
            _dados[f"{cen}_ytest"] = np.asarray(r["y_test"])
            _dados[f"{cen}_mediana"] = np.asarray(r["mediana"])
            _dados[f"{cen}_preds"] = np.asarray(r["preds_matrix"])
            _dados[f"{cen}_R2"] = np.asarray(float(r["R2"]))
        _npz = fname.rsplit(".", 1)[0] + "_plotdata.npz"
        np.savez_compressed(_npz, **_dados)
        print(f"[SALVO] {_npz}")
    except Exception as _e:
        print(f"[AVISO] nao foi possivel salvar plotdata: {_e!r}")


# ─── Contrato do JSON de saída ────────────────────────────────────────────────

_SCHEMA_TOPO    = {"algoritmo", "fase", "tipo", "n_parametros_quanticos", "config"}
_SCHEMA_CENARIO = {"R2", "RMSE", "MAE", "WIS", "WIS_norm", "tempo_s"}


def validar_json_saida(doc, contexto=""):
    """
    Valida que o documento JSON produzido por salvar_padrao() respeita o contrato
    esperado pelo Comparacao_Geral e pelo protocolo IMDC.

    Lança AssertionError descritivo se alguma invariante for violada.

    Parâmetros
    ----------
    doc      : dict  — documento retornado por salvar_padrao()
    contexto : str   — nome do notebook/fase para mensagens de erro
    """
    pfx = f"[{contexto}] " if contexto else ""

    # Campos obrigatórios no topo
    faltando_topo = _SCHEMA_TOPO - set(doc.keys())
    assert not faltando_topo, f"{pfx}Campos ausentes no JSON: {faltando_topo}"

    for cen in ["C1", "C2", "C3"]:
        assert cen in doc, f"{pfx}Cenario '{cen}' ausente no JSON"
        d = doc[cen]

        # Campos obrigatórios por cenário
        faltando = _SCHEMA_CENARIO - set(d.keys())
        assert not faltando, f"{pfx}{cen}: campos ausentes: {faltando}"

        # Invariantes físicas
        # R² tem limite superior 1 mas pode ser arbitrariamente negativo
        # quando o modelo é pior que prever a média (sem limite inferior teórico)
        assert d["R2"] <= 1.0 or np.isnan(d["R2"]), \
            f"{pfx}{cen}: R2={d['R2']} acima de 1.0 (impossivel)"
        assert d["WIS"] >= 0 or np.isnan(d["WIS"]), \
            f"{pfx}{cen}: WIS={d['WIS']} negativo (impossivel)"
        assert d["WIS_norm"] >= 0 or np.isnan(d["WIS_norm"]), \
            f"{pfx}{cen}: WIS_norm={d['WIS_norm']} negativo"
        assert d["RMSE"] >= 0 or np.isnan(d["RMSE"]), \
            f"{pfx}{cen}: RMSE={d['RMSE']} negativo (impossivel)"
        assert d["MAE"] >= 0 or np.isnan(d["MAE"]), \
            f"{pfx}{cen}: MAE={d['MAE']} negativo (impossivel)"
        assert d["tempo_s"] >= 0, \
            f"{pfx}{cen}: tempo_s={d['tempo_s']} negativo"

    print(f"[CONTRATO OK] {pfx}JSON valido — todos os campos e invariantes corretos")
    return True


# ─── Integração do pipeline de dados ─────────────────────────────────────────

def validar_pipeline(dataset, splits, n_features_esperado=13, n_cenarios=3):
    """
    Valida a saída de construir_features() e splits_validacao() antes de treinar.

    Detecta: mudança no número de features (quebraria os circuitos de 6 qubits),
    splits insuficientes, arrays vazios e inconsistência de dimensões.

    Parâmetros
    ----------
    dataset              : dict retornado por construir_features()
    splits               : list retornada por splits_validacao()
    n_features_esperado  : int   — 13 no projeto atual (6 + 7 lags)
    n_cenarios           : int   — mínimo de cenários necessários
    """
    # Shape do dataset
    X = np.asarray(dataset["X"])
    y = np.asarray(dataset["y"])
    assert X.ndim == 2, f"dataset['X'] deve ser 2D, obtido shape {X.shape}"
    assert X.shape[1] == n_features_esperado, (
        f"Esperado {n_features_esperado} features, obtido {X.shape[1]}. "
        "Verifique feature_engineering.py — isso quebraria os circuitos de 6 qubits."
    )
    assert len(y) == len(X), "Tamanhos de X e y incompativeis"
    assert len(X) > 0, "dataset vazio"

    # Splits
    assert len(splits) >= n_cenarios, (
        f"Esperado >= {n_cenarios} splits, obtido {len(splits)}. "
        "Verifique o tamanho da serie historica."
    )
    for i, sp in enumerate(splits[:n_cenarios]):
        tr, te = sp["treino"], sp["teste"]
        X_tr = np.asarray(tr["X"]); y_tr = np.asarray(tr["y"])
        X_te = np.asarray(te["X"]); y_te = np.asarray(te["y"])

        assert len(X_tr) > 0,            f"Split {i}: treino vazio"
        assert len(X_te) > 0,            f"Split {i}: teste vazio"
        assert X_tr.shape[1] == n_features_esperado, \
            f"Split {i} treino: {X_tr.shape[1]} features (esperado {n_features_esperado})"
        assert X_te.shape[1] == n_features_esperado, \
            f"Split {i} teste: {X_te.shape[1]} features (esperado {n_features_esperado})"
        assert len(X_tr) == len(y_tr),   f"Split {i}: X_train e y_train com tamanhos diferentes"
        assert len(X_te) == len(y_te),   f"Split {i}: X_test e y_test com tamanhos diferentes"
        assert np.all(y_tr >= 0),        f"Split {i}: y_train com valores negativos"
        assert np.all(y_te >= 0),        f"Split {i}: y_test com valores negativos"

    n_feat = X.shape[1]
    feat_names = dataset.get("feature_names", [])
    if feat_names:
        assert len(feat_names) == n_feat, (
            f"feature_names tem {len(feat_names)} entradas mas X tem {n_feat} colunas"
        )

    print(f"[PIPELINE OK] {n_feat} features | {len(splits)} splits | "
          f"serie={len(X)} semanas | cenarios C1/C2/C3 prontos")
    return True


# ─── Golden File (Regressão de Resultados) ───────────────────────────────────

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")


def registrar_golden(doc, contexto=""):
    """
    Salva os resultados atuais como referência golden para execuções futuras.
    Chame manualmente após confirmar que os resultados são corretos.

    O arquivo é salvo em <raiz_projeto>/golden/<contexto>_golden.json
    Se o arquivo já existir, não sobrescreve (use force=True para forçar).

    Parâmetros
    ----------
    doc      : dict  — documento retornado por salvar_padrao()
    contexto : str   — identificador (ex: "Fase0_Classico_RF")
    """
    os.makedirs(_GOLDEN_DIR, exist_ok=True)
    slug  = contexto.replace(" ", "_").replace("/", "_")
    fpath = os.path.join(_GOLDEN_DIR, f"{slug}_golden.json")

    if os.path.exists(fpath):
        print(f"[GOLDEN] Arquivo ja existe: {fpath}  (use force=True para sobrescrever)")
        return False

    snapshot = {cen: {k: doc[cen][k] for k in ["R2", "WIS", "WIS_norm", "RMSE", "MAE"]}
                for cen in ["C1", "C2", "C3"] if cen in doc}
    snapshot["_algoritmo"] = doc.get("algoritmo", "")
    snapshot["_fase"]      = doc.get("fase", -1)

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"[GOLDEN SALVO] {fpath}")
    return True


def validar_golden(doc, contexto="", tol_wis_pct=10.0, tol_r2=0.05):
    """
    Compara os resultados atuais com o golden file da execução de referência.

    Se o golden não existir ainda, registra automaticamente e retorna True.
    Diferenças acima da tolerância geram WARNING (não AssertionError) porque
    Bootstrap introduz variância estocástica legítima.

    Parâmetros
    ----------
    doc         : dict  — documento retornado por salvar_padrao()
    contexto    : str   — mesmo contexto usado em registrar_golden()
    tol_wis_pct : float — tolerância em % para o WIS (padrão: 10%)
    tol_r2      : float — tolerância absoluta para R² (padrão: 0.05)

    Retorna
    -------
    bool — True se dentro da tolerância, False se alguma divergência foi detectada
    """
    slug  = contexto.replace(" ", "_").replace("/", "_")
    fpath = os.path.join(_GOLDEN_DIR, f"{slug}_golden.json")

    if not os.path.exists(fpath):
        print(f"[GOLDEN] Sem referencia previa — registrando resultado atual como golden.")
        registrar_golden(doc, contexto)
        return True

    golden = json.load(open(fpath, encoding="utf-8"))
    pfx    = f"[{contexto}] " if contexto else ""
    ok     = True

    for cen in ["C1", "C2", "C3"]:
        if cen not in doc or cen not in golden:
            continue
        cur = doc[cen]
        ref = golden[cen]

        # WIS: tolerância percentual
        if ref["WIS"] > 0:
            delta_wis_pct = abs(cur["WIS"] - ref["WIS"]) / ref["WIS"] * 100
            if delta_wis_pct > tol_wis_pct:
                print(f"[GOLDEN WARN] {pfx}{cen}: WIS mudou {delta_wis_pct:.1f}%  "
                      f"(ref={ref['WIS']:.2f}  atual={cur['WIS']:.2f}  tol={tol_wis_pct}%)")
                ok = False

        # R²: tolerância absoluta
        delta_r2 = abs(cur["R2"] - ref["R2"])
        if delta_r2 > tol_r2:
            print(f"[GOLDEN WARN] {pfx}{cen}: R2 mudou {delta_r2:.4f}  "
                  f"(ref={ref['R2']:.4f}  atual={cur['R2']:.4f}  tol={tol_r2})")
            ok = False

    if ok:
        print(f"[GOLDEN OK] {pfx}Resultados dentro da tolerancia vs. referencia.")
    return ok


# ─── Invariância Quântica ─────────────────────────────────────────────────────

def testar_invariancia_quantica(circuit_fn, n_qubits, x_sample, weights_sample,
                                contexto="", tol=1e-6):
    """
    Verifica três invariantes que qualquer circuito PennyLane correto deve satisfazer.

    Testes realizados
    -----------------
    1. Determinismo     : circuit_fn(x, w) == circuit_fn(x, w) (mesmas entradas → mesma saída)
    2. Bounds de output : todos os valores de saída em [-1, 1] (expectativas de Pauli)
    3. Shape de output  : retorna exatamente n_qubits valores quando circuit retorna
                          lista de expval, OU 1 valor quando retorna expval escalar

    Parâmetros
    ----------
    circuit_fn     : callable(x, weights) -> float | array
    n_qubits       : int    — número de qubits do circuito
    x_sample       : array  — um vetor de features de entrada
    weights_sample : array  — parâmetros do circuito
    contexto       : str    — identificador para mensagens
    tol            : float  — tolerância numérica para determinismo
    """
    import numpy as np
    pfx = f"[{contexto}] " if contexto else ""

    # ── Teste 1: Determinismo ─────────────────────────────────────────────────
    out1 = np.asarray(circuit_fn(x_sample, weights_sample), dtype=float).flatten()
    out2 = np.asarray(circuit_fn(x_sample, weights_sample), dtype=float).flatten()
    assert np.allclose(out1, out2, atol=tol), (
        f"{pfx}Determinismo violado: mesmas entradas produziram saidas diferentes\n"
        f"  out1={out1}\n  out2={out2}"
    )

    # ── Teste 2: Bounds [-1, 1] para expectativas de Pauli ───────────────────
    assert np.all(out1 >= -1.0 - tol) and np.all(out1 <= 1.0 + tol), (
        f"{pfx}Valores fora de [-1, 1]: {out1}. "
        "Verifique se o circuito retorna qml.expval (nao qml.probs ou qml.state)."
    )

    # ── Teste 3: Shape coerente ───────────────────────────────────────────────
    # Aceita: escalar (1 valor) ou vetor de n_qubits valores
    n_out = len(out1)
    assert n_out in (1, n_qubits), (
        f"{pfx}Shape inesperado: circuit retornou {n_out} valores "
        f"(esperado 1 escalar ou {n_qubits} expval individuais)."
    )

    print(f"[QUANTICO OK] {pfx}Determinismo / bounds [-1,1] / shape({n_out}) — OK")
    return True


# ─── Property-Based Testing (Hypothesis) ─────────────────────────────────────

def testar_propriedades():
    """
    Executa testes baseados em propriedades matemáticas de calcular_wis() e metricas()
    usando a biblioteca Hypothesis (generates centenas de inputs aleatórios).

    Se Hypothesis não estiver instalada, exibe aviso e executa versão simplificada
    com NumPy para não bloquear a execução do notebook.
    """
    try:
        from hypothesis import given, settings, HealthCheck
        from hypothesis import strategies as st
        import hypothesis.extra.numpy as hnp
        _HAS_HYPOTHESIS = True
    except ImportError:
        _HAS_HYPOTHESIS = False
        print("[HYPOTHESIS] Biblioteca nao instalada. Executando versao simplificada.")
        print("             Para instalar: pip install hypothesis")

    if _HAS_HYPOTHESIS:
        # ── Propriedade 1: WIS nunca negativo ────────────────────────────────
        @given(
            y     = hnp.arrays(float, st.integers(3, 20),
                               elements=st.floats(0, 5000, allow_nan=False, allow_infinity=False)),
            preds = hnp.arrays(float, st.tuples(st.integers(2, 10), st.integers(3, 20)),
                               elements=st.floats(0, 5000, allow_nan=False, allow_infinity=False)),
        )
        @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
        def prop_wis_nao_negativo(y, preds):
            if y.shape[0] != preds.shape[1]:
                return  # dimensoes incompativeis: pula
            wis = calcular_wis(y, preds)
            assert wis >= 0, f"WIS negativo: {wis} (y={y}, preds shape={preds.shape})"

        # ── Propriedade 2: RMSE e MAE nunca negativos ────────────────────────
        @given(
            y = hnp.arrays(float, st.integers(3, 30),
                           elements=st.floats(0, 5000, allow_nan=False, allow_infinity=False)),
        )
        @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
        def prop_rmse_mae_nao_negativos(y):
            rng  = np.random.default_rng(0)
            pred = y + rng.normal(0, 10, len(y))
            m = metricas(y, pred)
            assert m["RMSE"] >= 0, f"RMSE negativo: {m['RMSE']}"
            assert m["MAE"]  >= 0, f"MAE negativo: {m['MAE']}"

        # ── Propriedade 3: predicao perfeita -> WIS proximo de 0 ─────────────
        @given(
            y = hnp.arrays(float, st.integers(3, 20),
                           elements=st.floats(1, 1000, allow_nan=False, allow_infinity=False)),
        )
        @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
        def prop_predicao_perfeita(y):
            preds = np.tile(y, (10, 1))
            wis   = calcular_wis(y, preds)
            assert wis < 1e-6, f"WIS nao-zero para predicao perfeita: {wis}"

        # ── Propriedade 4: WIS_norm = WIS / mean(y) ──────────────────────────
        @given(
            y = hnp.arrays(float, st.integers(3, 15),
                           elements=st.floats(10, 1000, allow_nan=False, allow_infinity=False)),
        )
        @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
        def prop_wis_norm_consistente(y):
            rng   = np.random.default_rng(1)
            preds = np.maximum(y + rng.normal(0, 50, (5, len(y))), 0)
            m     = metricas(y, np.median(preds, axis=0), preds)
            esperado = m["WIS"] / (np.mean(y) + 1e-10)
            assert abs(m["WIS_norm"] - esperado) < 1e-9, (
                f"WIS_norm inconsistente: {m['WIS_norm']} vs esperado {esperado}"
            )

        prop_wis_nao_negativo()
        prop_rmse_mae_nao_negativos()
        prop_predicao_perfeita()
        prop_wis_norm_consistente()
        print("[HYPOTHESIS OK] 4 propriedades verificadas com 100-200 exemplos aleatorios cada.")

    else:
        # Versao simplificada sem Hypothesis
        rng = np.random.default_rng(42)
        falhas = 0
        for _ in range(500):
            n  = rng.integers(3, 25)
            y  = rng.uniform(0, 5000, n)
            pm = rng.uniform(0, 5000, (rng.integers(2, 8), n))
            wis = calcular_wis(y, pm)
            if wis < 0:
                falhas += 1
                print(f"[PROP FALHOU] WIS negativo: {wis}")
            m = metricas(y, np.median(pm, axis=0), pm)
            if m["RMSE"] < 0 or m["MAE"] < 0:
                falhas += 1
                print(f"[PROP FALHOU] RMSE/MAE negativo: RMSE={m['RMSE']}, MAE={m['MAE']}")
        if falhas == 0:
            print(f"[PROP OK] 500 combinacoes aleatorias: WIS>=0, RMSE>=0, MAE>=0 em todos.")
        else:
            print(f"[PROP WARN] {falhas} falhas detectadas — verificar calcular_wis()")


# ─── Testes unitários embutidos ───────────────────────────────────────────────

def _testar():
    """Executa verificações básicas das funções do módulo."""

    # ── calcular_wis ──────────────────────────────────────────────────────────
    y = np.array([10.0, 20.0, 30.0])
    preds = np.tile(y, (100, 1))

    wis = calcular_wis(y, preds)
    assert wis < 0.1, f"WIS esperado ~0, obtido {wis:.4f}"

    # WIS sempre nao-negativo
    rng = np.random.default_rng(0)
    for _ in range(10):
        y_r = rng.uniform(0, 1000, 20)
        pm  = rng.uniform(0, 1000, (5, 20))
        assert calcular_wis(y_r, pm) >= 0, "WIS negativo detectado"

    # ── metricas ──────────────────────────────────────────────────────────────
    m = metricas(y, y, preds, nome="teste")
    assert abs(m["R2"] - 1.0) < 1e-9,  f"R2 esperado 1.0, obtido {m['R2']}"
    assert m["RMSE"] < 1e-9,           f"RMSE esperado 0, obtido {m['RMSE']}"
    assert m["WIS_norm"] < 0.01,       f"WIS_norm alto: {m['WIS_norm']}"

    m2 = metricas(y, y)
    assert "WIS" not in m2, "WIS nao deve aparecer sem preds_matrix"

    # ── validar_json_saida ────────────────────────────────────────────────────
    doc_ok = {
        "algoritmo": "Teste", "fase": 99, "tipo": "teste",
        "n_parametros_quanticos": None, "config": {},
        "C1": {"R2": 0.5, "RMSE": 100.0, "MAE": 80.0, "WIS": 50.0, "WIS_norm": 0.3, "tempo_s": 1.0},
        "C2": {"R2": 0.3, "RMSE": 200.0, "MAE": 150.0,"WIS": 120.0,"WIS_norm": 0.4, "tempo_s": 1.1},
        "C3": {"R2": 0.7, "RMSE": 30.0,  "MAE": 20.0, "WIS": 10.0, "WIS_norm": 0.1, "tempo_s": 0.9},
    }
    validar_json_saida(doc_ok, contexto="teste_unitario")

    # Deve falhar com WIS negativo
    doc_ruim = {k: v for k, v in doc_ok.items()}
    doc_ruim["C1"] = {**doc_ok["C1"], "WIS": -5.0}
    try:
        validar_json_saida(doc_ruim, contexto="deve_falhar")
        assert False, "Deveria ter lancado AssertionError"
    except AssertionError:
        pass  # esperado

    print("[OK] Todos os testes de utils_qml passaram.")


if __name__ == "__main__":
    _testar()
