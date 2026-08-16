# -*- coding: utf-8 -*-
"""
Os 11 modelos do estudo, parametrizados para rodar sobre CENÁRIOS arbitrários
(aqui, as janelas T1/T2/T3 do 2º IMDC). Hiperparâmetros idênticos aos do estudo
original (verificados). Cada função recebe CEN {nome:{X_train,y_train,X_test,y_test}}
e devolve R {nome:{R2,RMSE,MAE,WIS,WIS_norm, mediana, preds_matrix, y_test}}.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time
from copy import deepcopy
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = r"C:/Users/julia/OneDrive/Área de Trabalho/PIBIT/Experimentos/Relatorio_Final_IC"
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, REPO)
from utils_qml import metricas
from imdc_splits import DESCRICAO

import pennylane as qml
from pennylane import numpy as pnp
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

try:
    import pennylane_lightning; DEVN = "lightning.qubit"
except ImportError:
    DEVN = "default.qubit"

_CORES = {"T1": "tab:blue", "T2": "tab:red", "T3": "tab:green"}


def _res(cen, yte, med, P):
    m = metricas(yte, med, P, nome=cen)
    return {**m, "mediana": med, "preds_matrix": P, "y_test": yte}


def plot_imdc(R, titulo, path, show=False):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for i, (cen, r) in enumerate(R.items()):
        ax = axes[i]; sem = np.arange(len(r["y_test"]))
        p10 = np.percentile(r["preds_matrix"], 10, axis=0)
        p90 = np.percentile(r["preds_matrix"], 90, axis=0)
        cor = _CORES.get(cen, "tab:purple")
        ax.fill_between(sem, p10, p90, alpha=0.2, color=cor, label="IC 80%")
        ax.plot(sem, r["y_test"], "k-", lw=1.5, label="casos_est (real)", zorder=5)
        ax.plot(sem, r["mediana"], "--", lw=1.5, color=cor, label=f"R²={r['R2']:.3f}", zorder=4)
        ax.set_title(DESCRICAO.get(cen, cen), fontweight="bold")
        ax.set_ylabel("casos_est"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Semana da temporada")
    plt.suptitle(titulo, fontsize=13, fontweight="bold"); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
    # sidecar npz
    try:
        dd = {}
        for cen, r in R.items():
            dd[f"{cen}_ytest"] = np.asarray(r["y_test"]); dd[f"{cen}_mediana"] = np.asarray(r["mediana"])
            dd[f"{cen}_preds"] = np.asarray(r["preds_matrix"]); dd[f"{cen}_R2"] = np.asarray(float(r["R2"]))
        np.savez_compressed(path.rsplit(".", 1)[0] + "_plotdata.npz", **dd)
    except Exception as e:
        print("[aviso npz]", repr(e))
    print("[SALVO]", path)


# ───────────────────────── CLÁSSICOS ─────────────────────────
def run_rf(CEN):
    R = {}
    for cen, d in CEN.items():
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
        P = np.zeros((5, len(Xte)))
        for b in range(5):
            rng = np.random.RandomState(42 + b); idx = rng.choice(len(Xtr), len(Xtr), True)
            rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42 + b, n_jobs=-1)
            rf.fit(Xtr[idx], np.log1p(ytr[idx])); P[b] = np.maximum(np.expm1(rf.predict(Xte)), 0)
        R[cen] = _res(cen, yte, np.median(P, 0), P)
    return R


def run_xgb(CEN):
    from xgboost import XGBRegressor
    R = {}
    for cen, d in CEN.items():
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]; n = len(Xtr)
        w = 1.0 + 2.0 * (np.arange(n) / max(n - 1, 1)); P = np.zeros((5, len(Xte)))
        for b in range(5):
            rng = np.random.RandomState(42 + b); idx = rng.choice(n, n, True)
            xg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=3, reg_lambda=1.0, random_state=42 + b, n_jobs=-1)
            xg.fit(Xtr[idx], np.log1p(ytr[idx]), sample_weight=w[idx]); P[b] = np.maximum(np.expm1(xg.predict(Xte)), 0)
        R[cen] = _res(cen, yte, np.median(P, 0), P)
    return R


def run_ensemble_rfxgb(CEN):
    from xgboost import XGBRegressor
    R = {}
    for cen, d in CEN.items():
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]; n = len(Xtr)
        w = 1.0 + 2.0 * (np.arange(n) / max(n - 1, 1)); P = np.zeros((10, len(Xte)))
        for b in range(5):
            rng = np.random.RandomState(42 + b); idx = rng.choice(n, n, True)
            rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42 + b, n_jobs=-1)
            rf.fit(Xtr[idx], np.log1p(ytr[idx])); P[b] = np.maximum(np.expm1(rf.predict(Xte)), 0)
            xg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=3, reg_lambda=1.0, random_state=42 + b, n_jobs=-1)
            xg.fit(Xtr[idx], np.log1p(ytr[idx]), sample_weight=w[idx]); P[5 + b] = np.maximum(np.expm1(xg.predict(Xte)), 0)
        R[cen] = _res(cen, yte, np.median(P, 0), P)
    return R


def run_sarima(CEN):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    R = {}; B = 30
    for cen, d in CEN.items():
        ytr, yte = d["y_train"], d["y_test"]; ylog = np.log1p(ytr.astype(float)); n = len(ylog)
        seasonal = (1, 0, 0, 52) if n >= 60 else (0, 0, 0, 0)
        try:
            res = SARIMAX(ylog, order=(2, 1, 2), seasonal_order=seasonal, enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False, maxiter=300)
        except Exception:
            res = SARIMAX(ylog, order=(1, 1, 1)).fit(disp=False, maxiter=300)
        fc = res.get_forecast(steps=len(yte)); ml = np.asarray(fc.predicted_mean); sl = np.minimum(np.asarray(fc.se_mean), 1.0)
        rng = np.random.RandomState(42)
        al = np.clip(rng.normal(ml[None, :], sl[None, :], size=(B, len(yte))), 0.0, np.log1p(100000))
        P = np.maximum(np.expm1(al), 0.0); med = np.maximum(np.expm1(ml), 0.0)
        R[cen] = _res(cen, yte, med, P)
    return R


def run_lstm(CEN, NF):
    W, H = 4, 4
    def sig(z): return 1.0 / (1.0 + pnp.exp(-z))
    def initp(seed):
        rng = np.random.RandomState(seed); p = {}; Din = H + NF; esc = 1.0 / np.sqrt(Din)
        for g in ["f", "i", "g", "o"]:
            p["W" + g] = pnp.array(rng.normal(0, esc, (H, Din)), requires_grad=True); p["b" + g] = pnp.array(np.zeros(H), requires_grad=True)
        p["Wout"] = pnp.array(rng.normal(0, 0.3, (H,)), requires_grad=True); p["bout"] = pnp.array(0.0, requires_grad=True); return p
    def fwd(seq, p):
        h = pnp.zeros(H); c = pnp.zeros(H)
        for tt in range(seq.shape[0]):
            v = pnp.concatenate([h, seq[tt]])
            f = sig(p["Wf"] @ v + p["bf"]); i = sig(p["Wi"] @ v + p["bi"]); g = pnp.tanh(p["Wg"] @ v + p["bg"]); o = sig(p["Wo"] @ v + p["bo"])
            c = f * c + i * g; h = o * pnp.tanh(c)
        return p["Wout"] @ h + p["bout"]
    def btr(Xn, y): return np.array([Xn[i - W + 1:i + 1] for i in range(W - 1, len(Xn))]), y[W - 1:]
    def bte(Xtr, Xte):
        ext = np.vstack([Xtr[-(W - 1):], Xte]) if W > 1 else Xte; return np.array([ext[i - W + 1:i + 1] for i in range(W - 1, len(ext))])
    def pred(seqs, p): return np.array([float(fwd(pnp.array(s), p)) for s in seqs])
    def cost(p, Xb, yb): return pnp.mean((pnp.stack([fwd(pnp.array(s), p) for s in Xb]) - yb) ** 2)
    def train1(seqs, yln, seed, ep):
        p = initp(seed); rng = np.random.RandomState(seed); idx = rng.choice(len(seqs), len(seqs), True); Xb, yb = seqs[idx], yln[idx]
        opt = qml.AdamOptimizer(0.05)
        for _ in range(ep): p, _ = opt.step_and_cost(lambda pp: cost(pp, Xb, yb), p)
        return p
    np.random.seed(42); R = {}
    for cen, d in CEN.items():
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtn, Xen = (Xtr - mu) / sd, (Xte - mu) / sd
        st, yst = btr(Xtn, ytr); se = bte(Xtn, Xen); ym, ys = np.log1p(yst).mean(), np.log1p(yst).std() + 1e-6; yln = (np.log1p(yst) - ym) / ys
        dn = lambda pn: np.maximum(np.expm1(pn * ys + ym), 0); P = np.zeros((3, len(se)))
        for b in range(3): P[b] = dn(pred(se, train1(st, yln, 42 + b, 30)))
        R[cen] = _res(cen, yte, np.median(P, 0), P)
    return R


# ───────────────────────── QUÂNTICOS ─────────────────────────
def run_qrc(CEN):
    NQ, NL = 8, 6
    rng = np.random.RandomState(42); Wres = rng.uniform(-np.pi, np.pi, (NL, NQ, 3))
    N_SINGLE = NQ; N_TWO_SUB = min(NQ * 2, NQ * (NQ - 1) // 2); N_MEAS = N_SINGLE + N_TWO_SUB
    PAIRS = [(i, j) for i in range(NQ) for j in range(i + 1, NQ)][:N_TWO_SUB]
    dev = qml.device(DEVN, wires=NQ)
    @qml.qnode(dev)
    def reserv(x, w):
        qml.AngleEmbedding(x[:NQ], wires=range(NQ), rotation="Y")
        for l in range(NL):
            for q in range(NQ): qml.RY(w[l, q, 0], wires=q); qml.RZ(w[l, q, 1], wires=q); qml.RX(w[l, q, 2], wires=q)
            for q in range(NQ): qml.CZ(wires=[q, (q + 1) % NQ])
            if l % 2 == 1:
                for q in range(NQ): qml.RY(x[q % len(x)] * 0.5, wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(NQ)] + [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for (i, j) in PAIRS]
    def proj(X, sc=None):
        if sc is None: sc = MinMaxScaler(feature_range=(0.01, np.pi)); Xs = sc.fit_transform(X)
        else: Xs = sc.transform(X)
        Rm = np.zeros((len(Xs), N_MEAS)); Wp = pnp.array(Wres)
        for i, x in enumerate(Xs):
            xp = np.pad(x, (0, max(0, NQ - len(x))))[:NQ]; Rm[i] = [float(v) for v in reserv(pnp.array(xp), Wp)]
        return Rm, sc
    R = {}
    for cen, d in CEN.items():
        Rtr, sc = proj(d["X_train"]); Rte, _ = proj(d["X_test"], sc); ytr, yte = d["y_train"], d["y_test"]; ylog = np.log1p(ytr)
        P = np.zeros((10, len(yte)))
        for b in range(10):
            rb = np.random.RandomState(42 + b); idx = rb.choice(len(Rtr), len(Rtr), True)
            Rb = Rtr[idx] + rb.normal(0, 0.01, Rtr[idx].shape)
            rg = Ridge(alpha=1e-3); rg.fit(Rb, ylog[idx]); P[b] = np.maximum(np.expm1(rg.predict(Rte)), 0)
        R[cen] = _res(cen, yte, np.median(P, 0), P)
    return R


def run_qsvm(CEN):
    from sklearn.svm import SVR, SVC
    from sklearn.metrics import accuracy_score
    NQ = 6; C = 10.0; EPS = 0.1; FEAT = [0, 1, 2, 3, 4, 5, 6, 7]; RT = 4
    dev = qml.device(DEVN, wires=NQ)
    @qml.qnode(dev)
    def fmap(x):
        xp = x[:NQ] if len(x) >= NQ else np.pad(x, (0, NQ - len(x)))
        for _ in range(2):
            for i in range(NQ): qml.Hadamard(wires=i); qml.RZ(xp[i % len(xp)], wires=i)
            for i in range(NQ - 1):
                qml.CZ(wires=[i, i + 1]); fi, fj = xp[i % len(xp)], xp[(i + 1) % len(xp)]
                qml.RZ((np.pi - fi) * (np.pi - fj), wires=i); qml.RZ((np.pi - fi) * (np.pi - fj), wires=i + 1)
        return qml.state()
    def K(a, b):
        Km = np.zeros((len(a), len(b)))
        for i in range(len(a)):
            si = fmap(a[i])
            for j in range(len(b)): Km[i, j] = float(np.abs(np.dot(np.conj(si), fmap(b[j]))) ** 2)
        return Km
    def regime(X):
        rt = X[:, RT]; lab = np.ones(len(rt), int); lab[rt < 0.85] = 0; lab[rt > 1.20] = 2; return lab
    np.random.seed(42); R = {}
    for cen, d in CEN.items():
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
        sc = MinMaxScaler(feature_range=(0.01, np.pi)); Xk = sc.fit_transform(Xtr[:, FEAT]); Xek = sc.transform(Xte[:, FEAT])
        Ktr = K(Xk, Xk); Kte = K(Xek, Xk); rtr = regime(Xtr)
        clf = SVC(kernel="precomputed", C=C, random_state=42); clf.fit(Ktr, rtr); rte = clf.predict(Kte)
        svrs = {}
        for r in [0, 1, 2]:
            mask = rtr == r
            if mask.sum() < 3: svrs[r] = None
            else:
                sv = SVR(kernel="precomputed", C=C, epsilon=EPS); sv.fit(Ktr[np.ix_(mask, mask)], np.log1p(ytr[mask])); svrs[r] = (sv, np.where(mask)[0])
        svg = SVR(kernel="precomputed", C=C, epsilon=EPS); svg.fit(Ktr, np.log1p(ytr))
        P = np.zeros((5, len(yte)))
        for b in range(5):
            rb = np.random.RandomState(42 + b); nm = rb.rand(len(rte)) < 0.10; rboot = rte.copy(); rboot[nm] = rb.randint(0, 3, nm.sum())
            pr = np.zeros(len(yte))
            for i, rg in enumerate(rboot):
                if svrs.get(rg) is not None:
                    svr_r, tri = svrs[rg]; ks = Kte[i, tri].reshape(1, -1)
                    try: pr[i] = np.expm1(float(svr_r.predict(ks)))
                    except Exception: pr[i] = np.expm1(float(svg.predict(Kte[i].reshape(1, -1))))
                else: pr[i] = np.expm1(float(svg.predict(Kte[i].reshape(1, -1))))
            P[b] = np.maximum(pr, 0)
        r = _res(cen, yte, np.median(P, 0), P)
        r["acc_regime"] = float(accuracy_score(regime(Xte), rte))
        R[cen] = r
    return R


def run_regime(CEN):
    from sklearn.svm import SVC
    NQ = 6; C = 10.0; FEAT = [0, 1, 2, 3, 4, 5, 6, 7]; RT = 4
    dev = qml.device("default.qubit", wires=NQ)
    @qml.qnode(dev)
    def fmap(x):
        xp = x[:NQ] if len(x) >= NQ else np.pad(x, (0, NQ - len(x)))
        for _ in range(2):
            for i in range(NQ): qml.Hadamard(wires=i); qml.RZ(xp[i % len(xp)], wires=i)
            for i in range(NQ - 1):
                qml.CZ(wires=[i, i + 1]); fi, fj = xp[i % len(xp)], xp[(i + 1) % len(xp)]
                qml.RZ((np.pi - fi) * (np.pi - fj), wires=i); qml.RZ((np.pi - fi) * (np.pi - fj), wires=i + 1)
        return qml.state()
    def K(a, b):
        Km = np.zeros((len(a), len(b)))
        for i in range(len(a)):
            si = fmap(a[i])
            for j in range(len(b)): Km[i, j] = float(np.abs(np.dot(np.conj(si), fmap(b[j]))) ** 2)
        return Km
    def regime(X):
        rt = X[:, RT]; lab = np.ones(len(rt), int); lab[rt < 0.85] = 0; lab[rt > 1.20] = 2; return lab
    def novo_rf(s): return RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=s, n_jobs=-1)
    np.random.seed(42); R = {}
    for cen, d in CEN.items():
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]; rtr = regime(Xtr)
        sc = MinMaxScaler(feature_range=(0.01, np.pi)); Ktrf = sc.fit_transform(Xtr[:, FEAT]); Ktef = sc.transform(Xte[:, FEAT])
        Ktr = K(Ktrf, Ktrf); Kte = K(Ktef, Ktrf); clf = SVC(kernel="precomputed", C=C, random_state=42); clf.fit(Ktr, rtr); rte = clf.predict(Kte)
        P = np.zeros((5, len(Xte)))
        for b in range(5):
            rng = np.random.RandomState(42 + b); rfs = {}
            for r in [0, 1, 2]:
                mm = np.where(rtr == r)[0]
                if len(mm) >= 3: idx = rng.choice(mm, len(mm), True); rf = novo_rf(42 + b); rf.fit(Xtr[idx], np.log1p(ytr[idx])); rfs[r] = rf
            idxg = rng.choice(len(Xtr), len(Xtr), True); rfg = novo_rf(42 + b); rfg.fit(Xtr[idxg], np.log1p(ytr[idxg]))
            for i in range(len(Xte)):
                mo = rfs.get(int(rte[i]), rfg); P[b, i] = max(np.expm1(float(mo.predict(Xte[i:i + 1])[0])), 0.0)
        R[cen] = _res(cen, yte, np.median(P, 0), P)
    return R


def run_vqr_reup(CEN):
    NQ, NL, NR, EP, SEED = 6, 6, 3, 80, 42
    dev = qml.device(DEVN, wires=NQ)
    @qml.qnode(dev)
    def vqr2(x, vw, rw):
        npure = NL - NR; qml.AmplitudeEmbedding(x, wires=range(NQ), normalize=True)
        if npure > 0: qml.StronglyEntanglingLayers(vw[:npure], wires=range(NQ))
        for i in range(NR):
            for q in range(NQ): qml.RY(x[q % len(x)] * rw[i, q, 0] + rw[i, q, 1], wires=q); qml.RZ(x[(q + 1) % len(x)] * rw[i, q, 2], wires=q)
            qml.StronglyEntanglingLayers(vw[npure + i:npure + i + 1], wires=range(NQ))
        return [qml.expval(qml.PauliZ(q)) for q in range(NQ)]
    def prep(X):
        n_feat = 64; sc = MinMaxScaler(feature_range=(0.01, 0.99))
        if X.shape[1] < n_feat: X = np.hstack([X, np.zeros((X.shape[0], n_feat - X.shape[1]))])
        else: X = X[:, :n_feat]
        return sc, X
    R = {}
    for cen, d in CEN.items():
        scf, _ = prep(d["X_train"]); Xtr = scf.fit_transform(prep(d["X_train"])[1]); Xte = scf.transform(prep(d["X_test"])[1])
        ytr, yte = d["y_train"], d["y_test"]; ylog = np.log1p(ytr).astype(float); mu, sg = ylog.mean(), ylog.std() + 1e-8; yn = (ylog - mu) / sg
        P = np.zeros((5, len(Xte)))
        for b in range(5):
            rng = np.random.RandomState(SEED + b); idx = rng.choice(len(Xtr), len(Xtr), True)
            Xb = pnp.array(Xtr[idx], requires_grad=False); yb = pnp.array(yn[idx], requires_grad=False)
            vw = pnp.array(rng.uniform(-np.pi, np.pi, (NL, NQ, 3)), requires_grad=True)
            rw = pnp.array(rng.uniform(-0.5, 0.5, (NR, NQ, 3)), requires_grad=True)
            osc = pnp.array(rng.uniform(0.5, 1.5, NQ), requires_grad=True); obi = pnp.array(0.0, requires_grad=True)
            best = (np.inf, None); patc = 0
            for ep in range(EP):
                lr = 0.001 + 0.5 * (0.05 - 0.001) * (1 + np.cos(np.pi * ep / EP)); opt = qml.AdamOptimizer(lr)
                bi_ = rng.choice(len(Xb), min(16, len(Xb)), False); Xba, yba = Xb[bi_], yb[bi_]
                def cost(vw_, rw_, sc_, bi2_):
                    p = pnp.array([pnp.sum(pnp.array(vqr2(Xba[i], vw_, rw_)) * sc_) + bi2_ for i in range(len(Xba))]); return pnp.mean((p - yba) ** 2)
                (vw, rw, osc, obi), loss = opt.step_and_cost(cost, vw, rw, osc, obi); lv = float(loss)
                if lv < best[0]: best = (lv, (deepcopy(vw.numpy()), deepcopy(rw.numpy()), deepcopy(osc.numpy()), float(obi))); patc = 0
                else:
                    patc += 1
                    if patc >= 15: break
            vwb, rwb, scb, bib = best[1]; vwb = pnp.array(vwb, requires_grad=False); rwb = pnp.array(rwb, requires_grad=False); scb = pnp.array(scb, requires_grad=False)
            raw = np.array([float(pnp.sum(pnp.array(vqr2(pnp.array(Xte[i], requires_grad=False), vwb, rwb)) * scb) + bib) for i in range(len(Xte))])
            P[b] = np.maximum(np.expm1(raw * sg + mu), 0)
        R[cen] = _res(cen, yte, np.median(P, 0), P)
    return R


def run_vqr_angle(CEN):
    NQ, NL, EP, LR, SEED = 6, 4, 80, 0.01, 42
    dev = qml.device(DEVN, wires=NQ)
    @qml.qnode(dev)
    def vqr1(x, w):
        qml.AngleEmbedding(x[:NQ], wires=range(NQ), rotation="Y"); qml.StronglyEntanglingLayers(w, wires=range(NQ))
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(5))
    R = {}
    for cen, d in CEN.items():
        sc = MinMaxScaler(feature_range=(0.01, 3.14)); Xtr = sc.fit_transform(d["X_train"][:, :6]); Xte = sc.transform(d["X_test"][:, :6])
        ytr, yte = d["y_train"], d["y_test"]; ylog = np.log1p(ytr).astype(float); mu, sg = ylog.mean(), ylog.std() + 1e-8; yn = (ylog - mu) / sg
        P = np.zeros((5, len(Xte)))
        for b in range(5):
            rng = np.random.RandomState(SEED + b); idx = rng.choice(len(Xtr), len(Xtr), True)
            Xb = pnp.array(Xtr[idx], requires_grad=False); yb = pnp.array(yn[idx], requires_grad=False)
            W = pnp.array(rng.uniform(-np.pi, np.pi, (NL, NQ, 3)) * 0.01, requires_grad=True)
            scp = pnp.array(rng.uniform(0.5, 1.5), requires_grad=True); bi = pnp.array(0.0, requires_grad=True)
            opt = qml.AdamOptimizer(LR); best, bw, patc = np.inf, deepcopy(W.numpy()), 0
            for ep in range(EP):
                def cost(W_, sc_, bi_):
                    p = pnp.array([vqr1(Xb[i], W_) * sc_ + bi_ for i in range(len(Xb))]); return pnp.mean((p - yb) ** 2)
                (W, scp, bi), loss = opt.step_and_cost(cost, W, scp, bi); lv = float(loss)
                if lv < best: best, bw, patc = lv, deepcopy(W.numpy()), 0
                else:
                    patc += 1
                    if patc >= 15: break
            Wb = pnp.array(bw, requires_grad=False)
            raw = np.array([float(vqr1(pnp.array(Xte[i], requires_grad=False), Wb)) * float(scp) + float(bi) for i in range(len(Xte))])
            P[b] = np.maximum(np.expm1(raw * sg + mu), 0)
        R[cen] = _res(cen, yte, np.median(P, 0), P)
    return R


def run_qlstm(CEN, NF):
    W, NQ, NLAY = 4, 4, 1
    dev = qml.device("default.qubit", wires=NQ)
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def gate(inp, weights):
        qml.AngleEmbedding(inp[:NQ], wires=range(NQ), rotation="Y")
        qml.AngleEmbedding(inp[NQ:2 * NQ], wires=range(NQ), rotation="Z")
        qml.StronglyEntanglingLayers(weights, wires=range(NQ))
        return [qml.expval(qml.PauliZ(w)) for w in range(NQ)]
    def sig(z): return 1.0 / (1.0 + pnp.exp(-z))
    def initp(seed=0):
        rng = np.random.RandomState(seed); p = {}
        p["Win"] = pnp.array(rng.normal(0, 0.3, (NF, NQ)), requires_grad=True)
        for g in ["f", "i", "g", "o"]: p[g] = pnp.array(rng.uniform(-np.pi, np.pi, (NLAY, NQ, 3)), requires_grad=True)
        p["Wout"] = pnp.array(rng.normal(0, 0.3, (NQ,)), requires_grad=True); p["bout"] = pnp.array(0.0, requires_grad=True); return p
    def fwd(seq, p):
        h = pnp.zeros(NQ); c = pnp.zeros(NQ)
        for t in range(seq.shape[0]):
            xp = pnp.tanh(seq[t] @ p["Win"]); v = pnp.concatenate([h, xp])
            f = sig(pnp.stack(gate(v, p["f"]))); i = sig(pnp.stack(gate(v, p["i"])))
            g = pnp.tanh(pnp.stack(gate(v, p["g"]))); o = sig(pnp.stack(gate(v, p["o"])))
            c = f * c + i * g; h = o * pnp.tanh(c)
        return p["Wout"] @ h + p["bout"]
    def btr(Xn, y): return np.array([Xn[i - W + 1:i + 1] for i in range(W - 1, len(Xn))]), y[W - 1:]
    def bte(Xtr, Xte):
        ext = np.vstack([Xtr[-(W - 1):], Xte]) if W > 1 else Xte; return np.array([ext[i - W + 1:i + 1] for i in range(W - 1, len(ext))])
    def pred(seqs, p): return np.array([float(fwd(pnp.array(s), p)) for s in seqs])
    def cost(p, Xb, yb): return pnp.mean((pnp.stack([fwd(pnp.array(s), p) for s in Xb]) - yb) ** 2)
    def train1(seqs, yln, seed, ep):
        p = initp(seed); rng = np.random.RandomState(seed); idx = rng.choice(len(seqs), len(seqs), True); Xb, yb = seqs[idx], yln[idx]
        opt = qml.AdamOptimizer(0.05)
        for _ in range(ep): p, _ = opt.step_and_cost(lambda pp: cost(pp, Xb, yb), p)
        return p
    np.random.seed(42); R = {}
    for cen, d in CEN.items():
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtn, Xen = (Xtr - mu) / sd, (Xte - mu) / sd
        st, yst = btr(Xtn, ytr); se = bte(Xtn, Xen); ym, ys = np.log1p(yst).mean(), np.log1p(yst).std() + 1e-6; yln = (np.log1p(yst) - ym) / ys
        dn = lambda pn: np.maximum(np.expm1(pn * ys + ym), 0); P = np.zeros((3, len(se)))
        for b in range(3): P[b] = dn(pred(se, train1(st, yln, 42 + b, 30)))
        R[cen] = _res(cen, yte, np.median(P, 0), P)
    return R
