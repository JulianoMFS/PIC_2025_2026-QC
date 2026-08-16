"""
submissao_mosqlimate.py
Upload das previsões QML para a plataforma Mosqlimate via API.

Requer mosqlient: pip install mosqlient
A chave da API é lida da variável de ambiente MOSQLIMATE_API_KEY.
"""

import json
import csv
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# ── Configuração ─────────────────────────────────────────────────────────────

# Nunca versionar segredos: definir MOSQLIMATE_API_KEY no ambiente.
API_KEY    = os.environ.get("MOSQLIMATE_API_KEY", "")
REPO       = "JulianoSilva/3rd_imdc_ceub_qml"   # ajustar ao usuário GitHub real
BASE_URL   = "https://api.mosqlimate.org"
SUBMISSOES = Path(__file__).parent.parent / "data" / "submissions" / "dengue_uf"


# ── Cliente HTTP leve (sem requests/pandas/numpy) ────────────────────────────

def _api_post(endpoint: str, payload: dict, tentativas: int = 3) -> dict:
    url  = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Token {API_KEY}",
        "Content-Type":  "application/json",
        "X-UID-Key":     API_KEY,
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    for t in range(1, tentativas + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            if e.code == 429:
                print(f"Rate limit — aguardando 10s (tentativa {t})")
                time.sleep(10)
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
        except Exception as e:
            if t == tentativas:
                raise
            time.sleep(5 * t)
    return {}


def _api_get(endpoint: str) -> dict:
    url  = f"{BASE_URL}{endpoint}"
    req  = urllib.request.Request(url, headers={"X-UID-Key": API_KEY})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


# ── Ler CSV de submissão ─────────────────────────────────────────────────────

def ler_csv_previsao(path: Path) -> list:
    """Lê o CSV de previsão e retorna lista de dicts prontos para a API."""
    registros = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            registros.append({
                "date":     row["date"],
                "pred":     float(row["pred"]),
                "lower_50": float(row["lower_50"]),
                "upper_50": float(row["upper_50"]),
                "lower_80": float(row["lower_80"]),
                "upper_80": float(row["upper_80"]),
                "lower_90": float(row["lower_90"]),
                "upper_90": float(row["upper_90"]),
                "lower_95": float(row["lower_95"]),
                "upper_95": float(row["upper_95"]),
            })
    return registros


# ── Obter hash do commit atual ───────────────────────────────────────────────

def _obter_commit() -> str:
    """Tenta ler o hash do commit Git mais recente. Retorna placeholder se falhar."""
    try:
        git_head = Path(__file__).parent.parent / ".git" / "HEAD"
        if not git_head.exists():
            return "abc123placeholder"
        head_content = git_head.read_text().strip()
        if head_content.startswith("ref: "):
            ref = head_content.replace("ref: ", "")
            ref_path = Path(__file__).parent.parent / ".git" / ref
            if ref_path.exists():
                return ref_path.read_text().strip()
        return head_content  # detached HEAD
    except Exception:
        return "abc123placeholder"


# ── Submeter previsão ────────────────────────────────────────────────────────

def submeter_previsao(rodada: str = "final", geocodigo: int = 53,
                      dry_run: bool = False) -> dict:
    """
    Envia a previsão do DF ao Mosqlimate via POST /api/registry/predictions/.

    Args:
        rodada:    "final" | "validation_1" | ... | "validation_4"
        geocodigo: geocódigo IBGE da UF (53 = DF)
        dry_run:   se True, mostra payload sem enviar
    """
    csv_path = SUBMISSOES / rodada / f"{geocodigo}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {csv_path}\n"
            "Execute primeiro: python src/predict.py --rodada all"
        )

    previsao = ler_csv_previsao(csv_path)
    commit   = _obter_commit()

    payload = {
        "repository":       REPO,
        "disease":          "A90",
        "description": (
            f"QML — Variational Quantum Regressor (PennyLane, data re-uploading) "
            f"para dengue no Distrito Federal (DF). "
            f"Rodada: {rodada}. "
            f"Features: casos_est (lags 1-4), Rt, p_rt1, receptivo, "
            f"transmissao, tempmed, umidmed, sazonalidade SE_sin/cos. "
            f"Incerteza via bootstrap log-normal (B=5 replicas). "
            f"Projeto PIBIT — CEUB, 2026."
        ),
        "commit":           commit,
        "case_definition":  "probable",
        "published":        True,
        "adm_level":        1,
        "adm_0":            "BRA",
        "adm_1":            geocodigo,
        "adm_2":            None,
        "adm_3":            None,
        "prediction":       previsao,
    }

    if dry_run:
        print("DRY RUN — payload que seria enviado:")
        print(json.dumps({k: v for k, v in payload.items() if k != "prediction"},
                         ensure_ascii=False, indent=2))
        print(f"prediction: {len(previsao)} linhas")
        print(f"Primeira: {previsao[0]}")
        print(f"Última:   {previsao[-1]}")
        return {}

    print(f"Enviando previsão ({len(previsao)} semanas) para rodada '{rodada}'...")
    resultado = _api_post("/api/registry/predictions/", payload)

    print(f"Previsão submetida com sucesso!")
    print(f"  ID:       {resultado.get('id')}")
    print(f"  Modelo:   {resultado.get('model', {}).get('repository', REPO)}")
    print(f"  Período:  {resultado.get('start')} a {resultado.get('end')}")
    print(f"  Commit:   {commit[:8]}...")

    # Salva comprovante
    comprovante = Path(__file__).parent.parent / "data" / f"submissao_{rodada}_{geocodigo}.json"
    comprovante.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Comprovante salvo: {comprovante.name}")

    return resultado


def verificar_modelo_registrado() -> bool:
    """Verifica se o modelo já está registrado na plataforma."""
    try:
        resp = _api_get(f"/api/registry/models/?repository={REPO}")
        models = resp.get("items", [])
        if models:
            m = models[0]
            print(f"Modelo registrado: {m.get('repository')}")
            print(f"  ID: {m.get('id')}")
            print(f"  Categoria: {m.get('category')}")
            print(f"  IMDC: {m.get('imdc_year')}")
            print(f"  Previsões: {m.get('predictions_count', 0)}")
            return True
        print(f"Modelo '{REPO}' NÃO está registrado.")
        print("Acesse https://api.mosqlimate.org e registre o modelo manualmente.")
        return False
    except Exception as e:
        print(f"Erro ao verificar: {e}")
        return False


# ── Tentativa de submissão com mosqlient (se disponível) ────────────────────

def submeter_com_mosqlient(rodada: str = "final", geocodigo: int = 53) -> bool:
    """
    Versão usando a biblioteca mosqlient (requer pip install mosqlient).
    Mais robusta que a versão urllib — preferir quando disponível.
    """
    try:
        import pandas as pd
        from mosqlient import upload_prediction, validate_prediction

        csv_path = SUBMISSOES / rodada / f"{geocodigo}.csv"
        pred_df  = pd.read_csv(csv_path)
        commit   = _obter_commit()

        params = {
            "api_key":         API_KEY,
            "repository":      REPO,
            "disease":         "A90",
            "description":     f"QML VQR — PIBIT CEUB — rodada {rodada}",
            "commit":          commit,
            "case_definition": "probable",
            "published":       True,
            "adm_level":       1,
            "adm_0":           "BRA",
            "adm_1":           geocodigo,
            "prediction":      pred_df,
        }

        print("Validando com mosqlient...")
        validate_prediction(**params)

        print("Enviando...")
        resultado = upload_prediction(**params)
        print(f"OK! ID: {resultado.id}")
        return True

    except ImportError:
        print("mosqlient não instalado. Usando cliente urllib.")
        submeter_previsao(rodada=rodada, geocodigo=geocodigo)
        return True
    except Exception as e:
        print(f"Erro com mosqlient: {e}")
        return False


# ── Instruções de registro ───────────────────────────────────────────────────

INSTRUCOES_REGISTRO = """
╔══════════════════════════════════════════════════════════════════════╗
║           COMO REGISTRAR O MODELO NO MOSQLIMATE                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. Acesse https://api.mosqlimate.org                              ║
║  2. Faça login com sua conta GitHub (JulianoSilva)                ║
║  3. Clique em "Register Model"                                     ║
║  4. Preencha os campos:                                            ║
║     • Repository: JulianoSilva/3rd_imdc_ceub_qml                  ║
║     • Description: VQR para dengue no DF — PIBIT CEUB 2026       ║
║     • Category: spatio_temporal_quantitative                       ║
║     • Time resolution: week                                        ║
║     • Disease: A90 (Dengue)                                        ║
║     • ADM level: 1 (estadual/UF)                                  ║
║     • IMDC year: 2026                                              ║
║  5. Clique em "Submit"                                             ║
║                                                                    ║
║  APÓS O REGISTRO: execute este script para enviar as previsões.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Submissão ao Mosqlimate IMDC")
    parser.add_argument("--rodada", default="final",
                        choices=["final", "validation_1", "validation_2",
                                 "validation_3", "validation_4", "all"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra payload sem enviar")
    parser.add_argument("--verificar", action="store_true",
                        help="Verifica se o modelo está registrado")
    parser.add_argument("--instrucoes", action="store_true",
                        help="Mostra como registrar o modelo")
    args = parser.parse_args()

    if args.instrucoes:
        print(INSTRUCOES_REGISTRO)
    elif args.verificar:
        verificar_modelo_registrado()
    elif args.rodada == "all":
        for rodada in ["validation_1", "validation_2", "validation_3", "validation_4", "final"]:
            print(f"\n{'='*60}")
            submeter_previsao(rodada=rodada, dry_run=args.dry_run)
            time.sleep(2)
    else:
        submeter_previsao(rodada=args.rodada, dry_run=args.dry_run)
