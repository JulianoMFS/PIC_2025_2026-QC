"""
predict.py
Gerador de previsões probabilísticas para submissão ao Mosqlimate IMDC 2026.

Gera 46 semanas de previsão (SE41/2026 a SE40/2027) com 9 quantis,
sem numpy/pandas — puro Python + math.
"""

import json
import math
import csv
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from dados_mosqlimate import carregar_dados, preparar_serie

SAIDA_DIR = Path(__file__).parent.parent / "data" / "submissions" / "dengue_uf"


# ── Utilidades ──────────────────────────────────────────────────────────────

def _lognormal_params(mu: float, sigma: float):
    """Estima parâmetros log-normais a partir de mu e sigma da escala linear."""
    if mu <= 0:
        mu = 0.1
    cv2 = (sigma / mu) ** 2 if mu > 0 else 1.0
    mu_ln  = math.log(mu) - 0.5 * math.log(1 + cv2)
    sig_ln = math.sqrt(math.log(1 + cv2))
    return mu_ln, sig_ln


def _lognormal_quantil(p: float, mu_ln: float, sig_ln: float) -> float:
    """Quantil de uma distribuição log-normal via inversa da normal padrão (aproximação Beasley-Springer-Moro)."""
    z = _probit(p)
    return max(0.0, math.exp(mu_ln + sig_ln * z))


def _probit(p: float) -> float:
    """Aproximação racional da inversa da normal padrão (Abramowitz & Stegun 26.2.17)."""
    p = max(1e-6, min(1 - 1e-6, p))
    if p < 0.5:
        t = math.sqrt(-2 * math.log(p))
        sign = -1
    else:
        t = math.sqrt(-2 * math.log(1 - p))
        sign = 1
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    num = c[0] + c[1] * t + c[2] * t**2
    den = 1 + d[0] * t + d[1] * t**2 + d[2] * t**3
    return sign * (t - num / den)


def gerar_quantis(mediana: float, coef_var: float = 0.4) -> dict:
    """
    Gera os 9 quantis obrigatórios do IMDC a partir da mediana e CV.
    Usa distribuição log-normal (padrão InfoDengue/IMDC).
    """
    sigma = mediana * coef_var
    mu_ln, sig_ln = _lognormal_params(mediana, sigma)

    q_vals = {}
    for p in [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]:
        q_vals[p] = round(_lognormal_quantil(p, mu_ln, sig_ln), 2)

    return {
        "pred":      q_vals[0.50],
        "lower_50":  q_vals[0.25],
        "upper_50":  q_vals[0.75],
        "lower_80":  q_vals[0.10],
        "upper_80":  q_vals[0.90],
        "lower_90":  q_vals[0.05],
        "upper_90":  q_vals[0.95],
        "lower_95":  q_vals[0.025],
        "upper_95":  q_vals[0.975],
    }


# ── Modelo de previsão Fase 0 (RF-based rule, sem PennyLane) ────────────────

def prever_serie_classica(serie: dict, n_semanas: int = 46) -> list:
    """
    Previsão clássica baseada em Rt com decaimento exponencial.
    Substitui o VQR quando o ambiente PennyLane não está disponível.
    Retorna lista de medianas para as próximas n_semanas.
    """
    if not serie["casos_est"]:
        return [100.0] * n_semanas

    # Pega últimas 8 semanas para estimar tendência
    ult_casos = serie["casos_est"][-8:]
    ult_rt    = serie["rt"][-4:]
    rt_atual  = sum(ult_rt) / len(ult_rt)
    tend_rt   = ult_rt[-1] - ult_rt[0] if len(ult_rt) >= 2 else 0

    base    = ult_casos[-1]
    prevsao = []

    rt_i = rt_atual
    for i in range(n_semanas):
        # Rt decai em direção a 1.0 com constante de tempo ~4 semanas
        rt_i = 1.0 + (rt_atual - 1.0) * math.exp(-0.15 * i)
        # Sazonal: padrão real do DF — pico em fev/mar (SE 5-12)
        se_base = 41 + i
        se_mod  = ((se_base - 1) % 52) + 1
        sazonal = 1.0 + 0.5 * math.cos(2 * math.pi * (se_mod - 10) / 52)
        # Previsão
        casos_i = base * (rt_i ** (i / 4)) * sazonal
        prevsao.append(max(0.0, round(casos_i, 1)))

    return prevsao


def domingo_da_se(ano: int, se: int) -> datetime:
    """
    Retorna o domingo de início da SE (semana epidemiológica brasileira).
    SE 1 começa no domingo mais próximo de 1 de janeiro.
    weekday(): 0=segunda ... 6=domingo.
    """
    jan1 = datetime(ano, 1, 1)
    # Acha o domingo anterior (ou o próprio dia se for domingo)
    dias_ate_domingo = jan1.weekday() + 1  # +1 porque weekday 6=domingo mas precisamos recuar
    if jan1.weekday() == 6:
        dias_ate_domingo = 0
    inicio_se1 = jan1 - timedelta(days=dias_ate_domingo)
    return inicio_se1 + timedelta(weeks=se - 1)


# ── Gera CSV de submissão ────────────────────────────────────────────────────

def gerar_csv_submissao(geocodigo: int = 53,
                        rodada: str = "final",
                        usar_vqr: bool = False) -> Path:
    """
    Gera o CSV de previsão no formato exato do IMDC 2026.

    Colunas: date, pred, lower_50, upper_50, lower_80, upper_80,
             lower_90, upper_90, lower_95, upper_95

    Período: SE41/2026 (05/out/2026) → SE40/2027 (05/out/2027), 46 semanas.
    """
    # Carrega dados reais
    print("Carregando dados...")
    dados   = carregar_dados(usar_cache=True)
    serie   = preparar_serie(dados)

    if usar_vqr:
        print("AVISO: VQR requer conda activate qml_dengue. Usando modelo clássico.")

    print("Gerando previsões...")
    medianas = prever_serie_classica(serie, n_semanas=46)

    # CV cresce com o horizonte (incerteza aumenta com o tempo)
    linhas = []
    for i, mediana in enumerate(medianas):
        se_num  = 41 + i
        ano_se  = 2026 if se_num <= 52 else 2027
        se_real = se_num if se_num <= 52 else se_num - 52
        data_se = domingo_da_se(ano_se, se_real)

        # Coef. de variação aumenta de 0.3 a 0.8 ao longo das 46 semanas
        cv = 0.3 + (0.5 * i / 45)
        quantis = gerar_quantis(mediana, coef_var=cv)

        linhas.append({
            "date":     data_se.strftime("%Y-%m-%d"),
            **quantis,
        })

    # Salva CSV
    saida_path = SAIDA_DIR / rodada / f"{geocodigo}.csv"
    saida_path.parent.mkdir(parents=True, exist_ok=True)

    colunas = ["date", "pred", "lower_50", "upper_50",
               "lower_80", "upper_80", "lower_90", "upper_90",
               "lower_95", "upper_95"]

    with open(saida_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(linhas)

    print(f"CSV salvo: {saida_path}")
    print(f"  Período: {linhas[0]['date']} a {linhas[-1]['date']}")
    print(f"  Mediana semana 1: {linhas[0]['pred']:.1f} casos")
    print(f"  IC 95% semana 1: [{linhas[0]['lower_95']:.1f}, {linhas[0]['upper_95']:.1f}]")
    print(f"  Mediana semana 46: {linhas[-1]['pred']:.1f} casos")

    return saida_path


def gerar_todos_splits(geocodigo: int = 53):
    """Gera CSVs para todas as 4 rodadas de validação + final."""
    rodadas = ["validation_1", "validation_2", "validation_3", "validation_4", "final"]
    for rodada in rodadas:
        print(f"\n--- Rodada: {rodada} ---")
        gerar_csv_submissao(geocodigo=geocodigo, rodada=rodada)


def validar_csv(path: Path) -> bool:
    """Verifica se o CSV gerado está em conformidade com o schema IMDC."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        erros  = []
        for i, row in enumerate(reader, 1):
            try:
                vals = {k: float(v) for k, v in row.items() if k != "date"}
                # Verifica aninhamento obrigatório
                if not (vals["lower_95"] <= vals["lower_90"] <= vals["lower_80"]
                        <= vals["lower_50"] <= vals["pred"]
                        <= vals["upper_50"] <= vals["upper_80"]
                        <= vals["upper_90"] <= vals["upper_95"]):
                    erros.append(f"Linha {i}: quantis fora de ordem")
                # Verifica não-negatividade
                if any(v < 0 for v in vals.values()):
                    erros.append(f"Linha {i}: valor negativo")
            except Exception as e:
                erros.append(f"Linha {i}: {e}")

    if erros:
        print(f"ERROS DE VALIDAÇÃO ({len(erros)}):")
        for e in erros[:10]:
            print(f"  {e}")
        return False

    print(f"CSV validado OK — {i} semanas, todos os quantis em ordem.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de previsões IMDC")
    parser.add_argument("--geocode", type=int, default=53, help="Geocódigo da UF (53=DF)")
    parser.add_argument("--rodada",  type=str, default="final",
                        choices=["final", "validation_1", "validation_2",
                                 "validation_3", "validation_4", "all"])
    parser.add_argument("--validar", action="store_true", help="Valida o CSV após gerar")
    args = parser.parse_args()

    if args.rodada == "all":
        gerar_todos_splits(args.geocode)
    else:
        path = gerar_csv_submissao(args.geocode, args.rodada)
        if args.validar:
            validar_csv(path)
