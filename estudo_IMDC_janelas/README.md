# Estudo IMDC — Replicação do benchmark com as janelas dos Testes 1/2/3 do 2º IMDC

Reproduz **integralmente** o benchmark clássico × quântico de previsão de dengue no Distrito Federal
(estudo original em `../notebooks/Fase*.ipynb`), porém trocando as janelas de validação C1/C2/C3 pelas
**janelas oficiais do 2º Infodengue-Mosqlimate Dengue Challenge (IMDC)**, para comparação direta com as
17 equipes que submeteram modelos para o DF.

- **Alvo:** `casos_est` (InfoDengue) — mesmo alvo do estudo original.
- **Métrica primária:** WIS (Weighted Interval Score).
- **Modelos:** os mesmos 11 (RF, XGBoost, LSTM, SARIMA, Ensemble RF+XGB; QRC, QSVM, VQR-ReUp,
  VQR-Angle, QLSTM; Ensemble por regime QSVM→RF).

## Janelas de validação (DF)

| Cenário | Janela | Semanas | Treino | Característica |
|---|---|---|---|---|
| **T1** | 2022-10-09 → 2023-10-01 | 52 | 36 sem | temporada típica |
| **T2** | 2023-10-08 → 2024-09-29 | 52 | 88 sem | **surto recorde 2024** (pico 25.714) |
| **T3** | 2024-10-06 → 2025-09-28 | 52 | 140 sem | pós-surto |

Treino **expansivo** desde jan/2022; cada teste é uma temporada completa. Janelas extraídas do arquivo
oficial `preds_2nd_sprint_update.csv` do repositório do 2º IMDC.

## Proveniência — o que é REUSO e o que é NOVO

### Reusado do estudo original (sem alteração)
- **`feature_engineering.py`** (`../src/`) — `construir_features`: mesmas 13 features.
- **`utils_qml.py`** (`../`) — `metricas` e `calcular_wis`: mesmo cálculo de R² e WIS.
- **`dados_mosqlimate.py`** (`../src/`) — coleta na API InfoDengue/Mosqlimate (usada para estender a
  série até out/2025 e cobrir o Teste 3 completo).
- **Código dos 11 modelos** — copiado *verbatim* dos scripts validados nesta sessão, que reproduzem
  fielmente os notebooks `Fase*.ipynb` originais (R² conferido contra os JSONs salvos do estudo original).

### Novo (escrito para este estudo)
- **`imdc_splits.py`** — define apenas as janelas T1/T2/T3 e monta os cenários (treino expansivo).
- **`modelos_imdc.py`** — os 11 modelos **parametrizados** para receber cenários arbitrários (em vez de
  fixar C1/C2/C3). A lógica interna de cada modelo é idêntica à original.
- **`run_estudo_imdc.py`** — runner (`python run_estudo_imdc.py rapidos|lentos|todos`).
- **`build_doc_imdc.py`** — gera o documento `.docx` (tabelas + figuras, data-driven).
- **`notebooks/01..04`** — "casca fina": importam `modelos_imdc` e apenas executam/exibem; não recontêm a
  lógica dos modelos.

> **Nota de fidelidade:** as primeiras versões de QSVM, ensemble-regime e VQR-Re-up em `modelos_imdc.py`
> foram reescritas de memória e divergiram do original; foram **substituídas pelo código exato** dos
> scripts validados. Hoje os 11 modelos são idênticos aos do primeiro estudo — a única variável é a janela.

## Diferença metodológica frente ao IMDC (importante para a comparação)

Este estudo (herdado do original) prevê **um passo à frente** usando os *lags* reais de cada semana,
enquanto as equipes do IMDC preveem a **temporada inteira** a partir de um único ponto de origem, sem
dados futuros. Isso favorece este estudo nos regimes estáveis (T1, T3) e reduz a vantagem no surto (T2).
A comparação de WIS deve ser lida com essa ressalva.

## Estrutura da pasta

```
estudo_IMDC_janelas/
├── data/dados_dengue_df_2022_2025out.json   série DF 2022→out/2025 (197 semanas)
├── imdc_splits.py            janelas T1/T2/T3
├── modelos_imdc.py           os 11 modelos parametrizados
├── run_estudo_imdc.py        runner
├── build_doc_imdc.py         gerador do documento .docx
├── notebooks/                01_dados_e_janelas · 02_modelos_classicos · 03_modelos_quanticos · 04_comparacao_IMDC
├── figuras/                  <modelo>_pred_vs_obs.png + sidecar _plotdata.npz
├── resultados/               <modelo>_resultados.json (R²/WIS/RMSE/MAE por cenário)
└── imdc_ref/                 metrics_wis.csv e demais fontes oficiais do 2º IMDC
```

## Reprodução

```bash
# kernel/conda: qml_dengue
python run_estudo_imdc.py todos        # roda os 11 modelos (quânticos são lentos, ~2h)
python build_doc_imdc.py               # gera o documento .docx
# notebooks: usar o kernel "qml_dengue (conda)"
```
