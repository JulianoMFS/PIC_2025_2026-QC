# Dengue-DF: Benchmark Clássico × Quântico

**Benchmark de modelos clássicos e quânticos para previsão probabilística e detecção de mudanças de regime da dengue no Distrito Federal.**

Projeto de Iniciação Científica (PIBIT/CEUB, 2025-2026). Compara modelos clássicos, quânticos e híbridos na previsão semanal de casos de dengue no DF, com validação temporal reprodutível, e explora um *Quantum Autoencoder* (QAE) para detecção antecipada de mudanças de regime. Parte do pipeline foi executada em hardware quântico real (IBM Quantum).

> ⚠️ **Achados em resumo:** no escopo deste benchmark, os modelos baseados em árvores apresentaram o melhor desempenho na previsão de magnitude; os regressores quânticos **não** demonstraram vantagem; e o QAE produziu um sinal antecipatório relevante no surto de 2024 — resultado **exploratório**, a validar externamente. A execução em hardware é **prova de exequibilidade técnica**, não de robustez quântica.

---

## Estrutura

```
.
├── src/                      # Pipeline de dados e features (Mosqlimate/InfoDengue)
│   ├── dados_mosqlimate.py         # coleta via API (urllib)
│   ├── feature_engineering.py      # construção de features + splits temporais
│   ├── predict.py / submissao_mosqlimate.py
├── utils_qml.py              # métricas (WIS), plots, validação de pipeline
├── notebooks/                # ESTUDO PRINCIPAL (cenários C1/C2/C3)
│   ├── Fase0a_RF · Fase0b_XGBoost           # clássicos
│   ├── Fase1_VQR_AngleEmbedding · Fase2_VQR_ReUploading
│   ├── Fase3_QSVM · Fase4a_QRC · Fase4b_QLSTM · Fase4c_LSTM
│   └── Fase5_Dengue_QAE_AnomalyDetector     # detector de anomalia (QAE)
├── ensembles/                # RF+XGBoost, regime QSVM→RF, SARIMA baseline
├── hardware/                 # Execução em QPU IBM (VQR)
├── estudo_IMDC_janelas/      # ESTUDO-SATÉLITE: réplica nas janelas oficiais do 2º IMDC
│   ├── notebooks/ (01–05) · modelos_imdc.py · imdc_splits.py · resultados/
├── figuras/                  # Figuras do relatório
├── scripts/baixar_dados_infodengue.py   # baixa os dados brutos grandes (FTP)
├── environment.yml · requirements.txt   # ambiente reprodutível
├── DATA_AVAILABILITY.md · CITATION.cff · LICENSE
```

## Dois estudos

- **Estudo principal** (`notebooks/`, `ensembles/`, `hardware/`): validação temporal com **janela expansiva** e previsão *rolling one-step-ahead*. Três cenários — **C1** (rotina), **C2** (transição para o surto de 2024) e **C3** (pós-surto) — com testes de 143/91/54 semanas até 29/06/2025.
- **Estudo-satélite** (`estudo_IMDC_janelas/`): reprodução parcial do protocolo da 2ª edição do *InfoDengue–Mosqlimate Dengue Challenge*, com as janelas oficiais dos três testes retrospectivos.

## Instalação

```bash
conda env create -f environment.yml
conda activate qml_dengue
# ou: pip install -r requirements.txt
```

Python 3.10 · PennyLane **0.40.0** (estudo principal) / 0.42.3 (hardware e satélite IMDC) · Qiskit 1.2 · scikit-learn 1.7 · XGBoost 3.2.

## Credenciais (variáveis de ambiente)

Nenhum segredo é versionado. Copie `.env.example` para `.env` e preencha:

```bash
MOSQLIMATE_API_KEY=usuario:uuid      # coleta de dados
IBM_QUANTUM_TOKEN=...                 # apenas notebooks de hardware
IBM_QUANTUM_INSTANCE=...
```

## Dados

Dados públicos e agregados do InfoDengue–Mosqlimate. O JSON do estudo principal já está
versionado (`data/`). Os arquivos brutos grandes (série 2010–2025) são baixados à parte:

```bash
python scripts/baixar_dados_infodengue.py
```

Ver [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) para checksums e detalhes.

## Como reproduzir

1. Criar o ambiente e definir as variáveis de ambiente.
2. `python scripts/baixar_dados_infodengue.py` (para os notebooks do QAE).
3. Executar os notebooks de `notebooks/` (kernel `qml_dengue`). Cada fase salva um
   `*_resultados.json` na sua pasta.
4. Ensembles e SARIMA em `ensembles/`; comparação IMDC em `estudo_IMDC_janelas/`.
5. Hardware (opcional): `hardware/Fase_IBM_Hardware_VQR.ipynb` — valide primeiro no
   simulador/ruído; só então defina `RUN_ON_HARDWARE = True`.

## Uso de inteligência artificial

O desenvolvimento contou com apoio de IA (assistência de programação, reorganização de código,
geração de componentes auxiliares e revisão de texto), sob validação e responsabilidade autoral
dos autores. Detalhes na seção de uso de IA do relatório final.

## Como citar

Ver [CITATION.cff](CITATION.cff).

## Licença

Código sob [Licença MIT](LICENSE). Dados pertencem ao InfoDengue–Mosqlimate, sob os termos da fonte.
