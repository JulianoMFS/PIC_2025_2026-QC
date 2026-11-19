# Regras para agentes neste repositório

Projeto de Iniciação Científica (PIBITI): benchmark clássico × quântico para previsão
semanal de dengue no DF. Os resultados já estão em relatório; preservá-los é prioridade.

## Ambiente e testes
- Python 3.10. Para testes, instale `requirements-ci.txt` (não precisa de PennyLane/Qiskit).
- Rode `python -m pytest` antes de abrir ou atualizar qualquer PR. PR com teste falhando não sai.
- Toda mudança de comportamento vem com um teste em `tests/` que falharia sem ela.

## O que não tocar sem pedido explícito na issue
- `golden/`, `*/resultados/`, `notebooks/*.json`, `ensembles/*.json`, `hardware/` — resultados de referência.
- `*.ipynb`, figuras (`*.png`, `*_plotdata.npz`) e `data/`.
- `environment.yml` e `requirements.txt` (versões congeladas para reprodutibilidade).

## Contratos que os notebooks assumem
- `construir_features` devolve 13 features nesta ordem fixa (ver docstring). Não reordene.
- Splits são cronológicos: nenhuma informação posterior à data de corte pode entrar no treino,
  inclusive em estatísticas de normalização.
- Métrica primária: WIS (`utils_qml.calcular_wis`), menor é melhor.

## Segurança e dados
- Nunca versione `.env`, tokens (`MOSQLIMATE_API_KEY`, `IBM_QUANTUM_TOKEN`) ou chaves.
- O repositório só contém séries agregadas por semana. Nenhum dado identificado de pessoa.

## Fluxo da fábrica
- Branches dos agentes: `fabrica/issue-<n>`. Título de PR: `[fabrica] <resumo>`.
- Agentes nunca fazem merge nem aprovam PR. Isso é do mantenedor.
- Em caso de dúvida sobre escopo, comente na issue e pare.
