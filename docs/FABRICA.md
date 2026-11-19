# Piloto da fábrica de software (caminho 2: Claude Code + GitHub Actions)

## Fluxo

```
Issue (template "Demanda para a fábrica", etiqueta `demanda`)
   │
   ▼  fabrica-triagem.yml ── agente comenta classificação, tamanho, plano e riscos
   │                          aplica `triada` ou `exige-humano`
   ▼  ★ DECISÃO HUMANA: você aplica `fabrica`
   │
   ▼  fabrica-implementacao.yml ── branch fabrica/issue-N, teste primeiro, pytest, PR
   │
   ├─ ci.yml ── pytest (portão objetivo)
   └─ fabrica-revisao.yml ── revisor independente (outro modelo, sem o contexto do implementador)
   │
   ▼  ★ DECISÃO HUMANA: você revisa, aprova e faz merge
```

## Configuração (uma vez, ~15 min)

1. **Instalar o Claude GitHub App** no repositório: <https://github.com/apps/claude>.
   Sem ele, o PR aberto pelo agente usa o `GITHUB_TOKEN` padrão, e PR criado por esse token
   não dispara a CI nem a revisão.
2. **Segredo** `ANTHROPIC_API_KEY` em *Settings → Secrets and variables → Actions*.
   (Alternativa com assinatura: gerar `CLAUDE_CODE_OAUTH_TOKEN` com `claude setup-token` e trocar
   o input `anthropic_api_key` por `claude_code_oauth_token` nos três workflows.)
3. **Etiquetas**: criar `demanda`, `triada`, `exige-humano` e `fabrica`:
   ```bash
   gh label create demanda      --color 0E8A16 --description "Entrada da fábrica"
   gh label create triada       --color 1D76DB --description "Triagem OK, aguardando decisão"
   gh label create exige-humano --color D93F0B --description "Fora do escopo dos agentes"
   gh label create fabrica      --color 5319E7 --description "Autoriza a implementação pelo agente"
   ```
4. **Proteção da `main`**: exigir PR, 1 aprovação e o check `CI / testes` verde.
   Assim o agente não tem como fazer merge, mesmo com `contents: write`.
5. Limitar quem pode aplicar `fabrica`: em repositório público, só quem tem permissão de
   *triage* ou acima consegue pôr etiquetas, o que já restringe a você.

## Métricas do piloto (2 semanas)

| Métrica | Como medir | Meta para continuar |
|---|---|---|
| PRs aceitos sem retrabalho | PR mesclado sem commit humano depois da abertura | ≥ 50% |
| PRs aceitos com ajuste | mesclado após pedido de ajuste | registrar |
| PRs descartados | fechado sem merge | ≤ 30% |
| Concordância do revisor | veredito do agente revisor = sua decisão | ≥ 70% |
| Tempo issue → PR | `created_at` da issue até o PR | registrar |
| Custo por PR | painel de uso da API ÷ nº de PRs | registrar |

Volume sugerido: 6 a 10 issues pequenas (tamanho P ou M). Consulta rápida:
```bash
gh pr list --state all --search "[fabrica] in:title" --json number,title,state,mergedAt,createdAt
```

## Primeiras issues sugeridas

1. **Vazamento na normalização** (bom primeiro caso real). `construir_features` calcula média e
   desvio do z-score sobre a série inteira, antes do split, então o treino "vê" a escala do surto
   de 2024. Critério de aceite: teste que prova que as estatísticas usadas no treino dependem só
   de datas ≤ corte. Marcar "pode alterar resultados": a mudança mexe em números publicados, então
   a triagem deve mandar para `exige-humano` ou pedir que a correção venha atrás de um parâmetro
   opcional que mantém o comportamento atual por padrão.
2. Testes para `src/predict.py` (tamanho P/M, sem risco científico).
3. Remover arquivos de trabalho versionados (`notebooks/nbc*.log`, `estudo_IMDC_janelas/notebooks/_t.png`,
   `*_exec.log`, `run_lentos.log`) e cobri-los no `.gitignore`.
4. Docstrings e tipos em `src/dados_mosqlimate.py`, sem mudar comportamento.

## Limites e governança

- Os agentes rodam em runners do GitHub e chamam a API da Anthropic. Este repositório só tem
  séries agregadas e públicas; **não** usar este fluxo com código ou dados institucionais da SES-DF
  antes de validar com a área de TI/segurança.
- `golden/`, resultados, notebooks e dados estão fora do escopo padrão (ver `CLAUDE.md`).
