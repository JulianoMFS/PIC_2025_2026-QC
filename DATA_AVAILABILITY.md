# Disponibilidade de dados e código

## Dados

Todos os dados são **públicos e agregados**, sem identificação individual, provenientes do
sistema **InfoDengue–Mosqlimate**.

| Conjunto | Uso | Onde está | Origem |
|---|---|---|---|
| `data/dados_dengue_df_real.json` | Estudo principal (C1/C2/C3) — 183 SE, 30/01/2022 a 29/06/2025 | **Versionado** (leve) | API Mosqlimate (via `urllib`) |
| `notebooks/data/dengue.csv.gz` | QAE — série longa do DF (2010–2025) | **Não versionado** (23 MB) | FTP InfoDengue `data_sprint_2025` |
| `notebooks/data/climate.csv.gz` | QAE — variáveis climáticas | **Não versionado** (281 MB > limite do GitHub) | FTP InfoDengue `data_sprint_2025` |

Para obter os arquivos grandes:

```bash
python scripts/baixar_dados_infodengue.py
```

O script imprime o **SHA-256** de cada arquivo — registre-o abaixo para fixar o snapshot usado.

- Data de extração: **[preencher]**
- `dengue.csv.gz`  SHA-256: **[preencher após rodar o script]**
- `climate.csv.gz` SHA-256: **[preencher após rodar o script]**

## Código

O código, os arquivos de configuração (`environment.yml`, `requirements.txt`), os resultados
intermediários (`**/resultados/*.json`) e as figuras encontram-se neste repositório.
Versão correspondente ao relatório: **[commit/tag/DOI]**.

## Aspectos éticos

O estudo utilizou exclusivamente dados secundários, agregados e sem identificação individual,
enquadrando-se nas normas institucionais do CEUB para uso de dados públicos agregados.

## Segredos

Nenhuma credencial é versionada. Chaves de API (Mosqlimate, IBM Quantum) são lidas de
variáveis de ambiente — ver `.env.example`.

> **Aviso de segurança:** chaves que estiveram expostas em versões anteriores destes arquivos
> (token IBM Quantum e API Key do Mosqlimate) devem ser **revogadas e regeradas** nos
> respectivos painéis, pois podem permanecer no histórico local até a limpeza do repositório.
