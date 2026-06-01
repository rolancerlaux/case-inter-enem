# Dados

Os arquivos de dados não estão versionados por causa do tamanho (> 1.6 GB). Siga os passos abaixo para reproduzir o ambiente completo.

## Fonte

Microdados do ENEM 2023 — disponibilizados pelo INEP:
https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem

## Arquivos necessários

Após baixar e descompactar, coloque os arquivos nas pastas indicadas:

```
data/
├── DADOS/
│   ├── MICRODADOS_ENEM_2023.csv      (~1.6 GB)  — notas + questionário socioeconômico
│   └── ITENS_PROVA_2023.csv          (~323 KB)  — metadados dos itens das provas
└── DICIONÁRIO/
    ├── Dicionário_Microdados_Enem_2023.ods
    └── Dicionário_Microdados_Enem_2023.xlsx
```

Os caminhos são centralizados em `src/paths.py` — se renomear os arquivos, atualize lá.

## Ordem de execução

1. Baixar e posicionar os dados conforme a estrutura acima
2. Rodar `notebooks/01_eda.ipynb` → gera `reports/base_analitica.parquet`
3. Rodar `python src/train.py` → gera `reports/metrics.json` e as figuras em `reports/figs/`
