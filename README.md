# Controlando pelo berço: o efeito incremental da escola privada sobre P(nota ≥ 700)

Case técnico, análise dos microdados do ENEM 2023 para estimar o efeito de frequentar uma escola privada sobre a probabilidade de atingir média ≥ 700, controlando pelo contexto socioeconômico do aluno (renda, escolaridade parental, acesso digital).

## Sobre o projeto

A pergunta central é: **quanto a escola move a agulha além do que o berço já determinou?**

Para responder, o modelo separa o efeito da escola (`TP_DEPENDENCIA_ADM_ESC`) dos controles socioeconômicos (renda familiar, escolaridade dos pais, acesso a computador e internet) usando uma abordagem de modelagem sequencial, primeiro o SES baseline, depois adicionando as variáveis acionáveis e mede o delta de PR-AUC entre os dois blocos.

## Estrutura do projeto

```
case-inter-enem/
├── data/                      # Dados brutos (ver data/README.md)
├── notebooks/
│   └── 01_eda.ipynb           # EDA + visualizações + gera base_analitica.parquet
├── reports/
│   ├── metrics.json           # Métricas dos modelos e cenários
│   └── figs/                  # Visualizações geradas (PNG)
├── src/
│   ├── paths.py               # Constantes de caminho centralizadas
│   └── train.py               # Feature engineering, modelos, calibração, figuras
└── requirements.txt
```

## Como rodar

```bash
# 1. Clonar e criar ambiente
git clone <repo>
cd case-inter-enem
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Baixar os dados (ver data/README.md para instruções)

# 3. Rodar o EDA — gera reports/base_analitica.parquet
jupyter notebook notebooks/01_eda.ipynb

# 4. Rodar o treinamento — gera reports/metrics.json e reports/figs/
python src/train.py
```

## Dados

Os microdados do ENEM 2023 não estão no repositório. Veja [`data/README.md`](data/README.md) para instruções de download e organização dos arquivos.

## Resultados

- `reports/metrics.json` — PR-AUC, Brier Score, F1 e cenários narrativos (baseline SES vs. SES + escola)
- `reports/figs/` — 8 visualizações prontas para o deck executivo
