# Controlando pelo berço: o efeito incremental da escola privada sobre P(nota ≥ 700)

Case técnico, análise dos microdados do ENEM 2023 para estimar o efeito de frequentar uma escola privada sobre a probabilidade de atingir média ≥ 700, controlando pelo contexto socioeconômico do aluno (renda, escolaridade parental, acesso digital).

## Sobre o projeto

A pergunta central é: **quanto a escola move a agulha além do que o berço já determinou?**

Para responder, dois blocos de modelo são treinados em sequência. Primeiro, uma **Regressão Logística (L2)** usando só as variáveis socioeconômicas (renda, escolaridade dos pais, acesso digital) esse baseline representa o quanto o contexto já explica. Depois, o mesmo modelo recebe o tipo de escola e dummies de UF; o delta de PR-AUC entre os dois blocos é o efeito incremental da escola depois de descontado o SES. Em cima disso, um **HistGradientBoosting** é treinado no bloco completo para maximizar a capacidade preditiva individual.

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

**Efeito médio da escola privada (AME): +5,44 pp** sobre P(nota ≥ 700), após controlar renda, escolaridade parental e acesso digital. O IC 95% é estreito (5,43–5,46 pp) não confundir precisão estatística com garantia causal; seleção não observada não está descartada.

O efeito é heterogêneo por SES:

| Perfil do aluno | Escola pública | Escola privada | Ganho |
|---|---|---|---|
| Baixo SES (renda ≤ R$1.320, mãe sem EM) | 0,64% | 1,68% | +1,04 pp |
| Médio SES (renda ~R$2.800, mãe com EM) | 1,15% | 3,00% | +1,85 pp |
| Alto SES (renda ~R$7.600, mãe com superior) | 4,67% | 11,48% | +6,81 pp |

O efeito absoluto cresce com a renda — a escola privada amplifica o que já existe, não equaliza.

**Hierarquia de importância (HGB):** renda familiar lidera com folga (importância 0,10). Escola privada aparece em 7º (0,013), atrás de escolaridade da mãe, do pai e acesso a computador. Isso quantifica o achado central: o berço pesa mais que a escola, mas a escola tem sinal próprio e mensurável após descontar o SES.

**Qualidade do modelo:** PR-AUC 0,32 no teste; lift de 3,76× no top 20%. O bloco de escola + UF adiciona +0,0117 de PR-AUC sobre o baseline SES.

Detalhes completos em [`reports/metrics.json`](reports/metrics.json) e visualizações em [`reports/figs/`](reports/figs/).
