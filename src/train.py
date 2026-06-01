"""
src/train.py

Treino e avaliação dos modelos de classificação para P(nota_media >= 700).

Approach:
  Bloco A: LR(SES apenas)          -> baseline PR-AUC
  Bloco B: LR(SES + escola + UF)   -> efeito incremental da escola privada
  HGB:     HistGradientBoosting     -> validacao performativa + importancia de features

Saidas:
  reports/metrics.json
  reports/figs/viz5_calibracao_lift.png
  reports/figs/viz6_features.png

Uso:
  python src/train.py
"""

import json
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)

from src.paths import REPORTS_DIR, FIGS_DIR

FIGS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 700
RANDOM_STATE = 42

plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
AZUL = '#1a4e8c'
LARANJA = '#e87722'


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA E FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

print("Carregando base analitica...")
df = pd.read_parquet(REPORTS_DIR / 'base_analitica.parquet')

# Filtro defensivo (base ja deve estar filtrada pelo notebook 01)
if 'IN_TREINEIRO' in df.columns:
    df = df[df['IN_TREINEIRO'] == 0].copy()
if 'TP_ST_CONCLUSAO' in df.columns:
    df = df[df['TP_ST_CONCLUSAO'].isin([1, 2])].copy()

print(f"  Base: {len(df):,} registros")

# --- Encoding ordinal SES ---
# Q006: renda familiar (A=sem renda ... Q=>20SM)
# Q001/Q002: escolaridade pai/mae (A=nunca estudou ... G=pos-grad; H=nao sei → NaN)
# Q024: computador em casa (A=nao; B=um; C=dois; D=tres; E=quatro ou mais)
# Q025: internet em casa (A=nao, B=sim)
Q_ORDINAL = {
    'Q001': {c: i for i, c in enumerate('ABCDEFG')},
    'Q002': {c: i for i, c in enumerate('ABCDEFG')},
    'Q006': {c: i for i, c in enumerate('ABCDEFGHIJKLMNOPQ')},
    'Q024': {c: i for i, c in enumerate('ABCDE')},
    'Q025': {'A': 0, 'B': 1},
}
for col, mapping in Q_ORDINAL.items():
    if col in df.columns:
        df[f'{col}_num'] = df[col].map(mapping)

SES_FEATURES = [f'{c}_num' for c in ['Q001', 'Q002', 'Q006', 'Q024', 'Q025'] if f'{c}_num' in df.columns]

# TP_COR_RACA → flag binaria "nao-branca" (0=nd/Branca/Amarela, 1=Preta/Parda/Indigena)
if 'TP_COR_RACA' in df.columns:
    df['nao_branca'] = (df['TP_COR_RACA'].isin([2, 3, 5])).astype(int)
    SES_FEATURES.append('nao_branca')

# TP_SEXO → binaria feminino
if 'TP_SEXO' in df.columns:
    df['feminino'] = (df['TP_SEXO'] == 'F').astype(int)
    SES_FEATURES.append('feminino')

# --- Flag escola privada ---
# Preferência: TP_ESCOLA (autodeclarada na inscrição — cobre egressos e concluintes sem nulls)
#   1=Não Respondeu, 2=Pública, 3=Privada
# Fallback: TP_DEPENDENCIA_ADM_ESC (dado administrativo — nulo para 100% dos egressos)
if 'TP_ESCOLA' in df.columns:
    df['escola_privada']  = (df['TP_ESCOLA'] == 3).astype(int)
    df['escola_nao_resp'] = (df['TP_ESCOLA'] == 1).astype(int)
    SES_FEATURES.append('escola_nao_resp')   # controle de não-resposta, não alavanca operacional
    ESCOLA_FEATURES = ['escola_privada']
elif 'TP_DEPENDENCIA_ADM_ESC' in df.columns:
    df['escola_privada'] = (df['TP_DEPENDENCIA_ADM_ESC'] == 4).astype(int)
    df['escola_federal']  = (df['TP_DEPENDENCIA_ADM_ESC'] == 1).astype(int)
    ESCOLA_FEATURES = ['escola_privada', 'escola_federal']
else:
    ESCOLA_FEATURES = []

# --- Features de política (acionáveis operacionalmente pela rede) ---
# concluindo_ano: aluno cursando o 3o ano em 2023 — alvo de preparação intensiva
# Q022_num: celular em casa — proxy de inclusão digital
# (eja removido: TP_ENSINO=3 não existe no dicionário ENEM 2024; valores válidos: 1=Regular, 2=Especial)
POLICY_FEATURES = []

if 'TP_ST_CONCLUSAO' in df.columns:
    df['concluindo_ano'] = (df['TP_ST_CONCLUSAO'] == 2).astype(int)
    POLICY_FEATURES.append('concluindo_ano')

Q022_MAP = {c: i for i, c in enumerate('ABCDE')}
if 'Q022' in df.columns:
    df['Q022_num'] = df['Q022'].map(Q022_MAP)
    POLICY_FEATURES.append('Q022_num')

# --- Dummies de UF (drop_first para evitar multicolinearidade) ---
if 'SG_UF_PROVA' in df.columns:
    uf_dummies = pd.get_dummies(df['SG_UF_PROVA'], prefix='uf', drop_first=True)
    df = pd.concat([df.reset_index(drop=True), uf_dummies.reset_index(drop=True)], axis=1)
    UF_FEATURES = list(uf_dummies.columns)
else:
    UF_FEATURES = []

ACTIONABLE_FEATURES = ESCOLA_FEATURES + POLICY_FEATURES  # controlavel pela rede
REGIONAL_FEATURES = UF_FEATURES                           # contexto geografico

# --- Remover missings no target e SES; imputar 0 nos acionaveis e regionais ---
df_clean = df.dropna(subset=['alta_performance'] + SES_FEATURES).copy()
for f in ACTIONABLE_FEATURES + REGIONAL_FEATURES:
    if f in df_clean.columns:
        df_clean[f] = df_clean[f].fillna(0)
df_clean = df_clean.reset_index(drop=True)

print(f"  Apos drop missings: {len(df_clean):,}")
print(f"  SES features ({len(SES_FEATURES)}): {SES_FEATURES}")
print(f"  Acionaveis: {ESCOLA_FEATURES} + {POLICY_FEATURES}")
print(f"  Regionais: {len(REGIONAL_FEATURES)} dummies UF")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT ESTRATIFICADO (salva indices para sensitivity check)
# ═══════════════════════════════════════════════════════════════════════════════

y = df_clean['alta_performance'].values
X_ses = df_clean[SES_FEATURES].values
X_full = df_clean[SES_FEATURES + ACTIONABLE_FEATURES + REGIONAL_FEATURES].values
indices = np.arange(len(df_clean))

(idx_train, idx_test,
 X_ses_train, X_ses_test,
 X_full_train, X_full_test,
 y_train, y_test) = train_test_split(
    indices, X_ses, X_full, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,
)
print(f"\nSplit: train={len(y_train):,} | test={len(y_test):,} | prevalencia={y_train.mean():.2%}")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {'pr_auc': 'average_precision', 'brier': 'neg_brier_score'}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BLOCO A — LR COM SES APENAS (BASELINE)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Bloco A] LR — SES apenas...")
lr_a = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, class_weight=None, max_iter=500, random_state=RANDOM_STATE)),
])
cv_a = cross_validate(lr_a, X_ses_train, y_train, cv=CV, scoring=SCORING)
lr_a.fit(X_ses_train, y_train)
prob_a_test = lr_a.predict_proba(X_ses_test)[:, 1]
pr_auc_a_test = average_precision_score(y_test, prob_a_test)
brier_a_test = brier_score_loss(y_test, prob_a_test)
print(f"  PR-AUC CV: {cv_a['test_pr_auc'].mean():.4f} +/- {cv_a['test_pr_auc'].std():.4f}")
print(f"  PR-AUC test: {pr_auc_a_test:.4f} | Brier: {brier_a_test:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BLOCO B — LR COM SES + ESCOLA + UF (EFEITO INCREMENTAL)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Bloco B] LR — SES + escola + UF...")
lr_b = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, class_weight=None, max_iter=500, random_state=RANDOM_STATE)),
])
cv_b = cross_validate(lr_b, X_full_train, y_train, cv=CV, scoring=SCORING)
lr_b.fit(X_full_train, y_train)
prob_b_test = lr_b.predict_proba(X_full_test)[:, 1]
pr_auc_b_test = average_precision_score(y_test, prob_b_test)
brier_b_test = brier_score_loss(y_test, prob_b_test)
delta_pr_auc = pr_auc_b_test - pr_auc_a_test
print(f"  PR-AUC CV: {cv_b['test_pr_auc'].mean():.4f} +/- {cv_b['test_pr_auc'].std():.4f}")
print(f"  PR-AUC test: {pr_auc_b_test:.4f} | Brier: {brier_b_test:.4f}")
print(f"  Delta PR-AUC (B - A): +{delta_pr_auc:.4f}")

# Coeficiente parcial de escola_privada (escala padronizada)
feature_names_full = SES_FEATURES + ACTIONABLE_FEATURES + REGIONAL_FEATURES
coef_b = pd.Series(lr_b.named_steps['clf'].coef_[0], index=feature_names_full)
coef_escola_privada = float(coef_b.get('escola_privada', np.nan)) if 'escola_privada' in coef_b.index else None
if coef_escola_privada is not None:
    print(f"  Coef. parcial escola_privada (padronizado): {coef_escola_privada:+.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HGB — CAMADA PERFORMATIVA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[HGB] HistGradientBoostingClassifier...")
hgb = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=RANDOM_STATE,
)
cv_hgb = cross_validate(hgb, X_full_train, y_train, cv=CV, scoring=SCORING)
hgb.fit(X_full_train, y_train)
prob_hgb_test = hgb.predict_proba(X_full_test)[:, 1]
pr_auc_hgb_test = average_precision_score(y_test, prob_hgb_test)
brier_hgb_test = brier_score_loss(y_test, prob_hgb_test)
print(f"  PR-AUC CV: {cv_hgb['test_pr_auc'].mean():.4f} +/- {cv_hgb['test_pr_auc'].std():.4f}")
print(f"  PR-AUC test: {pr_auc_hgb_test:.4f} | Brier: {brier_hgb_test:.4f}")

# Calibracao condicional
if brier_hgb_test > 0.15:
    print("  Brier > 0.15 — aplicando calibracao isotonica...")
    hgb_cal = CalibratedClassifierCV(hgb, method='isotonic', cv=5)
    hgb_cal.fit(X_full_train, y_train)        # k-fold interno no treino — sem data leakage
    prob_hgb_test = hgb_cal.predict_proba(X_full_test)[:, 1]
    brier_hgb_test = brier_score_loss(y_test, prob_hgb_test)
    print(f"  Brier apos calibracao: {brier_hgb_test:.4f}")

# Threshold otimo por F1
precision_c, recall_c, thresholds_c = precision_recall_curve(y_test, prob_hgb_test)
f1_scores_c = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-9)
optimal_threshold = float(thresholds_c[np.argmax(f1_scores_c[:-1])])
print(f"  Threshold otimo (F1): {optimal_threshold:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LIFT NO TOP-20%
# ═══════════════════════════════════════════════════════════════════════════════

top20_cut = int(len(y_test) * 0.2)
top20_idx = np.argsort(prob_hgb_test)[::-1][:top20_cut]
lift_top20 = float(y_test[top20_idx].mean() / y_test.mean())
print(f"\nLift top-20%: {lift_top20:.2f}x a taxa base")


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. GUARD CONTRA COLAPSO DO MODELO
# ═══════════════════════════════════════════════════════════════════════════════

prob_std_hgb = float(prob_hgb_test.std())
y_pred_binary = (prob_hgb_test >= optimal_threshold).astype(int)
tp_hgb = int(((y_pred_binary == 1) & (y_test == 1)).sum())
fn_hgb = int(((y_pred_binary == 0) & (y_test == 1)).sum())
recall_minority_hgb = tp_hgb / max(tp_hgb + fn_hgb, 1)
predicted_positive_count_hgb = int(y_pred_binary.sum())

if prob_std_hgb < 0.01:
    print(f"  ALERTA: std(prob)={prob_std_hgb:.4f} — possivel colapso para classe majoritaria")
if recall_minority_hgb < 0.05:
    print(f"  ALERTA: recall_positivos={recall_minority_hgb:.2%} — modelo nao discrimina alta performance")

n_test = len(y_test)
pct_pred = predicted_positive_count_hgb / n_test
print(f"\n[Saude do modelo HGB]")
print(f"  std(prob): {prob_std_hgb:.4f} | recall positivos: {recall_minority_hgb:.2%}"
      f" | positivos preditos: {predicted_positive_count_hgb:,}/{n_test:,} ({pct_pred:.2%})")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

print("\nComputando permutation importance (HGB, 10 repeticoes)...")
perm = permutation_importance(
    hgb, X_full_test, y_test,
    n_repeats=10, random_state=RANDOM_STATE,
    scoring='average_precision',
)
FEATURE_PT = {
    'Q006_num': 'Renda familiar',
    'Q002_num': 'Escolaridade da mae',
    'Q001_num': 'Escolaridade do pai',
    'Q024_num': 'Computador em casa',
    'Q025_num': 'Acesso a internet',
    'Q022_num': 'Celular em casa',
    'nao_branca': 'Raca (nao-branca)',
    'feminino': 'Sexo (feminino)',
    'escola_privada': 'Escola privada (autodeclarada)',
    'escola_nao_resp': 'Escola nao declarada',
    'escola_federal': 'Escola federal',
    'concluindo_ano': 'Concluindo EM em 2023',
    'eja': 'EJA (jovens e adultos)',
}
perm_df = pd.DataFrame({
    'feature': feature_names_full,
    'importancia_media': perm.importances_mean,
    'importancia_std': perm.importances_std,
})
perm_df['feature_pt'] = perm_df['feature'].apply(
    lambda f: FEATURE_PT.get(f, f.replace('uf_', 'UF '))
)
perm_df = perm_df.sort_values('importancia_media', ascending=False)
print(perm_df[['feature_pt', 'importancia_media']].head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CALIBRAÇÃO lr_b + CENÁRIOS EMPÍRICOS + AME
# ═══════════════════════════════════════════════════════════════════════════════

# 8a. Calibração condicional (Brier > 0.05 após remoção de class_weight)
lr_b_for_scenarios = lr_b
calibration_applied_lr_b = False
brier_b_calibrated = None

print(f"\n[Calibração lr_b] Brier={brier_b_test:.4f}", end='')
if brier_b_test > 0.05:
    print(" > 0.05 — aplicando calibracao isotonica...")
    lr_b_cal = CalibratedClassifierCV(lr_b, method='isotonic', cv=5)
    lr_b_cal.fit(X_full_train, y_train)
    _prob_b_cal = lr_b_cal.predict_proba(X_full_test)[:, 1]
    brier_b_calibrated = round(float(brier_score_loss(y_test, _prob_b_cal)), 4)
    print(f"  Brier apos calibracao: {brier_b_calibrated}")
    lr_b_for_scenarios = lr_b_cal
    calibration_applied_lr_b = True
else:
    print(" — calibracao nao necessaria.")

# 8b. Perfis empíricos: percentis observados de Q006_num no conjunto de treino
_q006_train = df_clean.iloc[idx_train]['Q006_num'].dropna()
q006_p25 = int(_q006_train.quantile(0.25))
q006_p50 = int(_q006_train.quantile(0.50))
q006_p75 = int(_q006_train.quantile(0.75))
print(f"\nPercentis Q006_num (renda) no treino: P25={q006_p25}, P50={q006_p50}, P75={q006_p75}")
print("  (ordinal: A=0=sem renda, D=3~R$1.980, G=6~R$3.960, L=11~R$9.240, Q=16>R$20.040)")

def _ses_medians_at(q006_target: int) -> dict:
    """Mediana de cada feature SES entre alunos com Q006_num dentro de ±1 do alvo."""
    mask = df_clean['Q006_num'].between(q006_target - 1, q006_target + 1)
    result = {}
    for f in SES_FEATURES:
        if f in df_clean.columns and f != 'escola_nao_resp':
            val = df_clean.loc[mask, f].median()
            result[f] = int(round(float(val))) if not np.isnan(val) else 0
    return result

low_profile_base  = {**_ses_medians_at(q006_p25), 'Q006_num': q006_p25}
mid_profile_base  = {**_ses_medians_at(q006_p50), 'Q006_num': q006_p50}
high_profile_base = {**_ses_medians_at(q006_p75), 'Q006_num': q006_p75}

# Contagem de alunos reais com Q006_num exatamente no percentil (na base completa)
profile_counts = {}
for label, pval in [('low_ses', q006_p25), ('mid_ses', q006_p50), ('high_ses', q006_p75)]:
    n = int((df_clean['Q006_num'] == pval).sum())
    pct = float((df_clean['Q006_num'] == pval).mean())
    profile_counts[label] = {'q006_num': pval, 'n_alunos': n, 'pct_base': round(pct, 4)}
    print(f"  Perfil {label}: Q006_num={pval} → {n:,} alunos ({pct:.1%} da base)")

# 8c. Predição dos cenários
def predict_profile(profile: dict) -> float:
    """P(>=700) para perfil dado. Features ausentes = 0 (ref: masc, branca, UF-referencia)."""
    row = np.zeros(len(feature_names_full))
    for fname, val in profile.items():
        if fname in feature_names_full:
            row[feature_names_full.index(fname)] = val
    return float(lr_b_for_scenarios.predict_proba(row.reshape(1, -1))[0, 1])

scenarios = {
    'low_ses_publica':  predict_profile({**low_profile_base,  'escola_privada': 0}),
    'low_ses_privada':  predict_profile({**low_profile_base,  'escola_privada': 1}),
    'mid_ses_publica':  predict_profile({**mid_profile_base,  'escola_privada': 0}),
    'mid_ses_privada':  predict_profile({**mid_profile_base,  'escola_privada': 1}),
    'high_ses_publica': predict_profile({**high_profile_base, 'escola_privada': 0}),
    'high_ses_privada': predict_profile({**high_profile_base, 'escola_privada': 1}),
}

print("\nCenarios narrativos P(>=700) — perfis empiricos (P25/P50/P75 de Q006_num):")
for k, v in scenarios.items():
    print(f"  {k:25s}: {v:.1%}")
print("\nDelta publico → privado por nivel de renda:")
for ses_lbl in ['low_ses', 'mid_ses', 'high_ses']:
    delta = (scenarios[f'{ses_lbl}_privada'] - scenarios[f'{ses_lbl}_publica']) * 100
    print(f"  {ses_lbl}: {scenarios[f'{ses_lbl}_publica']:.1%} → {scenarios[f'{ses_lbl}_privada']:.1%}  (+{delta:.1f} pp)")

# 8d. AME — Average Marginal Effect de escola_privada sobre o conjunto de teste
print("\nCalculando AME de escola_privada (500 bootstrap)...")
X_full_test_df = pd.DataFrame(X_full_test, columns=feature_names_full)
X_ame_1 = X_full_test_df.copy()
X_ame_0 = X_full_test_df.copy()
X_ame_1['escola_privada'] = 1.0
X_ame_0['escola_privada'] = 0.0
delta_per_student = (
    lr_b_for_scenarios.predict_proba(X_ame_1.values)[:, 1]
    - lr_b_for_scenarios.predict_proba(X_ame_0.values)[:, 1]
)
ame_pp = float(delta_per_student.mean() * 100)

rng = np.random.default_rng(RANDOM_STATE)
ame_boots = np.array([
    rng.choice(delta_per_student, size=len(delta_per_student), replace=True).mean() * 100
    for _ in range(500)
])
ame_ci_lower = float(np.percentile(ame_boots, 2.5))
ame_ci_upper = float(np.percentile(ame_boots, 97.5))
print(f"  AME escola_privada: {ame_pp:+.2f} pp  [IC 95%: {ame_ci_lower:+.2f}, {ame_ci_upper:+.2f}]")
print("  (media de delta P(>=700) ao trocar escola publica->privada para cada aluno do teste)")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SENSITIVITY CHECK — THRESHOLDS 680 E 720
# ═══════════════════════════════════════════════════════════════════════════════

print("\nSensitivity check nos thresholds...")
sensitivity = {}
if 'nota_media' in df_clean.columns:
    y_nota = df_clean['nota_media'].values
    for thr in [680, 720]:
        y_thr = (y_nota >= thr).astype(int)
        y_thr_train = y_thr[idx_train]
        y_thr_test = y_thr[idx_test]

        lr_thr_a = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, class_weight=None, max_iter=500, random_state=RANDOM_STATE)),
        ])
        lr_thr_b = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, class_weight=None, max_iter=500, random_state=RANDOM_STATE)),
        ])
        lr_thr_a.fit(X_ses_train, y_thr_train)
        lr_thr_b.fit(X_full_train, y_thr_train)

        pa_auc_a = average_precision_score(y_thr_test, lr_thr_a.predict_proba(X_ses_test)[:, 1])
        pa_auc_b = average_precision_score(y_thr_test, lr_thr_b.predict_proba(X_full_test)[:, 1])
        delta_thr = pa_auc_b - pa_auc_a

        coef_thr = pd.Series(lr_thr_b.named_steps['clf'].coef_[0], index=feature_names_full)
        top_feat = coef_thr.abs().idxmax()

        sensitivity[f'threshold_{thr}'] = {
            'delta_pr_auc': round(delta_thr, 4),
            'top_feature': FEATURE_PT.get(top_feat, top_feat),
        }
        print(f"  threshold={thr}: delta_PR-AUC={delta_thr:+.4f}, top_feature={top_feat}")
else:
    print("  nota_media nao disponivel na base — sensitivity check ignorado.")


# ═══════════════════════════════════════════════════════════════════════════════
# 9b. CENÁRIO DO PERSONAGEM — slide executivo
# ═══════════════════════════════════════════════════════════════════════════════

# Perfil: renda faixa C (~1-1.5 SM), mae com EM completo, sem computador, sem internet.
# Demais features SES = mediana da base (contexto tipico). UF = referencia (zeros).
_char_medians = {
    f: int(round(float(df_clean[f].median())))
    for f in SES_FEATURES
    if f in df_clean.columns
}
character_base = {
    **_char_medians,
    'Q006_num': 2,    # faixa C: ~R$1.320-1.980/mes (~1 SM)
    'Q002_num': 6,    # mae com EM completo (opcao G)
    'Q024_num': 0,    # sem computador
    'Q025_num': 0,    # sem internet
    'concluindo_ano': 1,   # aluno no ultimo ano do EM — janela de intervencao
    'escola_nao_resp': 0,
}

char_p_publico = predict_profile({**character_base, 'escola_privada': 0})
char_p_privado = predict_profile({**character_base, 'escola_privada': 1})
char_delta_pp = (char_p_privado - char_p_publico) * 100

print("\n" + "=" * 60)
print("CENÁRIO DO PERSONAGEM")
print("Perfil: renda faixa C (~1 SM), mae EM completo, sem PC/internet, concluindo EM")
print(f"  Escola publica:  P(>=700) = {char_p_publico:.2%}")
print(f"  Escola privada:  P(>=700) = {char_p_privado:.2%}  (+{char_delta_pp:.1f} pp, {char_p_privado/char_p_publico:.1f}x mais provavel)")


# ─── Race/gender stats — P(≥700) por grupo ────────────────────────────────────
_RACA_LABELS_STATS = {1: 'Branca', 2: 'Preta', 3: 'Parda', 4: 'Amarela', 5: 'Indigena'}
race_gender_stats: dict = {'raca': [], 'genero': []}

if 'TP_COR_RACA' in df_clean.columns:
    _rg = (
        df_clean[df_clean['TP_COR_RACA'].isin(_RACA_LABELS_STATS)]
        .groupby('TP_COR_RACA')
        .agg(media=('nota_media', 'mean'), p700=('alta_performance', 'mean'), n=('alta_performance', 'count'))
        .reset_index()
    )
    for _, _row in _rg.iterrows():
        race_gender_stats['raca'].append({
            'label': _RACA_LABELS_STATS[int(_row['TP_COR_RACA'])],
            'media': round(float(_row['media']), 1),
            'p700_pct': round(float(_row['p700']) * 100, 2),
            'n': int(_row['n']),
        })

if 'TP_SEXO' in df_clean.columns:
    _SEXO_LABELS = {'M': 'Masculino', 'F': 'Feminino'}
    _sg = (
        df_clean[df_clean['TP_SEXO'].isin(_SEXO_LABELS)]
        .groupby('TP_SEXO')
        .agg(media=('nota_media', 'mean'), p700=('alta_performance', 'mean'), n=('alta_performance', 'count'))
        .reset_index()
    )
    for _, _row in _sg.iterrows():
        race_gender_stats['genero'].append({
            'label': _SEXO_LABELS[_row['TP_SEXO']],
            'media': round(float(_row['media']), 1),
            'p700_pct': round(float(_row['p700']) * 100, 2),
            'n': int(_row['n']),
        })

print("\nRace/gender P(≥700):")
for _g in race_gender_stats['raca']:
    print(f"  {_g['label']:10s}: media={_g['media']:.1f}  P(≥700)={_g['p700_pct']:.2f}%  n={_g['n']:,}")
for _g in race_gender_stats['genero']:
    print(f"  {_g['label']:10s}: media={_g['media']:.1f}  P(≥700)={_g['p700_pct']:.2f}%  n={_g['n']:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SALVAR metrics.json
# ═══════════════════════════════════════════════════════════════════════════════

metrics = {
    'lr_ses_only': {
        'pr_auc_cv': round(float(cv_a['test_pr_auc'].mean()), 4),
        'pr_auc_cv_std': round(float(cv_a['test_pr_auc'].std()), 4),
        'pr_auc_test': round(pr_auc_a_test, 4),
        'brier_test': round(brier_a_test, 4),
    },
    'lr_full': {
        'pr_auc_cv': round(float(cv_b['test_pr_auc'].mean()), 4),
        'pr_auc_cv_std': round(float(cv_b['test_pr_auc'].std()), 4),
        'pr_auc_test': round(pr_auc_b_test, 4),
        'brier_test': round(brier_b_test, 4),
        'delta_pr_auc': round(delta_pr_auc, 4),
        'partial_coef_escola_privada': round(coef_escola_privada, 4) if coef_escola_privada is not None else None,
        'calibration_applied': calibration_applied_lr_b,
        'brier_after_calibration': brier_b_calibrated,
    },
    'hgb': {
        'pr_auc_cv': round(float(cv_hgb['test_pr_auc'].mean()), 4),
        'pr_auc_cv_std': round(float(cv_hgb['test_pr_auc'].std()), 4),
        'pr_auc_test': round(pr_auc_hgb_test, 4),
        'brier_test': round(brier_hgb_test, 4),
    },
    'lift_top20': round(lift_top20, 3),
    'threshold_f1_optimal': round(optimal_threshold, 3),
    'scenarios': {k: round(v, 4) for k, v in scenarios.items()},
    'scenarios_profile_counts': profile_counts,
    'sensitivity': sensitivity,
    'top10_features': [
        {'feature': row.feature, 'feature_pt': row.feature_pt, 'importancia': round(row.importancia_media, 5)}
        for row in perm_df.head(10).itertuples()
    ],
    'escola_effect': {
        'partial_coef_escola_privada': round(coef_escola_privada, 4) if coef_escola_privada is not None else None,
        'delta_pr_auc_escola_uf': round(delta_pr_auc, 4),
        'ame_pp': round(ame_pp, 2),
        'ame_ci_lower_pp': round(ame_ci_lower, 2),
        'ame_ci_upper_pp': round(ame_ci_upper, 2),
        'interpretacao': (
            'escola_privada derivada de TP_ESCOLA (autodeclarada na inscricao). '
            'ame_pp = Average Marginal Effect: media de [P(escola=privada|X_i) - P(escola=publica|X_i)] '
            'sobre todos os alunos do conjunto de teste (n~431k). '
            'IC 95% por bootstrap com 500 iteracoes (resampling do vetor delta). '
            'Estimativa observacional, nao causal — viés de selecao esperado. '
            'Modelos LR treinados sem class_weight (class_weight=None): '
            'predict_proba representa probabilidade de frequencia real.'
        ),
    },
    'model_health': {
        'prob_std_hgb': round(prob_std_hgb, 4),
        'recall_minority_hgb': round(recall_minority_hgb, 4),
        'predicted_positive_count_hgb': predicted_positive_count_hgb,
        'total_test_rows': len(y_test),
        'alerts': (
            ['prob_std < 0.01: possivel colapso'] if prob_std_hgb < 0.01 else []
        ) + (
            ['recall_minority < 0.05: modelo nao discrimina positivos'] if recall_minority_hgb < 0.05 else []
        ),
    },
    'features_policy': {
        'disponivel_na_base': POLICY_FEATURES,
        'nota_eja': (
            'TP_ENSINO=3 (EJA) ausente na base filtrada (TP_ST_CONCLUSAO IN [1,2]). '
            'Feature eja criada mas sem variancia — sem impacto no modelo atual.'
        ) if 'eja' in POLICY_FEATURES else 'TP_ENSINO nao disponivel.',
        'nota_q023': (
            'Q023 = telefone fixo (dicionario ENEM 2023), NAO escola declarada. '
            'CLAUDE.md tem descricao incorreta desta variavel.'
        ),
    },
    'cenario_personagem': {
        'descricao': 'Renda faixa C (~R$1.320-1.980/mes, ~1 SM), mae com EM completo (Q002=G), sem computador (Q024=A), sem internet (Q025=A), concluindo EM no ano do ENEM. Modelo LR-Full (SES + escola + UF). UF = categoria de referencia.',
        'escola_publica_pct': round(char_p_publico * 100, 2),
        'escola_privada_pct': round(char_p_privado * 100, 2),
        'delta_pp': round(char_delta_pp, 2),
        'razao': round(char_p_privado / char_p_publico, 2),
    },
    'race_gender_stats': race_gender_stats,
}

metrics_path = REPORTS_DIR / 'metrics.json'
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"\nMetrics salvas: {metrics_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. VIZ 5 — CALIBRACAO + LIFT
# ═══════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Calibracao
for probs, label, color in [
    (prob_b_test, 'LR (SES + escola)', LARANJA),
    (prob_hgb_test, 'HGB', AZUL),
]:
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    ax1.plot(mean_pred, frac_pos, marker='o', linewidth=2, color=color, label=label)
ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, label='Perfeito')
ax1.set_xlabel('Probabilidade prevista', fontsize=11)
ax1.set_ylabel('Fracao de positivos observados', fontsize=11)
ax1.set_title('Calibracao do Modelo', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)

# Lift por decil
n_decis = 10
decil_idx = np.argsort(prob_hgb_test)[::-1]
decil_size = len(decil_idx) // n_decis
lift_vals = [
    y_test[decil_idx[d * decil_size:(d + 1) * decil_size]].mean() / y_test.mean()
    for d in range(n_decis)
]
cores_lift = [AZUL if i < 2 else '#a8c4e0' for i in range(n_decis)]
ax2.bar(range(1, n_decis + 1), lift_vals, color=cores_lift, edgecolor='white')
ax2.axhline(1, color='gray', linestyle='--', linewidth=1.5, label='Baseline (sem modelo)')
ax2.set_xlabel('Decil (1 = maior probabilidade prevista)', fontsize=11)
ax2.set_ylabel('Lift vs. taxa base', fontsize=11)
ax2.set_title(
    f'Lift por Decil — HGB\n(Top-20%: {lift_top20:.1f}x a taxa base)',
    fontsize=12, fontweight='bold',
)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(FIGS_DIR / 'viz5_calibracao_lift.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Viz 5 salva: {FIGS_DIR / 'viz5_calibracao_lift.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. VIZ 6 — IMPORTANCIA DE FEATURES (TOP-10, DESTACANDO ACIONAVEIS)
# ═══════════════════════════════════════════════════════════════════════════════

top10 = perm_df.head(10).sort_values('importancia_media').copy()

def _classify_feature(f):
    if f in ESCOLA_FEATURES + POLICY_FEATURES:
        return 'acionavel'   # controlavel operacionalmente pela rede
    elif f in REGIONAL_FEATURES:
        return 'regional'    # efeito geografico — alavanca estrategica
    return 'ses'             # contexto socioeconomico

top10['categoria'] = top10['feature'].apply(_classify_feature)
COR_MAP = {'acionavel': AZUL, 'regional': LARANJA, 'ses': '#a8c4e0'}
cores_feat = [COR_MAP[c] for c in top10['categoria']]

fig, ax = plt.subplots(figsize=(11, 5))
ax.barh(
    top10['feature_pt'], top10['importancia_media'],
    xerr=top10['importancia_std'],
    color=cores_feat,
    error_kw={'elinewidth': 1.2, 'capsize': 4},
    edgecolor='white',
)
legend_elements = [
    Patch(facecolor=AZUL,      label='Variavel acionavel (operacional)'),
    Patch(facecolor=LARANJA,   label='Efeito regional (estrategico)'),
    Patch(facecolor='#a8c4e0', label='Contexto socioeconomico'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
ax.set_xlabel('Importancia (queda no PR-AUC ao permutar)', fontsize=11)
ax.set_title('Importancia das Features — HGB (10 repeticoes)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGS_DIR / 'viz6_features.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Viz 6 salva: {FIGS_DIR / 'viz6_features.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. VIZ 7 — RAÇA/ETNIA E GÊNERO
# ═══════════════════════════════════════════════════════════════════════════════

RACA_LABELS = {
    1: 'Branca',
    2: 'Preta',
    3: 'Parda',     # dicionário ENEM real: 3=Parda, 4=Amarela (CLAUDE.md tinha invertido)
    4: 'Amarela',
    5: 'Indigena',
}

if 'nota_media' in df_clean.columns and 'TP_COR_RACA' in df_clean.columns and 'TP_SEXO' in df_clean.columns:
    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Desempenho por Raca/Etnia e Genero — ENEM 2023',
                 fontsize=13, fontweight='bold', y=1.01)

    # Painel A: nota_media por grupo racial (excluindo nao declarado = 0)
    raca_df = df_clean[df_clean['TP_COR_RACA'].isin(RACA_LABELS.keys())].copy()
    raca_stats = (
        raca_df.groupby('TP_COR_RACA')['nota_media']
        .agg(media='mean', std='std', n='count')
        .reset_index()
    )
    raca_stats['ci95'] = 1.96 * raca_stats['std'] / raca_stats['n'] ** 0.5
    raca_stats['label'] = raca_stats['TP_COR_RACA'].map(RACA_LABELS)
    _raca_p700 = (
        raca_df.groupby('TP_COR_RACA')['alta_performance']
        .mean()
        .reset_index()
        .rename(columns={'alta_performance': 'p700'})
    )
    raca_stats = (
        raca_stats.merge(_raca_p700, on='TP_COR_RACA', how='left')
        .sort_values('media', ascending=True)
        .reset_index(drop=True)
    )

    cores_raca = [AZUL if lab == 'Branca' else '#a8c4e0' for lab in raca_stats['label']]
    ax_r.barh(
        raca_stats['label'], raca_stats['media'],
        xerr=raca_stats['ci95'],
        color=cores_raca,
        error_kw={'elinewidth': 1.2, 'capsize': 4},
        edgecolor='white',
    )
    _media_geral = df_clean['nota_media'].mean()
    ax_r.axvline(_media_geral, color=LARANJA, linestyle='--', linewidth=1.5,
                 label=f'Media geral ({_media_geral:.0f})')
    ax_r.legend(fontsize=10)
    for i, row in raca_stats.iterrows():
        ax_r.text(
            row['media'] + 1, i,
            f"{row['media']:.0f}  |  P(≥700): {row['p700']*100:.1f}%",
            va='center', fontsize=8.5,
        )
    ax_r.set_xlabel('Nota media (0–1000)', fontsize=11)
    ax_r.set_title('Nota Media por Raca/Etnia', fontsize=12, fontweight='bold')

    # Painel B: nota_media por sexo
    sexo_stats = (
        df_clean.groupby('TP_SEXO')['nota_media']
        .agg(media='mean', std='std', n='count')
        .reset_index()
    )
    sexo_stats['ci95'] = 1.96 * sexo_stats['std'] / sexo_stats['n'] ** 0.5
    sexo_stats['label'] = sexo_stats['TP_SEXO'].map({'M': 'Masculino', 'F': 'Feminino'})
    _sexo_p700 = (
        df_clean[df_clean['TP_SEXO'].isin(['M', 'F'])]
        .groupby('TP_SEXO')['alta_performance']
        .mean()
        .reset_index()
        .rename(columns={'alta_performance': 'p700'})
    )
    sexo_stats = (
        sexo_stats.merge(_sexo_p700, on='TP_SEXO', how='left')
        .sort_values('media', ascending=True)
        .reset_index(drop=True)
    )

    cores_sexo = [AZUL if lab == 'Masculino' else LARANJA for lab in sexo_stats['label']]
    ax_s.barh(
        sexo_stats['label'], sexo_stats['media'],
        xerr=sexo_stats['ci95'],
        color=cores_sexo,
        error_kw={'elinewidth': 1.2, 'capsize': 4},
        edgecolor='white',
    )
    ax_s.axvline(_media_geral, color='gray', linestyle='--', linewidth=1.5,
                 label=f'Media geral ({_media_geral:.0f})')
    ax_s.legend(fontsize=10)
    for i, row in sexo_stats.iterrows():
        ax_s.text(
            row['media'] + 1, i,
            f"{row['media']:.0f}  |  P(≥700): {row['p700']*100:.1f}%",
            va='center', fontsize=10,
        )
    ax_s.set_xlabel('Nota media (0–1000)', fontsize=11)
    ax_s.set_title('Nota Media por Sexo', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'viz7_raca_genero.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Viz 7 salva: {FIGS_DIR / 'viz7_raca_genero.png'}")
else:
    print("Viz 7 ignorada: colunas nota_media, TP_COR_RACA ou TP_SEXO ausentes.")


# ─── Resumo final ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f"PR-AUC SES-only:   {pr_auc_a_test:.4f}")
print(f"PR-AUC SES+escola: {pr_auc_b_test:.4f}  (delta: +{delta_pr_auc:.4f})")
print(f"PR-AUC HGB:        {pr_auc_hgb_test:.4f}")
print(f"Brier HGB:         {brier_hgb_test:.4f}")
print(f"Lift top-20%:      {lift_top20:.2f}x")
if coef_escola_privada is not None:
    print(f"Coef. parcial escola_privada: {coef_escola_privada:+.4f}")
print(f"AME escola_privada: {ame_pp:+.1f} pp  [IC 95%: {ame_ci_lower:+.1f}, {ame_ci_upper:+.1f}]")
print(f"Cenario personagem: publico={char_p_publico:.2%} | privado={char_p_privado:.2%} (+{char_delta_pp:.1f} pp)")
print(f"\nArtefatos:")
print(f"  {metrics_path}")
print(f"  {FIGS_DIR / 'viz5_calibracao_lift.png'}")
print(f"  {FIGS_DIR / 'viz6_features.png'}")
print(f"  {FIGS_DIR / 'viz7_raca_genero.png'}")
