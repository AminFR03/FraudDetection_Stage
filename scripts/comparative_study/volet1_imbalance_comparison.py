"""
volet1_imbalance_comparison.py — Volet 1 : Comparaison des techniques de rééquilibrage
Stage : Détection de Fraude Bancaire avec Federated Learning & Agentic AI

Compare SMOTE, ADASYN, RandomUnderSampling et Baseline (aucun resampling)
sur les deux datasets (ULB et Synthétique), avec XGBoost ensemble.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE = 42
TEST_SIZE    = 0.2
ENSEMBLE_SEEDS = [42, 123, 456]

XGB_PARAMS = dict(
    max_depth          = 7,
    learning_rate      = 0.05,
    subsample          = 0.85,
    colsample_bytree   = 0.85,
    min_child_weight   = 3,
    gamma              = 0.05,
    reg_alpha          = 0.05,
    reg_lambda         = 1.0,
    eval_metric        = 'logloss',
    use_label_encoder  = False,
    verbosity          = 0,
    tree_method        = 'hist',
    nthread            = -1,
)

TECHNIQUES = ['Baseline', 'SMOTE', 'ADASYN', 'Undersampling']


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Calcule la distance haversine entre deux points (en km)."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def find_optimal_threshold(y_true, y_prob, n_points=1000):
    """Cherche le seuil qui maximise le F1-score."""
    thresholds = np.linspace(0.001, 0.999, n_points)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def ensemble_predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# ULB Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_ulb(path):
    """Prépare le dataset ULB avec features d'interaction et score IF."""
    print("\n" + "=" * 70)
    print("  CHARGEMENT DU DATASET ULB")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Fraudes: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

    # Features d'interaction sur les composantes PCA les plus discriminantes
    # (V14, V12, V10, V17, V4, V11 sont connues comme les plus importantes)
    df['V14_V12'] = df['V14'] * df['V12']
    df['V14_V17'] = df['V14'] * df['V17']
    df['V10_V14'] = df['V10'] * df['V14']
    df['V4_V11']  = df['V4']  * df['V11']
    df['V14_sq']  = df['V14'] ** 2
    df['V12_sq']  = df['V12'] ** 2
    df['V17_sq']  = df['V17'] ** 2
    df['V10_sq']  = df['V10'] ** 2
    df['Amount_log'] = np.log1p(df['Amount'])

    X_raw = df.drop('Class', axis=1).values
    y     = df['Class'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Scaling
    scaler   = RobustScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    # Isolation Forest
    print("  Fitting Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()),
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_tr_s)
    X_tr_f = np.hstack([X_tr_s, iso.score_samples(X_tr_s).reshape(-1, 1)])
    X_te_f = np.hstack([X_te_s, iso.score_samples(X_te_s).reshape(-1, 1)])

    print(f"  Features: {X_tr_f.shape[1]} | Train: {len(y_tr):,} | Test: {len(y_te):,}")
    print(f"  Fraudes train: {y_tr.sum():,} | Fraudes test: {y_te.sum():,}")
    return X_tr_f, X_te_f, y_tr, y_te


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def build_synthetic_features(df):
    """
    Ingénierie complète des features pour le dataset synthétique.
    Les statistiques cartes / marchands / catégories sont calculées sur le
    DATASET COMPLET (simule les données historiques disponibles en production).
    """
    feat = pd.DataFrame(index=df.index)

    # ── Numériques bruts ──
    for col in ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
                'Customer_Age', 'Customer_Satisfaction_Score', 'Loyalty_Points_Earned']:
        if col in df.columns:
            feat[col] = df[col].fillna(df[col].median())

    # ── Temporel ──
    if 'trans_date_trans_time' in df.columns:
        dt = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        feat['hour']        = dt.dt.hour.fillna(12).astype(float)
        feat['day_of_week'] = dt.dt.dayofweek.fillna(2).astype(float)
        feat['month']       = dt.dt.month.fillna(6).astype(float)
        feat['is_night']    = ((dt.dt.hour >= 22) | (dt.dt.hour <= 4)).astype(float)
        feat['is_weekend']  = (dt.dt.dayofweek >= 5).astype(float)
    elif 'unix_time' in df.columns:
        unix = df['unix_time'].fillna(df['unix_time'].median())
        feat['hour']        = ((unix // 3600) % 24).astype(float)
        feat['day_of_week'] = ((unix // 86400) % 7).astype(float)
        feat['is_night']    = ((feat['hour'] >= 22) | (feat['hour'] <= 4)).astype(float)
        feat['is_weekend']  = (feat['day_of_week'] >= 5).astype(float)

    # ── Âge du client ──
    if 'dob' in df.columns and 'trans_date_trans_time' in df.columns:
        dob_dt = pd.to_datetime(df['dob'], errors='coerce')
        tx_dt  = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        feat['age'] = ((tx_dt - dob_dt).dt.days / 365.25).fillna(40.0).astype(float)

    # ── Distance haversine (km) ──
    if all(c in df.columns for c in ['lat', 'long', 'merch_lat', 'merch_long']):
        feat['dist_km'] = haversine_km(df['lat'], df['long'], df['merch_lat'], df['merch_long']).fillna(0)

    # ── Montant ──
    if 'amt' in df.columns:
        feat['amt_log'] = np.log1p(df['amt'])

    # ── Statistiques par carte (historique complet) ──
    if 'cc_num' in df.columns and 'amt' in df.columns:
        cc_stats = df.groupby('cc_num')['amt'].agg(['mean', 'std', 'max'])
        cc_stats.columns = ['cc_amt_mean', 'cc_amt_std', 'cc_amt_max']
        cc_stats['cc_amt_std'] = cc_stats['cc_amt_std'].fillna(1.0)
        merged = df[['cc_num', 'amt']].join(cc_stats, on='cc_num')
        feat['cc_amt_mean']     = merged['cc_amt_mean'].values
        feat['cc_amt_std']      = merged['cc_amt_std'].values
        feat['cc_amt_max']      = merged['cc_amt_max'].values
        feat['amt_ratio_cc']    = (df['amt'] / (merged['cc_amt_mean'] + 1e-3)).values
        feat['amt_zscore_cc']   = ((df['amt'] - merged['cc_amt_mean']) / (merged['cc_amt_std'] + 1e-3)).values

    # ── Statistiques par marchand ──
    if 'merchant' in df.columns and 'amt' in df.columns:
        merch_mean = df.groupby('merchant')['amt'].transform('mean')
        feat['amt_ratio_merch'] = (df['amt'] / (merch_mean + 1e-3)).values

    # ── Statistiques par catégorie ──
    if 'category' in df.columns and 'amt' in df.columns:
        cat_mean = df.groupby('category')['amt'].transform('mean')
        feat['amt_ratio_cat'] = (df['amt'] / (cat_mean + 1e-3)).values

    # ── One-hot encoding (conserve la richesse des catégories) ──
    for col in ['category', 'gender', 'Merchant_Category',
                'Transaction_Type', 'Payment_Method', 'state']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            feat = pd.concat([feat, dummies], axis=1)

    return feat


def load_synthetic(path):
    """Prépare le dataset synthétique avec toutes les features."""
    print("\n" + "=" * 70)
    print("  CHARGEMENT DU DATASET SYNTHÉTIQUE")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Fraudes: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.3f}%)")

    y = df['is_fraud'].values

    print("  Ingénierie des features...")
    feat = build_synthetic_features(df)
    feat = feat.fillna(0).astype(np.float32)
    print(f"  Features finales (avant IF): {feat.shape[1]}")

    X = feat.values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Scaling
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Isolation Forest
    print("  Fitting Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()),
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_tr_s)
    X_tr_f = np.hstack([X_tr_s, iso.score_samples(X_tr_s).reshape(-1, 1)])
    X_te_f = np.hstack([X_te_s, iso.score_samples(X_te_s).reshape(-1, 1)])

    print(f"  Features totales (avec IF): {X_tr_f.shape[1]}")
    print(f"  Train: {len(y_tr):,} | Test: {len(y_te):,}")
    print(f"  Fraudes train: {y_tr.sum():,} | Fraudes test: {y_te.sum():,}")
    return X_tr_f, X_te_f, y_tr, y_te


# ═══════════════════════════════════════════════════════════════════════════════
# Resampling
# ═══════════════════════════════════════════════════════════════════════════════

def apply_resampling(X_tr, y_tr, technique, dataset_name):
    strategy = 0.5 if dataset_name == 'ULB' else 0.3

    if technique == 'Baseline':
        print(f"    [Baseline] Aucun resampling — {len(y_tr):,} samples")
        return X_tr, y_tr

    elif technique == 'SMOTE':
        s = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
        X_r, y_r = s.fit_resample(X_tr, y_tr)
        print(f"    [SMOTE] {len(y_tr):,} → {len(y_r):,} samples")
        return X_r, y_r

    elif technique == 'ADASYN':
        try:
            s = ADASYN(sampling_strategy=strategy, random_state=RANDOM_STATE)
            X_r, y_r = s.fit_resample(X_tr, y_tr)
            print(f"    [ADASYN] {len(y_tr):,} → {len(y_r):,} samples")
            return X_r, y_r
        except ValueError as e:
            print(f"    [ADASYN] failed ({e}) → SMOTE fallback")
            s = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
            X_r, y_r = s.fit_resample(X_tr, y_tr)
            return X_r, y_r

    elif technique == 'Undersampling':
        s = RandomUnderSampler(random_state=RANDOM_STATE)
        X_r, y_r = s.fit_resample(X_tr, y_tr)
        print(f"    [Undersampling] {len(y_tr):,} → {len(y_r):,} samples")
        return X_r, y_r

    raise ValueError(f"Technique inconnue: {technique}")


# ═══════════════════════════════════════════════════════════════════════════════
# Train & Evaluate
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(X_tr, y_tr, X_te, y_te, technique, dataset_name):
    """Entraîne un ensemble de modèles XGBoost et retourne les métriques."""
    X_res, y_res = apply_resampling(X_tr, y_tr, technique, dataset_name)

    n_neg = int((y_res == 0).sum())
    n_pos = int((y_res == 1).sum())
    spw   = n_neg / n_pos if technique == 'Baseline' else 1.0
    if technique == 'Baseline':
        print(f"    [Baseline] scale_pos_weight = {spw:.1f}")

    # Split validation interne pour early stopping
    try:
        X_t, X_v, y_t, y_v = train_test_split(
            X_res, y_res, test_size=0.1, stratify=y_res, random_state=RANDOM_STATE
        )
    except ValueError:
        X_t, X_v, y_t, y_v = X_res, X_res[-20:], y_res, y_res[-20:]

    # Ensemble de 3 modèles
    models = []
    iters  = []
    for seed in ENSEMBLE_SEEDS:
        params = {**XGB_PARAMS,
                  'random_state': seed, 'scale_pos_weight': spw,
                  'n_estimators': 2000, 'early_stopping_rounds': 50}
        m = xgb.XGBClassifier(**params)
        m.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        models.append(m)
        iters.append(m.best_iteration + 1)

    print(f"    Arbres utilisés: {iters} (moy={int(np.mean(iters))})")

    y_prob = ensemble_predict(models, X_te)
    best_t, best_f1 = find_optimal_threshold(y_te, y_prob)
    y_pred = (y_prob >= best_t).astype(int)
    print(f"    Seuil optimal: {best_t:.3f} → F1={best_f1:.4f}")

    metrics = {
        'Dataset':   dataset_name,
        'Technique': technique,
        'Accuracy':  round(accuracy_score(y_te, y_pred), 4),
        'Precision': round(precision_score(y_te, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_te, y_pred, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_te, y_pred, zero_division=0), 4),
        'AUC-ROC':   round(roc_auc_score(y_te, y_prob), 4),
        'AUC-PR':    round(average_precision_score(y_te, y_prob), 4),
    }
    print(f"    → Acc={metrics['Accuracy']:.4f}  Prec={metrics['Precision']:.4f}  "
          f"Rec={metrics['Recall']:.4f}  F1={metrics['F1-Score']:.4f}  "
          f"AUC={metrics['AUC-ROC']:.4f}  AUPRC={metrics['AUC-PR']:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results_df, output_dir):
    datasets = results_df['Dataset'].unique()
    metrics  = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
    colors   = {'Baseline': '#7f8c8d', 'SMOTE': '#3498db',
                'ADASYN': '#e67e22',   'Undersampling': '#e74c3c'}

    fig, axes = plt.subplots(1, len(datasets), figsize=(20, 8))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = results_df[results_df['Dataset'] == ds]
        x   = np.arange(len(metrics))
        w   = 0.18
        n   = len(sub)
        for i, tech in enumerate(sub['Technique'].tolist()):
            vals   = [sub[sub['Technique'] == tech][m].values[0] for m in metrics]
            offset = (i - (n - 1) / 2) * w
            bars   = ax.bar(x + offset, vals, w, label=tech,
                            color=colors.get(tech, '#95a5a6'),
                            alpha=0.85, edgecolor='white', linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=7, fontweight='bold', rotation=45)

        ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Seuil 0.9')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Volet 1 — {ds}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Comparaison Techniques de Rééquilibrage (XGBoost Ensemble + IF)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, 'volet1_barplot.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [✓] Graphique sauvegardé : {p}")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Run Volet 1
# ═══════════════════════════════════════════════════════════════════════════════

def run_volet1(ulb_path, synthetic_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    print("\n" + "═" * 70)
    print("  VOLET 1 — DATASET ULB")
    print("═" * 70)
    X_tr_u, X_te_u, y_tr_u, y_te_u = load_ulb(ulb_path)
    for tech in TECHNIQUES:
        print(f"\n  ▸ Technique: {tech}")
        all_results.append(train_and_evaluate(X_tr_u, y_tr_u, X_te_u, y_te_u, tech, 'ULB'))

    print("\n" + "═" * 70)
    print("  VOLET 1 — DATASET SYNTHÉTIQUE")
    print("═" * 70)
    X_tr_s, X_te_s, y_tr_s, y_te_s = load_synthetic(synthetic_path)
    for tech in TECHNIQUES:
        print(f"\n  ▸ Technique: {tech}")
        all_results.append(train_and_evaluate(X_tr_s, y_tr_s, X_te_s, y_te_s, tech, 'Synthétique'))

    df_res = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'volet1_imbalance_results.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"\n  [✓] Résultats CSV sauvegardés : {csv_path}")
    plot_results(df_res, output_dir)

    print("\n" + "=" * 90)
    print("  TABLEAU COMPARATIF — VOLET 1 : TECHNIQUES DE RÉÉQUILIBRAGE")
    print("=" * 90)
    for ds in ['ULB', 'Synthétique']:
        sub = df_res[df_res['Dataset'] == ds]
        print(f"\n  ── {ds} ──")
        print(sub[['Technique', 'Accuracy', 'Precision', 'Recall',
                    'F1-Score', 'AUC-ROC', 'AUC-PR']].to_string(index=False))

    return df_res


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_volet1(
        ulb_path       = os.path.join(BASE_DIR, 'data', 'creditcard.csv'),
        synthetic_path = os.path.join(BASE_DIR, 'data', 'fraud_detection_credit_card_small.csv'),
        output_dir     = os.path.join(BASE_DIR, 'results', 'comparative_study'),
    )
    print("\n  [✓] Volet 1 terminé avec succès !")
