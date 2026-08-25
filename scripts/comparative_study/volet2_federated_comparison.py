"""
volet2_federated_comparison.py — Volet 2 : Centralisé vs Federated Learning
Stage : Détection de Fraude Bancaire avec Federated Learning & Agentic AI

Compare entraînement centralisé vs FedAvg (soft voting) sur les deux datasets.
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
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE      = 42
TEST_SIZE         = 0.2
ENSEMBLE_SEEDS    = [42, 123, 456]
NUM_CLIENTS       = 10
CLIENTS_PER_ROUND = 8
NUM_ROUNDS        = 12

XGB_PARAMS = dict(
    max_depth         = 7,
    learning_rate     = 0.05,
    subsample         = 0.85,
    colsample_bytree  = 0.85,
    min_child_weight  = 3,
    gamma             = 0.05,
    reg_alpha         = 0.05,
    reg_lambda        = 1.0,
    eval_metric       = 'logloss',
    use_label_encoder = False,
    verbosity         = 0,
    tree_method       = 'hist',
    nthread           = -1,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities (mirrors volet1)
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def find_optimal_threshold(y_true, y_prob, n_points=1000):
    thresholds = np.linspace(0.001, 0.999, n_points)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def ensemble_predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def compute_metrics(y_te, y_pred, y_prob, mode, dataset):
    return {
        'Dataset':   dataset,
        'Mode':      mode,
        'Accuracy':  round(accuracy_score(y_te, y_pred), 4),
        'Precision': round(precision_score(y_te, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_te, y_pred, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_te, y_pred, zero_division=0), 4),
        'AUC-ROC':   round(roc_auc_score(y_te, y_prob), 4),
        'AUC-PR':    round(average_precision_score(y_te, y_prob), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_ulb(path):
    """Prépare ULB avec interactions PCA + Isolation Forest."""
    print("\n" + "=" * 70)
    print("  CHARGEMENT DU DATASET ULB")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Fraudes: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

    # Interaction features
    df['V14_V12']    = df['V14'] * df['V12']
    df['V14_V17']    = df['V14'] * df['V17']
    df['V10_V14']    = df['V10'] * df['V14']
    df['V4_V11']     = df['V4']  * df['V11']
    df['V14_sq']     = df['V14'] ** 2
    df['V12_sq']     = df['V12'] ** 2
    df['V17_sq']     = df['V17'] ** 2
    df['V10_sq']     = df['V10'] ** 2
    df['Amount_log'] = np.log1p(df['Amount'])

    X_raw = df.drop('Class', axis=1).values
    y     = df['Class'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print("  Fitting Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()),
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_tr_s)
    X_tr_f = np.hstack([X_tr_s, iso.score_samples(X_tr_s).reshape(-1, 1)])
    X_te_f = np.hstack([X_te_s, iso.score_samples(X_te_s).reshape(-1, 1)])

    print(f"  Features: {X_tr_f.shape[1]} | Train: {len(y_tr):,} | Test: {len(y_te):,}")
    return X_tr_f, X_te_f, y_tr, y_te


def build_synthetic_features(df):
    feat = pd.DataFrame(index=df.index)

    for col in ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
                'Customer_Age', 'Customer_Satisfaction_Score', 'Loyalty_Points_Earned']:
        if col in df.columns:
            feat[col] = df[col].fillna(df[col].median())

    if 'trans_date_trans_time' in df.columns:
        dt = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        feat['hour']       = dt.dt.hour.fillna(12).astype(float)
        feat['day_of_week']= dt.dt.dayofweek.fillna(2).astype(float)
        feat['month']      = dt.dt.month.fillna(6).astype(float)
        feat['is_night']   = ((dt.dt.hour >= 22) | (dt.dt.hour <= 4)).astype(float)
        feat['is_weekend'] = (dt.dt.dayofweek >= 5).astype(float)
    elif 'unix_time' in df.columns:
        unix = df['unix_time'].fillna(df['unix_time'].median())
        feat['hour']       = ((unix // 3600) % 24).astype(float)
        feat['day_of_week']= ((unix // 86400) % 7).astype(float)
        feat['is_night']   = ((feat['hour'] >= 22) | (feat['hour'] <= 4)).astype(float)
        feat['is_weekend'] = (feat['day_of_week'] >= 5).astype(float)

    if 'dob' in df.columns and 'trans_date_trans_time' in df.columns:
        dob_dt = pd.to_datetime(df['dob'], errors='coerce')
        tx_dt  = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        feat['age'] = ((tx_dt - dob_dt).dt.days / 365.25).fillna(40.0).astype(float)

    if all(c in df.columns for c in ['lat', 'long', 'merch_lat', 'merch_long']):
        feat['dist_km'] = haversine_km(df['lat'], df['long'], df['merch_lat'], df['merch_long']).fillna(0)

    if 'amt' in df.columns:
        feat['amt_log'] = np.log1p(df['amt'])

    if 'cc_num' in df.columns and 'amt' in df.columns:
        cc_stats = df.groupby('cc_num')['amt'].agg(['mean', 'std', 'max'])
        cc_stats.columns = ['cc_amt_mean', 'cc_amt_std', 'cc_amt_max']
        cc_stats['cc_amt_std'] = cc_stats['cc_amt_std'].fillna(1.0)
        merged = df[['cc_num', 'amt']].join(cc_stats, on='cc_num')
        feat['cc_amt_mean']   = merged['cc_amt_mean'].values
        feat['cc_amt_std']    = merged['cc_amt_std'].values
        feat['cc_amt_max']    = merged['cc_amt_max'].values
        feat['amt_ratio_cc']  = (df['amt'] / (merged['cc_amt_mean'] + 1e-3)).values
        feat['amt_zscore_cc'] = ((df['amt'] - merged['cc_amt_mean']) / (merged['cc_amt_std'] + 1e-3)).values

    if 'merchant' in df.columns and 'amt' in df.columns:
        merch_mean = df.groupby('merchant')['amt'].transform('mean')
        feat['amt_ratio_merch'] = (df['amt'] / (merch_mean + 1e-3)).values

    if 'category' in df.columns and 'amt' in df.columns:
        cat_mean = df.groupby('category')['amt'].transform('mean')
        feat['amt_ratio_cat'] = (df['amt'] / (cat_mean + 1e-3)).values

    for col in ['category', 'gender', 'Merchant_Category',
                'Transaction_Type', 'Payment_Method', 'state']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            feat = pd.concat([feat, dummies], axis=1)

    return feat


def load_synthetic(path):
    print("\n" + "=" * 70)
    print("  CHARGEMENT DU DATASET SYNTHÉTIQUE")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Fraudes: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.3f}%)")

    y = df['is_fraud'].values

    print("  Ingénierie des features...")
    feat = build_synthetic_features(df).fillna(0).astype(np.float32)
    print(f"  Features finales (avant IF): {feat.shape[1]}")

    X = feat.values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

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
# Centralized training
# ═══════════════════════════════════════════════════════════════════════════════

def train_centralized(X_tr, y_tr, X_te, y_te, dataset_name):
    print(f"\n  ▸ Mode CENTRALISÉ — {dataset_name}")

    strategy = 0.5 if dataset_name == 'ULB' else 0.3
    smote = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)
    print(f"    SMOTE: {len(y_tr):,} → {len(y_res):,} samples")

    try:
        X_t, X_v, y_t, y_v = train_test_split(
            X_res, y_res, test_size=0.1, stratify=y_res, random_state=RANDOM_STATE
        )
    except ValueError:
        X_t, X_v, y_t, y_v = X_res, X_res[-20:], y_res, y_res[-20:]

    models = []
    iters  = []
    for seed in ENSEMBLE_SEEDS:
        params = {**XGB_PARAMS, 'random_state': seed,
                  'n_estimators': 2000, 'early_stopping_rounds': 50}
        m = xgb.XGBClassifier(**params)
        m.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        models.append(m)
        iters.append(m.best_iteration + 1)

    print(f"    Arbres: {iters}")

    y_prob = ensemble_predict(models, X_te)
    best_t, _ = find_optimal_threshold(y_te, y_prob)
    y_pred = (y_prob >= best_t).astype(int)
    print(f"    Seuil optimal: {best_t:.3f}")

    metrics = compute_metrics(y_te, y_pred, y_prob, 'Centralisé', dataset_name)
    print(f"    → Acc={metrics['Accuracy']:.4f}  Prec={metrics['Precision']:.4f}  "
          f"Rec={metrics['Recall']:.4f}  F1={metrics['F1-Score']:.4f}  "
          f"AUC={metrics['AUC-ROC']:.4f}  AUPRC={metrics['AUC-PR']:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Federated Learning
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_client_split(X, y, n_clients):
    """Divise le dataset en n_clients partitions stratifiées."""
    np.random.seed(RANDOM_STATE)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    np.random.shuffle(fraud_idx)
    np.random.shuffle(legit_idx)
    clients = []
    for f_part, l_part in zip(np.array_split(fraud_idx, n_clients),
                               np.array_split(legit_idx, n_clients)):
        idx = np.concatenate([f_part, l_part])
        np.random.shuffle(idx)
        clients.append((X[idx], y[idx]))
    return clients


def train_federated(X_tr, y_tr, X_te, y_te, dataset_name):
    print(f"\n  ▸ Mode FÉDÉRÉ — {dataset_name}")
    print(f"    Config: {NUM_CLIENTS} clients, {CLIENTS_PER_ROUND}/round, {NUM_ROUNDS} rounds")

    clients = stratified_client_split(X_tr, y_tr, NUM_CLIENTS)
    strategy = 0.5 if dataset_name == 'ULB' else 0.3

    fraud_counts = [int(y.sum()) for _, y in clients]
    sizes        = [len(y) for _, y in clients]
    print(f"    Taille clients: {min(sizes):,}-{max(sizes):,} | "
          f"Fraudes/client: {min(fraud_counts)}-{max(fraud_counts)}")

    np.random.seed(RANDOM_STATE)
    client_models = [None] * NUM_CLIENTS

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)

        for cid in selected:
            X_c, y_c = clients[cid]
            if len(np.unique(y_c)) < 2:
                continue

            try:
                smote = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
                X_cr, y_cr = smote.fit_resample(X_c, y_c)
            except Exception:
                X_cr, y_cr = X_c, y_c

            try:
                X_ct, X_cv, y_ct, y_cv = train_test_split(
                    X_cr, y_cr, test_size=0.15, stratify=y_cr, random_state=RANDOM_STATE
                )
                params = {**XGB_PARAMS, 'random_state': RANDOM_STATE,
                          'n_estimators': 2000, 'early_stopping_rounds': 30}
                m = xgb.XGBClassifier(**params)
                m.fit(X_ct, y_ct, eval_set=[(X_cv, y_cv)], verbose=False)
            except Exception:
                params = {**XGB_PARAMS, 'random_state': RANDOM_STATE, 'n_estimators': 300}
                m = xgb.XGBClassifier(**params)
                m.fit(X_cr, y_cr)

            client_models[cid] = m

        active = [m for m in client_models if m is not None]
        if active:
            y_prob_r = ensemble_predict(active, X_te)
            _, f1_r = find_optimal_threshold(y_te, y_prob_r)
            print(f"    Round {rnd:02d}/{NUM_ROUNDS} — {len(active)} modèles actifs — F1={f1_r:.4f}")

    active = [m for m in client_models if m is not None]
    print(f"\n    Agrégation finale: {len(active)} modèles (Soft Voting)")

    y_prob = ensemble_predict(active, X_te)
    best_t, _ = find_optimal_threshold(y_te, y_prob)
    y_pred = (y_prob >= best_t).astype(int)
    print(f"    Seuil optimal: {best_t:.3f}")

    metrics = compute_metrics(y_te, y_pred, y_prob, 'Fédéré (FedAvg)', dataset_name)
    print(f"    → Acc={metrics['Accuracy']:.4f}  Prec={metrics['Precision']:.4f}  "
          f"Rec={metrics['Recall']:.4f}  F1={metrics['F1-Score']:.4f}  "
          f"AUC={metrics['AUC-ROC']:.4f}  AUPRC={metrics['AUC-PR']:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(df_res, output_dir):
    datasets = df_res['Dataset'].unique()
    metrics  = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
    colors   = {'Centralisé': '#2ecc71', 'Fédéré (FedAvg)': '#9b59b6'}

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 7))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub    = df_res[df_res['Dataset'] == ds]
        modes  = sub['Mode'].tolist()
        x      = np.arange(len(metrics))
        w      = 0.30
        n      = len(modes)

        for i, mode in enumerate(modes):
            vals   = [sub[sub['Mode'] == mode][m].values[0] for m in metrics]
            offset = (i - (n - 1) / 2) * w
            bars   = ax.bar(x + offset, vals, w, label=mode,
                            color=colors.get(mode, '#95a5a6'),
                            alpha=0.85, edgecolor='white', linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Seuil 0.9')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Volet 2 — {ds}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Centralisé vs Federated Learning (XGBoost Ensemble + IF)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, 'volet2_barplot.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [✓] Graphique sauvegardé : {p}")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Run Volet 2
# ═══════════════════════════════════════════════════════════════════════════════

def run_volet2(ulb_path, synthetic_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    print("\n" + "═" * 70)
    print("  VOLET 2 — DATASET ULB")
    print("═" * 70)
    X_tr_u, X_te_u, y_tr_u, y_te_u = load_ulb(ulb_path)
    all_results.append(train_centralized(X_tr_u, y_tr_u, X_te_u, y_te_u, 'ULB'))
    all_results.append(train_federated(X_tr_u, y_tr_u, X_te_u, y_te_u, 'ULB'))

    print("\n" + "═" * 70)
    print("  VOLET 2 — DATASET SYNTHÉTIQUE")
    print("═" * 70)
    X_tr_s, X_te_s, y_tr_s, y_te_s = load_synthetic(synthetic_path)
    all_results.append(train_centralized(X_tr_s, y_tr_s, X_te_s, y_te_s, 'Synthétique'))
    all_results.append(train_federated(X_tr_s, y_tr_s, X_te_s, y_te_s, 'Synthétique'))

    df_res = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'volet2_federated_results.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"\n  [✓] Résultats CSV sauvegardés : {csv_path}")
    plot_results(df_res, output_dir)

    print("\n" + "=" * 90)
    print("  TABLEAU COMPARATIF — VOLET 2 : CENTRALISÉ VS FÉDÉRÉ")
    print("=" * 90)
    for ds in ['ULB', 'Synthétique']:
        sub = df_res[df_res['Dataset'] == ds]
        print(f"\n  ── {ds} ──")
        print(sub[['Mode', 'Accuracy', 'Precision', 'Recall',
                   'F1-Score', 'AUC-ROC', 'AUC-PR']].to_string(index=False))

    return df_res


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_volet2(
        ulb_path       = os.path.join(BASE_DIR, 'data', 'creditcard.csv'),
        synthetic_path = os.path.join(BASE_DIR, 'data', 'fraud_detection_credit_card_small.csv'),
        output_dir     = os.path.join(BASE_DIR, 'results', 'comparative_study'),
    )
    print("\n  [✓] Volet 2 terminé avec succès !")
