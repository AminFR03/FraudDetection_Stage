"""
volet2_federated_comparison.py — Volet 2 : Centralisé vs Federated Learning
Stage : Détection de Fraude Bancaire avec Federated Learning & Agentic AI

Compare un entraînement centralisé vs FedAvg (soft voting) sur les deux datasets,
avec les MÊMES hyperparamètres XGBoost pour garantir une comparaison équitable.
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
from sklearn.preprocessing import RobustScaler, StandardScaler
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

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Hyperparamètres XGBoost IDENTIQUES pour centralisé ET fédéré
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

# Paramètres Federated Learning
NUM_CLIENTS = 25
CLIENTS_PER_ROUND = 10
NUM_ROUNDS = 10

# Colonnes du dataset synthétique
SYNTHETIC_NUMERIC_COLS = [
    'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
    'Customer_Age', 'Customer_Satisfaction_Score', 'Loyalty_Points_Earned'
]
SYNTHETIC_CATEGORICAL_COLS = [
    'category', 'gender', 'Transaction_Type', 'Payment_Method',
    'Merchant_Category'
]


# ═══════════════════════════════════════════════════════════════════════════════
# Chargement des données (identique au Volet 1 pour cohérence)
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_ulb(path):
    """Charge et prépare le dataset ULB."""
    print("\n" + "=" * 70)
    print("  CHARGEMENT DU DATASET ULB")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Fraudes: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


def load_and_prepare_synthetic(path):
    """Charge et prépare le dataset Synthétique."""
    print("\n" + "=" * 70)
    print("  CHARGEMENT DU DATASET SYNTHÉTIQUE")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Fraudes: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.3f}%)")

    num_cols = [c for c in SYNTHETIC_NUMERIC_COLS if c in df.columns]
    cat_cols = [c for c in SYNTHETIC_CATEGORICAL_COLS if c in df.columns]

    df_features = df[num_cols].copy()
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df_features = pd.concat([df_features, dummies], axis=1)

    print(f"  Features: {df_features.shape[1]}")

    mask = ~df_features.isna().any(axis=1)
    X = df_features[mask].values
    y = df.loc[mask, 'is_fraud'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# Mode Centralisé
# ═══════════════════════════════════════════════════════════════════════════════

def train_centralized(X_train, y_train, X_test, y_test, dataset_name):
    """Entraîne XGBoost de manière centralisée avec SMOTE."""
    print(f"\n  ▸ Mode CENTRALISÉ — {dataset_name}")

    # SMOTE sur tout le train
    strategy = 0.1 if dataset_name == 'ULB' else 'auto'
    smote = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"    SMOTE: {X_train.shape[0]:,} → {X_train_res.shape[0]:,} samples")

    # Entraînement
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_res, y_train_res)

    # Évaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob, 'Centralisé', dataset_name)
    print(f"    → Acc={metrics['Accuracy']:.4f}  Prec={metrics['Precision']:.4f}  "
          f"Rec={metrics['Recall']:.4f}  F1={metrics['F1-Score']:.4f}  "
          f"AUC={metrics['AUC-ROC']:.4f}  AUPRC={metrics['AUC-PR']:.4f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Mode Federated Learning (FedAvg — Soft Voting)
# ═══════════════════════════════════════════════════════════════════════════════

def federated_split(X_train, y_train, num_clients):
    """Divise le train en partitions client (IID aléatoire)."""
    np.random.seed(RANDOM_STATE)
    indices = np.random.permutation(len(X_train))
    splits = np.array_split(indices, num_clients)
    clients = []
    for i, idx in enumerate(splits):
        clients.append((X_train[idx], y_train[idx]))
    return clients


def train_federated(X_train, y_train, X_test, y_test, dataset_name):
    """Entraîne XGBoost en mode fédéré (FedAvg / Soft Voting)."""
    print(f"\n  ▸ Mode FÉDÉRÉ — {dataset_name}")
    print(f"    Config: {NUM_CLIENTS} clients, {CLIENTS_PER_ROUND}/round, {NUM_ROUNDS} rounds")

    # Partitionnement en clients
    clients_data = federated_split(X_train, y_train, NUM_CLIENTS)

    strategy = 0.1 if dataset_name == 'ULB' else 'auto'

    # Stats par client
    fraud_counts = [int(y.sum()) for _, y in clients_data]
    sizes = [len(y) for _, y in clients_data]
    print(f"    Taille clients: {min(sizes):,}-{max(sizes):,} samples")
    print(f"    Fraudes/client: {min(fraud_counts)}-{max(fraud_counts)}")

    np.random.seed(RANDOM_STATE)
    all_client_models = [None] * NUM_CLIENTS

    for round_num in range(1, NUM_ROUNDS + 1):
        # Sélection aléatoire des clients
        selected = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)

        trained_count = 0
        for client_id in selected:
            X_c, y_c = clients_data[client_id]

            # Skip si le client n'a qu'une seule classe
            if len(np.unique(y_c)) < 2:
                continue

            # SMOTE local sur les données du client
            try:
                smote = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
                X_c_res, y_c_res = smote.fit_resample(X_c, y_c)
            except ValueError:
                # Pas assez de samples minoritaires pour SMOTE
                X_c_res, y_c_res = X_c, y_c

            # Entraînement local avec les MÊMES hyperparamètres
            model = xgb.XGBClassifier(**XGB_PARAMS)
            model.fit(X_c_res, y_c_res)
            all_client_models[client_id] = model
            trained_count += 1

        # Évaluation intermédiaire
        active_models = [m for m in all_client_models if m is not None]
        if active_models:
            y_prob = np.zeros(len(X_test))
            for m in active_models:
                y_prob += m.predict_proba(X_test)[:, 1]
            y_prob /= len(active_models)
            y_pred_round = (y_prob >= 0.5).astype(int)
            f1_round = f1_score(y_test, y_pred_round, zero_division=0)
            print(f"    Round {round_num:02d}/{NUM_ROUNDS} — "
                  f"{trained_count} clients entraînés — "
                  f"F1={f1_round:.4f} ({len(active_models)} modèles actifs)")

    # Évaluation finale — Soft Voting
    active_models = [m for m in all_client_models if m is not None]
    print(f"\n    Agrégation finale: {len(active_models)} modèles (Soft Voting)")

    y_prob_final = np.zeros(len(X_test))
    for m in active_models:
        y_prob_final += m.predict_proba(X_test)[:, 1]
    y_prob_final /= len(active_models)
    y_pred_final = (y_prob_final >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred_final, y_prob_final,
                              'Fédéré (FedAvg)', dataset_name)
    print(f"    → Acc={metrics['Accuracy']:.4f}  Prec={metrics['Precision']:.4f}  "
          f"Rec={metrics['Recall']:.4f}  F1={metrics['F1-Score']:.4f}  "
          f"AUC={metrics['AUC-ROC']:.4f}  AUPRC={metrics['AUC-PR']:.4f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Métriques
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_test, y_pred, y_prob, mode_name, dataset_name):
    """Calcule toutes les métriques."""
    return {
        'Dataset': dataset_name,
        'Mode': mode_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'AUC-ROC': round(roc_auc_score(y_test, y_prob), 4),
        'AUC-PR': round(average_precision_score(y_test, y_prob), 4)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_volet2_results(results_df, output_dir):
    """Génère les barplots comparatifs du Volet 2."""
    datasets = results_df['Dataset'].unique()
    metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
    colors = {
        'Centralisé': '#2ecc71',
        'Fédéré (FedAvg)': '#9b59b6'
    }

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 7))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        df_sub = results_df[results_df['Dataset'] == dataset]
        modes = df_sub['Mode'].tolist()

        x = np.arange(len(metrics_to_plot))
        width = 0.30
        n_modes = len(modes)

        for i, mode in enumerate(modes):
            vals = [df_sub[df_sub['Mode'] == mode][m].values[0] for m in metrics_to_plot]
            offset = (i - (n_modes - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                          label=mode, color=colors.get(mode, '#95a5a6'),
                          alpha=0.85, edgecolor='white', linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                        fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Volet 2 — {dataset}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Centralisé vs Federated Learning (XGBoost, mêmes hyperparamètres)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, 'volet2_barplot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [✓] Graphique sauvegardé : {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Exécution du Volet 2
# ═══════════════════════════════════════════════════════════════════════════════

def run_volet2(ulb_path, synthetic_path, output_dir):
    """Exécute l'étude comparative complète du Volet 2."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    # ── Dataset ULB ──
    print("\n" + "═" * 70)
    print("  VOLET 2 — DATASET ULB")
    print("═" * 70)
    X_train_ulb, X_test_ulb, y_train_ulb, y_test_ulb = load_and_prepare_ulb(ulb_path)

    metrics_cent = train_centralized(
        X_train_ulb, y_train_ulb, X_test_ulb, y_test_ulb, 'ULB')
    all_results.append(metrics_cent)

    metrics_fed = train_federated(
        X_train_ulb, y_train_ulb, X_test_ulb, y_test_ulb, 'ULB')
    all_results.append(metrics_fed)

    # ── Dataset Synthétique ──
    print("\n" + "═" * 70)
    print("  VOLET 2 — DATASET SYNTHÉTIQUE")
    print("═" * 70)
    X_train_syn, X_test_syn, y_train_syn, y_test_syn = load_and_prepare_synthetic(synthetic_path)

    metrics_cent = train_centralized(
        X_train_syn, y_train_syn, X_test_syn, y_test_syn, 'Synthétique')
    all_results.append(metrics_cent)

    metrics_fed = train_federated(
        X_train_syn, y_train_syn, X_test_syn, y_test_syn, 'Synthétique')
    all_results.append(metrics_fed)

    # ── Résultats ──
    results_df = pd.DataFrame(all_results)

    csv_path = os.path.join(output_dir, 'volet2_federated_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  [✓] Résultats CSV sauvegardés : {csv_path}")

    plot_path = plot_volet2_results(results_df, output_dir)

    # Affichage
    print("\n" + "=" * 90)
    print("  TABLEAU COMPARATIF — VOLET 2 : CENTRALISÉ VS FÉDÉRÉ")
    print("=" * 90)

    for dataset in ['ULB', 'Synthétique']:
        df_ds = results_df[results_df['Dataset'] == dataset]
        print(f"\n  ── {dataset} ──")
        print(df_ds[['Mode', 'Accuracy', 'Precision', 'Recall',
                      'F1-Score', 'AUC-ROC', 'AUC-PR']].to_string(index=False))

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ULB_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
    SYNTHETIC_PATH = os.path.join(BASE_DIR, 'data', 'fraud_detection_credit_card_small.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'comparative_study')

    results = run_volet2(ULB_PATH, SYNTHETIC_PATH, OUTPUT_DIR)
    print("\n  [✓] Volet 2 terminé avec succès !")
