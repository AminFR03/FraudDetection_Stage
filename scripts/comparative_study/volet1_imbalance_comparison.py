"""
volet1_imbalance_comparison.py — Volet 1 : Comparaison des techniques de rééquilibrage
Stage : Détection de Fraude Bancaire avec Federated Learning & Agentic AI

Compare SMOTE, ADASYN, RandomUnderSampling et Baseline (aucun resampling)
sur les deux datasets (ULB et Synthétique), avec XGBoost à hyperparamètres fixes.
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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
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
TEST_SIZE = 0.2

# Hyperparamètres XGBoost FIXES (identiques pour toutes les combinaisons)
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

# Colonnes numériques du dataset synthétique
SYNTHETIC_NUMERIC_COLS = [
    'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
    'Customer_Age', 'Customer_Satisfaction_Score', 'Loyalty_Points_Earned'
]

# Colonnes catégorielles du dataset synthétique (one-hot encoding)
SYNTHETIC_CATEGORICAL_COLS = [
    'category', 'gender', 'Transaction_Type', 'Payment_Method',
    'Merchant_Category'
]

TECHNIQUES = ['Baseline', 'SMOTE', 'ADASYN', 'Undersampling']


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions de chargement et prétraitement
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_ulb(path):
    """Charge et prépare le dataset ULB (creditcard.csv)."""
    print("\n" + "=" * 70)
    print("  CHARGEMENT DU DATASET ULB")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Fraudes: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Scaling avec RobustScaler (résistant aux outliers)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"  Fraudes train: {y_train.sum():,} | Fraudes test: {y_test.sum():,}")

    return X_train, X_test, y_train, y_test


def load_and_prepare_synthetic(path):
    """Charge et prépare le dataset Synthétique avec encodage catégoriel."""
    print("\n" + "=" * 70)
    print("  CHARGEMENT DU DATASET SYNTHÉTIQUE")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Fraudes: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.3f}%)")

    # Sélection des colonnes numériques disponibles
    num_cols = [c for c in SYNTHETIC_NUMERIC_COLS if c in df.columns]
    cat_cols = [c for c in SYNTHETIC_CATEGORICAL_COLS if c in df.columns]

    # Préparation features numériques
    df_features = df[num_cols].copy()

    # Encodage one-hot des colonnes catégorielles
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df_features = pd.concat([df_features, dummies], axis=1)

    print(f"  Features finales: {df_features.shape[1]} "
          f"({len(num_cols)} num + {df_features.shape[1] - len(num_cols)} cat encodées)")

    # Suppression des NaN
    mask = ~df_features.isna().any(axis=1)
    X = df_features[mask].values
    y = df.loc[mask, 'is_fraud'].values

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Scaling avec StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"  Fraudes train: {y_train.sum():,} | Fraudes test: {y_test.sum():,}")

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# Application du resampling
# ═══════════════════════════════════════════════════════════════════════════════

def apply_resampling(X_train, y_train, technique, dataset_name):
    """Applique la technique de resampling spécifiée."""
    strategy = 0.1 if dataset_name == 'ULB' else 'auto'
    
    if technique == 'Baseline':
        print(f"    [{technique}] Aucun resampling — {X_train.shape[0]:,} samples")
        return X_train, y_train

    elif technique == 'SMOTE':
        sampler = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        print(f"    [{technique}] {X_train.shape[0]:,} → {X_res.shape[0]:,} samples")
        return X_res, y_res

    elif technique == 'ADASYN':
        try:
            sampler = ADASYN(sampling_strategy=strategy, random_state=RANDOM_STATE)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            print(f"    [{technique}] {X_train.shape[0]:,} → {X_res.shape[0]:,} samples")
            return X_res, y_res
        except ValueError as e:
            print(f"    [{technique}] ADASYN a échoué ({e}), fallback vers SMOTE")
            sampler = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            print(f"    [SMOTE fallback] {X_train.shape[0]:,} → {X_res.shape[0]:,} samples")
            return X_res, y_res

    elif technique == 'Undersampling':
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        print(f"    [{technique}] {X_train.shape[0]:,} → {X_res.shape[0]:,} samples")
        return X_res, y_res

    else:
        raise ValueError(f"Technique inconnue: {technique}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entraînement et évaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(X_train, y_train, X_test, y_test, technique, dataset_name):
    """Entraîne XGBoost et calcule toutes les métriques."""
    # Resampling
    X_train_res, y_train_res = apply_resampling(X_train, y_train, technique, dataset_name)

    # Entraînement
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_res, y_train_res)

    # Prédictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Métriques
    metrics = {
        'Dataset': dataset_name,
        'Technique': technique,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'AUC-ROC': round(roc_auc_score(y_test, y_prob), 4),
        'AUC-PR': round(average_precision_score(y_test, y_prob), 4)
    }

    print(f"    → Acc={metrics['Accuracy']:.4f}  Prec={metrics['Precision']:.4f}  "
          f"Rec={metrics['Recall']:.4f}  F1={metrics['F1-Score']:.4f}  "
          f"AUC={metrics['AUC-ROC']:.4f}  AUPRC={metrics['AUC-PR']:.4f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_volet1_results(results_df, output_dir):
    """Génère les barplots comparatifs du Volet 1."""
    datasets = results_df['Dataset'].unique()
    metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
    colors = {
        'Baseline': '#7f8c8d',
        'SMOTE': '#3498db',
        'ADASYN': '#e67e22',
        'Undersampling': '#e74c3c'
    }

    fig, axes = plt.subplots(1, len(datasets), figsize=(20, 8))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        df_sub = results_df[results_df['Dataset'] == dataset]
        techniques = df_sub['Technique'].tolist()

        x = np.arange(len(metrics_to_plot))
        width = 0.18
        n_tech = len(techniques)

        for i, tech in enumerate(techniques):
            vals = [df_sub[df_sub['Technique'] == tech][m].values[0] for m in metrics_to_plot]
            offset = (i - (n_tech - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                          label=tech, color=colors.get(tech, '#95a5a6'),
                          alpha=0.85, edgecolor='white', linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                        fontweight='bold', rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Volet 1 — {dataset}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Comparaison des Techniques de Rééquilibrage (XGBoost)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Graphique standard
    path = os.path.join(output_dir, 'volet1_barplot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [✓] Graphique barplot sauvegardé : {path}")
    return path


def plot_volet1_confusion_matrices(all_cms, output_dir):
    """Génère les matrices de confusion pour toutes les combinaisons."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (title, cm) in enumerate(all_cms.items()):
        if idx < len(axes):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Prédit')
            axes[idx].set_ylabel('Vrai')

    plt.suptitle('Matrices de Confusion — Volet 1 (Techniques de Rééquilibrage)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'volet1_confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Matrices de confusion sauvegardées : {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Exécution du Volet 1
# ═══════════════════════════════════════════════════════════════════════════════

def run_volet1(ulb_path, synthetic_path, output_dir):
    """Exécute l'étude comparative complète du Volet 1."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    # ── Dataset ULB ──
    print("\n" + "═" * 70)
    print("  VOLET 1 — DATASET ULB")
    print("═" * 70)
    X_train_ulb, X_test_ulb, y_train_ulb, y_test_ulb = load_and_prepare_ulb(ulb_path)

    for tech in TECHNIQUES:
        print(f"\n  ▸ Technique: {tech}")
        metrics = train_and_evaluate(
            X_train_ulb, y_train_ulb, X_test_ulb, y_test_ulb,
            technique=tech, dataset_name='ULB'
        )
        all_results.append(metrics)

    # ── Dataset Synthétique ──
    print("\n" + "═" * 70)
    print("  VOLET 1 — DATASET SYNTHÉTIQUE")
    print("═" * 70)
    X_train_syn, X_test_syn, y_train_syn, y_test_syn = load_and_prepare_synthetic(synthetic_path)

    for tech in TECHNIQUES:
        print(f"\n  ▸ Technique: {tech}")
        metrics = train_and_evaluate(
            X_train_syn, y_train_syn, X_test_syn, y_test_syn,
            technique=tech, dataset_name='Synthétique'
        )
        all_results.append(metrics)

    # ── Résultats ──
    results_df = pd.DataFrame(all_results)

    # Sauvegarde CSV
    csv_path = os.path.join(output_dir, 'volet1_imbalance_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  [✓] Résultats CSV sauvegardés : {csv_path}")

    # Graphiques
    plot_path = plot_volet1_results(results_df, output_dir)

    # Affichage du tableau
    print("\n" + "=" * 90)
    print("  TABLEAU COMPARATIF — VOLET 1 : TECHNIQUES DE RÉÉQUILIBRAGE")
    print("=" * 90)

    # Tableau pivot pour affichage
    for dataset in ['ULB', 'Synthétique']:
        df_ds = results_df[results_df['Dataset'] == dataset]
        print(f"\n  ── {dataset} ──")
        print(df_ds[['Technique', 'Accuracy', 'Precision', 'Recall',
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

    results = run_volet1(ULB_PATH, SYNTHETIC_PATH, OUTPUT_DIR)
    print("\n  [✓] Volet 1 terminé avec succès !")
