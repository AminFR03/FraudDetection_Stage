"""
preprocessing.py — Utilitaires de prétraitement des données
Stage : Détection de Fraude Bancaire avec Federated Learning & Agentic AI
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement et exploration basique
# ─────────────────────────────────────────────────────────────────────────────

def load_ulb_dataset(path):
    """
    Charge le dataset ULB (creditcard.csv).
    Retourne: DataFrame complet.
    """
    df = pd.read_csv(path)
    print(f"[ULB] Chargé: {df.shape[0]:,} transactions × {df.shape[1]} colonnes")
    print(f"[ULB] Fraudes: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    return df


def load_synthetic_dataset(path, sample_frac=None, random_state=42):
    """
    Charge le dataset synthétique.
    sample_frac: fraction à échantillonner (ex: 0.1 pour 10%) pour les tests rapides.
    """
    df = pd.read_csv(path)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=random_state)
    target_col = 'is_fraud' if 'is_fraud' in df.columns else df.columns[-1]
    print(f"[Synthétique] Chargé: {df.shape[0]:,} transactions × {df.shape[1]} colonnes")
    print(f"[Synthétique] Fraudes: {df[target_col].sum():,} ({df[target_col].mean()*100:.3f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prétraitement du dataset ULB
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_ulb(df, test_size=0.2, val_size=0.15, top_k_features=None,
                   resampling='smote', scaler_type='robust', random_state=42):
    """
    Pipeline complet pour le dataset ULB.

    Args:
        df: DataFrame brut
        test_size: fraction test
        val_size: fraction validation (du train restant)
        top_k_features: int ou None (ex: 20 pour top-20 ANOVA)
        resampling: 'smote', 'adasyn', 'undersampling', ou None
        scaler_type: 'robust' ou 'standard'
        random_state: seed

    Returns:
        dict avec X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    feature_names = X.columns.tolist()

    # --- Split stratifié ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state)

    # --- Sélection de features (ANOVA) ---
    if top_k_features:
        selector = SelectKBest(f_classif, k=top_k_features)
        X_train = selector.fit_transform(X_train, y_train)
        X_val   = selector.transform(X_val)
        X_test  = selector.transform(X_test)
        feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
        print(f"[Features] Top {top_k_features} features sélectionnées: {feature_names}")
    else:
        X_train = X_train.values
        X_val   = X_val.values
        X_test  = X_test.values

    # --- Scaling ---
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    y_train_arr = y_train.values
    y_val_arr   = y_val.values
    y_test_arr  = y_test.values

    # --- Resampling (sur le train uniquement) ---
    if resampling == 'smote':
        rs = SMOTE(random_state=random_state)
        X_train, y_train_arr = rs.fit_resample(X_train, y_train_arr)
        print(f"[SMOTE] Après resampling: {X_train.shape[0]:,} samples")
    elif resampling == 'adasyn':
        rs = ADASYN(random_state=random_state)
        X_train, y_train_arr = rs.fit_resample(X_train, y_train_arr)
        print(f"[ADASYN] Après resampling: {X_train.shape[0]:,} samples")
    elif resampling == 'undersampling':
        rs = RandomUnderSampler(random_state=random_state)
        X_train, y_train_arr = rs.fit_resample(X_train, y_train_arr)
        print(f"[UnderSampling] Après resampling: {X_train.shape[0]:,} samples")
    else:
        print("[Resampling] Aucun resampling appliqué.")

    return {
        'X_train': X_train, 'y_train': y_train_arr,
        'X_val':   X_val,   'y_val':   y_val_arr,
        'X_test':  X_test,  'y_test':  y_test_arr,
        'feature_names': feature_names,
        'scaler': scaler
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prétraitement du dataset Synthétique
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_NUMERIC_COLS = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
                           'Customer_Age', 'Customer_Satisfaction_Score', 'Loyalty_Points_Earned']

SYNTHETIC_CATEGORICAL_COLS = ['category', 'gender', 'Transaction_Type', 'Payment_Method',
                               'Merchant_Category']


def preprocess_synthetic(df, use_cols=None, test_size=0.2, resampling='smote',
                         random_state=42):
    """
    Prétraitement simplifié pour le dataset synthétique.
    Conserve uniquement les colonnes numériques par défaut.
    """
    target_col = 'is_fraud' if 'is_fraud' in df.columns else df.columns[-1]

    if use_cols is None:
        use_cols = [c for c in SYNTHETIC_NUMERIC_COLS if c in df.columns]

    df_clean = df[use_cols + [target_col]].dropna()
    X = df_clean[use_cols]
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if resampling == 'smote':
        rs = SMOTE(random_state=random_state)
        X_train_sc, y_train_arr = rs.fit_resample(X_train_sc, y_train.values)
    else:
        y_train_arr = y_train.values

    return {
        'X_train': X_train_sc, 'y_train': y_train_arr,
        'X_test':  X_test_sc,  'y_test':  y_test.values,
        'feature_names': use_cols,
        'scaler': scaler,
        'df_test_raw': df_clean.iloc[len(X_train):]  # Pour explications lisibles
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Split fédéré (simulation multi-clients)
# ─────────────────────────────────────────────────────────────────────────────

def federated_split(X_train, y_train, num_clients=25, random_state=42):
    """
    Divise le dataset d'entraînement en num_clients partitions (non-IID simulé).
    Chaque client reçoit une fraction des données.

    Returns:
        list de (X_client_i, y_client_i)
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(X_train))
    splits = np.array_split(indices, num_clients)
    clients = []
    for i, idx in enumerate(splits):
        clients.append((X_train[idx], y_train[idx]))
        print(f"  Client {i+1:02d}: {len(idx):,} samples | "
              f"Fraudes: {y_train[idx].sum()} ({y_train[idx].mean()*100:.1f}%)")
    return clients
