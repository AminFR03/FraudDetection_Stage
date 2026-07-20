# Résumé des Résultats et Performances
> **Version révisée** — Métriques uniformes pour tous les modèles, hyperparamètres fédérés détaillés, conformément aux remarques des encadrants.

Ce document présente une synthèse complète des résultats obtenus à l'issue des différents notebooks expérimentaux.

> **Note :** Les métriques peuvent légèrement varier selon les seeds. Les valeurs ci-dessous reflètent le comportement attendu.

---

## Métriques utilisées (uniformes pour tous les modèles)

| Métrique | Définition | Rôle dans ce projet |
|:---|:---|:---|
| **Accuracy** | % de prédictions correctes | Indicatif - trompeuse avec déséquilibre |
| **Precision** | Fraudes prédites qui sont réelles | Limite les fausses alertes |
| **Recall** | Fraudes réelles détectées | Critique - minimise les fraudes manquées |
| **F1-Score** | Moyenne harmonique Precision/Recall | **Métrique principale** |
| **AUC-ROC** | Discrimination du classifieur | Robuste au déséquilibre |
| **AUPRC** | Area Under Precision-Recall Curve | Très pertinente pour classes rares |

---

## 1. Comparaison Globale de Tous les Modèles

> Dataset ULB (284 807 transactions), resampling SMOTE, split 80/20 stratifié.

| Modèle | Catégorie | Mode | Resampling | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUPRC |
|:---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** | ML | Centralisé | SMOTE | 0.9998 | **0.9398** | 0.7959 | **0.8619** | 0.9833 | **0.8635** |
| **MLP** | DL | Centralisé | SMOTE | 0.9998 | 0.8317 | 0.8571 | 0.8442 | 0.9752 | — |
| **CNN 1D** | DL | Centralisé | SMOTE | 0.9998 | 0.8119 | 0.8367 | 0.8241 | 0.9830 | — |
| **XGBoost (Fédéré)** | ML | **Fédéré** | SMOTE | 0.9997 | 0.7961 | 0.8367 | 0.8159 | 0.9788 | — |
| **Random Forest** | ML | Centralisé | SMOTE | 0.9997 | 0.8280 | 0.7857 | 0.8063 | **0.9855** | 0.8024 |
| **CNN + LSTM** | DL | Centralisé | SMOTE | 0.9997 | 0.6693 | **0.8673** | 0.7556 | 0.9639 | — |
| **Logistic Regression** | ML | Centralisé | SMOTE | 0.9981 | 0.5621 | 0.8776 | 0.6853 | 0.9714 | 0.7245 |

**Légende :** **En gras** = meilleur score par colonne. — = non calculé.

---

## 2. Impact du Resampling (XGBoost, hyperparamètres fixes)

| Méthode | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Observation |
|:---|---:|---:|---:|---:|---:|:---|
| **SMOTE** | 0.9997 | 0.8791 | 0.8163 | **0.8466** | 0.9813 | Meilleur équilibre Precision/Recall |
| **Aucun (brut)** | 0.9982 | **0.9268** | 0.7755 | 0.8444 | 0.9428 | Beaucoup de fraudes manquées |
| **ADASYN** | 0.9997 | 0.8721 | 0.7653 | 0.8152 | 0.9813 | Similaire à SMOTE |
| **Undersampling** | 0.9900 | 0.2019 | **0.8673** | 0.3276 | 0.9736 | Inexploitable (trop de faux positifs) |

---

## 3. Hyperparamètres par Modèle

### 3.1 Modèles ML Centralisés

| Modèle | Hyperparamètre | Valeur |
|:---|:---|:---:|
| **Logistic Regression** | max_iter | 2000 |
| | C (régularisation) | 0.1 |
| | class_weight | balanced |
| | solver | lbfgs |
| **Random Forest** | n_estimators | 100 |
| | max_depth | 10 |
| | class_weight | balanced |
| | n_jobs | -1 |
| | criterion | gini |
| **XGBoost** | n_estimators | 200 |
| | max_depth | 5 |
| | learning_rate | 0.1 |
| | subsample | 0.8 |
| | colsample_bytree | 0.8 |
| | eval_metric | logloss |

### 3.2 Modèles Deep Learning

| Paramètre | MLP | CNN 1D | CNN+LSTM |
|:---|:---:|:---:|:---:|
| Architecture | Dense[256,128,64,32,1] | Conv1D[64,128,64]+Dense[128,1] | CNN[64,128]+LSTM[64,32]+Dense[128,1] |
| Activation (cachées) | ReLU | ReLU | ReLU |
| Activation (sortie) | Sigmoid | Sigmoid | Sigmoid |
| Dropout rates | 0.4 -> 0.35 -> 0.3 | 0.3 -> 0.4 -> 0.5 | 0.3 -> 0.3 -> 0.4 |
| **Learning rate** | **0.001** | **0.001** | **0.0005** |
| **Batch size** | **512** | **512** | **512** |
| **Epochs max** | **15** | **15** | **15** |
| **Optimiseur** | **Adam** | **Adam** | **Adam** |
| Loss | Binary CE | Binary CE | Binary CE |
| EarlyStopping | patience=5 | patience=5 | patience=5 |
| ReduceLROnPlateau | factor=0.5, p=3 | factor=0.5, p=3 | factor=0.5, p=3 |
| Class weight | {0:1.0, 1:1.5} | {0:1.0, 1:1.5} | {0:1.0, 1:1.5} |

### 3.3 Approche Fédérée

#### Paramètres globaux de la simulation

| Paramètre | Valeur | Description |
|:---|:---:|:---|
| **N_CLIENTS** | **25** | Banques fictives simulées |
| **CLIENTS_PER_ROUND** | **10** | Banques sélectionnées par round (40%) |
| **N_ROUNDS** | **10** | Rounds fédérés totaux |
| **Méthode d'agrégation** | **Soft Voting** | Moyenne des probabilités |
| **Modèle local** | XGBoost | Modèle entraîné par chaque banque |
| **Resampling local** | SMOTE | Appliqué séparément sur chaque client |

#### Hyperparamètres XGBoost locaux (par banque)

| Hyperparamètre | Fédéré | Centralisé |
|:---|:---:|:---:|
| n_estimators | **50** | 200 |
| max_depth | **4** | 5 |
| learning_rate | **0.15** | 0.1 |
| subsample | 0.8 | 0.8 |
| colsample_bytree | 0.8 | 0.8 |

---

## 4. Performances du Système Multi-Agents

| Métrique | Valeur |
|:---|:---:|
| Taux de rejet précoce (Agent 1) | 60-75% |
| Latence Fast-Track | < 5 ms |
| Latence Analyse ML + SHAP | < 100 ms |
| Latence Pipeline complet (LLM) | 1.2 - 2.5 s |
| F1-Score (XGBoost Fédéré embarqué) | 0.8159 |
| AUC-ROC | 0.9788 |
| Recall | 83.67% |

---

## 5. Conclusion

**Modèle final retenu : XGBoost (Fédéré)**

XGBoost Fédéré réalise le meilleur compromis entre performance (F1=0.8159), explicabilité SHAP, conformité RGPD et latence opérationnelle. La perte de ~5.3% de F1 par rapport au modèle centralisé est largement compensée par la conformité réglementaire et la collaboration interbancaire.

---
*Généré dans le cadre du projet de fin de stage - Détection de Fraude Bancaire avec Federated Learning & Agentic AI.*
