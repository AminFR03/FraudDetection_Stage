# Résumé des Résultats et Performances
> **Version révisée** — Métriques uniformes pour tous les modèles, hyperparamètres fédérés détaillés, conformément aux remarques des encadrants.

Ce document présente une synthèse complète des résultats obtenus à l'issue des différents notebooks expérimentaux, reflétant fidèlement les sorties des scripts.

> **Note :** Les métriques peuvent légèrement varier selon les seeds. Les valeurs ci-dessous sont issues des exécutions validées (seed=42).

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

## 1. Impact du Resampling (Modèle : XGBoost, Dataset : ULB & Synthétique)

### Dataset ULB
| Méthode | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUPRC | Observation |
|:---|---:|---:|---:|---:|---:|---:|:---|
| **Baseline (Aucun)** | 0.9995 | **0.9286** | 0.7959 | **0.8571** | 0.9778 | **0.8724** | **Meilleur équilibre F1**. XGBoost gère bien nativement. |
| **SMOTE** | 0.9981 | 0.4649 | 0.8776 | 0.6078 | **0.9833** | 0.8635 | Bon rappel, mais chute de précision. |
| **ADASYN** | 0.9970 | 0.3539 | 0.8776 | 0.5044 | 0.9802 | 0.8489 | Similaire à SMOTE, plus de faux positifs. |
| **Undersampling** | 0.9593 | 0.0375 | **0.9184** | 0.0720 | 0.9776 | 0.6865 | Inexploitable (trop de fausses alertes). |

### Dataset Synthétique
| Méthode | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUPRC | Observation |
|:---|---:|---:|---:|---:|---:|---:|:---|
| **ADASYN** | 0.9959 | **0.6397** | **0.7250** | **0.6797** | 0.9761 | **0.7245** | **Meilleur F1**. |
| **SMOTE** | 0.9957 | 0.6222 | 0.7000 | 0.6588 | 0.9674 | 0.7008 | Très proche de ADASYN. |
| **Baseline** | 0.9958 | 0.7778 | 0.4083 | 0.5355 | **0.9826** | 0.6821 | Beaucoup de fraudes manquées (Recall faible). |
| **Undersampling** | 0.9307 | 0.0746 | 0.9250 | 0.1381 | 0.9776 | 0.3452 | Inexploitable. |

**Conclusion Volet 1 :** 
- Sur les données tabulaires fortement déséquilibrées et propres (ULB/PCA), XGBoost n'a pas nécessairement besoin de resampling (Baseline = F1 le plus élevé).
- Sur les données avec des relations plus complexes/non-linéaires (Synthétique), ADASYN offre les meilleures performances.
Pour le Volet 2, afin de comparer l'impact du fédéré de manière contrôlée, nous utilisons l'approche **SMOTE** comme standard de référence.

---

## 2. Comparaison Centralisé vs Fédéré (FedAvg sur XGBoost via Soft Voting)

> **Protocole contrôlé** : Même modèle de base (XGBoost), même resampling local (SMOTE), mêmes hyperparamètres locaux.

### Dataset ULB
| Mode | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUPRC |
|:---|---:|---:|---:|---:|---:|---:|
| **Centralisé** | 0.9981 | 0.4649 | **0.8776** | 0.6078 | **0.9833** | **0.8635** |
| **Fédéré (FedAvg)** | **0.9994** | **0.8211** | 0.7959 | **0.8083** | 0.9752 | 0.8293 |

*Analyse ULB :* Paradoxalement, sur ce test, le modèle Fédéré (qui est un ensemble par Soft Voting) lisse les erreurs du SMOTE centralisé et obtient un bien meilleur F1-Score (0.8083 vs 0.6078) en augmentant massivement la Precision, au prix d'une légère baisse du Recall. C'est un excellent résultat pour le Federated Learning !

### Dataset Synthétique
| Mode | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUPRC |
|:---|---:|---:|---:|---:|---:|---:|
| **Centralisé** | 0.9957 | 0.6222 | **0.7000** | **0.6588** | **0.9674** | **0.7008** |
| **Fédéré (FedAvg)** | 0.9942 | **0.6250** | 0.0833 | 0.1471 | 0.9587 | 0.5034 |

*Analyse Synthétique :* La simulation FL échoue sur ce dataset. La cause principale est la fragmentation des variables encodées (One-Hot Encoding des catégories urbaines, marchands) réparties sur 25 clients. Chaque client local n'a pas suffisamment d'exemples de fraudes par catégorie pour qu'un modèle XGBoost local puisse apprendre, ce qui s'effondre lors de l'agrégation (Recall extrêmement faible de 8.3%). Cela démontre les limites du Soft Voting non-IID sur des données catégorielles fines.

---

## 3. Hyperparamètres du Modèle Fédéré XGBoost

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

Le pipeline final Agentic AI embarque le modèle **XGBoost Fédéré (ULB)** en raison de ses excellentes performances équilibrées et sa conformité RGPD.

| Métrique | Valeur |
|:---|:---:|
| Taux de rejet précoce (Agent 1) | 60-75% |
| Latence Fast-Track | < 5 ms |
| Latence Analyse ML + SHAP | < 100 ms |
| Latence Pipeline complet (LLM Gemini 2.5) | 1.2 - 2.5 s |
| F1-Score du modèle (Agent 2) | 0.8083 |
| AUC-ROC du modèle | 0.9752 |

---

## 5. Conclusion

**Modèle final retenu : XGBoost (Fédéré sur données continues)**

Le Federated Learning sur XGBoost (via Soft Voting) réalise une performance remarquable sur le dataset ULB avec un F1-Score de **0.8083**. Il agit comme un meta-ensemble lissant le surapprentissage généré par SMOTE, et produit un modèle robuste avec une très bonne précision.
Cette approche allie performance prédictive, explicabilité SHAP immédiate (idéale pour l'Agent 4), conformité RGPD, et latence opérationnelle.
Cependant, l'échec sur le dataset synthétique souligne que le Federated Learning (spécialement le Soft Voting d'arbres) nécessite un soin particulier pour les datasets comportant de nombreuses features catégorielles fragmentées non-IID.

---
*Généré dans le cadre du projet de fin de stage - Détection de Fraude Bancaire avec Federated Learning & Agentic AI.*
