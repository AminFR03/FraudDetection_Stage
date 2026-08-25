# 📊 Project Parameters & Results — Fraud Detection System
> **Référence complète** : Tous les paramètres du système, hyperparamètres ML, règles métier et résultats expérimentaux.
> *Généré dans le cadre du stage : Détection de Fraude Bancaire avec Agentic AI, LLMs et Federated Learning.*
> *Seed global : `42` | Test split : `20%` | Dataset principal : ULB Credit Card*

---

## 📁 Où trouver quoi

| Contenu | Fichier |
|:---|:---|
| Ce document (référence complète) | `results/project_parameters_and_results.md` |
| Résultats Volet 1 (CSV brut) | `results/comparative_study/volet1_imbalance_results.csv` |
| Résultats Volet 2 (CSV brut) | `results/comparative_study/volet2_federated_results.csv` |
| Analyse critique (Volets 1&2) | `results/comparative_study/analyse_critique.md` |
| Barplot Volet 1 | `results/comparative_study/volet1_barplot.png` |
| Barplot Volet 2 | `results/comparative_study/volet2_barplot.png` |
| Technique retenue (JSON) | `results/best_technique.json` |
| Alertes du pipeline (JSONL) | `results/alerts.jsonl` *(généré à l exécution)* |
| Base de données décisions | `scripts/data/agent_memory.db` |
| Notebooks d exploration | `notebooks/01_Data_Exploration.ipynb` → `08_Results_Comparison.ipynb` |
| Scripts comparatifs | `scripts/comparative_study/volet1_imbalance_comparison.py`, `volet2_federated_comparison.py` |
| Algorithme FedAvg | `scripts/federated/fed_avg.py` |
| Pipeline agents (démo) | `scripts/agents/pipeline_demo.py` |

---

## 🗃️ Datasets

| Dataset | Fichier | Lignes | Features | Fraudes | Taux |
|:---|:---|---:|---:|---:|---:|
| **ULB Credit Card** | `data/creditcard.csv` | 284 807 | 30 (V1–V28 + Amount + Time) | 492 | 0.172% |
| **Synthetic Credit Card** | `data/fraud_detection_credit_card_small.csv` | ~120 000 | 21 (transactionnelles brutes) | ~1 200 | ~1.0% |

**Prétraitement commun** : `RobustScaler` sur Amount/Time (ULB) et features numériques (Synthetic).

---

## ⚙️ Volet 1 — Paramètres des Techniques de Rééquilibrage

**Script** : `scripts/comparative_study/volet1_imbalance_comparison.py`

### Hyperparamètres XGBoost (Ensemble de 3 modèles, Early Stopping)

| Paramètre | Valeur |
|:---|:---:|
| `n_estimators` (max) | **2000** |
| `early_stopping_rounds` | **50** |
| `max_depth` | **7** |
| `learning_rate` | **0.05** |
| `subsample` | **0.85** |
| `colsample_bytree` | **0.85** |
| `min_child_weight` | **3** |
| `gamma` | **0.05** |
| `eval_metric` | `logloss` |
| Ensemble | **3 seeds** (42, 123, 456) |
| Seuil décision | **Optimal F1** (1000 points) |
| `tree_method` | `hist` |

### Ingénierie des features

| Dataset | Ajouts clés |
|:---|:---|
| **ULB** | Interactions PCA (V14×V12, V14×V17, V10×V14, V4×V11, V14², V12², V17², V10²), log(Amount), score Isolation Forest |
| **Synthétique** | Ratios montant carte (cc_amt_mean/std/max, amt_ratio_cc, amt_zscore_cc), ratios marchand/catégorie, distance Haversine (km), âge client, features temporelles, score Isolation Forest |

### Techniques comparées

| Technique | Librairie | `sampling_strategy` |
|:---|:---|:---:|
| **Baseline** | — | — (scale_pos_weight auto) |
| **SMOTE** | `imbalanced-learn` | 0.5 (ULB) / 0.3 (Syn.) |
| **ADASYN** | `imbalanced-learn` | 0.5 (ULB) / 0.3 (Syn.) |
| **Undersampling** | `imbalanced-learn.RandomUnderSampler` | balanced |

---

## ⚙️ Volet 2 — Paramètres Federated Learning

**Script** : `scripts/comparative_study/volet2_federated_comparison.py`

### Paramètres de la simulation FL

| Paramètre | Valeur | Description |
|:---|:---:|:---|
| `NUM_CLIENTS` | **10** | Banques fictives simulées |
| `CLIENTS_PER_ROUND` | **8** | Banques participantes par round (80%) |
| `NUM_ROUNDS` | **12** | Rounds fédérés totaux |
| Méthode d agrégation | **Soft Voting** | Moyenne des probabilités sur tous les modèles |
| Partition clients | **Stratifiée** | Fraudes garanties dans chaque partition |
| Resampling local | **SMOTE** | Appliqué indépendamment chez chaque client |
| `random_state` | **42** | Reproductibilité |

### Hyperparamètres XGBoost (identiques centralisé ET fédéré)

| Paramètre | Valeur |
|:---|:---:|
| `n_estimators` (max) | **2000** |
| `early_stopping_rounds` | **30** (client) / **50** (centralisé) |
| `max_depth` | **7** |
| `learning_rate` | **0.05** |
| `subsample` | **0.85** |
| `colsample_bytree` | **0.85** |

---

## 📈 Résultats Volet 1 — Impact du Resampling

> Modèle : **XGBoost Ensemble (3 modèles)** + Isolation Forest | Seed : `42` | Test split : 20%

### Dataset ULB

| Technique | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|:---|---:|---:|---:|---:|---:|---:|
| **Baseline** | 0.9996 | **0.9747** | 0.7857 | 0.8701 | 0.9788 | 0.8817 |
| **SMOTE** | 0.9996 | 0.9518 | 0.8061 | 0.8729 | **0.9828** | 0.8802 |
| **ADASYN** | 0.9996 | 0.9750 | 0.7959 | **0.8764** | 0.9812 | 0.8739 |
| Undersampling | 0.9993 | 0.8333 | 0.7653 | 0.7979 | 0.9757 | **0.8817** |

**Meilleur F1 sur ULB : ADASYN (F1 = 0.8764) | Meilleure Précision : ADASYN (0.9750)**

### Dataset Synthétique

| Technique | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|:---|---:|---:|---:|---:|---:|---:|
| **Baseline** | 0.9986 | **0.9694** | 0.7917 | **0.8716** | **0.9911** | **0.9001** |
| SMOTE | 0.9983 | 0.9135 | 0.7917 | 0.8482 | 0.9899 | 0.8773 |
| ADASYN | 0.9982 | 0.8621 | **0.8333** | 0.8475 | 0.9899 | 0.8811 |
| Undersampling | 0.9943 | 0.5203 | 0.6417 | 0.5746 | 0.9842 | 0.5518 |

**Meilleur F1 sur Synthétique : Baseline (F1 = 0.8716) | AUC-PR = 0.9001 ✓**

**Technique retenue pour Volet 2** : **SMOTE** (standard de référence contrôlé).

---

## 📈 Résultats Volet 2 — Centralisé vs Federated Learning

> Modèle : **XGBoost Ensemble + SMOTE** | Seed : `42` | 10 clients, 12 rounds, Soft Voting

### Dataset ULB

| Mode | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|:---|---:|---:|---:|---:|---:|---:|
| **Centralisé (SMOTE)** | 0.9996 | **0.9518** | **0.8061** | **0.8729** | **0.9828** | **0.8802** |
| Fédéré (FedAvg) | 0.9994 | 0.9024 | 0.7551 | 0.8222 | 0.9665 | 0.8522 |

**Résultat clé ULB** : Centralisé surpasse Fédéré de 0.0507 en F1 (−6.2%). FL conserve Précision > 0.90 (0.9024).

### Dataset Synthétique

| Mode | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|:---|---:|---:|---:|---:|---:|---:|
| **Centralisé (SMOTE)** | 0.9983 | **0.9135** | **0.7917** | **0.8482** | **0.9899** | **0.8773** |
| Fédéré (FedAvg) | 0.9962 | 0.6964 | 0.6500 | 0.6724 | 0.9823 | 0.6859 |

**Résultat clé Synthétique** : Centralisé surpasse Fédéré de 0.1758 en F1 (−20.7%). Le FL maintient AUC-ROC = 0.9823 — excellente discrimination globale.

---

## 🤖 Système Multi-Agents — Paramètres

### Modèle ML final retenu

| Paramètre | Valeur |
|:---|:---|
| Modèle | **XGBoost Ensemble Centralisé + SMOTE (ULB)** |
| F1-Score | **0.8729** |
| Précision | **0.9518** |
| Recall | **0.8061** |
| AUC-ROC | **0.9828** |
| AUC-PR | **0.8802** |
| Justification | Meilleur équilibre F1/Précision, Précision > 0.95, état de l’art ULB |

---

### Agent 1 — SurveillanceAgent

**Fichier** : `scripts/agents/surveillance_agent.py`

#### Seuils ULB (`ULB_THRESHOLDS`)

| Feature | Condition | Seuil |
|:---|:---:|:---:|
| `Amount` | > | 2.0 σ |
| `V14` | < | −2.5 |
| `V12` | < | −2.0 |
| `V10` | < | −2.0 |
| `V4` | > | +2.5 |
| `V3` (abs) | > | 3.5 |

#### Seuils Synthétique (`SYNTHETIC_THRESHOLDS`)

| Feature | Condition | Seuil |
|:---|:---:|:---:|
| `amt` | > | $1 500 |
| `hour` | entre | 0h–4h |
| `category` | dans | travel, online_retail, shopping_net, misc_net, home, personal_care |

---

### Agent 2 — AnalysisAgent

**Fichier** : `scripts/agents/analysis_agent.py`

| Paramètre | Valeur |
|:---|:---|
| Modèle ML | XGBoost Fédéré (ULB) |
| Explainabilité | `shap.TreeExplainer` |
| Top features | **5** (`top_n=5`) |
| Scaler | `RobustScaler` |

#### Niveaux de risque

| Niveau | Seuil |
|:---|:---:|
| **CRITIQUE** | >= 85% |
| **ÉLEVÉ** | >= 60% |
| **MOYEN** | >= 35% |
| **FAIBLE** | < 35% |

---

### Agent 3 — DecisionAgent

**Fichier** : `scripts/agents/decision_agent.py`

| Décision | Seuil | Sévérité | SLA | Notifications |
|:---|:---:|:---:|:---:|:---|
| BLOCK | >= 85% | CRITIQUE | 0 min | equipe_fraude, client, conformite |
| REVIEW | >= 60% | ÉLEVÉ | 15 min | analyste_fraude, client |
| ALERT | >= 35% | MOYEN | 60 min | equipe_surveillance |
| ALLOW | < 35% | FAIBLE | — | — |

---

### Agent 4 — ExplanationAgent (LLM)

**Fichier** : `scripts/agents/explanation_agent.py`

| Paramètre | Valeur |
|:---|:---|
| Modèle LLM | `qwen2.5:7b` (local via Ollama) |
| Backend | Ollama — http://localhost:11434 |
| `temperature` | **0.3** |
| `max_tokens` | **500** |
| Langue | Français professionnel |
| Fallback | Template hors-ligne structuré |
| Latence typique (CPU) | 9–17 secondes |

---

### Agent 5 — FeedbackAgent

**Fichier** : `scripts/agents/feedback_agent.py`

| Type de feedback | Valeur |
|:---|:---|
| Fraude confirmée | `CONFIRMED_FRAUD` |
| Faux positif | `FALSE_POSITIVE` |
| Légitime | `LEGITIMATE` |
| Stockage | `scripts/data/agent_memory.db` (SQLite) |

---

### Agent 6 — MonitoringAgent

**Fichier** : `scripts/agents/monitoring_agent.py`

| Paramètre | Valeur |
|:---|:---|
| Fenêtre de monitoring | 24h |
| Source | DatabaseTool -> SQLite |
| Status nominal | `HEALTHY` |

---

### Orchestrateur — Workflows

**Fichier** : `scripts/agents/core/orchestrator.py`

| Workflow | Agents | Cas d usage |
|:---|:---|:---|
| `fast_track` | Surveillance seul | Transaction claire, < 5 ms |
| `standard` | Surveillance > Analyse > Décision > Explication | Pipeline normal |
| `escalation` | + Feedback | Cas douteux, validation humaine |
| `deep_analysis` | + Monitoring | Audit périodique |

---

### Système de Mémoire (4 couches)

**Fichier** : `scripts/agents/core/memory.py`

| Couche | Type | TTL | Stockage |
|:---|:---|:---|:---|
| Working Memory | Travail (transaction) | Pipeline en cours | RAM |
| Episodic Memory | Historique sessions | Session | deque (max 100) |
| Semantic Memory | Patterns fraude | Session | Features clés |
| Long-Term Memory | Décisions + feedback | Persistant | SQLite |

---

## 🚀 Performances Opérationnelles

| Etape | Latence typique |
|:---|:---:|
| Agent 1 Fast-Track | < 5 ms |
| Agent 1 Surveillance | < 1 ms |
| Agent 2 Analyse ML + SHAP | < 100 ms |
| Agent 3 Décision | < 1 ms |
| Agent 4 LLM Qwen (CPU) | 9–17 secondes |
| Pipeline complet (avec LLM) | ~9–18 secondes |
| Taux Fast-Track estimé | 60–75% |

---

*Dernière mise à jour : Août 2026 — Stage FDL Agentic AI*
