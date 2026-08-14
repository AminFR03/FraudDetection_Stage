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

### Hyperparamètres XGBoost (identiques pour toutes les techniques)

| Paramètre | Valeur |
|:---|:---:|
| `n_estimators` | **200** |
| `max_depth` | **5** |
| `learning_rate` | **0.1** |
| `subsample` | **0.8** |
| `colsample_bytree` | **0.8** |
| `eval_metric` | `logloss` |
| `random_state` | **42** |

### Techniques comparées

| Technique | Librairie | Paramètres spécifiques |
|:---|:---|:---|
| **Baseline** | — | Aucun resampling |
| **SMOTE** | `imbalanced-learn` | `random_state=42` |
| **ADASYN** | `imbalanced-learn` | `random_state=42` |
| **Undersampling** | `imbalanced-learn.RandomUnderSampler` | `random_state=42` |

---

## ⚙️ Volet 2 — Paramètres Federated Learning

**Script** : `scripts/comparative_study/volet2_federated_comparison.py`

### Paramètres de la simulation FL

| Paramètre | Valeur | Description |
|:---|:---:|:---|
| `NUM_CLIENTS` | **25** | Banques fictives simulées |
| `CLIENTS_PER_ROUND` | **10** | Banques participantes par round (40%) |
| `NUM_ROUNDS` | **10** | Rounds fédérés totaux |
| Méthode d agrégation | **Soft Voting** | Moyenne des probabilités sur tous les clients |
| Resampling local | **SMOTE** | Appliqué indépendamment chez chaque client |
| `random_state` | **42** | Reproductibilité |

### Hyperparamètres XGBoost (identiques centralisé ET fédéré)

| Paramètre | Valeur |
|:---|:---:|
| `n_estimators` | **200** |
| `max_depth` | **5** |
| `learning_rate` | **0.1** |
| `subsample` | **0.8** |
| `colsample_bytree` | **0.8** |

---

## 📈 Résultats Volet 1 — Impact du Resampling

> Modèle : **XGBoost** | Seed : `42` | Test split : 20%

### Dataset ULB

| Technique | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|:---|---:|---:|---:|---:|---:|---:|
| **Baseline** | 0.9995 | **0.9286** | 0.7959 | **0.8571** | 0.9778 | 0.8724 |
| **SMOTE** | 0.9992 | 0.7143 | 0.8673 | 0.7834 | **0.9822** | **0.8766** |
| ADASYN | 0.9990 | 0.6667 | **0.8776** | 0.7577 | 0.9797 | 0.8648 |
| Undersampling | 0.9593 | 0.0375 | 0.9184 | 0.0720 | 0.9776 | 0.6865 |

**Meilleur F1 sur ULB : Baseline (F1 = 0.8571)**

### Dataset Synthétique

| Technique | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|:---|---:|---:|---:|---:|---:|---:|
| **ADASYN** | 0.9959 | **0.6397** | **0.7250** | **0.6797** | 0.9761 | **0.7245** |
| SMOTE | 0.9957 | 0.6222 | 0.7000 | 0.6588 | 0.9674 | 0.7008 |
| Baseline | 0.9958 | 0.7778 | 0.4083 | 0.5355 | **0.9826** | 0.6821 |
| Undersampling | 0.9307 | 0.0746 | 0.9250 | 0.1381 | 0.9776 | 0.3452 |

**Meilleur F1 sur Synthétique : ADASYN (F1 = 0.6797)**

**Technique retenue pour Volet 2** : **SMOTE** (standard de référence contrôlé).

---

## 📈 Résultats Volet 2 — Centralisé vs Federated Learning

> Modèle : **XGBoost + SMOTE** | Seed : `42` | 25 clients, 10 rounds, Soft Voting

### Dataset ULB

| Mode | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|:---|---:|---:|---:|---:|---:|---:|
| Centralisé (SMOTE) | 0.9992 | 0.7143 | **0.8673** | 0.7834 | **0.9822** | **0.8766** |
| **Fédéré (FedAvg)** | **0.9993** | **0.8261** | 0.7755 | **0.8000** | 0.9737 | 0.8392 |

**Résultat clé** : FL agit comme un meta-ensemble (+15% Precision, +2% F1 vs centralisé SMOTE).

### Dataset Synthétique

| Mode | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|:---|---:|---:|---:|---:|---:|---:|
| **Centralisé (SMOTE)** | 0.9957 | 0.6222 | **0.7000** | **0.6588** | **0.9674** | **0.7008** |
| Fédéré (FedAvg) | 0.9942 | **0.6250** | 0.0833 | 0.1471 | 0.9587 | 0.5034 |

**Echec FL sur Synthétique** : Recall = 8.3% — fragmentation des features One-Hot sur 25 clients (non-IID).

---

## 🤖 Système Multi-Agents — Paramètres

### Modèle ML final retenu

| Paramètre | Valeur |
|:---|:---|
| Modèle | **XGBoost Fédéré (ULB)** |
| F1-Score | **0.8000** |
| AUC-ROC | **0.9737** |
| Justification | Meilleur équilibre Performance / Precision / RGPD |

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
