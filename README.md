# 🛡️ Détection de Fraude Bancaire basée sur l'Agentic AI, les LLMs et le Federated Learning

**Projet de recherche et d'expérimentation** visant à comparer rigoureusement les techniques de rééquilibrage de données, l'entraînement centralisé vs fédéré (FedAvg), et l'intégration d'un système multi-agents intelligent orienté explainabilité et gouvernance.

![Architecture Status](https://img.shields.io/badge/Architecture-Multi--Agents-blue)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest%20%7C%20DeepLearning-orange)
![Federated Learning](https://img.shields.io/badge/Federated%20Learning-FedAvg-green)
![LLM](https://img.shields.io/badge/LLM-Qwen%202.5%207B%20(Ollama)-yellow)

---

## 🌟 Fonctionnalités Clés

1. **Volet 1 — Gestion du Déséquilibre des Classes**
   - Comparaison rigoureuse de **Baseline**, **SMOTE**, **ADASYN**, et **Undersampling**.
   - Évaluation sur deux datasets : **ULB Credit Card** (données réelles anonymisées PCA) et **Synthetic Credit Card** (features transactionnelles brutes).
   - Métriques : Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUPRC.

2. **Volet 2 — Centralisé vs Federated Learning**
   - Comparaison d'un modèle **Centralisé** vs **Federated Learning** (Simulation 25 clients bancaires, FedAvg via Soft Voting).
   - Validation de l'impact de la fragmentation et de la confidentialité RGPD sur les performances.

3. **Agentic AI Framework (Blueprint 8 Étapes)**
   - **OrchestratorAgent** : Routing adaptatif (Fast-Track < 5ms, Standard, Escalation).
   - **SurveillanceAgent** : Ingestion & filtrage rapide par règles métier.
   - **AnalysisAgent** : Scoring ML & explicabilité localisée via SHAP (TreeExplainer).
   - **DecisionAgent** : Règles de gouvernance & déclenchement d'alertes.
   - **ExplanationAgent** : Génération de rapports NLG en français avec **Ollama (Qwen 2.5)** (+ fallback offline).
   - **FeedbackAgent & MonitoringAgent** : Boucle rétroactive Human-in-the-Loop et détection de drift.
   - **Memory Systems** : Architecture mémoire 4 couches (Working, Episodic, Semantic, Long-Term SQLite).

---

## 📊 Résultats Clés (Dataset ULB)

Le framework final utilise un **XGBoost Fédéré (FedAvg sur 25 banques)** qui réussit à lisser les effets du surapprentissage de SMOTE.

| Métrique | Modèle Centralisé (SMOTE) | Modèle Fédéré (SMOTE) | Modèle Centralisé (Baseline) |
|:---|---:|---:|---:|
| **F1-Score** | 0.6078 | **0.8083** | **0.8571** |
| **Precision** | 0.4649 | **0.8211** | 0.9286 |
| **Recall** | **0.8776** | 0.7959 | 0.7959 |
| **AUC-ROC** | 0.9833 | 0.9752 | 0.9778 |

> **Conclusion principale :** Le Federated Learning par Soft Voting apporte une robustesse inattendue en agissant comme un meta-ensemble, corrigeant la perte de précision massive induite par SMOTE en mode centralisé. Le modèle final (F1=0.8083) offre un compromis exceptionnel entre confidentialité (RGPD), performance et explicabilité.

---

## 📁 Structure du Projet

```
FraudDetection_Stage/
├── data/                         # Datasets (creditcard.csv, fraud_detection_credit_card_small.csv)
├── notebooks/                    # Notebooks Jupyter de recherche (exploration & tuning)
├── report/                       # Documentation, architecture & analyse détaillée
│   ├── agentic_ai_architecture.md
│   ├── architecture_diagram.html
│   ├── rapport_complet_stage.md
│   └── rapport_technique.md
├── results/                      # Résultats comparatifs & alertes (.csv, .jsonl, .md)
│   ├── comparative_study/        # Logs de l'étude (Volets 1 & 2)
│   └── project_parameters_and_results.md  # Paramètres complets & résultats
├── scripts/
│   ├── agents/                   # Framework Multi-Agents & Démo
│   │   ├── core/                 # Agent Base, Memory, Tools, Orchestrator
│   │   ├── surveillance_agent.py # Agent 1
│   │   ├── analysis_agent.py     # Agent 2
│   │   ├── decision_agent.py     # Agent 3
│   │   ├── explanation_agent.py  # Agent 4
│   │   ├── feedback_agent.py     # Agent 5
│   │   ├── monitoring_agent.py   # Agent 6
│   │   └── pipeline_demo.py      # Démo d'exécution bout-en-bout
│   ├── comparative_study/       # Scripts Volets 1 & 2
│   ├── federated/               # Algorithmes FedAvg (Soft Voting, Agrégation PyTorch)
│   └── utils/                   # Utilitaires métriques & prétraitement
├── requirements.txt              # Dépendances Python
└── README.md
```

---

## 🚀 Prise en Main & Exécution

### 1. Installation des dépendances
```bash
# Il est recommandé d'utiliser un environnement virtuel (venv ou conda)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configuration (LLM)
Créez un fichier `.env` à la racine du projet :
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```
Assurez-vous qu'**Ollama est installé et démarré** (`ollama pull qwen2.5:7b`) avant d'exécuter le pipeline.

### 3. Exécuter l'étude comparative complète (Volets 1 & 2)
> ⚠️ Nécessite la présence de `data/creditcard.csv`.
```bash
python scripts/comparative_study/run_full_study.py
```
*Les résultats CSV et Markdown seront générés dans le dossier `results/comparative_study/`.*

### 4. Lancer la démonstration Agentic AI Pipeline
```bash
python scripts/agents/pipeline_demo.py
```
*Ce script charge les données, exécute le workflow (Surveillance → Analyse → Décision → LLM) et affiche le rapport généré.*

### 5. Consulter l'Architecture Interactive
Ouvrez le fichier [report/architecture_diagram.html](file:///c:/Users/aminf/Desktop/Stage/FraudDetection_Stage/report/architecture_diagram.html) dans votre navigateur pour visualiser le flux du système Multi-Agents.

---

## 🛡️ Licence & Contribution

Ce projet a été développé dans le cadre d'un stage de fin d'études. Il s'appuie sur le framework conceptuel d'**Agentic AI** (Blueprint "How to Build an AI Agent").

- **Auteur :** [Votre Nom/Stagiaire]
- **Encadrant :** [Nom de l'encadrant]
- **Technologies :** Python, XGBoost, Scikit-learn, SHAP, PyTorch, Ollama (Qwen).
