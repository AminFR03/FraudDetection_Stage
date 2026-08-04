# 🛡️ Détection de Fraude Bancaire basée sur l'Agentic AI, les LLMs et le Federated Learning

Projet de recherche et d'expérimentation visant à comparer rigoureusement les techniques de rééquilibrage de données, l'entraînement centralisé vs fédéré (FedAvg), et l'intégration d'un système multi-agents intelligent orienté explainabilité et gouvernance.

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
   - **ExplanationAgent** : Génération de rapports NLG en français avec **Google Gemini 2.5 Flash** (+ fallback offline).
   - **FeedbackAgent & MonitoringAgent** : Boucle rétroactive Human-in-the-Loop et détection de drift.
   - **Memory Systems** : Architecture mémoire 4 couches (Working, Episodic, Semantic, Long-Term SQLite).

---

## 📁 Structure du Projet

```
FraudDetection_Stage/
├── data/                         # Datasets (creditcard.csv, synthetic)
├── notebooks/                    # Notebooks Jupyter de recherche
├── report/                       # Rapports & Diagrammes d'Architecture
│   ├── agentic_ai_architecture.md
│   ├── architecture_diagram.html
│   └── rapport_complet_stage.md
├── results/                      # Résultats comparatifs & alertes
│   └── comparative_study/
├── scripts/
│   ├── agents/                   # Framework Multi-Agents & Démo
│   │   ├── core/                 # Agent Base, Memory, Tools, Orchestrator
│   │   ├── surveillance_agent.py
│   │   ├── analysis_agent.py
│   │   ├── decision_agent.py
│   │   ├── explanation_agent.py
│   │   ├── feedback_agent.py
│   │   ├── monitoring_agent.py
│   │   └── pipeline_demo.py
│   ├── comparative_study/       # Scripts Volets 1 & 2
│   ├── federated/               # Algorithmes FedAvg
│   └── utils/                   # Utilitaires métriques & prétraitement
├── requirements.txt              # Dépendances Python
└── README.md
```

---

## 🚀 Prise en Main & Exécution

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancer la démonstration Agentic AI Pipeline
```bash
python scripts/agents/pipeline_demo.py
```

### 3. Exécuter l'étude comparative complète (Volets 1 & 2)
```bash
python scripts/comparative_study/run_full_study.py
```

### 4. Consulter l'Architecture Interactive
Ouvrez le fichier [architecture_diagram.html](file:///c:/Users/aminf/Desktop/Stage/FraudDetection_Stage/report/architecture_diagram.html) dans votre navigateur.
