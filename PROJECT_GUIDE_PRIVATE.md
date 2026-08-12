# 🗺️ PROJECT GUIDE — Explication Complète du Projet
> Fichier PRIVÉ — Ne pas partager. Référence personnelle complète.
> Détection de Fraude Bancaire avec Agentic AI, LLMs et Federated Learning.

---

## PIPELINE GLOBAL (Vue d ensemble)

```
[Données brutes CSV]
        │
        ▼
[Notebooks 01→06] ──► Exploration, modèles ML, FL, SHAP (recherche)
        │
        ▼ (meilleur modèle XGBoost exporté)
[Notebook 07 / pipeline_demo.py]
        │
        ├─► Transaction entrante
        │         │
        │    [Agent 1 — Surveillance]   ──► Règles métier simples
        │         │ (si suspect)
        │    [Agent 2 — Analyse ML]     ──► XGBoost + SHAP
        │         │
        │    [Agent 3 — Décision]       ──► BLOCK/REVIEW/ALERT/ALLOW
        │         │
        │    [Agent 4 — Explication]    ──► Rapport NLG via Qwen (Ollama)
        │         │
        │    [Agent 5 — Feedback]       ──► Correction humaine
        │    [Agent 6 — Monitoring]     ──► Surveillance performances
        │
        ▼
[Résultats → results/ + SQLite DB]
```

---

## 📂 RACINE DU PROJET

### `.env`
**Rôle** : Variables d environnement chargées au démarrage par python-dotenv.
**Contenu actuel** :
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```
**Utilisé par** : `LLMGatewayTool` (dans `tools.py`) qui lit ces variables pour savoir où appeler Ollama.
**Important** : Ce fichier n est PAS commité dans git (protégé par `.gitignore`).

---

### `.gitignore`
**Rôle** : Indique à Git quels fichiers NE PAS versionner.
**Ce qu il exclut** : `.env`, `__pycache__/`, `*.pyc`, données CSV (trop lourds pour Git), dossiers venv.
**Pourquoi** : Sécurité (ne pas pousser de credentials) et légèreté du repo.

---

### `README.md`
**Rôle** : Page d accueil du projet. Visible en premier sur GitHub.
**Contenu** : Description du projet, tableau de résultats clés, structure des dossiers, instructions d installation et d exécution.
**Audience** : Encadrants, recruteurs, collègues. Doit rester concis et professionnel.

---

### `requirements.txt`
**Rôle** : Liste de toutes les dépendances Python du projet avec leur version minimale.
**Commande** : `pip install -r requirements.txt`
**Librairies clés** :
- `pandas`, `numpy` : manipulation de données
- `scikit-learn` : modèles ML de base, métriques, prétraitement
- `xgboost` : modèle principal (XGBoostClassifier)
- `imbalanced-learn` : SMOTE, ADASYN, RandomUnderSampler
- `shap` : calcul des valeurs SHAP pour l explicabilité
- `tensorflow` : utilisé dans les notebooks Deep Learning (NB03)
- `matplotlib`, `seaborn` : visualisations
- `python-dotenv` : chargement du fichier `.env`
- `jupyter`, `ipywidgets` : exécution des notebooks
**Note** : Ollama n est PAS un package Python — c est une application externe à installer séparément.

---

### `test_ollama.py`
**Rôle** : Script de test rapide pour vérifier que Ollama fonctionne correctement.
**Ce qu il fait** : Instancie `LLMGatewayTool`, affiche le backend actif et le modèle, envoie un prompt de test simple.
**Quand l utiliser** : Toujours lancer ce script en premier si Ollama semble ne pas répondre.
**Commande** : `python test_ollama.py`

---

### `architecture_framework_global.png`
**Rôle** : Image d architecture globale du système (vue haut niveau).
**Utilisé dans** : Rapports, présentations, potentiellement dans les notebooks ou le README.
**C est une image statique** — elle n est pas générée par le code, c est un asset visuel.

---

## 📂 `data/` — Datasets

### `data/creditcard.csv`
**Rôle** : Dataset principal du projet. Transactions de cartes de crédit réelles (anonymisées par PCA).
**Source** : ULB Machine Learning Group (Kaggle). Données 2013 de cartes européennes.
**Contenu** : 284 807 transactions, 30 colonnes (V1 à V28 = composantes PCA anonymes, Amount, Time, Class).
**Class** : 0 = légitime, 1 = fraude (492 fraudes = 0.172%).
**Utilisé dans** : Tous les notebooks sauf NB04 (FL utilise les deux). Dataset de référence pour les résultats finaux.
**Taille** : ~144 MB.

---

### `data/fraud_detection_credit_card_small.csv`
**Rôle** : Dataset synthétique avec features transactionnelles brutes (non-PCA).
**Contenu** : ~120 000 transactions, ~21 colonnes (amt, category, merchant, lat/long, etc.).
**Particularité** : Contient des features catégorielles (category, gender, Payment_Method...) nécessitant un encodage One-Hot.
**Utilisé dans** : Volets 1 et 2 en parallèle du dataset ULB, pour tester la généralisation des méthodes.
**Taille** : ~30 MB.

---

## 📂 `notebooks/` — Notebooks Jupyter (Recherche & Exploration)

Les notebooks sont la phase de RECHERCHE du projet. Ils sont exécutés séquentiellement et produisent des insights et des artefacts (modèles, résultats) utilisés dans la phase de PRODUCTION (scripts/).

### `01_Data_Exploration.ipynb`
**Rôle** : Exploration complète des deux datasets.
**Ce qu il fait** : Distribution des classes, corrélations, boxplots, analyse temporelle, statistiques descriptives, visualisation de l imbalance.
**Output** : Compréhension profonde des données — c est la base de toutes les décisions de preprocessing qui suivent.
**Résultat clé** : Confirmation que ULB est fortement déséquilibré (0.172%), que V14/V12 sont les features les plus corrélées avec la fraude.

---

### `02_Baseline_Models.ipynb`
**Rôle** : Entraînement et comparaison des modèles ML classiques sur ULB.
**Modèles testés** : Logistic Regression, Random Forest, XGBoost (avec hyperparamètre tuning).
**Ce qu il fait** : Train/test split 80/20, scaling RobustScaler, évaluation F1/AUC/Precision/Recall.
**Output clé** : XGBoost identifié comme le meilleur modèle de base. Sert de référence pour tous les volets suivants.

---

### `03_Deep_Learning_Models.ipynb`
**Rôle** : Exploration des modèles Deep Learning pour la détection de fraude.
**Modèles testés** : Autoencoder (détection d anomalie non-supervisée), MLP classifieur, potentiellement LSTM.
**Ce qu il fait** : Construction du modèle avec TensorFlow/Keras, entraînement, comparaison avec XGBoost.
**Conclusion** : Les modèles DL ne surpassent pas significativement XGBoost sur des données tabulaires PCA — résultat classique en ML.

---

### `04_Federated_Learning.ipynb`
**Rôle** : Exploration conceptuelle et implémentation expérimentale du Federated Learning.
**Ce qu il fait** : Simulation d un environnement FL avec simulation de clients, FedAvg manuel, comparaison centralisé vs fédéré sur les deux datasets.
**Lien avec les scripts** : Les découvertes de ce notebook ont servi à concevoir `scripts/federated/fed_avg.py` et `scripts/comparative_study/volet2_federated_comparison.py`.

---

### `05_Class_Imbalance.ipynb`
**Rôle** : Étude de l impact des techniques de rééquilibrage des classes.
**Techniques comparées** : Baseline, SMOTE, ADASYN, RandomUnderSampling.
**Ce qu il fait** : Application de chaque technique sur XGBoost, évaluation multi-métriques, sélection de la meilleure approche.
**Output clé** : `results/best_technique.json` — contient `{"best_technique": "SMOTE"}` utilisé par le NB07 pour choisir le resampler automatiquement.

---

### `06_SHAP_Explainability.ipynb`
**Rôle** : Analyse complète de l explicabilité du modèle via SHAP.
**Ce qu il fait** : Calcul des valeurs SHAP avec TreeExplainer sur XGBoost, waterfall plots, summary plots, bee swarm, analyse feature importance globale.
**Résultats clés** : V14 est de loin la feature la plus influente (SHAP le plus élevé en valeur absolue), suivie par V17, V12, V10. Ces découvertes alimentent les seuils de l Agent 1.
**Lien avec les agents** : L Agent 2 (AnalysisAgent) utilise exactement le même TreeExplainer pour produire les Top-5 SHAP features en temps réel.

---

### `07_MultiAgent_System.ipynb`
**Rôle** : Démonstration du système multi-agents directement dans Jupyter (version auto-contenue).
**Particularité** : Ce notebook redéfinit les agents localement (classes Python inline) — il ne dépend PAS des scripts/ pour être exécutable seul.
**Ce qu il fait** : Instancie les 4 agents (Surveillance, Analyse, Décision, Explication), crée un orchestrateur, traite plusieurs transactions réelles et affiche les rapports.
**Agent 4 dans ce notebook** : Utilise `urllib` directement vers Ollama (pas LLMGatewayTool) — les deux approches sont équivalentes.
**Public** : Ce notebook est ce qu on montre à un encadrant pour une démo rapide.

---

### `08_Results_Comparison.ipynb`
**Rôle** : Récapitulatif final de tous les résultats expérimentaux avec visualisations.
**Ce qu il fait** : Charge les CSV de `results/comparative_study/`, génère des graphiques comparatifs propres, consolide les conclusions des Volets 1 et 2.
**Output** : Visualisations finales utilisables dans le rapport de stage.

---

## 📂 `report/` — Documentation et Rapports

### `report/rapport_complet_stage.md`
**Rôle** : Rapport de stage complet (~30 pages). Document principal à remettre.
**Contenu** : Contexte, problématique, état de l art, méthodologie, résultats, conclusions, perspectives.
**Public** : Encadrants académiques et industriels.

---

### `report/rapport_technique.md`
**Rôle** : Documentation technique approfondie du système.
**Contenu** : Architecture des agents, spécification des interfaces (inputs/outputs de chaque agent), description des outils (Tools), protocole d orchestration.
**Public** : Développeurs, équipe technique.

---

### `report/agentic_ai_architecture.md`
**Rôle** : Document dédié à l architecture Agentic AI.
**Contenu** : Les 8 étapes du blueprint "How to Build an AI Agent" appliquées au projet, justification des choix de design (mémoire, outils, orchestration).
**Public** : Encadrants, jury de soutenance.

---

### `report/architecture_diagram.html`
**Rôle** : Diagramme interactif de l architecture du pipeline multi-agents.
**Format** : Page HTML autonome (pas de serveur requis) — s ouvre directement dans le navigateur.
**Contenu** : Flux visuel de bout en bout avec les 6 agents, les outils, les flux de données.
**Comment l utiliser** : Double-cliquer sur le fichier pour l ouvrir dans Chrome/Firefox.

---

## 📂 `results/` — Résultats et Tracking

### `results/best_technique.json`
**Rôle** : Fichier de configuration généré automatiquement par le NB05.
**Contenu** : `{"best_technique": "SMOTE"}`
**Utilisé par** : Le NB07 au démarrage — il lit ce fichier pour choisir automatiquement le bon resampler (SMOTE vs ADASYN) sans paramètre manuel.
**C est un lien vivant** entre la phase de recherche (NB05) et la phase de démo (NB07).

---

### `results/project_parameters_and_results.md`
**Rôle** : Référence complète de tous les paramètres et résultats du projet.
**Contenu** : Hyperparamètres XGBoost, paramètres FL, règles des agents, seuils de décision, tableaux de métriques complets.
**Public** : Usage personnel et rapports internes.

---

### `results/comparative_study/volet1_imbalance_results.csv`
**Rôle** : Résultats bruts du Volet 1 (impact du resampling).
**Format** : CSV avec colonnes Dataset, Technique, Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR.
**Généré par** : `scripts/comparative_study/volet1_imbalance_comparison.py`

---

### `results/comparative_study/volet2_federated_results.csv`
**Rôle** : Résultats bruts du Volet 2 (centralisé vs fédéré).
**Format** : CSV avec colonnes Dataset, Mode, Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR.
**Généré par** : `scripts/comparative_study/volet2_federated_comparison.py`

---

### `results/comparative_study/volet1_barplot.png`
**Rôle** : Graphique comparatif des techniques de resampling (Volet 1).
**Généré automatiquement** par le script volet1 avec matplotlib/seaborn.
**Usage** : Insérer dans les slides ou le rapport.

---

### `results/comparative_study/volet2_barplot.png`
**Rôle** : Graphique comparatif centralisé vs fédéré (Volet 2).
**Généré automatiquement** par le script volet2.

---

### `results/comparative_study/analyse_critique.md`
**Rôle** : Analyse critique textuelle des résultats des deux volets.
**Contenu** : Interprétation des chiffres, explications des anomalies (ex: pourquoi le FL échoue sur le dataset synthétique), recommandations.

---

## 📂 `scripts/` — Code de Production

### `scripts/utils/metrics.py`
**Rôle** : Utilitaire partagé de calcul de métriques.
**Fonctions principales** :
- `evaluate_model(y_true, y_prob)` : calcule Accuracy, Precision, Recall, F1, AUC-ROC, AUPRC en une seule fois
- Fonctions de visualisation : ROC curves, confusion matrix, barplots comparatifs
**Utilisé par** : Les scripts `volet1_imbalance_comparison.py` et `volet2_federated_comparison.py` pour éviter la duplication de code.

---

### `scripts/utils/preprocessing.py`
**Rôle** : Utilitaire partagé de prétraitement des données.
**Fonctions principales** :
- `load_ulb_dataset(path)` : charge creditcard.csv avec affichage des stats
- `load_synthetic_dataset(path)` : charge le dataset synthétique
- `prepare_features_ulb()` : scaling RobustScaler, split train/test
- `apply_resampling(X, y, technique)` : applique SMOTE/ADASYN/Undersampling selon le paramètre
**Utilisé par** : Les scripts comparatifs pour éviter de réécrire le preprocessing à chaque fois.

---

### `scripts/federated/fed_avg.py`
**Rôle** : Implémentation de l algorithme FedAvg (Federated Averaging) adapté aux modèles scikit-learn.
**Contexte** : FedAvg original (McMahan 2017) est conçu pour les réseaux de neurones (moyenne des poids). Pour XGBoost qui est un ensemble d arbres, on ne peut pas faire la moyenne des poids directement. Solution : **Soft Voting** (chaque client entraîne un modèle local, la prédiction finale est la moyenne de leurs probabilités).
**Classe principale** : `FederatedSoftVoting`
- `fit(X, y)` : partitionne les données entre les clients, entraîne un modèle local par client sur ses données + SMOTE, sélectionne aléatoirement des clients par round
- `predict_proba(X)` : moyenne les probabilités de tous les modèles clients
**Utilisé par** : `volet2_federated_comparison.py` et le NB04.

---

### `scripts/comparative_study/volet1_imbalance_comparison.py`
**Rôle** : Script de l étude comparative Volet 1 — impact des techniques de rééquilibrage.
**Ce qu il fait** :
1. Charge ULB et Synthétique
2. Pour chaque dataset × technique (Baseline, SMOTE, ADASYN, Undersampling) :
   - Applique le preprocessing et le resampling
   - Entraîne XGBoost avec les mêmes hyperparamètres fixes
   - Évalue sur le test set
3. Sauvegarde les résultats dans `results/comparative_study/volet1_imbalance_results.csv`
4. Génère `volet1_barplot.png`
**Commande** : `python scripts/comparative_study/run_full_study.py` (ou le volet seul)

---

### `scripts/comparative_study/volet2_federated_comparison.py`
**Rôle** : Script de l étude comparative Volet 2 — centralisé vs Federated Learning.
**Ce qu il fait** :
1. Charge ULB et Synthétique
2. Pour chaque dataset :
   - Entraîne un XGBoost centralisé (SMOTE global)
   - Simule FL avec `FederatedSoftVoting` (25 clients, 10 rounds, SMOTE local)
   - Compare les métriques
3. Sauvegarde dans `results/comparative_study/volet2_federated_results.csv`
4. Génère `volet2_barplot.png`

---

### `scripts/comparative_study/run_full_study.py`
**Rôle** : Runner qui lance les deux volets à la suite automatiquement.
**Commande** : `python scripts/comparative_study/run_full_study.py`
**Ce qu il fait** : Appelle volet1 puis volet2, génère tous les CSV et PNG dans `results/comparative_study/`.

---

### `scripts/comparative_study/__init__.py`
**Rôle** : Transforme le dossier en module Python pour permettre les imports relatifs.
**Contenu** : Juste une déclaration de module (quelques lignes).

---

### `scripts/data/agent_memory.db`
**Rôle** : Base de données SQLite persistante pour le système multi-agents.
**Tables** :
- `decisions` : toutes les décisions prises (transaction_id, decision, probability, risk_level, timestamp)
- `feedback` : retours humains (transaction_id, feedback_type, was_fraud, timestamp)
**Utilisé par** :
- `LongTermMemory` (dans `memory.py`) pour sauvegarder les décisions
- `DatabaseTool` (dans `tools.py`) pour les requêtes du MonitoringAgent
- `FeedbackAgent` pour enregistrer les corrections humaines
**Survit aux redémarrages** — contrairement à la mémoire RAM des agents.

---

## 📂 `scripts/agents/` — Le Cœur du Système Multi-Agents

### `scripts/agents/pipeline_demo.py`
**Rôle** : Script de démonstration bout-en-bout du pipeline complet.
**Ce qu il fait** :
1. Entraîne un XGBoost rapidement sur des données dummy (ou réelles si disponibles)
2. Instancie le `MemorySystem`
3. Crée les 6 agents et les connecte
4. Instancie l `OrchestratorAgent`
5. Traite une transaction suspecte (V14=-4.5 déclenche l alerte)
6. Affiche la décision finale, le rapport LLM, le feedback et le monitoring
**Commande** : `python scripts/agents/pipeline_demo.py`
**C est le point d entrée principal** pour tester que tout le pipeline fonctionne.

---

## 📂 `scripts/agents/core/` — Le Framework des Agents

### `scripts/agents/core/__init__.py`
**Rôle** : Expose publiquement toutes les classes du framework core.
**Ce qu il permet** : `from scripts.agents.core import BaseAgent, LLMGatewayTool, MemorySystem` (import propre depuis n importe où).

---

### `scripts/agents/core/agent_base.py`
**Rôle** : Fondation de tout le système. Définit les classes de base dont héritent tous les agents.
**Classes définies** :
- `AgentStatus` (Enum) : IDLE, PROCESSING, SUCCESS, ERROR, TIMEOUT, ESCALATED
- `Priority` (Enum) : CRITICAL=1, HIGH=2, MEDIUM=3, LOW=4
- `AgentResponse` (dataclass) : structure standardisée de réponse de chaque agent (data, status, latency_ms, errors, trace_id...)
- `AgentMetrics` : collecteur de métriques temps réel (avg_latency, p99_latency, throughput, success_rate)
- `AgentLogger` : logger structuré JSON compatible ELK/Datadog, produit les logs que tu vois dans le terminal
- `BaseAgent` (ABC) : classe abstraite parente. Définit `run()`, `process()` (abstraite), `use_tool()`, `log_action()`. Tout agent DOIT hériter de cette classe et implémenter `process()`.
**Pourquoi c est important** : C est le contrat que tous les agents respectent, ce qui permet à l Orchestrateur de les traiter de manière uniforme.

---

### `scripts/agents/core/memory.py`
**Rôle** : Système de mémoire multi-couches pour les agents.
**Classes définies** :
- `WorkingMemory` : dictionnaire RAM partagé entre agents pendant UNE transaction. Effacé à chaque nouveau pipeline. Stocke : transaction_id, raw_transaction, état intermédiaire.
- `EpisodicMemory` : `deque` (max 100 éléments) des dernières transactions traitées. Permet au LLM de contextualiser avec l historique récent.
- `SemanticMemory` : stockage de patterns de fraude connus. Recherche par similarité de features.
- `LongTermMemory` : interface SQLite. Méthodes : `save_decision()`, `save_feedback()`, `get_feedback_accuracy()`, `get_recent_decisions()`. Données survivent aux redémarrages.
- `MemorySystem` : façade unifiée qui regroupe les 4 couches. C est cet objet qui est injecté dans chaque agent.

---

### `scripts/agents/core/tools.py`
**Rôle** : Bibliothèque d outils réutilisables injectables dans les agents.
**Classes définies** :
- `BaseTool` (ABC) : interface abstraite pour tous les outils. Méthodes : `execute()`, `get_stats()`.
- `MLModelTool` : encapsule un modèle scikit-learn. Méthode `execute(transaction)` → retourne `{probability, prediction, risk_level, latency_ms}`. Gère le scaling automatiquement.
- `SHAPExplainabilityTool` : encapsule un `shap.TreeExplainer`. Méthode `execute(features_array, top_n=5)` → retourne `{shap_top_n, all_shap, base_value}`. Classe le top N features par valeur SHAP absolue.
- `LLMGatewayTool` : passerelle vers Ollama. Au démarrage, teste la connexion à `http://localhost:11434/api/tags`. Si OK → mode online (appels HTTP vers `/api/chat`). Sinon → mode offline (template structuré). Paramètres : temperature=0.3, max_tokens=500.
- `AlertTool` : enregistre les alertes dans `results/alerts.jsonl` et affiche en console les alertes CRITIQUE/ÉLEVÉ.
- `DatabaseTool` : exécute des requêtes prédéfinies sur l SQLite (`recent_decisions`, `decision_stats`).
- `FederatedLearningTool` : interface vers `FederatedSoftVoting` pour les prédictions du modèle fédéré.

---

### `scripts/agents/core/orchestrator.py`
**Rôle** : Le chef d orchestre — coordonne l exécution de tous les agents.
**Classes définies** :
- `PipelineResult` (dataclass) : résultat consolidé d une transaction (transaction_id, workflow_used, total_latency_ms, final_decision, probability, risk_level, explanation, agent_responses).
- `OrchestratorAgent` : classe principale.
  - `route(transaction)` : détermine quel workflow utiliser (fast_track, standard, escalation, deep_analysis)
  - `process_transaction(transaction, transaction_id)` : boucle principale. Pour chaque agent du workflow choisi, appelle `agent.run(current_data, context)`, récupère la réponse, met à jour `current_data` avec les données de la réponse (c est ainsi que les agents se passent l information). Si Agent 1 décide ALLOW directement → coupe le pipeline (fast-track < 5ms). Sauvegarde le résultat en mémoire épisodique et long-terme.
**Flux de données** : chaque agent reçoit TOUT ce que les agents précédents ont produit (les données s accumulent dans `current_data`).

---

## 📂 `scripts/agents/` — Les 6 Agents

### `scripts/agents/surveillance_agent.py` — Agent 1
**Rôle** : Premier filtre. Règles métier codées en dur basées sur les seuils SHAP découverts dans le NB06.
**Hérite de** : `BaseAgent`
**Outils** : Aucun (logique pure Python)
**Input** : `{transaction: dict, transaction_id: str}`
**Traitement** : Vérifie chaque feature contre les seuils (`ULB_THRESHOLDS` ou `SYNTHETIC_THRESHOLDS`).
**Output** : `{suspicious: bool, reasons: list, fast_decision: "ALLOW"|None, risk_score: int}`
**Logique Fast-Track** : Si `suspicious=False` → l Orchestrateur coupe le pipeline ici (ALLOW immédiat < 1ms, pas d appel ML ni LLM).
**Pourquoi** : 60-75% des transactions sont légitimes et ne nécessitent pas d analyse ML coûteuse.

---

### `scripts/agents/analysis_agent.py` — Agent 2
**Rôle** : Scoring ML + explicabilité SHAP.
**Hérite de** : `BaseAgent`
**Outils injectés** : `MLModelTool` (XGBoost), `SHAPExplainabilityTool` (TreeExplainer)
**Input** : Données de l Agent 1 + transaction originale
**Traitement** :
  1. `MLModelTool.execute(transaction)` → probabilité de fraude
  2. `SHAPExplainabilityTool.execute(features_array, top_n=5)` → top 5 features SHAP
**Output** : `{probability, risk_level, shap_top5, all_shap, base_value}`
**Latence** : < 100ms (modèle en RAM, pas de réseau).

---

### `scripts/agents/decision_agent.py` — Agent 3
**Rôle** : Transformation de la probabilité en décision opérationnelle + déclenchement d alertes.
**Hérite de** : `BaseAgent`
**Outils injectés** : `AlertTool`
**Input** : `probability` de l Agent 2
**Traitement** : Compare la probabilité aux seuils `DEFAULT_THRESHOLDS` (BLOCK>=0.85, REVIEW>=0.60, ALERT>=0.35, ALLOW<0.35).
**Output** : `{decision, probability, severity, action, notify, sla_min, color, icon, alert_details}`
**Effet de bord** : Pour BLOCK/REVIEW/ALERT, appelle `AlertTool` qui logge dans `results/alerts.jsonl`.

---

### `scripts/agents/explanation_agent.py` — Agent 4
**Rôle** : Génération du rapport NLG en français via le LLM local Qwen.
**Hérite de** : `BaseAgent`
**Outils injectés** : `LLMGatewayTool`
**Input** : Contexte complet (surveillance reasons, SHAP top5, décision)
**Traitement** :
  1. Construit un prompt structuré avec toutes les données des agents précédents
  2. Appelle `LLMGatewayTool.execute(prompt)` → requête HTTP vers Ollama
  3. Si Ollama offline → génère un rapport template formaté localement
**Output** : `{rapport: str, mode: "online"|"offline", transaction_id}`
**Latence** : 9-17 secondes sur CPU (le goulot d étranglement du pipeline).

---

### `scripts/agents/feedback_agent.py` — Agent 5
**Rôle** : Capture du retour humain pour améliorer le système (Human-in-the-Loop).
**Hérite de** : `BaseAgent`
**Outils** : Aucun (écrit directement dans la mémoire long-terme)
**Input** : `{transaction_id, feedback: "CONFIRMED_FRAUD"|"FALSE_POSITIVE"|"LEGITIMATE", was_fraud: bool}`
**Traitement** : Appelle `memory.long_term.save_feedback(...)` → écrit dans la table `feedback` du SQLite.
**Output** : `{transaction_id, feedback_saved: True, feedback_type, was_fraud}`
**Pourquoi** : Les corrections humaines permettent de mesurer le taux de faux positifs/négatifs réels du modèle en production.

---

### `scripts/agents/monitoring_agent.py` — Agent 6
**Rôle** : Surveillance de la santé du pipeline et détection de drift.
**Hérite de** : `BaseAgent`
**Outils injectés** : `DatabaseTool`
**Input** : `{hours: int}` (fenêtre de monitoring, défaut 24h)
**Traitement** :
  1. `DatabaseTool.execute(query_type="decision_stats")` → statistiques par décision (COUNT, AVG proba)
  2. `memory.long_term.get_feedback_accuracy()` → précision des décisions validées par les humains
**Output** : `{monitoring_period_hours, decision_stats, human_feedback_accuracy, status: "HEALTHY"}`
**Usage** : Appelé périodiquement ou après le pipeline pour surveiller les dérives.

---

## 🔄 RÉSUMÉ DU FLUX COMPLET (avec les fichiers)

```
Transaction (dict Python)
    │
    │ [orchestrator.py] route() → "standard"
    │
    ├─── [surveillance_agent.py].process()
    │       Lit: ULB_THRESHOLDS (hardcoded)
    │       Écrit dans: current_data["suspicious"], ["reasons"]
    │
    ├─── [analysis_agent.py].process()
    │       Appelle: tools.py → MLModelTool → XGBoost.predict_proba()
    │       Appelle: tools.py → SHAPExplainabilityTool → shap.TreeExplainer
    │       Écrit dans: current_data["probability"], ["shap_top5"]
    │
    ├─── [decision_agent.py].process()
    │       Lit: DEFAULT_THRESHOLDS (hardcoded)
    │       Appelle: tools.py → AlertTool → results/alerts.jsonl
    │       Écrit dans: current_data["decision"], ["action"]
    │
    ├─── [explanation_agent.py].process()
    │       Appelle: tools.py → LLMGatewayTool → HTTP → Ollama → qwen2.5:7b
    │       Fallback: template local si Ollama offline
    │       Écrit dans: current_data["rapport"]
    │
    └─── [orchestrator.py] consolidate()
            Sauvegarde: memory.py → LongTermMemory → scripts/data/agent_memory.db
            Retourne: PipelineResult
                        ├── final_decision
                        ├── probability
                        ├── risk_level
                        ├── explanation (rapport LLM)
                        └── total_latency_ms
```

---

## ⚠️ POINTS D ATTENTION IMPORTANTS

1. **Ollama doit être démarré** avant d exécuter le pipeline. Sinon l Agent 4 bascule en mode offline.
2. **creditcard.csv n est pas dans le repo Git** (trop lourd). Il faut le télécharger séparément depuis Kaggle.
3. **Le modèle XGBoost n est pas sauvegardé** — il est réentraîné à chaque exécution de `pipeline_demo.py` (sur des données dummy). En production, il faudrait le sauvegarder avec `joblib.dump()`.
4. **`best_technique.json`** est lu par le NB07 — si ce fichier est absent, il prend SMOTE par défaut.
5. **La latence LLM (9-17s)** vient du fait que Qwen 7B tourne sur CPU. Sur GPU elle serait < 2s.
6. **`scripts/data/agent_memory.db`** s accumule à chaque exécution. Pour réinitialiser, supprimer ce fichier.

---

*Créé le : Août 2026 — Pour usage privé uniquement.*
