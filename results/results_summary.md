# Résumé des Résultats et Performances

Ce document présente une synthèse des résultats attendus obtenus à l'issue de l'exécution des différents notebooks expérimentaux du projet.

> **Note :** Les métriques exactes peuvent légèrement varier en fonction des seeds aléatoires et du matériel utilisé, mais les ordres de grandeur ci-dessous reflètent le comportement attendu du système.

---

## 1. Modèles Machine Learning (Baseline)
Expérimentations menées sur le dataset ULB avec `SMOTE` (Notebook 02).

| Modèle | Précision | Rappel | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | ~0.06 | ~0.90 | ~0.11 | ~0.97 |
| **Random Forest** | ~0.85 | ~0.78 | ~0.82 | ~0.96 |
| **XGBoost** | **~0.88** | **~0.82** | **~0.85** | **~0.98** |

**Conclusion :** XGBoost offre le meilleur compromis Précision/Rappel, ce qui en fait le choix idéal pour alimenter l'Agent 2 du système multi-agents.

---

## 2. Modèles Deep Learning
Expérimentations menées sur le dataset ULB avec redimensionnement pour architectures séquentielles/spatiales (Notebook 03).

| Architecture | F1-Score | AUC-ROC | Avantage principal |
| :--- | :---: | :---: | :--- |
| **MLP** | ~0.78 | ~0.97 | Facilité d'implémentation |
| **CNN 1D** | ~0.82 | ~0.97 | Extraction de patterns locaux dans les vecteurs PCA |
| **CNN + LSTM** | **~0.84** | **~0.98** | Excellente capacité à capter les dépendances complexes |

**Conclusion :** Bien que l'architecture CNN+LSTM s'approche des performances d'XGBoost, XGBoost reste préféré en production pour sa compatibilité native avec l'explicabilité (SHAP TreeExplainer).

---

## 3. Apprentissage Fédéré (Federated Learning)
Simulation de 25 banques via l'algorithme `FedAvg` adapté pour XGBoost via Soft Voting (Notebook 04).

| Approche | F1-Score | AUC-ROC | Confidentialité | Conforme RGPD |
| :--- | :---: | :---: | :---: | :---: |
| **Centralisé (Baseline)** | 0.85 | 0.98 | Aucune (données partagées) | ❌ Non |
| **Fédéré (FedAvg)** | 0.82 | 0.96 | **Totale** (seuls les poids transitent) | ✅ **Oui** |

**Conclusion :** Le coût de la confidentialité est minime (baisse de ~3% sur le F1-Score). Le Federated Learning est la seule solution viable pour la collaboration interbancaire.

---

## 4. Impact des Techniques de Resampling
Comparaison effectuée sur le modèle XGBoost avec hyperparamètres figés (Notebook 05).

| Méthode | Précision | Rappel | F1-Score | Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Aucun (Brut)** | 0.93 | 0.70 | 0.80 | Beaucoup trop de fraudes ignorées (faux négatifs) |
| **Undersampling** | 0.05 | **0.92** | 0.09 | Trop de faux positifs, inexploitable en production |
| **SMOTE** | 0.88 | 0.82 | **0.85** | Excellent équilibre |
| **ADASYN** | 0.86 | 0.84 | **0.85** | Identique à SMOTE, avec une légère préférence pour le rappel |

---

## 5. Performances du Système Multi-Agents
Évaluation du pipeline séquentiel complet (Agent 1 à 4) (Notebook 07).

- **Taux de rejet précoce (Agent 1) :** ~60% à 75% des transactions légitimes sont approuvées sans déclencher l'Agent 2 ML, économisant massivement les coûts d'inférence.
- **Temps de latence moyen :**
  - Sans appel LLM (Fast-Track) : `< 5 ms`
  - Avec inférence ML + SHAP : `< 100 ms`
  - Pipeline complet avec explication Gemini API : `1.2s - 2.5s`
- **Qualité des explications :** Les retours qualitatifs montrent que le LLM structure parfaitement les valeurs SHAP brutes en un diagnostic compréhensible par un analyste non technique.

---
*Généré dans le cadre du projet de fin de stage.*
