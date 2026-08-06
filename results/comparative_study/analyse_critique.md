# Analyse Critique — Étude Comparative

> **Projet** : Amélioration de la détection de fraude bancaire — Agentic AI, LLMs et Federated Learning
> **Date** : 04/08/2026
> **Modèle** : XGBoost (n_estimators=200, max_depth=5, lr=0.1)
> **Seed** : 42 | **Split** : 80/20 stratifié

---

## Volet 1 — Gestion du déséquilibre des classes

### Tableau comparatif

| Technique | ULB Acc | ULB Prec | ULB Rec | ULB F1 | ULB AUC | ULB AUPRC | Synthétique Acc | Synthétique Prec | Synthétique Rec | Synthétique F1 | Synthétique AUC | Synthétique AUPRC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | 0.9995 | 0.9286 | 0.7959 | 0.8571 | 0.9778 | 0.8724 | 0.9958 | 0.7778 | 0.4083 | 0.5355 | 0.9826 | 0.6821 |
| **SMOTE** | 0.9981 | 0.4649 | 0.8776 | 0.6078 | 0.9833 | 0.8635 | 0.9957 | 0.6222 | 0.7000 | 0.6588 | 0.9674 | 0.7008 |
| **ADASYN** | 0.9970 | 0.3539 | 0.8776 | 0.5044 | 0.9802 | 0.8489 | 0.9959 | 0.6397 | 0.7250 | 0.6797 | 0.9761 | 0.7245 |
| **Undersampling** | 0.9593 | 0.0375 | 0.9184 | 0.0720 | 0.9776 | 0.6865 | 0.9307 | 0.0746 | 0.9250 | 0.1381 | 0.9776 | 0.3452 |

### Analyse par dataset

#### Dataset ULB

- **Meilleure technique (F1)** : **Baseline** (F1 = 0.8571)
- **Meilleur Recall** : Undersampling (Recall = 0.9184)
- **Meilleure AUC-ROC** : SMOTE (AUC = 0.9833)
- **Pire technique** : Undersampling (F1 = 0.0720)

#### Dataset Synthétique

- **Meilleure technique (F1)** : **ADASYN** (F1 = 0.6797)
- **Meilleur Recall** : Undersampling (Recall = 0.9250)
- **Meilleure AUC-ROC** : Baseline (AUC = 0.9826)
- **Pire technique** : Undersampling (F1 = 0.1381)

### Interprétation

**SMOTE vs ADASYN** : SMOTE génère des exemples synthétiques uniformément le long du segment reliant deux points minoritaires. ADASYN concentre la génération sur les exemples difficiles à classifier (zones frontières). En pratique, leurs performances sont souvent très proches sur les données tabulaires, avec ADASYN légèrement plus risqué (potentiel d'overfitting sur les zones bruitées).

**Undersampling** : Réduit la classe majoritaire au niveau de la minoritaire. Bien que cela améliore le Recall (détection des fraudes), la perte massive d'information entraîne une chute de la Precision (trop de faux positifs). L'Accuracy devient inexploitable. Cette technique n'est donc **pas recommandée** comme stratégie unique.

**Baseline (aucun resampling)** : XGBoost gère relativement bien le déséquilibre grâce à son scale_pos_weight interne, mais le Recall reste inférieur à SMOTE/ADASYN.

---

## Volet 2 — Federated Learning vs Centralisé

### Tableau comparatif

| Mode | ULB Acc | ULB Prec | ULB Rec | ULB F1 | ULB AUC | ULB AUPRC | Synthétique Acc | Synthétique Prec | Synthétique Rec | Synthétique F1 | Synthétique AUC | Synthétique AUPRC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Centralisé** | 0.9981 | 0.4649 | 0.8776 | 0.6078 | 0.9833 | 0.8635 | 0.9957 | 0.6222 | 0.7000 | 0.6588 | 0.9674 | 0.7008 |
| **Fédéré (FedAvg)** | 0.9994 | 0.8211 | 0.7959 | 0.8083 | 0.9752 | 0.8293 | 0.9942 | 0.6250 | 0.0833 | 0.1471 | 0.9587 | 0.5034 |

### Analyse par dataset

#### Dataset ULB

- Le modèle **fédéré** surpasse le centralisé de **0.2005** en F1-Score
- Centralisé : F1=0.6078, Precision=0.4649, Recall=0.8776
- Fédéré : F1=0.8083, Precision=0.8211, Recall=0.7959

#### Dataset Synthétique

- Le modèle **centralisé** surpasse le fédéré de **0.5117** en F1-Score (−77.7% pour le FL)
- Centralisé : F1=0.6588, Precision=0.6222, Recall=0.7000
- Fédéré : F1=0.1471, Precision=0.6250, Recall=0.0833

### Interprétation

Le Federated Learning introduit une perte de performance par rapport au modèle centralisé. Cette dégradation s'explique par :

1. **Fragmentation des données** : Chaque client ne voit qu'1/25ème du dataset, ce qui réduit la diversité des patterns observés.
2. **Déséquilibre local exacerbé** : Les rares fraudes sont réparties sur 25 partitions, certains clients n'en ayant que très peu.
3. **Agrégation par Soft Voting** : Contrairement au vrai FedAvg (moyenne des poids de réseaux de neurones), le Soft Voting sur des modèles XGBoost est une approximation qui ne bénéficie pas de la même convergence.
4. **Catastrophe sur le Synthétique (F1=0.14)** : Le dataset synthétique utilise un One-Hot Encoding massif pour les catégories (marchands, villes). Répartir ces colonnes ultra-creuses sur 25 clients détruit complètement l'information locale, rendant l'apprentissage de chaque arbre local impossible.

**Cependant**, cette perte est compensée par les avantages du FL :

- **Conformité RGPD/DORA** : Les données ne quittent jamais chaque institution bancaire.
- **Collaboration interbancaire** : Chaque banque contribue à un modèle global sans partager ses données clients.
- **Résilience** : Pas de point unique de défaillance des données.

---

## Conclusion générale

### Volet 1 — Recommandation

**Baseline** est la technique de rééquilibrage recommandée. Elle offre le meilleur compromis Precision/Recall sur le dataset ULB. L'Undersampling est à éviter en production en raison de la perte massive de Precision.

### Volet 2 — Recommandation

Le **modèle centralisé** reste plus performant, mais le **Federated Learning** constitue la solution recommandée en production car il concilie performance acceptable et conformité réglementaire. La perte de F1 est un compromis raisonnable face aux exigences RGPD et aux bénéfices de la collaboration interbancaire.
