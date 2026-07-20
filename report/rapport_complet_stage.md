# 📊 Rapport Complet de Stage - Détection de Fraude Bancaire
## Framework : Federated Learning & Système Multi-Agents

---

## 🎯 Objectif du Projet

Ce projet vise à développer un système complet de détection de fraude bancaire combinant :
- **Federated Learning** pour la collaboration interbancaire préservant la confidentialité
- **Système Multi-Agents** pour l'analyse intelligente des transactions
- **Explainable AI** via SHAP et LLM pour des explications compréhensibles

---

## 🏗️ Schéma Global du Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DONNÉES BRUTES                                      │
│  ┌─────────────────────┐        ┌──────────────────────────────────┐      │
│  │ creditcard.csv      │        │ fraud_detection_credit_card_     │      │
│  │ (ULB - 284K trans.) │        │ small.csv (Synthétique - 100K)   │      │
│  └─────────────────────┘        └──────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRÉTRAITEMENT & FEATURE ENGINEERING                       │
│  • Train/Test Split (stratifié)                                              │
│  • Scaling : RobustScaler / StandardScaler                                   │
│  • Resampling : SMOTE / ADASYN / Aucun / Undersampling                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────────────┐
│   MODÈLES ML      │    │   MODÈLES DL      │    │   FEDERATED LEARNING       │
│  Centralisés      │    │  Centralisés      │    │   (FedAvg)                 │
├───────────────────┤    ├───────────────────┤    ├───────────────────────────┤
│ • Logistic Reg.   │    │ • MLP             │    │ • XGBoost (Soft Voting)    │
│ • Random Forest   │    │ • CNN 1D          │    │ • 25 banques simulées      │
│ • XGBoost         │    │ • CNN + LSTM      │    │ • 10 clients/round         │
└───────────────────┘    └───────────────────┘    │ • 10 rounds                │
                                                          └───────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SÉLECTION DU MEILLEUR MODÈLE                               │
│                 (XGBoost - Meilleur compromis Précision/Rappel)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SYSTÈME MULTI-AGENTS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │  AGENT 1     │    │  AGENT 2     │    │  AGENT 3     │    │ AGENT 4  │ │
│  │Surveillance  │───▶│  Analyse     │───▶│  Décision    │───▶│ Explic.  │ │
│  │  + Filtrage  │    │  ML + SHAP   │    │  Métier      │    │   LLM    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘ │
│       │                   │                   │                   │        │
│       ▼                   ▼                   ▼                   ▼        │
│  Règles métier      Probabilité +      BLOCK/REVIEW/      Rapport          │
│  (fast-track)       SHAP values         ALERT/ALLOW       naturel          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              ALERTE & RAPPORT POUR ANALYSTE BANCAIRE                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Tableau Comparatif des Résultats

### Métriques utilisées pour tous les modèles
- **Accuracy** : Pourcentage de prédictions correctes (indicatif - artificiellement élevé)
- **Precision** : Pourcentage de fraudes prédites qui sont réellement des fraudes
- **Recall (Rappel)** : Pourcentage de fraudes réelles détectées
- **F1-Score** : Moyenne harmonique de Precision et Recall (métrique principale)
- **AUC-ROC** : Area Under Curve - Receiver Operating Characteristic
- **AUPRC** : Area Under Precision-Recall Curve

| Modèle | Catégorie | Entraînement | Dataset | Resampling | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUPRC |
|:---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** | ML | Centralisé | ULB | SMOTE | 0.9998 | 0.9398 | 0.7959 | **0.8619** | 0.9833 | 0.8635 |
| **MLP** | DL | Centralisé | ULB | SMOTE | 0.9998 | 0.8317 | 0.8571 | 0.8442 | 0.9752 | - |
| **Random Forest** | ML | Centralisé | ULB | SMOTE | 0.9997 | 0.8280 | 0.7857 | 0.8063 | 0.9855 | 0.8024 |
| **CNN 1D** | DL | Centralisé | ULB | SMOTE | 0.9998 | 0.8119 | 0.8367 | 0.8241 | 0.9830 | - |
| **XGBoost (Fédéré)** | ML | Fédéré | ULB | SMOTE | 0.9997 | 0.7961 | 0.8367 | 0.8159 | 0.9788 | - |
| **CNN+LSTM** | DL | Centralisé | ULB | SMOTE | 0.9997 | 0.6693 | 0.8673 | 0.7556 | 0.9639 | - |
| **Logistic Regression** | ML | Centralisé | ULB | SMOTE | 0.9981 | 0.5621 | 0.8776 | 0.6853 | 0.9714 | 0.7245 |
| **XGBoost (Sans resampling)** | ML | Centralisé | ULB | Aucun | 0.9982 | 0.9268 | 0.7755 | 0.8444 | 0.9428 | - |
| **XGBoost (ADASYN)** | ML | Centralisé | ULB | ADASYN | 0.9997 | 0.8721 | 0.7653 | 0.8152 | 0.9813 | - |
| **XGBoost (Undersampling)** | ML | Centralisé | ULB | Undersampling | 0.9900 | 0.2019 | 0.8673 | 0.3276 | 0.9736 | - |

**Légende :**
- **En gras** : Meilleur score F1 (métrique principale)
- **ULB** : Dataset anonymisé avec features PCA (30 features)
- **SMOTE** : Synthetic Minority Over-sampling Technique
- **ADASYN** : Adaptive Synthetic Sampling

---

## 🔧 Tableau Récapitulatif des Hyperparamètres

### Modèles Machine Learning (Centralisés)

| Modèle | Hyperparamètres principaux | Valeurs |
|:---|:---|:---|
| **Logistic Regression** | Max iterations | 2000 |
| | Regularisation (C) | 0.1 |
| | Class weight | balanced |
| | Optimiseur | lbfgs |
| | Solver | lbfgs |
| **Random Forest** | N estimators | 100 |
| | Max depth | 10 |
| | Class weight | balanced |
| | N jobs | -1 (parallèle) |
| | Criterion | gini |
| **XGBoost** | N estimators | 200 |
| | Max depth | 5 |
| | Learning rate | 0.1 |
| | Subsample | 0.8 |
| | Colsample bytree | 0.8 |
| | Eval metric | logloss |
| | Use label encoder | False |

### Modèles Deep Learning (Centralisés)

| Modèle | Architecture | Couches | Activation | Dropout | Learning Rate | Batch Size | Epochs | Optimiseur | Loss |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **MLP** | Dense[256,128,64,32,1] | 5 | ReLU/Sigmoid | 0.4,0.35,0.3 | 0.001 | 512 | 15 (ES p=5) | Adam | Binary Crossentropy |
| **CNN 1D** | Conv1D[64,128,64]+Dense[128,1] | 6 | ReLU/Sigmoid | 0.3,0.4,0.5 | 0.001 | 512 | 15 (ES p=5) | Adam | Binary Crossentropy |
| **CNN+LSTM** | CNN[64,128]+LSTM[64,32]+Dense[128,1] | 7 | ReLU/Sigmoid | 0.3,0.3,0.4 | 0.0005 | 512 | 15 (ES p=5) | Adam | Binary Crossentropy |

**Callbacks communs DL :**
- EarlyStopping(patience=5, restore_best_weights=True)
- ReduceLROnPlateau(factor=0.5, patience=3)
- Class weights : {0: 1.0, 1: 1.5}

### Approche Fédérée (FedAvg)

| Paramètre | Valeur | Description |
|:---|:---:|:---|
| **Nombre de clients (N)** | 25 | Banques fictives simulées |
| **Clients par round (C)** | 10 | Clients sélectionnés aléatoirement par round |
| **Nombre de rounds (R)** | 10 | Itérations fédérées totales |
| **Méthode d'agrégation** | Soft Voting | Moyenne des probabilités de tous les clients |
| **Modèle local** | XGBoost | Modèle entraîné par chaque client |
| **N_estimators (local)** | 50 | Arbres par modèle local |
| **Max_depth (local)** | 4 | Profondeur maximale des arbres |
| **Learning Rate (local)** | 0.15 | Taux d'apprentissage local |
| **Resampling** | SMOTE | Appliqué sur chaque client |
| **Random state** | 42 | Reproductibilité |

**Note technique :** XGBoost ne supporte pas la moyenne directe des poids (FedAvg classique). L'agrégation est réalisée par **Soft Voting** : chaque modèle client vote avec ses probabilités, et la moyenne est utilisée pour la décision finale.

### Techniques de Resampling

| Méthode | Paramètres | Description |
|:---|:---|:---|
| **SMOTE** | k_neighbors=5, random_state=42 | Génère des échantillons synthétiques de la classe minoritaire |
| **ADASYN** | random_state=42 | Adaptatif - génère plus d'échantillons dans les zones difficiles |
| **RandomUnderSampler** | random_state=42 | Sous-échantillonne la classe majoritaire aléatoirement |
| **Aucun** | - | Données brutes (déséquilibrées) |

---

## 🤖 Système Multi-Agents - Documentation Détaillée

### Architecture Globale

Le système multi-agents fonctionne en **pipeline séquentiel** où chaque agent a une responsabilité spécifique :

```
Transaction → Agent 1 → Agent 2 → Agent 3 → Agent 4 → Rapport Final
```

---

### Agent 1 : Surveillance Agent (Pré-filtrage)

**Objectif :**
Filtrage rapide des transactions légitimes pour réduire les coûts d'inférence ML. Cet agent utilise des règles métier simples pour identifier les transactions manifestement normales.

**Input (Entrée) :**
- `transaction` : Dictionnaire contenant les features de la transaction
  - Exemple : `{'V14': -1.5, 'V12': 0.8, 'Amount': 150.0, 'V4': 2.1, ...}`

**Output (Sortie) :**
```python
{
    'suspicious': bool,           # True si transaction suspecte
    'reasons': list[str],         # Liste des règles déclenchées
    'fast_decision': str or None, # 'ALLOW' si non suspecte, None sinon
    'risk_score': int             # Nombre de règles déclenchées
}
```

**Règles métier configurées :**
- `V14 < -2.5` : Fort indicateur de fraude
- `V12 < -2.0` : Corrélation avec fraude carte
- `V10 < -2.0` : Valeur anormale
- `V4 > 2.5` : Pattern de fraude connu
- `Amount > 2.0` : Montant standardisé élevé
- Règle composite : ≥2 anomalies mineures simultanées (V3, V7, V11, V16)

**Métriques d'efficacité :**
- **Taux de rejet précoce** : Pourcentage de transactions autorisées sans ML
- **Latence fast-track** : Temps de traitement pour transactions non suspectes (< 5ms)
- **Taux de suspicion** : Pourcentage de transactions transmises à l'Agent 2

**Avantages :**
- Réduction massive des coûts d'inférence (60-75% des transactions)
- Latence minimale pour transactions légitimes
- Filtrage basé sur l'expertise métier

---

### Agent 2 : Analysis Agent (ML + SHAP)

**Objectif :**
Analyser les transactions suspectes avec un modèle ML (XGBoost) et extraire les valeurs SHAP pour l'explicabilité. Cet agent fournit une probabilité de fraude et identifie les features les plus influentes.

**Input (Entrée) :**
- `transaction` : Dictionnaire de features (même format que Agent 1)
- Les valeurs doivent être dans l'espace ORIGINAL (le scaler est appliqué en interne)

**Output (Sortie) :**
```python
{
    'probability': float,        # Probabilité de fraude [0, 1]
    'risk_level': str,           # 'CRITIQUE', 'ÉLEVÉ', 'MOYEN', 'FAIBLE'
    'shap_top5': list[dict],    # Top 5 features les plus influentes
    'all_shap': dict,            # Valeurs SHAP pour toutes les features
}
```

**Format de `shap_top5` :**
```python
[
    {
        'feature': str,          # Nom de la feature (ex: 'V14')
        'value': float,          # Valeur de la feature
        'shap': float,           # Valeur SHAP
        'direction': str         # 'AUGMENTE risque fraude' ou 'RÉDUIT risque fraude'
    },
    ...
]
```

**Niveaux de risque :**
- **CRITIQUE** : probabilité ≥ 0.85
- **ÉLEVÉ** : probabilité ≥ 0.60
- **MOYEN** : probabilité ≥ 0.35
- **FAIBLE** : probabilité < 0.35

**Modèle utilisé :**
- XGBoost entraîné avec SMOTE
- SHAP TreeExplainer pour l'explicabilité
- RobustScaler pour la normalisation

**Métriques d'efficacité :**
- **AUC-ROC** : Capacité de discrimination du modèle
- **F1-Score** : Équilibre précision/rappel
- **Latence d'analyse** : Temps de prédiction + calcul SHAP (< 100ms)
- **Stabilité SHAP** : Cohérence des explications pour des transactions similaires

**Avantages :**
- Probabilité de fraude précise
- Explicabilité native via SHAP
- Compatible avec l'Agent 4 pour génération de rapports

---

### Agent 3 : Decision Agent (Règles Métier)

**Objectif :**
Appliquer les règles métier pour prendre une décision automatisée basée sur la probabilité de fraude. Cet agent transforme la probabilité en action concrète.

**Input (Entrée) :**
- `probability` : Float [0, 1] - Probabilité de fraude (sortie de l'Agent 2)

**Output (Sortie) :**
```python
{
    'decision': str,             # 'BLOCK', 'REVIEW', 'ALERT', 'ALLOW'
    'probability': float,        # Probabilité utilisée
    'threshold_used': float,     # Seuil appliqué
    'severity': str,             # 'CRITIQUE', 'ÉLEVÉ', 'MOYEN', 'FAIBLE'
    'action': str,               # Description de l'action
    'notify': list[str],         # Parties à notifier
    'icon': str,                 # Emoji pour affichage
    'sla_min': int or None       # SLA en minutes
}
```

**Seuils configurables (défaut) :**
| Décision | Seuil | Sévérité | Action | SLA | Notifications |
|:---|---:|:---|:---|---:|:---|
| **BLOCK** | ≥ 0.85 | CRITIQUE | Transaction refusée. Dossier fraude ouvert. | 0 min | équipe_fraude, client, conformite |
| **REVIEW** | ≥ 0.60 | ÉLEVÉ | Transaction suspendue. Analyste assigné. | 15 min | analyste_fraude, client |
| **ALERT** | ≥ 0.35 | MOYEN | Autorisée avec surveillance renforcée. | 60 min | equipe_surveillance |
| **ALLOW** | < 0.35 | FAIBLE | Transaction approuvée normalement. | None | - |

**Métriques d'efficacité :**
- **Distribution des décisions** : Pourcentage de BLOCK/REVIEW/ALERT/ALLOW
- **Taux de faux positifs** : Transactions légitimes bloquées
- **Taux de faux négatifs** : Fraudes autorisées
- **Respect des SLA** : Temps de traitement dans les limites

**Avantages :**
- Automatisation complète des décisions
- Configurable selon la politique de risque de la banque
- Notifications automatiques aux parties concernées

---

### Agent 4 : Explanation Agent (LLM)

**Objectif :**
Transformer la sortie mathématique du modèle ML (boîte noire) en un diagnostic compréhensible par un analyste bancaire non-technique. L'agent explique *pourquoi* une transaction a été bloquée ou signalée, accélérant l'investigation.

**Input (Entrée) :**
```python
{
    'transaction': dict,         # Features de la transaction
    'score_fraude': float,       # Probabilité de fraude (ex: 0.92)
    'shap_top5': list[dict],     # Top 5 valeurs SHAP
    'decision': str,             # Décision de l'Agent 3
    'surveillance_alerts': list[str]  # Alertes de l'Agent 1
}
```

**Output (Sortie) :**
```python
{
    'rapport': str,              # Texte du rapport en français
    'resume': str,               # Résumé en 1 phrase
    'features_cles': list[str],  # Features les plus importantes
    'confidence': str,           # Niveau de confiance
    'mode': str                  # 'online' (LLM) ou 'offline' (template)
}
```

**Exemple de rapport généré :**
```
RAPPORT — TXN-0840 [Vrai:FRAUDE]
Décision: 🚫 BLOCK (Prob fraude: 99.9%, Risque: CRITIQUE)
Action: Transaction refusée. Dossier fraude ouvert.

Feature clé: V14=-6.781 (SHAP=+4.229 → AUGMENTE risque fraude)
2e facteur: V17=-7.324 (SHAP=+1.295)

Alertes surveillance:
- V14 très bas : fort indicateur de fraude (V14=-6.781)
- V12 anormal : corrélé avec fraude carte (V12=-4.726)

Notifications envoyées: equipe_fraude, client, conformite
```

**LLM utilisé :**
- Google Gemini 1.5 Flash
- Mode fallback hors-ligne si API non disponible
- Prompt structuré pour garantir des réponses professionnelles

**Métriques d'efficacité :**
- **Qualité des explications** : Évaluation qualitative par analystes
- **Temps de génération** : Latence du rapport (1.2s - 2.5s avec LLM)
- **Compréhensibilité** : Score de satisfaction des utilisateurs
- **Pertinence** : Cohérence avec les valeurs SHAP

**Avantages :**
- Explications naturelles en français
- Adaptation au contexte (dataset synthétique avec features lisibles)
- Accélération des investigations manuelles
- Traçabilité complète des décisions

---

### Métriques Globales du Pipeline

**Performance opérationnelle :**
- **Taux de rejet précoce (Agent 1)** : 60-75% des transactions légitimes
- **Latence moyenne** :
  - Fast-track (Agent 1 uniquement) : < 5 ms
  - Analyse ML + SHAP : < 100 ms
  - Pipeline complet avec LLM : 1.2s - 2.5s
- **Distribution des décisions** : BLOCK/REVIEW/ALERT/ALLOW

**Performance de détection :**
- **F1-Score global** : 0.8159 (XGBoost fédéré)
- **AUC-ROC** : 0.9788
- **Taux de détection de fraude** : 83.67% (Recall)
- **Taux de fausses alertes** : 20.39% (1 - Precision)

---

## 📊 Analyse Comparative des Performances

### Avantages et Limites de Chaque Approche

#### 1. Logistic Regression
**Avantages :**
- Très rapide à entraîner
- Excellente baseline pour comparaison
- Fort rappel (87.76%) - détecte la plupart des fraudes

**Limites :**
- Trop de faux positifs (precision 56.21%)
- F1-Score faible (0.6853)
- Inutilisable en production à cause des fausses alertes massives

#### 2. Random Forest
**Avantages :**
- Robuste au surapprentissage
- Gère bien le déséquilibre (class_weight='balanced')
- Bonne performance globale (F1=0.8063)

**Limites :**
- Moins performant qu'XGBoost sur ces données
- Inadapté au FedAvg classique (pas d'agrégation de poids simple)
- Plus lent qu'XGBoost à l'inférence

#### 3. XGBoost (Centralisé)
**Avantages :**
- **Meilleur compromis Precision/Rappel** (F1=0.8619)
- Compatibilité native avec SHAP via TreeExplainer (calcul quasi-instantané)
- Rapide à l'entraînement et à l'inférence
- Gère nativement le déséquilibre des classes

**Limites :**
- Agrégation fédérée par Soft Voting (pas de FedAvg paramétrique standard)
- Nécessite un réglage fin des hyperparamètres

#### 4. MLP (Deep Learning)
**Avantages :**
- Bonne performance (F1=0.8442)
- Simple à implémenter
- Architecture légère (53K paramètres)

**Limites :**
- Boîte noire (moins interprétable que XGBoost)
- Moins performant qu'XGBoost sur données tabulaires
- Nécessite DeepExplainer SHAP (plus lent)

#### 5. CNN 1D
**Avantages :**
- Détecte des patterns locaux dans les features PCA
- Performance correcte (F1=0.8241)
- Architecture adaptée aux données séquentielles

**Limites :**
- Boîte noire complexe
- Architecture plus complexe que MLP sans gain significatif
- SHAP DeepExplainer coûteux en calcul

#### 6. CNN + LSTM
**Avantages :**
- Capture patterns spatiaux (CNN) + séquentiels (LSTM)
- Meilleur rappel parmi les modèles DL (86.73%)
- Architecture hybride puissante

**Limites :**
- Architecture lourde et complexe
- SHAP DeepExplainer très coûteux
- F1 inférieur à XGBoost (0.7556 vs 0.8619)
- Latence d'inférence élevée

#### 7. XGBoost (Fédéré)
**Avantages :**
- **Respect de la confidentialité** (RGPD compliant)
- Collaboration interbancaire sans partage de données
- Performance proche du centralisé (F1=0.8159 vs 0.8619)
- Perte de performance minimale (~3% F1)

**Limites :**
- Légère perte de performance vs centralisé
- Complexité de mise en œuvre (coordination des clients)
- Soft Voting moins optimal que FedAvg vrai pour réseaux de neurones

### Impact du Resampling (sur XGBoost avec hyperparamètres fixes)

| Méthode | Precision | Recall | F1-Score | Observation |
|:---|---:|---:|---:|:---|
| **Aucun (brut)** | 0.9268 | 0.7755 | 0.8444 | Beaucoup trop de fraudes ignorées (faux négatifs). Precision artificiellement élevée. |
| **SMOTE** | 0.8791 | 0.8163 | **0.8466** | **Excellent équilibre**. Meilleur F1 global. Recommandé. |
| **ADASYN** | 0.8721 | 0.7653 | 0.8152 | Performances similaires à SMOTE, légère préférence pour le rappel. |
| **Undersampling** | 0.2019 | 0.8673 | 0.3276 | Trop de faux positifs, inexploitable en production. Perte d'information massive. |

**Conclusion sur le resampling :**
- **SMOTE** est la méthode recommandée pour ce projet
- Équilibre optimal entre précision et rappel
- ADASYN est une alternative valide si on privilégie le rappel
- Undersampling est à éviter (perte d'information trop importante)

---

## 🏆 Conclusion et Choix du Modèle Final

### Modèle Final Retenu : XGBoost (Fédéré)

Les expérimentations confirment que le **Federated Learning est une solution viable** pour la détection de fraude interbancaire, avec une perte de performance minimale par rapport à l'approche centralisée.

### Justification du Choix

#### 1. Performance de Détection
- **F1-Score** : 0.8159 (meilleur parmi les approches fédérées)
- **AUC-ROC** : 0.9788 (excellente capacité de discrimination)
- **Perte de performance** : seulement ~3% vs centralisé (0.8619 → 0.8159)

#### 2. Explicabilité Native
- **Compatibilité SHAP** : TreeExplainer quasi-instantané
- **Essentiel pour l'Agent 4** : Génération de rapports en temps réel
- **Avantage vs DL** : DeepExplainer pour CNN/LSTM est nettement plus lent

#### 3. Respect de la Confidentialité
- **Conformité RGPD** : Aucun partage de données brutes
- **Collaboration interbancaire** : 25 banques peuvent collaborer
- **Soft Voting sécurisé** : Seules les probabilités transitent

#### 4. Performance Opérationnelle
- **Latence d'inférence** : < 100ms (incluant SHAP)
- **Scalabilité** : Supporte 25 clients fédérés
- **Robustesse** : Fonctionne même si certains clients sont défaillants

#### 5. Intégration Multi-Agents
- **Agent 2** : Analyse ML rapide et précise
- **Agent 3** : Décisions métier automatisées
- **Agent 4** : Explications LLM basées sur SHAP

### Comparaison avec les Alternatives

| Critère | XGBoost Fédéré | XGBoost Centralisé | CNN+LSTM |
|:---|:---:|:---:|:---:|
| **F1-Score** | 0.8159 | 0.8619 | 0.7556 |
| **Confidentialité** | ✅ Total | ❌ Aucune | ❌ Aucune |
| **RGPD Compliant** | ✅ Oui | ❌ Non | ❌ Non |
| **Explicabilité SHAP** | ✅ Rapide | ✅ Rapide | ⚠️ Lent |
| **Latence** | < 100ms | < 100ms | > 200ms |
| **Collaboration** | ✅ Multi-banques | ❌ Centralisé | ❌ Centralisé |

### Coût de la Confidentialité

La différence de performance entre l'approche centralisée et fédérée représente le **coût de la confidentialité** :

```
Δ F1-Score  = -0.0460 (perte de 5.3%)
Δ AUC-ROC   = -0.0045 (perte de 0.5%)
```

Cette perte est **acceptable** compte tenu des gains :
- ✅ Conformité RGPD
- ✅ Collaboration interbancaire
- ✅ Confiance des clients
- ✅ Réduction des risques juridiques

### Recommandations de Déploiement

1. **Phase pilote** : Déployer sur 5-10 banques partenaires
2. **Monitoring** : Suivre les métriques F1, AUC, et latence en production
3. **Feedback** : Collecter les retours des analystes sur les explications LLM
4. **Optimisation** : Ajuster les seuils de décision (Agent 3) selon la politique de risque
5. **Extension** : Intégrer progressivement d'autres banques au réseau fédéré

### Perspectives d'Amélioration

1. **FedAvg vrai pour réseaux de neurones** : Implémenter l'agrégation de poids pour PyTorch
2. **Personnalisation locale** : Permettre à chaque banque d'adapter le modèle à ses données
3. **Détection d'anomalies** : Ajouter une couche de détection d'anomalies non supervisée
4. **Active Learning** : Sélectionner intelligemment les transactions à étiqueter
5. **Explainabilité avancée** : Intégrer d'autres méthodes (LIME, Counterfactuals)

---

## 📚 Références

### Datasets
- **ULB Credit Card Fraud Detection** : Kaggle dataset (284,807 transactions)
- **Synthetic Dataset** : fraud_detection_credit_card_small.csv (100,000 transactions)

### Algorithmes
- **FedAvg** : McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
- **SMOTE** : Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique", JAIR 2002
- **SHAP** : Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NIPS 2017

### Technologies
- **XGBoost** : Library pour gradient boosting
- **TensorFlow/Keras** : Deep learning framework
- **SHAP** : Explainable AI library
- **Google Gemini** : LLM pour génération d'explications

---

*Ce rapport a été généré dans le cadre du projet de fin de stage sur la détection de fraude bancaire avec Federated Learning et Agentic AI.*
