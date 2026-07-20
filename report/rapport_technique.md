# Rapport Technique — Système Multi-Agents (Agentic AI)
## Projet : Détection de Fraude Bancaire avec Federated Learning

> **Version complète** — Objectifs, Inputs, Outputs et métriques pour chaque agent du pipeline.

---

## 1. Vue d'ensemble

Le système multi-agents constitue le cœur opérationnel du framework. Son objectif est de transformer le modèle ML (boîte noire) en un **pipeline de décision intelligent, explicable et automatisé**, capable de traiter des milliers de transactions bancaires en temps réel.

### Pipeline séquentiel

```
Transaction --> Agent 1 --> (si suspect) Agent 2 --> Agent 3 --> Agent 4 --> Rapport
              (Filtrage)    (Analyse ML)           (Décision)    (LLM)
```

---

## 2. Agent 1 — Surveillance & Filtrage (Pré-filtrage rapide)

### Objectif
Identifier et éliminer rapidement les transactions **manifestement légitimes** via des règles métier simples, **avant** toute inférence ML coûteuse. Réduit les coûts de calcul de 60 à 75%.

### Input
```python
transaction = {
    'V14': -1.5,    # Feature PCA anonyme
    'V12':  0.8,
    'V10':  0.3,
    'V4':   2.1,
    'Amount': 150.0  # Montant (standardisé)
}
```

### Output
```python
{
    'suspicious':    bool,        # True = transaction suspecte -> vers Agent 2
    'fast_decision': str | None,  # 'ALLOW' si non suspecte, None sinon
    'reasons':       list[str],   # Règles métier déclenchées
    'risk_score':    int          # Nombre de règles activées (0 à N)
}
```

### Règles métier configurées

| Règle | Condition | Signification |
|:---|:---|:---|
| R1 | V14 < -2.5 | Fort indicateur statistique de fraude carte |
| R2 | V12 < -2.0 | Corrélation connue avec fraude sur carte |
| R3 | V10 < -2.0 | Valeur anormale dans l'espace PCA |
| R4 | V4 > 2.5 | Pattern de comportement frauduleux connu |
| R5 | Amount > 2.0 | Montant standardisé anormalement élevé |
| R6 | >= 2 anomalies mineures (V3,V7,V11,V16) | Combinaison de signaux faibles |

### Métriques d'efficacité

| Métrique | Définition | Valeur observée |
|:---|:---|:---:|
| **Taux de rejet précoce** | % transactions légitimes autorisées sans ML | 60–75% |
| **Latence fast-track** | Temps de traitement pour transactions normales | < 5 ms |
| **Taux de suspicion** | % transactions envoyées à l'Agent 2 | 25–40% |
| **Taux de faux négatifs Agent 1** | Fraudes manquées par les règles métier | À mesurer |

---

## 3. Agent 2 — Analyse ML + Explicabilité SHAP

### Objectif
Analyser en profondeur les transactions suspectes avec le **modèle XGBoost fédéré** et calculer les valeurs **SHAP** pour l'explicabilité quantitative.

### Input
```python
# Toutes les features de la transaction (le RobustScaler est appliqué internement)
transaction = {'V1': ..., 'V2': ..., ..., 'V28': ..., 'Amount': ...}
```

### Output
```python
{
    'probability': float,      # Probabilité de fraude [0, 1]
    'risk_level':  str,        # 'CRITIQUE' | 'ÉLEVÉ' | 'MOYEN' | 'FAIBLE'
    'shap_top5':   list[dict], # Top 5 features influentes
    'all_shap':    dict        # SHAP pour toutes les features
}

# Format shap_top5 :
[{
    'feature':   'V14',
    'value':     -6.78,
    'shap':      +4.23,
    'direction': 'AUGMENTE risque fraude'
}]
```

### Seuils de niveau de risque

| Niveau | Condition | Couleur |
|:---|:---|:---:|
| **CRITIQUE** | probabilité >= 0.85 | Rouge |
| **ÉLEVÉ** | 0.60 <= probabilité < 0.85 | Orange |
| **MOYEN** | 0.35 <= probabilité < 0.60 | Jaune |
| **FAIBLE** | probabilité < 0.35 | Vert |

### Métriques d'efficacité

| Métrique | Valeur |
|:---|:---:|
| **F1-Score** (XGBoost fédéré) | 0.8159 |
| **AUC-ROC** | 0.9788 |
| **Recall** | 83.67% |
| **Precision** | 79.61% |
| **Latence analyse** (pred + SHAP) | < 100 ms |

---

## 4. Agent 3 — Décision Métier Automatisée

### Objectif
Transformer la **probabilité brute** en une **décision métier concrète** avec SLA et notifications automatiques.

### Input
```python
probability = 0.92  # Float [0,1], issu de l'Agent 2
```

### Output
```python
{
    'decision':       'BLOCK',
    'probability':    0.92,
    'threshold_used': 0.85,
    'severity':       'CRITIQUE',
    'action':         'Transaction refusée. Dossier fraude ouvert.',
    'notify':         ['equipe_fraude', 'client', 'conformite'],
    'icon':           'BLOCK',
    'sla_min':        0
}
```

### Matrice de décision (seuils configurables)

| Décision | Condition | Sévérité | Action | SLA | Notifications |
|:---|:---|:---|:---|---:|:---|
| **BLOCK** | prob >= 0.85 | CRITIQUE | Transaction refusée, dossier fraude ouvert | 0 min | equipe_fraude, client, conformite |
| **REVIEW** | 0.60 <= prob < 0.85 | ÉLEVÉ | Transaction suspendue, analyste assigné | 15 min | analyste_fraude, client |
| **ALERT** | 0.35 <= prob < 0.60 | MOYEN | Autorisée avec surveillance renforcée | 60 min | equipe_surveillance |
| **ALLOW** | prob < 0.35 | FAIBLE | Transaction approuvée normalement | — | — |

### Métriques d'efficacité

| Métrique | Définition |
|:---|:---|
| **Distribution des décisions** | % BLOCK / REVIEW / ALERT / ALLOW |
| **Taux de faux positifs** | Transactions légitimes bloquées (coût opérationnel) |
| **Taux de faux négatifs** | Fraudes ayant reçu ALLOW ou ALERT (risque financier) |
| **Respect des SLA** | % de décisions traitées dans les délais |
| **Taux d'automatisation** | % décisions sans intervention humaine |

---

## 5. Agent 4 — Explication LLM (Google Gemini 1.5 Flash)

### Objectif
Transformer la sortie mathématique en un **rapport en langage naturel français**, compréhensible par un analyste non technique. L'agent répond à : *"Pourquoi cette transaction a-t-elle été signalée ?"*

### Input
```python
{
    'transaction':         dict,       # Features de la transaction (Agent 1)
    'score_fraude':        float,      # Probabilité de fraude (Agent 2)
    'shap_top5':           list[dict], # Top 5 SHAP (Agent 2)
    'decision':            str,        # Décision (Agent 3)
    'action':              str,        # Description action (Agent 3)
    'surveillance_alerts': list[str]   # Règles déclenchées (Agent 1)
}
```

### Output
```python
{
    'rapport':       str,   # Rapport complet en français (~200-400 mots)
    'resume':        str,   # Résumé en 1 phrase pour dashboard
    'features_cles': list,  # Noms des features les plus déterminantes
    'confidence':    str,   # Niveau de confiance de l'explication
    'mode':          str    # 'online' (Gemini API) | 'offline' (template)
}
```

### Exemple de rapport généré
```
RAPPORT — TXN-0840 | Décision : BLOCK
Score de fraude : 99.9% | Risque : CRITIQUE

ANALYSE DES FACTEURS DÉTERMINANTS

1. V14 = -6.781 (SHAP = +4.229 => AUGMENTE risque fraude)
   Cette variable présente une valeur extrêmement basse, caractéristique
   d'un comportement frauduleux selon les patterns appris par le modèle.

2. V17 = -7.324 (SHAP = +1.295)
   Second facteur contributeur, confirmant la fraude.

ALERTES SURVEILLANCE
• V14 très bas : fort indicateur de fraude (V14=-6.781)
• V12 anormal : corrélé avec fraude carte (V12=-4.726)

ACTION : Transaction refusée. Dossier fraude ouvert.
Notifications : equipe_fraude, client, conformite | SLA : 0 min
```

### LLM utilisé : Google Gemini 1.5 Flash
- Prompt structuré garantissant des réponses professionnelles en français
- Mode **fallback offline** (template Python) si API indisponible

### Métriques d'efficacité

| Métrique | Valeur |
|:---|:---:|
| **Latence de génération** | 1.2 – 2.5 s |
| **Taux de disponibilité** (LLM vs template) | Dépend de l'API |
| **Pertinence SHAP** | Cohérence texte vs valeurs SHAP |
| **Compréhensibilité** | Score analystes (1–5) — À mesurer |
| **Complétude** | % rapports couvrant Top-5 SHAP — À mesurer |

---

## 6. Métriques Globales du Pipeline Multi-Agents

| Métrique | Valeur |
|:---|:---:|
| Taux de rejet précoce (Agent 1) | 60–75% |
| Latence Fast-Track (Agent 1 seul) | < 5 ms |
| Latence Analyse ML + SHAP (Agent 2) | < 100 ms |
| Latence Pipeline complet avec LLM | 1.2 – 2.5 s |
| F1-Score (XGBoost fédéré embarqué) | 0.8159 |
| AUC-ROC | 0.9788 |
| Recall (détection fraude) | 83.67% |
| Taux de fausses alertes | 20.39% (1 – Precision) |

---
*Document généré dans le cadre du projet de fin de stage — Détection de Fraude Bancaire avec Federated Learning & Agentic AI.*
