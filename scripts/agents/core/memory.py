"""
memory.py — Système de Mémoire Multi-Couches pour le Framework Agentic AI
Stage : Détection de Fraude Bancaire avec Agentic AI, LLMs et Federated Learning

Étape 5 du blueprint "How to Build an AI Agent" — Memory Systems :
┌─────────────────────────────────────────────────────────────────────┐
│                      MEMORY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐   Données de la transaction en cours          │
│  │  Working Memory   │   TTL: durée du pipeline (~secondes)          │
│  │  (Court terme)    │   Partagée entre agents pendant le traitement │
│  └──────────────────┘                                                │
│                                                                      │
│  ┌──────────────────┐   N dernières transactions traitées            │
│  │  Episodic Memory  │   TTL: session ou fenêtre glissante           │
│  │  (Conversation)   │   Contexte conversationnel pour le LLM        │
│  └──────────────────┘                                                │
│                                                                      │
│  ┌──────────────────┐   Patterns de fraude connus                    │
│  │  Semantic Memory  │   Basé sur des embeddings ou features clés    │
│  │  (Patterns)       │   Recherche par similarité                    │
│  └──────────────────┘                                                │
│                                                                      │
│  ┌──────────────────┐   Base de données persistante                  │
│  │  Long-Term Memory │   SQLite : décisions, feedback, statistiques  │
│  │  (Persistante)    │   Survit aux redémarrages                     │
│  └──────────────────┘                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
"""

import time
import json
import sqlite3
import os
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Working Memory — Mémoire de travail (transaction en cours)
# ═══════════════════════════════════════════════════════════════════════════════

class WorkingMemory:
    """
    Mémoire de travail pour la transaction en cours de traitement.

    Stocke l'état intermédiaire partagé entre les agents pendant
    le traitement d'une seule transaction. Effacée après chaque pipeline.

    Analogie : la RAM d'un ordinateur — rapide, volatile, limitée.
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._history: List[Dict] = []  # Historique des modifications
        self._created_at: float = time.time()

    def set(self, key: str, value: Any):
        """Stocke une valeur dans la mémoire de travail."""
        self._store[key] = value
        self._history.append({
            'action': 'set',
            'key': key,
            'timestamp': time.time() - self._created_at,
        })

    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de la mémoire de travail."""
        return self._store.get(key, default)

    def update(self, data: dict):
        """Met à jour plusieurs valeurs à la fois."""
        self._store.update(data)
        for key in data:
            self._history.append({
                'action': 'update',
                'key': key,
                'timestamp': time.time() - self._created_at,
            })

    def get_all(self) -> dict:
        """Retourne tout le contenu de la mémoire de travail."""
        return dict(self._store)

    def get_pipeline_state(self) -> dict:
        """Retourne l'état résumé du pipeline pour le contexte."""
        state = {}
        for key, value in self._store.items():
            if key.endswith('_result') and isinstance(value, dict):
                agent_name = key.replace('_result', '')
                state[agent_name] = {
                    k: v for k, v in value.items()
                    if not k.startswith('_')
                }
        return state

    def clear(self):
        """Vide la mémoire de travail (entre deux transactions)."""
        self._store.clear()
        self._history.clear()
        self._created_at = time.time()

    @property
    def age_ms(self) -> float:
        """Âge de la mémoire de travail en millisecondes."""
        return (time.time() - self._created_at) * 1000


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Episodic Memory — Mémoire épisodique (historique récent)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """Un épisode = une transaction traitée + son résultat."""
    transaction_id: str
    transaction_data: dict
    pipeline_result: dict
    decision: str
    probability: float
    was_fraud: Optional[bool] = None  # Vérité terrain (si connue)
    feedback: Optional[str] = None    # Feedback humain (si donné)
    timestamp: float = field(default_factory=time.time)


class EpisodicMemory:
    """
    Mémoire épisodique — historique des N dernières transactions.

    Utilisée pour :
    - Fournir du contexte au LLM (few-shot examples récents)
    - Détecter des patterns récurrents (même client, même merchant)
    - Calculer des statistiques glissantes (taux de fraude récent)

    Taille fixe avec politique FIFO (First In, First Out).
    """

    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self._episodes: deque = deque(maxlen=max_episodes)
        self._index: Dict[str, Episode] = {}  # Index par transaction_id

    def add(self, episode: Episode):
        """Ajoute un épisode à la mémoire."""
        self._episodes.append(episode)
        self._index[episode.transaction_id] = episode

        # Nettoyage de l'index si la deque a éjecté des éléments
        if len(self._index) > self.max_episodes * 1.5:
            active_ids = {e.transaction_id for e in self._episodes}
            self._index = {k: v for k, v in self._index.items()
                           if k in active_ids}

    def get_recent(self, n: int = 10) -> List[Episode]:
        """Retourne les N épisodes les plus récents."""
        return list(self._episodes)[-n:]

    def get_by_id(self, transaction_id: str) -> Optional[Episode]:
        """Retrouve un épisode par son ID."""
        return self._index.get(transaction_id)

    def get_similar_decisions(self, decision: str, n: int = 5) -> List[Episode]:
        """Retourne les N derniers épisodes avec la même décision."""
        return [e for e in reversed(self._episodes)
                if e.decision == decision][:n]

    def get_fraud_rate(self, window: int = 100) -> float:
        """Taux de fraude sur les N dernières transactions."""
        recent = list(self._episodes)[-window:]
        if not recent:
            return 0.0
        fraud_count = sum(1 for e in recent
                          if e.probability >= 0.5 or e.was_fraud)
        return fraud_count / len(recent)

    def get_context_for_llm(self, n: int = 3) -> str:
        """
        Génère un résumé des transactions récentes pour le contexte LLM.

        Format optimisé pour être injecté dans le prompt du LLM
        comme few-shot examples.
        """
        recent = self.get_recent(n)
        if not recent:
            return "Aucune transaction récente en mémoire."

        lines = ["Transactions récentes traitées :"]
        for ep in recent:
            lines.append(
                f"  • {ep.transaction_id}: Décision={ep.decision}, "
                f"P(fraude)={ep.probability:.1%}"
                + (f", Feedback: {ep.feedback}" if ep.feedback else "")
            )
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._episodes)

    def get_stats(self) -> dict:
        """Statistiques globales de la mémoire épisodique."""
        if not self._episodes:
            return {'size': 0}

        decisions = {}
        for ep in self._episodes:
            decisions[ep.decision] = decisions.get(ep.decision, 0) + 1

        probs = [ep.probability for ep in self._episodes]
        return {
            'size': len(self._episodes),
            'max_capacity': self.max_episodes,
            'decision_distribution': decisions,
            'avg_probability': round(np.mean(probs), 4),
            'fraud_rate_100': f"{self.get_fraud_rate(100):.1%}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Semantic Memory — Mémoire sémantique (patterns de fraude)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FraudPattern:
    """Un pattern de fraude connu."""
    pattern_id: str
    name: str
    description: str
    features: Dict[str, Any]     # Caractéristiques du pattern
    severity: str                 # CRITIQUE, ÉLEVÉ, MOYEN
    frequency: int = 0            # Nombre de fois observé
    last_seen: float = 0.0
    embedding: Optional[np.ndarray] = None  # Embedding pour similarité


class SemanticMemory:
    """
    Mémoire sémantique — base de connaissances des patterns de fraude.

    Stocke les patterns de fraude connus avec leurs caractéristiques.
    Permet la recherche par similarité pour identifier des transactions
    qui ressemblent à des fraudes connues.

    Dans un système de production, cette mémoire serait alimentée par :
    - Les cas de fraude confirmés
    - Les règles métier des experts
    - Les résultats de l'analyse de drift
    """

    # Patterns de fraude prédéfinis (expertise métier)
    DEFAULT_PATTERNS = [
        FraudPattern(
            pattern_id="FP001",
            name="Transaction nocturne à montant élevé",
            description="Transaction effectuée entre 0h et 4h avec un montant "
                        "supérieur à 1500€, typique de fraude par carte volée.",
            features={'hour_range': (0, 4), 'min_amount': 1500},
            severity='ÉLEVÉ',
        ),
        FraudPattern(
            pattern_id="FP002",
            name="Anomalie PCA V14-V12",
            description="Combinaison de V14 < -3.0 et V12 < -2.5, fortement "
                        "corrélée avec les fraudes dans le dataset ULB.",
            features={'V14_max': -3.0, 'V12_max': -2.5},
            severity='CRITIQUE',
        ),
        FraudPattern(
            pattern_id="FP003",
            name="Rafale de micro-transactions",
            description="Plusieurs transactions de petit montant (<50€) en "
                        "moins de 5 minutes, technique de test de carte.",
            features={'max_amount': 50, 'max_interval_sec': 300, 'min_count': 3},
            severity='ÉLEVÉ',
        ),
        FraudPattern(
            pattern_id="FP004",
            name="Catégorie à risque + géolocalisation anormale",
            description="Transaction dans une catégorie à risque (travel, "
                        "online_retail) depuis une ville très peu peuplée.",
            features={'categories': ['travel', 'online_retail'], 'max_city_pop': 500},
            severity='MOYEN',
        ),
        FraudPattern(
            pattern_id="FP005",
            name="Fraude par usurpation d'identité",
            description="Changement brusque de comportement : montant très "
                        "supérieur à la moyenne du client, nouvelle catégorie.",
            features={'amount_multiplier': 5.0, 'new_category': True},
            severity='CRITIQUE',
        ),
    ]

    def __init__(self):
        self._patterns: Dict[str, FraudPattern] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Charge les patterns de fraude prédéfinis."""
        for pattern in self.DEFAULT_PATTERNS:
            self._patterns[pattern.pattern_id] = pattern

    def add_pattern(self, pattern: FraudPattern):
        """Ajoute un nouveau pattern de fraude."""
        self._patterns[pattern.pattern_id] = pattern

    def get_pattern(self, pattern_id: str) -> Optional[FraudPattern]:
        return self._patterns.get(pattern_id)

    def match_patterns(self, transaction: dict) -> List[Tuple[FraudPattern, float]]:
        """
        Recherche les patterns correspondant à une transaction.

        Returns:
            Liste de (pattern, score_matching) triée par score décroissant.
        """
        matches = []
        for pattern in self._patterns.values():
            score = self._compute_match_score(transaction, pattern)
            if score > 0.3:  # Seuil minimum de correspondance
                pattern.frequency += 1
                pattern.last_seen = time.time()
                matches.append((pattern, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _compute_match_score(self, transaction: dict,
                             pattern: FraudPattern) -> float:
        """Calcule un score de correspondance simple entre 0 et 1."""
        features = pattern.features
        score_components = []

        # Vérification des montants
        if 'min_amount' in features:
            amt = transaction.get('amt', transaction.get('Amount', 0))
            if amt >= features['min_amount']:
                score_components.append(1.0)
            else:
                score_components.append(max(0, amt / features['min_amount']))

        if 'max_amount' in features:
            amt = transaction.get('amt', transaction.get('Amount', 0))
            if amt <= features['max_amount']:
                score_components.append(1.0)
            else:
                score_components.append(0.0)

        # Vérification des features PCA
        for key in ['V14_max', 'V12_max']:
            if key in features:
                feat_name = key.replace('_max', '')
                val = transaction.get(feat_name, 0)
                if val <= features[key]:
                    score_components.append(1.0)
                else:
                    score_components.append(
                        max(0, 1 - abs(val - features[key]) / 3)
                    )

        # Vérification des catégories
        if 'categories' in features:
            cat = transaction.get('category', '').lower()
            if any(c in cat for c in features['categories']):
                score_components.append(1.0)
            else:
                score_components.append(0.0)

        if not score_components:
            return 0.0
        return sum(score_components) / len(score_components)

    def get_all_patterns(self) -> List[FraudPattern]:
        return list(self._patterns.values())

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Long-Term Memory — Mémoire persistante (SQLite)
# ═══════════════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    Mémoire à long terme — persistance SQLite.

    Stocke de manière permanente :
    - Toutes les décisions prises (audit trail)
    - Le feedback des analystes humains
    - Les statistiques agrégées
    - Les configurations des seuils adaptatifs

    Survit aux redémarrages du système.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            base = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(base, 'data', 'agent_memory.db')

        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Crée les tables si elles n'existent pas."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    decision TEXT NOT NULL,
                    probability REAL NOT NULL,
                    risk_level TEXT,
                    agent_latencies TEXT,
                    shap_top3 TEXT,
                    pipeline_trace TEXT,
                    feedback TEXT,
                    was_fraud INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metrics_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT NOT NULL,
                    config_value TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    reason TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_txn
                ON decisions(transaction_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_decision
                ON decisions(decision)
            """)

    def save_decision(self, transaction_id: str, decision: str,
                      probability: float, risk_level: str = None,
                      agent_latencies: dict = None, shap_top3: list = None,
                      pipeline_trace: dict = None):
        """Sauvegarde une décision de manière persistante."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO decisions
                (transaction_id, timestamp, decision, probability,
                 risk_level, agent_latencies, shap_top3, pipeline_trace)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction_id,
                time.time(),
                decision,
                float(probability),
                risk_level,
                json.dumps(agent_latencies or {}, default=str),
                json.dumps(shap_top3 or [], default=str),
                json.dumps(pipeline_trace or {}, default=str),
            ))

    def save_feedback(self, transaction_id: str, feedback: str,
                      was_fraud: bool = None):
        """Enregistre le feedback d'un analyste humain."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE decisions
                SET feedback = ?, was_fraud = ?
                WHERE transaction_id = ?
            """, (feedback, int(was_fraud) if was_fraud is not None else None,
                  transaction_id))

    def get_decision_stats(self, hours: float = 24) -> dict:
        """Statistiques des décisions sur les N dernières heures."""
        cutoff = time.time() - hours * 3600
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT decision, COUNT(*) as cnt,
                       AVG(probability) as avg_prob
                FROM decisions
                WHERE timestamp > ?
                GROUP BY decision
            """, (cutoff,)).fetchall()

        stats = {}
        total = 0
        for decision, count, avg_prob in rows:
            stats[decision] = {
                'count': count,
                'avg_probability': round(avg_prob, 4),
            }
            total += count

        return {
            'period_hours': hours,
            'total_decisions': total,
            'distribution': stats,
        }

    def get_recent_decisions(self, n: int = 20) -> List[dict]:
        """Retourne les N décisions les plus récentes."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT transaction_id, timestamp, decision, probability,
                       risk_level, feedback, was_fraud
                FROM decisions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (n,)).fetchall()

        return [
            {
                'transaction_id': r[0],
                'timestamp': r[1],
                'decision': r[2],
                'probability': r[3],
                'risk_level': r[4],
                'feedback': r[5],
                'was_fraud': bool(r[6]) if r[6] is not None else None,
            }
            for r in rows
        ]

    def save_agent_stats(self, agent_name: str, metrics: dict):
        """Sauvegarde les métriques d'un agent."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO agent_stats (agent_name, timestamp, metrics_json)
                VALUES (?, ?, ?)
            """, (agent_name, time.time(), json.dumps(metrics)))

    def get_feedback_accuracy(self) -> dict:
        """Calcule la précision des décisions vs feedback humain."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT decision, was_fraud, COUNT(*)
                FROM decisions
                WHERE was_fraud IS NOT NULL
                GROUP BY decision, was_fraud
            """).fetchall()

        if not rows:
            return {'message': 'Aucun feedback disponible'}

        matrix = {}
        for decision, was_fraud, count in rows:
            if decision not in matrix:
                matrix[decision] = {'correct': 0, 'incorrect': 0}
            is_block = decision in ('BLOCK', 'REVIEW')
            if (is_block and was_fraud) or (not is_block and not was_fraud):
                matrix[decision]['correct'] += count
            else:
                matrix[decision]['incorrect'] += count

        return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MemorySystem — Système intégré multi-couches
# ═══════════════════════════════════════════════════════════════════════════════

class MemorySystem:
    """
    Système de mémoire intégré multi-couches.

    Fournit un point d'accès unique à toutes les couches de mémoire
    pour les agents du pipeline.

    Usage:
        memory = MemorySystem(db_path="data/agent_memory.db")

        # Mémoire de travail (transaction en cours)
        memory.working.set('transaction', tx_data)

        # Mémoire épisodique (historique récent)
        memory.episodic.add(Episode(...))

        # Mémoire sémantique (patterns)
        matches = memory.semantic.match_patterns(tx_data)

        # Mémoire long-terme (persistante)
        memory.long_term.save_decision(...)
    """

    def __init__(self, db_path: str = None, max_episodes: int = 1000):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(max_episodes=max_episodes)
        self.semantic = SemanticMemory()
        self.long_term = LongTermMemory(db_path=db_path)

    def reset_working(self):
        """Réinitialise la mémoire de travail entre deux transactions."""
        self.working.clear()

    def get_full_context(self) -> dict:
        """Retourne un contexte complet pour le LLM."""
        return {
            'working_state': self.working.get_pipeline_state(),
            'recent_episodes': self.episodic.get_context_for_llm(3),
            'fraud_patterns_count': self.semantic.pattern_count,
            'recent_fraud_rate': self.episodic.get_fraud_rate(100),
        }

    def get_summary(self) -> dict:
        return {
            'working_memory_keys': len(self.working._store),
            'episodic_memory_size': self.episodic.size,
            'semantic_patterns': self.semantic.pattern_count,
            'long_term_db': self.long_term.db_path,
        }
