"""
agent_base.py — Classe abstraite BaseAgent et types de données fondamentaux
Stage : Système Agentic AI de Détection de Fraude Bancaire

Suivant le blueprint "How to Build an AI Agent" (8 étapes) :
- Étape 1 : Purpose & Scope → Défini par chaque agent concret
- Étape 8 : Testing & Evals → Métriques intégrées dans BaseAgent

Chaque agent suit le cycle de vie :
    initialize() → process(input) → respond() → log()

Architecture inspirée des frameworks CrewAI, LangGraph et LlamaIndex,
adaptée au contexte spécifique de la détection de fraude bancaire.
"""

import time
import uuid
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Énumérations et Types
# ═══════════════════════════════════════════════════════════════════════════════

class AgentStatus(Enum):
    """États possibles d'un agent."""
    IDLE = "idle"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"


class Priority(Enum):
    """Niveaux de priorité pour le traitement."""
    CRITICAL = 1    # Fraude quasi-certaine → traitement immédiat
    HIGH = 2        # Transaction suspecte → priorité élevée
    MEDIUM = 3      # Alerte modérée → file normale
    LOW = 4         # Transaction normale → traitement standard


# ═══════════════════════════════════════════════════════════════════════════════
# AgentResponse — Réponse standardisée de chaque agent
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResponse:
    """
    Réponse standardisée retournée par chaque agent.

    Permet au pipeline de traiter uniformément les sorties
    de tous les agents, quelle que soit leur spécialisation.
    """
    agent_name: str                        # Nom de l'agent émetteur
    status: AgentStatus                    # Statut du traitement
    data: Dict[str, Any] = field(         # Données de sortie (spécifiques à l'agent)
        default_factory=dict)
    confidence: float = 1.0                # Confiance dans la réponse [0, 1]
    latency_ms: float = 0.0               # Temps de traitement en ms
    errors: List[str] = field(            # Erreurs rencontrées
        default_factory=list)
    warnings: List[str] = field(          # Avertissements
        default_factory=list)
    metadata: Dict[str, Any] = field(     # Métadonnées additionnelles
        default_factory=dict)
    timestamp: str = ""                    # Horodatage ISO 8601
    trace_id: str = ""                     # ID de traçabilité

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())[:8]

    @property
    def is_success(self) -> bool:
        return self.status == AgentStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        return self.status == AgentStatus.ERROR

    def to_dict(self) -> dict:
        return {
            'agent_name': self.agent_name,
            'status': self.status.value,
            'data': self.data,
            'confidence': self.confidence,
            'latency_ms': round(self.latency_ms, 2),
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'trace_id': self.trace_id,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AgentMetrics — Suivi de performance en temps réel
# ═══════════════════════════════════════════════════════════════════════════════

class AgentMetrics:
    """
    Collecteur de métriques de performance pour un agent.

    Étape 8 du blueprint : Testing & Evals
    - Latence moyenne et P99
    - Taux de succès / erreur
    - Throughput (transactions/seconde)
    - Distribution des décisions
    """

    def __init__(self):
        self._latencies: List[float] = []
        self._success_count: int = 0
        self._error_count: int = 0
        self._total_count: int = 0
        self._start_time: float = time.time()
        self._decision_counts: Dict[str, int] = {}

    def record(self, latency_ms: float, success: bool = True,
               decision: str = None):
        """Enregistre les métriques d'un traitement."""
        self._latencies.append(latency_ms)
        self._total_count += 1
        if success:
            self._success_count += 1
        else:
            self._error_count += 1
        if decision:
            self._decision_counts[decision] = \
                self._decision_counts.get(decision, 0) + 1

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def p99_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def success_rate(self) -> float:
        if self._total_count == 0:
            return 1.0
        return self._success_count / self._total_count

    @property
    def throughput(self) -> float:
        """Transactions par seconde."""
        elapsed = time.time() - self._start_time
        if elapsed == 0:
            return 0.0
        return self._total_count / elapsed

    def get_summary(self) -> dict:
        return {
            'total_processed': self._total_count,
            'success_count': self._success_count,
            'error_count': self._error_count,
            'success_rate': f"{self.success_rate * 100:.1f}%",
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'p99_latency_ms': round(self.p99_latency_ms, 2),
            'throughput_tps': round(self.throughput, 2),
            'decision_distribution': dict(self._decision_counts),
        }

    def reset(self):
        self.__init__()


# ═══════════════════════════════════════════════════════════════════════════════
# AgentLogger — Logging structuré pour les agents
# ═══════════════════════════════════════════════════════════════════════════════

class AgentLogger:
    """
    Logger structuré pour tracer les actions de chaque agent.

    Produit des logs au format JSON pour analyse et debugging.
    Compatible avec ELK Stack, Datadog, CloudWatch.
    """

    def __init__(self, agent_name: str, level: int = logging.INFO):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"agentic.{agent_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f"[%(asctime)s] [{agent_name}] %(levelname)s — %(message)s",
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(level)

    def info(self, message: str, **kwargs):
        self.logger.info(self._format(message, kwargs))

    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format(message, kwargs))

    def error(self, message: str, **kwargs):
        self.logger.error(self._format(message, kwargs))

    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format(message, kwargs))

    def action(self, action_name: str, details: dict = None):
        """Log une action spécifique de l'agent."""
        msg = f"ACTION: {action_name}"
        if details:
            msg += f" | {json.dumps(details, ensure_ascii=False, default=str)}"
        self.logger.info(msg)

    def _format(self, message: str, extra: dict) -> str:
        if extra:
            parts = " | ".join(f"{k}={v}" for k, v in extra.items())
            return f"{message} | {parts}"
        return message


# ═══════════════════════════════════════════════════════════════════════════════
# BaseAgent — Classe abstraite fondamentale
# ═══════════════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """
    Classe abstraite de base pour tous les agents du système.

    Suivant le blueprint "How to Build an AI Agent" :
    ┌────────────────────────────────────────────────────────────────┐
    │  Étape 1 — Purpose & Scope                                     │
    │  → Défini par chaque agent via `AGENT_PURPOSE` et `AGENT_ROLE` │
    │                                                                │
    │  Étape 2 — System Prompt Design                                │
    │  → `SYSTEM_PROMPT` pour les agents utilisant un LLM            │
    │                                                                │
    │  Étape 4 — Tools & Integrations                                │
    │  → `self.tools` — dict d'outils disponibles pour l'agent       │
    │                                                                │
    │  Étape 5 — Memory Systems                                      │
    │  → `self.memory` — accès au MemorySystem partagé               │
    │                                                                │
    │  Étape 8 — Testing & Evals                                     │
    │  → `self.metrics` — métriques de performance intégrées         │
    └────────────────────────────────────────────────────────────────┘

    Cycle de vie d'un agent :
        1. __init__()     → Configuration et injection de dépendances
        2. process()      → Traitement principal (à implémenter)
        3. _post_process() → Actions après traitement (logging, mémoire)

    Chaque agent retourne un `AgentResponse` standardisé.
    """

    # ── Attributs de classe (à surcharger dans chaque agent) ────────────────
    AGENT_NAME: str = "BaseAgent"
    AGENT_ROLE: str = "Agent de base"
    AGENT_PURPOSE: str = "Traitement générique"
    AGENT_VERSION: str = "2.0.0"

    # Guardrails : contraintes de l'agent
    MAX_LATENCY_MS: float = 5000.0    # Timeout max en millisecondes
    MAX_RETRIES: int = 2               # Nombre max de retry
    REQUIRES_LLM: bool = False         # True si l'agent utilise un LLM

    def __init__(self, config: dict = None, memory=None, tools: dict = None):
        """
        Initialise l'agent avec ses dépendances.

        Args:
            config: Configuration spécifique à l'agent (seuils, paramètres)
            memory: Instance de MemorySystem partagée entre agents
            tools:  dict de BaseTool disponibles pour cet agent
        """
        self.config = config or {}
        self.memory = memory
        self.tools = tools or {}
        self.metrics = AgentMetrics()
        self.logger = AgentLogger(self.AGENT_NAME)
        self._status = AgentStatus.IDLE
        self._initialized = True

        self.logger.info(
            f"Agent initialisé",
            role=self.AGENT_ROLE,
            version=self.AGENT_VERSION,
            tools=list(self.tools.keys()) if self.tools else "none"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # API Principale
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, input_data: dict, context: dict = None) -> AgentResponse:
        """
        Point d'entrée principal — exécute l'agent avec mesure de performance.

        Args:
            input_data: Données d'entrée (transaction, résultats d'agents précédents)
            context:    Contexte additionnel (pipeline state, trace_id, etc.)

        Returns:
            AgentResponse standardisé
        """
        self._status = AgentStatus.PROCESSING
        start_time = time.time()
        context = context or {}
        trace_id = context.get('trace_id', str(uuid.uuid4())[:8])

        self.logger.action("START_PROCESSING", {
            'trace_id': trace_id,
            'input_keys': list(input_data.keys())
        })

        try:
            # ── Exécution avec retry ──────────────────────────────────────
            result = self._execute_with_retry(input_data, context)

            latency_ms = (time.time() - start_time) * 1000

            # ── Vérification du timeout ──────────────────────────────────
            if latency_ms > self.MAX_LATENCY_MS:
                self.logger.warning(
                    f"Latence excessive : {latency_ms:.0f}ms > {self.MAX_LATENCY_MS}ms"
                )

            # ── Construction de la réponse ────────────────────────────────
            response = AgentResponse(
                agent_name=self.AGENT_NAME,
                status=AgentStatus.SUCCESS,
                data=result,
                latency_ms=latency_ms,
                trace_id=trace_id,
                confidence=result.get('_confidence', 1.0),
            )

            # ── Post-processing ───────────────────────────────────────────
            self._post_process(input_data, response)
            self.metrics.record(latency_ms, success=True,
                                decision=result.get('decision'))
            self._status = AgentStatus.SUCCESS

            self.logger.action("PROCESSING_COMPLETE", {
                'trace_id': trace_id,
                'latency_ms': round(latency_ms, 2),
                'status': 'SUCCESS'
            })

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._status = AgentStatus.ERROR
            self.metrics.record(latency_ms, success=False)

            self.logger.error(
                f"Erreur de traitement : {str(e)[:200]}",
                trace_id=trace_id
            )

            return AgentResponse(
                agent_name=self.AGENT_NAME,
                status=AgentStatus.ERROR,
                errors=[str(e)],
                latency_ms=latency_ms,
                trace_id=trace_id,
            )

    def _execute_with_retry(self, input_data: dict, context: dict) -> dict:
        """Exécute process() avec retry automatique en cas d'erreur."""
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return self.process(input_data, context)
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    self.logger.warning(
                        f"Tentative {attempt}/{self.MAX_RETRIES} échouée, retry...",
                        error=str(e)[:100]
                    )
                    time.sleep(0.1 * attempt)  # Backoff exponentiel léger
        raise last_error

    # ─────────────────────────────────────────────────────────────────────────
    # Méthodes abstraites (à implémenter par chaque agent)
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def process(self, input_data: dict, context: dict = None) -> dict:
        """
        Logique métier principale de l'agent.

        DOIT être implémenté par chaque agent concret.

        Args:
            input_data: Données d'entrée
            context:    Contexte du pipeline

        Returns:
            dict contenant les résultats (sera encapsulé dans AgentResponse)
        """
        raise NotImplementedError

    # ─────────────────────────────────────────────────────────────────────────
    # Hooks (optionnels, à surcharger si besoin)
    # ─────────────────────────────────────────────────────────────────────────

    def _post_process(self, input_data: dict, response: AgentResponse):
        """
        Hook exécuté après le traitement réussi.
        Utilisé pour sauvegarder en mémoire, émettre des événements, etc.
        """
        if self.memory:
            self.memory.working.update({
                f'{self.AGENT_NAME}_result': response.data,
                f'{self.AGENT_NAME}_latency': response.latency_ms,
            })

    def validate_input(self, input_data: dict) -> bool:
        """Valide les données d'entrée. Retourne True si valides."""
        return bool(input_data)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────────────────

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Invoque un outil enregistré auprès de l'agent.

        Args:
            tool_name: Nom de l'outil (clé dans self.tools)
            **kwargs:  Arguments passés à l'outil

        Returns:
            Résultat de l'outil
        """
        if tool_name not in self.tools:
            raise ValueError(
                f"[{self.AGENT_NAME}] Outil '{tool_name}' non disponible. "
                f"Outils enregistrés : {list(self.tools.keys())}"
            )
        tool = self.tools[tool_name]
        self.logger.action("USE_TOOL", {'tool': tool_name, 'args': list(kwargs.keys())})
        return tool.execute(**kwargs)

    def get_status(self) -> dict:
        """Retourne l'état complet de l'agent."""
        return {
            'name': self.AGENT_NAME,
            'role': self.AGENT_ROLE,
            'version': self.AGENT_VERSION,
            'status': self._status.value,
            'metrics': self.metrics.get_summary(),
            'tools': list(self.tools.keys()),
            'has_memory': self.memory is not None,
        }

    def __repr__(self) -> str:
        return (f"<{self.AGENT_NAME} v{self.AGENT_VERSION} "
                f"status={self._status.value} "
                f"processed={self.metrics._total_count}>")
