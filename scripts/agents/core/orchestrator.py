"""
orchestrator.py — Orchestrateur avec Routing Intelligent et Workflows Adaptatifs
Stage : Système Agentic AI de Détection de Fraude Bancaire

Étape 6 du blueprint "How to Build an AI Agent" — Orchestration :
┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    Transaction ──► [ROUTER] ──► Sélection du Workflow               │
│                        │                                             │
│       ┌────────────────┼────────────────┬────────────────┐          │
│       ▼                ▼                ▼                ▼          │
│  Fast-Track       Standard        Escalation       Monitoring       │
│  (Ingestion       (Ingestion ──►  (Séquentiel      (Agent           │
│   uniquement)      Analysis ──►    + Feedback       Monitoring      │
│                    Decision ──►    + Escalade       périodique)     │
│                    Explanation)    Humaine)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from scripts.agents.core.agent_base import (
    BaseAgent, AgentResponse, AgentStatus, AgentLogger, Priority
)
from scripts.agents.core.memory import MemorySystem, Episode


# ═══════════════════════════════════════════════════════════════════════════════
# PipelineResult — Résultat consolidé de l'exécution du pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """
    Résultat consolidé du traitement d'une transaction à travers le pipeline.
    """
    transaction_id: str
    workflow_used: str
    total_latency_ms: float
    final_decision: str
    probability: float
    risk_level: str
    agent_responses: Dict[str, AgentResponse]
    explanation: Optional[str] = None
    alerts_emitted: List[dict] = field(default_factory=list)
    success: bool = True
    trace_id: str = ""

    def to_dict(self) -> dict:
        return {
            'transaction_id': self.transaction_id,
            'workflow_used': self.workflow_used,
            'total_latency_ms': round(self.total_latency_ms, 2),
            'final_decision': self.final_decision,
            'probability': round(self.probability, 4),
            'risk_level': self.risk_level,
            'explanation': self.explanation,
            'alerts_emitted': self.alerts_emitted,
            'agent_responses': {
                name: resp.to_dict()
                for name, resp in self.agent_responses.items()
            },
            'success': self.success,
            'trace_id': self.trace_id,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OrchestratorAgent — Agent Chef d'Orchestre
# ═══════════════════════════════════════════════════════════════════════════════

class OrchestratorAgent:
    """
    Chef d'orchestre du système multi-agents (Étape 6 du blueprint).

    Rôle :
    1. Recevoir les transactions entrantes.
    2. Déterminer le workflow optimal via le Router (Fast-track vs Standard vs Escalation).
    3. Exécuter la séquence d'agents en transmettant le contexte et la mémoire.
    4. Gérer les erreurs, fallbacks et timeouts à chaque étape.
    5. Consolider les résultats et émettre le rapport/décision final(e).
    """

    WORKFLOWS = {
        'fast_track': ['surveillance'],
        'standard': ['surveillance', 'analysis', 'decision', 'explanation'],
        'escalation': ['surveillance', 'analysis', 'decision', 'explanation', 'feedback'],
        'deep_analysis': ['surveillance', 'analysis', 'decision', 'explanation', 'monitoring'],
    }

    def __init__(self, agents: Dict[str, BaseAgent], memory: MemorySystem = None,
                 config: dict = None):
        """
        Args:
            agents: dict {agent_name: BaseAgent_instance}
            memory: MemorySystem partagé
            config: Configuration d'orchestration (ex: seuils fast-track)
        """
        self.agents = agents
        self.memory = memory or MemorySystem()
        self.config = config or {}
        self.logger = AgentLogger("Orchestrator")
        self._total_pipelines = 0
        self._fast_track_count = 0
        self._standard_count = 0
        self._escalation_count = 0

    def route(self, transaction: dict) -> str:
        """
        Routing intelligent de la transaction (Détermine le workflow).

        Logique de routing :
        - Règle 1 : Si la transaction a un drapeau 'force_escalation' → 'escalation'
        - Règle 2 : Si pré-filtrage activé et pas de risque évident → 'fast_track' ou 'standard'
        """
        if transaction.get('force_escalation', False):
            return 'escalation'
        
        # Par défaut, on lance le pipeline standard pour une évaluation complète
        return 'standard'

    def process_transaction(self, transaction: dict, transaction_id: str = None,
                            workflow_override: str = None) -> PipelineResult:
        """
        Traite une transaction de bout en bout à travers le pipeline d'agents.

        Args:
            transaction: dict des caractéristiques de la transaction
            transaction_id: ID unique optionnel
            workflow_override: forcer un workflow spécifique

        Returns:
            PipelineResult consolidé
        """
        start_time = time.time()
        self._total_pipelines += 1
        tx_id = transaction_id or f"TXN-{self._total_pipelines:05d}"
        trace_id = str(uuid.uuid4())[:8]

        # ── 1. Initialisation de la mémoire de travail ───────────────────────
        self.memory.reset_working()
        self.memory.working.set('transaction_id', tx_id)
        self.memory.working.set('raw_transaction', transaction)

        # ── 2. Routing (Choix du workflow) ──────────────────────────────────
        workflow_name = workflow_override or self.route(transaction)
        pipeline_agents = self.WORKFLOWS.get(workflow_name, self.WORKFLOWS['standard'])

        self.logger.info(
            f"Début du traitement transaction {tx_id}",
            workflow=workflow_name,
            agents=pipeline_agents,
            trace_id=trace_id
        )

        # ── Compteurs de workflow ────────────────────────────────────────────
        if workflow_name == 'standard':
            self._standard_count += 1
        elif workflow_name == 'escalation':
            self._escalation_count += 1

        agent_responses: Dict[str, AgentResponse] = {}
        current_data = {'transaction': transaction, 'transaction_id': tx_id}

        # ── 3. Exécution séquentielle des agents ─────────────────────────────
        for agent_key in pipeline_agents:
            if agent_key not in self.agents:
                self.logger.warning(f"Agent '{agent_key}' introuvable dans le registre, ignoré.")
                continue

            agent = self.agents[agent_key]
            
            # Injection de l'état du pipeline précédent
            context = {
                'trace_id': trace_id,
                'transaction_id': tx_id,
                'previous_responses': agent_responses,
                'working_memory': self.memory.working.get_all(),
            }

            response = agent.run(current_data, context=context)
            agent_responses[agent_key] = response

            if response.is_error:
                self.logger.error(f"Échec de l'agent {agent_key}: {response.errors}")
                # Stratégie de fallback : Continuer ou basculer en mode dégradé

            # Mise à jour des données transmises à l'agent suivant
            if response.is_success and response.data:
                current_data.update(response.data)

            # Optimisation Fast-Track : Interrompre si l'agent 1 autorise directement
            if agent_key == 'surveillance' and response.data.get('fast_decision') == 'ALLOW':
                if workflow_name != 'escalation' and not self.config.get('force_full_pipeline', False):
                    self.logger.info(f"Transaction {tx_id} Fast-Track (ALLOW). Fin du pipeline.")
                    workflow_name = 'fast_track'
                    self._fast_track_count += 1
                    break

        total_latency = (time.time() - start_time) * 1000

        # ── 4. Consolidation des résultats ──────────────────────────────────
        surv_data = agent_responses.get('surveillance', AgentResponse('surveillance', AgentStatus.IDLE)).data
        anal_data = agent_responses.get('analysis', AgentResponse('analysis', AgentStatus.IDLE)).data
        dec_data = agent_responses.get('decision', AgentResponse('decision', AgentStatus.IDLE)).data
        expl_data = agent_responses.get('explanation', AgentResponse('explanation', AgentStatus.IDLE)).data

        final_decision = dec_data.get('decision', surv_data.get('fast_decision', 'ALLOW'))
        probability = anal_data.get('probability', 0.0)
        risk_level = anal_data.get('risk_level', 'FAIBLE' if final_decision == 'ALLOW' else 'MOYEN')
        explanation_text = expl_data.get('rapport', expl_data.get('text', None))

        result = PipelineResult(
            transaction_id=tx_id,
            workflow_used=workflow_name,
            total_latency_ms=total_latency,
            final_decision=final_decision,
            probability=probability,
            risk_level=risk_level,
            agent_responses=agent_responses,
            explanation=explanation_text,
            alerts_emitted=dec_data.get('alerts', []),
            success=all(resp.is_success for resp in agent_responses.values()),
            trace_id=trace_id
        )

        # ── 5. Sauvegarde en mémoires (Épisodique & Persistante) ──────────────
        episode = Episode(
            transaction_id=tx_id,
            transaction_data=transaction,
            pipeline_result=result.to_dict(),
            decision=final_decision,
            probability=probability
        )
        self.memory.episodic.add(episode)
        
        self.memory.long_term.save_decision(
            transaction_id=tx_id,
            decision=final_decision,
            probability=probability,
            risk_level=risk_level,
            agent_latencies={name: resp.latency_ms for name, resp in agent_responses.items()},
            shap_top3=anal_data.get('shap_top5', [])[:3]
        )

        return result

    def get_stats(self) -> dict:
        return {
            'total_pipelines': self._total_pipelines,
            'fast_track_count': self._fast_track_count,
            'standard_count': self._standard_count,
            'escalation_count': self._escalation_count,
            'agents_registered': list(self.agents.keys()),
        }
