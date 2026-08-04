"""
Core Agentic AI Framework — Module Init
Stage : Détection de Fraude Bancaire avec Agentic AI, LLMs et Federated Learning

Ce module fournit le framework de base pour le système multi-agents :
- BaseAgent : Classe abstraite pour tous les agents
- MemorySystem : Mémoire multi-couches (working, episodic, semantic, long-term)
- Tools : Outils réutilisables (ML, SHAP, LLM, DB, Alert)
- OrchestratorAgent : Chef d'orchestre du pipeline
"""

from scripts.agents.core.agent_base import BaseAgent, AgentResponse, AgentMetrics, AgentLogger
from scripts.agents.core.memory import (
    MemorySystem, WorkingMemory, EpisodicMemory,
    SemanticMemory, LongTermMemory
)
from scripts.agents.core.tools import (
    BaseTool, MLModelTool, SHAPExplainabilityTool,
    LLMGatewayTool, AlertTool, DatabaseTool
)
from scripts.agents.core.orchestrator import OrchestratorAgent, PipelineResult

__all__ = [
    'BaseAgent', 'AgentResponse', 'AgentMetrics', 'AgentLogger',
    'MemorySystem', 'WorkingMemory', 'EpisodicMemory',
    'SemanticMemory', 'LongTermMemory',
    'BaseTool', 'MLModelTool', 'SHAPExplainabilityTool',
    'LLMGatewayTool', 'AlertTool', 'DatabaseTool',
    'OrchestratorAgent', 'PipelineResult',
]
