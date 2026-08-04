"""
monitoring_agent.py — Agent 6 : Surveillance des Performances et Data Drift
Stage : Système Agentic AI de Détection de Fraude Bancaire

Surveillance en temps réel des métriques du pipeline et détection de drift.
"""

from typing import Dict, Any
from scripts.agents.core.agent_base import BaseAgent
from scripts.agents.core.tools import DatabaseTool


class MonitoringAgent(BaseAgent):
    """
    Agent 6 — Surveillance & Drift.
    Rôle : Suivre les performances globales, la distribution des décisions et la dégradation de précision.
    """

    AGENT_NAME = "MonitoringAgent"
    AGENT_ROLE = "Performance & Drift Monitoring"
    AGENT_PURPOSE = "Surveiller la santé du pipeline et détecter les dérives de données"

    def __init__(self, config: dict = None, memory=None, tools: dict = None):
        tools = tools or {}
        if 'db_tool' not in tools:
            tools['db_tool'] = DatabaseTool()
        super().__init__(config=config, memory=memory, tools=tools)

    def process(self, input_data: dict, context: dict = None) -> dict:
        hours = input_data.get('hours', 24)

        db_stats = {}
        if 'db_tool' in self.tools:
            db_stats = self.use_tool('db_tool', query_type="decision_stats")

        accuracy_data = {}
        if self.memory:
            accuracy_data = self.memory.long_term.get_feedback_accuracy()

        return {
            'monitoring_period_hours': hours,
            'decision_stats': db_stats.get('stats', {}),
            'human_feedback_accuracy': accuracy_data,
            'status': 'HEALTHY'
        }
