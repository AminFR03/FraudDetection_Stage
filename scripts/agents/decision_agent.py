"""
decision_agent.py — Agent 3 : Décision métier & Règles de gouvernance
Stage : Système Agentic AI de Détection de Fraude Bancaire

Transformé pour utiliser BaseAgent et intégrer AlertTool.
"""

from typing import Dict, Any
from scripts.agents.core.agent_base import BaseAgent
from scripts.agents.core.tools import AlertTool


class DecisionAgent(BaseAgent):
    """
    Agent 3 — Prise de décision automatique et déclenchement d'alertes.
    """

    AGENT_NAME = "DecisionAgent"
    AGENT_ROLE = "Gouvernance & Prise de Décision"
    AGENT_PURPOSE = "Transformer la probabilité en décision opérationnelle et alerter les équipes"

    DEFAULT_THRESHOLDS = {
        'BLOCK': 0.85,
        'REVIEW': 0.60,
        'ALERT': 0.35,
        'ALLOW': 0.00,
    }

    DECISION_METADATA = {
        'BLOCK': {
            'severity': 'CRITIQUE',
            'action': 'Transaction refusée automatiquement. Dossier fraude ouvert.',
            'sla_min': 0,
            'notify': ['equipe_fraude', 'client', 'conformite'],
            'color': '#e53935',
            'icon': '🚫',
        },
        'REVIEW': {
            'severity': 'ÉLEVÉ',
            'action': 'Transaction suspendue. Analyste humain assigné.',
            'sla_min': 15,
            'notify': ['analyste_fraude', 'client'],
            'color': '#f57c00',
            'icon': '⚠️',
        },
        'ALERT': {
            'severity': 'MOYEN',
            'action': 'Transaction autorisée avec surveillance renforcée.',
            'sla_min': 60,
            'notify': ['equipe_surveillance'],
            'color': '#fbc02d',
            'icon': '⚡',
        },
        'ALLOW': {
            'severity': 'FAIBLE',
            'action': 'Transaction approuvée normally.',
            'sla_min': None,
            'notify': [],
            'color': '#43a047',
            'icon': '✅',
        },
    }

    def __init__(self, thresholds: dict = None, config: dict = None,
                 memory=None, tools: dict = None):
        tools = tools or {}
        if 'alert_tool' not in tools:
            tools['alert_tool'] = AlertTool()

        super().__init__(config=config, memory=memory, tools=tools)
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()

    def process(self, input_data: dict, context: dict = None) -> dict:
        probability = input_data.get('probability', 0.0)
        tx_id = input_data.get('transaction_id', context.get('transaction_id', 'UNKNOWN') if context else 'UNKNOWN')

        decision = 'ALLOW'
        for action in ['BLOCK', 'REVIEW', 'ALERT', 'ALLOW']:
            if probability >= self.thresholds[action]:
                decision = action
                break

        meta = self.DECISION_METADATA[decision]

        # Déclenchement d'alerte si BLOCK ou REVIEW
        alert_info = {}
        if decision in ('BLOCK', 'REVIEW', 'ALERT') and 'alert_tool' in self.tools:
            alert_info = self.use_tool(
                'alert_tool',
                transaction_id=tx_id,
                severity=meta['severity'],
                decision=decision,
                message=meta['action'],
                notify=meta['notify']
            )

        return {
            'decision': decision,
            'probability': round(probability, 4),
            'threshold_used': self.thresholds[decision],
            'severity': meta['severity'],
            'action': meta['action'],
            'notify': meta['notify'],
            'sla_min': meta['sla_min'],
            'color': meta['color'],
            'icon': meta['icon'],
            'alert_details': alert_info,
        }
