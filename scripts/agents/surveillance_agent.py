"""
surveillance_agent.py — Agent 1 : Surveillance et filtrage rapide (Ingestion & Rule Filtering)
Stage : Système Agentic AI de Détection de Fraude Bancaire

Transformé pour hériter de BaseAgent et s'intégrer au framework core.
"""

from typing import Dict, List, Any
from scripts.agents.core.agent_base import BaseAgent, AgentResponse


class SurveillanceAgent(BaseAgent):
    """
    Agent 1 — Surveillance et filtrage par règles métier.
    Rôle : Ingestion, validation et pré-filtrage rapide des transactions.
    """

    AGENT_NAME = "SurveillanceAgent"
    AGENT_ROLE = "Ingestion & Filtrage Rapide"
    AGENT_PURPOSE = "Filtrer rapidement les transactions manifestement légitimes (Fast-Track)"

    # Seuils par défaut
    ULB_THRESHOLDS = {
        'Amount_high': 2.0,
        'V14_low': -2.5,
        'V12_low': -2.0,
        'V10_low': -2.0,
        'V4_high': 2.5,
        'V3_extreme': 3.5,
    }

    SYNTHETIC_THRESHOLDS = {
        'amt_high': 1500,
        'hour_unusual_min': 0,
        'hour_unusual_max': 4,
        'city_pop_low': 500,
    }

    HIGH_RISK_CATEGORIES = {
        'travel', 'online_retail', 'shopping_net', 'misc_net',
        'home', 'personal_care'
    }

    def __init__(self, dataset_type: str = 'ulb', config: dict = None,
                 memory=None, tools: dict = None):
        super().__init__(config=config, memory=memory, tools=tools)
        self.dataset_type = dataset_type

    def process(self, input_data: dict, context: dict = None) -> dict:
        transaction = input_data.get('transaction', input_data)

        if self.dataset_type == 'ulb':
            reasons = self._screen_ulb(transaction)
        else:
            reasons = self._screen_synthetic(transaction)

        suspicious = len(reasons) > 0

        if suspicious:
            return {
                'suspicious': True,
                'reasons': reasons,
                'fast_decision': None,
                'risk_score': len(reasons),
            }
        else:
            return {
                'suspicious': False,
                'reasons': [],
                'fast_decision': 'ALLOW',
                'risk_score': 0,
            }

    def _screen_ulb(self, tx: dict) -> list:
        reasons = []
        t = self.ULB_THRESHOLDS
        amount = tx.get('Amount', tx.get('amount', 0))
        if amount > t['Amount_high']:
            reasons.append(f"Montant anormalement élevé (Amount={amount:.2f} σ)")

        v14 = tx.get('V14', 0)
        if v14 < t['V14_low']:
            reasons.append(f"V14 anormal: {v14:.3f} < {t['V14_low']}")

        v12 = tx.get('V12', 0)
        if v12 < t['V12_low']:
            reasons.append(f"V12 anormal: {v12:.3f} < {t['V12_low']}")

        v10 = tx.get('V10', 0)
        if v10 < t['V10_low']:
            reasons.append(f"V10 anormal: {v10:.3f} < {t['V10_low']}")

        v4 = tx.get('V4', 0)
        if v4 > t['V4_high']:
            reasons.append(f"V4 élevé: {v4:.3f} > {t['V4_high']}")

        v3 = tx.get('V3', 0)
        if abs(v3) > t['V3_extreme']:
            reasons.append(f"V3 extrême: {v3:.3f}")

        return reasons

    def _screen_synthetic(self, tx: dict) -> list:
        reasons = []
        t = self.SYNTHETIC_THRESHOLDS
        amt = tx.get('amt', tx.get('amount', 0))
        if amt > t['amt_high']:
            reasons.append(f"Montant élevé: ${amt:.2f} > ${t['amt_high']}")

        hour = tx.get('hour', None)
        if hour is not None and t['hour_unusual_min'] <= hour <= t['hour_unusual_max']:
            reasons.append(f"Transaction nocturne: {hour}h")

        category = tx.get('category', '').lower()
        if any(cat in category for cat in self.HIGH_RISK_CATEGORIES):
            reasons.append(f"Catégorie à risque élevé: '{category}'")

        return reasons
