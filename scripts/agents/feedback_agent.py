"""
feedback_agent.py — Agent 5 : Boucle de Rétroaction et Human-in-the-Loop
Stage : Système Agentic AI de Détection de Fraude Bancaire

Gestion des corrections des analystes humains et réapprentissage.
"""

from typing import Dict, Any
from scripts.agents.core.agent_base import BaseAgent


class FeedbackAgent(BaseAgent):
    """
    Agent 5 — Feedback Human-in-the-Loop.
    Rôle : Capturer la validation ou l'infirmation des analystes et mettre à jour la mémoire persistante.
    """

    AGENT_NAME = "FeedbackAgent"
    AGENT_ROLE = "Human-in-the-Loop & Feedback"
    AGENT_PURPOSE = "Enregistrer la validation humaine et ajuster la mémoire long terme"

    def process(self, input_data: dict, context: dict = None) -> dict:
        tx_id = input_data.get('transaction_id')
        feedback = input_data.get('feedback', '')  # 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'LEGITIMATE'
        was_fraud = input_data.get('was_fraud', True if feedback == 'CONFIRMED_FRAUD' else False)

        if self.memory and tx_id:
            self.memory.long_term.save_feedback(
                transaction_id=tx_id,
                feedback=feedback,
                was_fraud=was_fraud
            )

        self.logger.info(f"Feedback enregistré pour {tx_id}: {feedback} (Was Fraud: {was_fraud})")

        return {
            'transaction_id': tx_id,
            'feedback_saved': True,
            'feedback_type': feedback,
            'was_fraud': was_fraud
        }
