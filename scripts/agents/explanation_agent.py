"""
explanation_agent.py — Agent 4 : Génération d'explications LLM (Qwen 2.5 via Ollama)
Stage : Système Agentic AI de Détection de Fraude Bancaire

Transformé pour utiliser BaseAgent et LLMGatewayTool.
"""

from typing import Dict, Any
from scripts.agents.core.agent_base import BaseAgent
from scripts.agents.core.tools import LLMGatewayTool


class ExplanationAgent(BaseAgent):
    """
    Agent 4 — Explication en langage naturel via LLM (Qwen 2.5 via Ollama).
    """

    AGENT_NAME = "ExplanationAgent"
    AGENT_ROLE = "Génération d'Explications & NLG"
    AGENT_PURPOSE = "Traduire les données SHAP et la décision ML en rapport compréhensible"

    def __init__(self, temperature: float = 0.3, max_tokens: int = 500,
                 config: dict = None, memory=None, tools: dict = None):
        
        tools = tools or {}
        if 'llm_tool' not in tools:
            tools['llm_tool'] = LLMGatewayTool(
                temperature=temperature, max_tokens=max_tokens
            )

        super().__init__(config=config, memory=memory, tools=tools)

    def process(self, input_data: dict, context: dict = None) -> dict:
        transaction = input_data.get('transaction', {})
        tx_id = input_data.get('transaction_id', 'TXN-0000')

        # Extraire de manière sécurisée les contextes précédents (peuvent être dicts ou AgentResponse)
        surv_data = input_data if isinstance(input_data, dict) else {}
        anal_data = input_data if isinstance(input_data, dict) else {}
        dec_data = input_data if isinstance(input_data, dict) else {}

        if context and 'previous_responses' in context:
            prev = context['previous_responses']
            if 'surveillance' in prev and hasattr(prev['surveillance'], 'data'):
                surv_data = prev['surveillance'].data
            if 'analysis' in prev and hasattr(prev['analysis'], 'data'):
                anal_data = prev['analysis'].data
            if 'decision' in prev and hasattr(prev['decision'], 'data'):
                dec_data = prev['decision'].data

        # Vérification si le LLM est disponible
        llm_tool = self.tools.get('llm_tool')
        if llm_tool and llm_tool.is_online:
            prompt = self._build_prompt(tx_id, transaction, surv_data, anal_data, dec_data)
            res = self.use_tool('llm_tool', prompt=prompt)
            rapport = res.get('text', '')
            mode = 'online'
        else:
            rapport = self._explain_offline(tx_id, transaction, surv_data, anal_data, dec_data)
            mode = 'offline'

        return {
            'rapport': rapport,
            'mode': mode,
            'transaction_id': tx_id
        }

    def _build_prompt(self, tx_id, transaction, surv, analysis, decision) -> str:
        surv_reasons = surv.get('reasons', [])
        surv_text = '\n'.join(f"  • {r}" for r in surv_reasons) if surv_reasons else "  • Aucune anomalie de règle détectée"

        shap_items = analysis.get('shap_top5', analysis.get('shap_top_n', []))
        shap_text = ""
        for item in shap_items:
            sign = "+" if item.get('shap', 0) > 0 else ""
            shap_text += f"  • {item.get('feature')}: {item.get('value')} (SHAP: {sign}{item.get('shap')} -> {item.get('direction')})\n"

        prompt = f"""
RAPPORT D'ANALYSE FRAUDE - Transaction {tx_id}

[AGENT 1 — Surveillance]
Alertes règles :
{surv_text}

[AGENT 2 — Analyse ML]
Probabilité de fraude : {analysis.get('probability', 0)*100:.1f}%
Niveau de risque : {analysis.get('risk_level', 'Inconnu')}
Top features SHAP :
{shap_text}

[AGENT 3 — Décision]
Décision : {decision.get('decision', 'ALLOW')}
Action : {decision.get('action', 'Aucune')}

Rédige un rapport de 4 à 6 phrases en français professionnel expliquant cette décision.
"""
        return prompt

    def _explain_offline(self, tx_id, transaction, surv, analysis, decision) -> str:
        prob = analysis.get('probability', 0.0) * 100
        dec = decision.get('decision', 'ALLOW')
        risk = analysis.get('risk_level', 'FAIBLE')

        return (
            f"═══ RAPPORT DE DÉTECTION HORS-LIGNE — {tx_id} ═══\n"
            f"La transaction {tx_id} a reçu la décision [{dec}] avec une probabilité "
            f"de fraude de {prob:.1f}% (Niveau de risque: {risk}).\n"
            f"Action requise: {decision.get('action', 'Aucune')}.\n"
            f"(Rapport généré via template hors-ligne sans LLM API)."
        )
