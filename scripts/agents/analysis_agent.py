"""
analysis_agent.py — Agent 2 : Analyse ML & Explicabilité SHAP
Stage : Système Agentic AI de Détection de Fraude Bancaire

Transformé pour intégrer BaseAgent et utiliser MLModelTool / SHAPExplainabilityTool.
"""

import numpy as np
from typing import Dict, Any
from scripts.agents.core.agent_base import BaseAgent
from scripts.agents.core.tools import MLModelTool, SHAPExplainabilityTool


class AnalysisAgent(BaseAgent):
    """
    Agent 2 — Analyse ML + SHAP Explainability.
    """

    AGENT_NAME = "AnalysisAgent"
    AGENT_ROLE = "Scoring ML & Explicabilité"
    AGENT_PURPOSE = "Calculer la probabilité de fraude et extraire les explications SHAP"

    def __init__(self, model=None, shap_explainer=None, scaler=None,
                 feature_names: list = None, top_n: int = 5,
                 config: dict = None, memory=None, tools: dict = None):
        
        tools = tools or {}
        if model and 'ml_tool' not in tools:
            tools['ml_tool'] = MLModelTool(model=model, scaler=scaler, feature_names=feature_names)
        if shap_explainer and 'shap_tool' not in tools:
            tools['shap_tool'] = SHAPExplainabilityTool(explainer=shap_explainer, feature_names=feature_names, top_n=top_n)

        super().__init__(config=config, memory=memory, tools=tools)
        self.feature_names = feature_names or []
        self.top_n = top_n

    def process(self, input_data: dict, context: dict = None) -> dict:
        transaction = input_data.get('transaction', input_data)

        # 1. Prédiction ML
        ml_res = {}
        if 'ml_tool' in self.tools:
            ml_res = self.use_tool('ml_tool', transaction=transaction)
        else:
            # Fallback direct si l'outil n'est pas instancié sous forme de classe Tool
            ml_res = {'probability': 0.5, 'risk_level': 'MOYEN'}

        probability = ml_res.get('probability', 0.0)
        risk_level = ml_res.get('risk_level', 'MOYEN')

        # 2. Explicabilité SHAP
        feat_values = np.array([[transaction.get(f, 0.0) for f in self.feature_names]])
        shap_res = {}
        if 'shap_tool' in self.tools:
            shap_res = self.use_tool('shap_tool', features_array=feat_values, top_n=self.top_n)

        top_features = shap_res.get('shap_top_n', [])
        all_shap = shap_res.get('all_shap', {})
        base_val = shap_res.get('base_value', 0.0)

        return {
            'probability': probability,
            'risk_level': risk_level,
            'shap_top5': top_features,
            'shap_top_n': top_features,
            'all_shap': all_shap,
            'base_value': base_val,
        }
