"""
pipeline_demo.py — Démonstrateur du Pipeline Agentic AI Complet
Stage : Système Agentic AI de Détection de Fraude Bancaire
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

# Ajouter la racine au path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from scripts.agents.core.memory import MemorySystem
from scripts.agents.core.orchestrator import OrchestratorAgent
from scripts.agents.surveillance_agent import SurveillanceAgent
from scripts.agents.analysis_agent import AnalysisAgent
from scripts.agents.decision_agent import DecisionAgent
from scripts.agents.explanation_agent import ExplanationAgent
from scripts.agents.feedback_agent import FeedbackAgent
from scripts.agents.monitoring_agent import MonitoringAgent


def run_demo():
    print("=" * 80)
    print("  DÉMONSTRATION DU SYSTÈME AGENTIC AI DE DÉTECTION DE FRAUDE")
    print("=" * 80)

    # 1. Préparation d'un faux modèle et explainer pour le test
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
    X_dummy = np.random.randn(100, len(feature_names))
    y_dummy = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
    model.fit(X_dummy, y_dummy)
    explainer = shap.TreeExplainer(model)

    # 2. Initialisation du MemorySystem
    memory = MemorySystem(max_episodes=100)

    # 3. Instanciation des agents
    surveillance = SurveillanceAgent(dataset_type='ulb', memory=memory)
    analysis = AnalysisAgent(model=model, shap_explainer=explainer, feature_names=feature_names, memory=memory)
    decision = DecisionAgent(memory=memory)
    explanation = ExplanationAgent(memory=memory)
    feedback = FeedbackAgent(memory=memory)
    monitoring = MonitoringAgent(memory=memory)

    agents = {
        'surveillance': surveillance,
        'analysis': analysis,
        'decision': decision,
        'explanation': explanation,
        'feedback': feedback,
        'monitoring': monitoring,
    }

    # 4. Instanciation de l'Orchestrateur
    orchestrator = OrchestratorAgent(agents=agents, memory=memory)

    # 5. Simulation de transactions
    print("\n[1] Traitement d'une transaction suspecte...")
    suspicious_tx = {f'V{i}': 0.1 for i in range(1, 29)}
    suspicious_tx['V14'] = -4.5  # Déclenche l'alerte V14
    suspicious_tx['Amount'] = 3.5

    result = orchestrator.process_transaction(suspicious_tx, transaction_id="TXN-99991")
    print(f"  • Décision finale : {result.final_decision}")
    print(f"  • Probabilité de fraude : {result.probability*100:.2f}%")
    print(f"  • Niveau de risque : {result.risk_level}")
    print(f"  • Latence totale : {result.total_latency_ms:.2f} ms")
    print(f"  • Workflow : {result.workflow_used}")
    print("\n--- RAPPORT GÉNÉRÉ ---")
    print(result.explanation)

    print("\n[2] Simulation de Feedback Humain...")
    fb_res = feedback.run({'transaction_id': "TXN-99991", 'feedback': 'CONFIRMED_FRAUD', 'was_fraud': True})
    print(f"  • Feedback enregistré : {fb_res.data}")

    print("\n[3] Rapport de Monitoring...")
    mon_res = monitoring.run({'hours': 24})
    print(f"  • Monitoring : {mon_res.data}")

    print("\n" + "=" * 80)
    print("  DÉMO TERMINÉE AVEC SUCCÈS")
    print("=" * 80)


if __name__ == '__main__':
    run_demo()
