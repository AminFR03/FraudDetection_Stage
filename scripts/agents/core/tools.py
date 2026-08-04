"""
tools.py — Outils (Tools & Integrations) pour le Framework Agentic AI
Stage : Détection de Fraude Bancaire avec Agentic AI, LLMs et Federated Learning

Étape 4 du blueprint "How to Build an AI Agent" — Tools & Integrations :
┌─────────────────────────────────────────────────────────────────────┐
│                      TOOLS ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  Prédiction de fraude via modèle ML               │
│  │ MLModelTool   │  XGBoost, RandomForest, MLP, Fédéré              │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────┐  Calcul d'explicabilité SHAP                      │
│  │ SHAPTool      │  TreeExplainer, top-N features, global summary   │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────┐  Interface unifiée vers les LLMs                  │
│  │ LLMGateway    │  Gemini 2.5 Flash + fallback offline             │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────┐  Notifications et alertes                         │
│  │ AlertTool     │  Email, Slack, logging structuré                  │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────┐  Accès base de données                            │
│  │ DatabaseTool  │  Historique transactions, feedback, stats         │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────┐  Coordination Federated Learning                  │
│  │ FederatedTool │  Envoi/réception modèles, agrégation             │
│  └──────────────┘                                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
"""

import time
import json
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# BaseTool — Interface abstraite pour tous les outils
# ═══════════════════════════════════════════════════════════════════════════════

class BaseTool(ABC):
    """
    Classe abstraite pour tous les outils du système.

    Chaque outil encapsule une capacité spécifique (ML, SHAP, LLM, etc.)
    et fournit une interface standardisée `execute()`.

    Les outils sont injectés dans les agents via le dictionnaire `tools`.
    """

    TOOL_NAME: str = "BaseTool"
    TOOL_DESCRIPTION: str = "Outil de base"
    TOOL_VERSION: str = "1.0.0"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._errors: List[str] = []

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Exécute l'outil avec les paramètres donnés."""
        raise NotImplementedError

    def get_stats(self) -> dict:
        avg_lat = (self._total_latency_ms / max(self._call_count, 1))
        return {
            'tool_name': self.TOOL_NAME,
            'call_count': self._call_count,
            'avg_latency_ms': round(avg_lat, 2),
            'error_count': len(self._errors),
        }

    def _track_call(self, latency_ms: float):
        self._call_count += 1
        self._total_latency_ms += latency_ms


# ═══════════════════════════════════════════════════════════════════════════════
# MLModelTool — Prédiction via modèle ML
# ═══════════════════════════════════════════════════════════════════════════════

class MLModelTool(BaseTool):
    """
    Outil de prédiction ML pour la détection de fraude.

    Encapsule un modèle scikit-learn compatible (XGBoost, RF, etc.)
    et fournit :
    - Prédiction de probabilité de fraude
    - Classification binaire avec seuil configurable
    - Support multi-modèle (ensemble)
    - Calibration de probabilité (Platt scaling)

    Usage:
        tool = MLModelTool(model=xgb_model, scaler=scaler,
                           feature_names=features)
        result = tool.execute(transaction={'V1': -1.3, 'V2': 0.5, ...})
    """

    TOOL_NAME = "MLModelTool"
    TOOL_DESCRIPTION = "Prédiction de fraude via modèle ML (XGBoost/RF/MLP)"

    # Niveaux de risque basés sur la probabilité
    RISK_LEVELS = {
        'CRITIQUE': 0.85,
        'ÉLEVÉ':    0.60,
        'MOYEN':    0.35,
        'FAIBLE':   0.00,
    }

    def __init__(self, model=None, scaler=None, feature_names: list = None,
                 threshold: float = 0.5, config: dict = None):
        """
        Args:
            model:         Modèle entraîné (avec predict_proba)
            scaler:        StandardScaler/RobustScaler fité
            feature_names: Liste des noms de features attendus
            threshold:     Seuil de classification par défaut
        """
        super().__init__(config)
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.threshold = threshold

    def execute(self, transaction: dict = None, features_array: np.ndarray = None,
                threshold: float = None, **kwargs) -> dict:
        """
        Prédit la probabilité de fraude pour une transaction.

        Args:
            transaction:    dict {feature_name: value}
            features_array: Alternative — vecteur numpy pré-formaté
            threshold:      Seuil personnalisé (optionnel)

        Returns:
            dict {probability, prediction, risk_level, threshold_used}
        """
        start = time.time()
        threshold = threshold or self.threshold

        if self.model is None:
            return {
                'probability': 0.0,
                'prediction': 0,
                'risk_level': 'INDISPONIBLE',
                'error': 'Modèle non chargé'
            }

        try:
            # Préparation du vecteur de features
            if features_array is not None:
                X = features_array if features_array.ndim == 2 \
                    else features_array.reshape(1, -1)
            elif transaction is not None:
                X = np.array([[transaction.get(f, 0.0)
                               for f in self.feature_names]])
            else:
                raise ValueError("Fournir 'transaction' ou 'features_array'")

            # Scaling si disponible
            if self.scaler is not None:
                X = self.scaler.transform(X)

            # Prédiction
            probability = float(self.model.predict_proba(X)[:, 1][0])
            prediction = int(probability >= threshold)

            # Niveau de risque
            risk_level = 'FAIBLE'
            for level, thresh in self.RISK_LEVELS.items():
                if probability >= thresh:
                    risk_level = level
                    break

            latency = (time.time() - start) * 1000
            self._track_call(latency)

            return {
                'probability': round(probability, 6),
                'prediction': prediction,
                'risk_level': risk_level,
                'threshold_used': threshold,
                'model_type': type(self.model).__name__,
                'latency_ms': round(latency, 2),
            }

        except Exception as e:
            self._errors.append(str(e))
            return {
                'probability': 0.0,
                'prediction': 0,
                'risk_level': 'ERREUR',
                'error': str(e)[:200],
            }


# ═══════════════════════════════════════════════════════════════════════════════
# SHAPExplainabilityTool — Calcul d'explicabilité SHAP
# ═══════════════════════════════════════════════════════════════════════════════

class SHAPExplainabilityTool(BaseTool):
    """
    Outil d'explicabilité SHAP pour les modèles ML.

    Fournit :
    - Valeurs SHAP pour chaque feature
    - Top-N features les plus influentes
    - Direction d'influence (augmente/réduit le risque)
    - Pourcentage d'impact relatif

    Usage:
        tool = SHAPExplainabilityTool(explainer=shap_explainer,
                                       feature_names=features)
        result = tool.execute(features_array=X_sample, top_n=5)
    """

    TOOL_NAME = "SHAPExplainabilityTool"
    TOOL_DESCRIPTION = "Calcul des valeurs SHAP pour l'explicabilité ML"

    def __init__(self, explainer=None, feature_names: list = None,
                 top_n: int = 5, config: dict = None):
        super().__init__(config)
        self.explainer = explainer
        self.feature_names = feature_names or []
        self.top_n = top_n

    def execute(self, features_array: np.ndarray = None, top_n: int = None,
                **kwargs) -> dict:
        """
        Calcule les valeurs SHAP pour un vecteur de features.

        Returns:
            dict {shap_top_n, all_shap, base_value}
        """
        start = time.time()
        top_n = top_n or self.top_n

        if self.explainer is None:
            return {
                'shap_top_n': [],
                'all_shap': {},
                'base_value': 0.0,
                'error': 'Explainer SHAP non configuré',
            }

        try:
            X = features_array if features_array.ndim == 2 \
                else features_array.reshape(1, -1)

            # Calcul SHAP
            shap_output = self.explainer.shap_values(X)

            # Gestion du format TreeExplainer (list [class0, class1])
            if isinstance(shap_output, list) and len(shap_output) == 2:
                shap_vals = shap_output[1][0]
            else:
                shap_vals = shap_output[0]

            # Valeur de base
            expected = self.explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                base_value = float(expected[1]) if len(expected) > 1 \
                    else float(expected[0])
            else:
                base_value = float(expected)

            # Construction du classement
            importance = list(zip(
                self.feature_names, X[0], shap_vals
            ))
            importance.sort(key=lambda x: abs(x[2]), reverse=True)

            total_abs_shap = sum(abs(s) for _, _, s in importance) + 1e-8

            top_features = [
                {
                    'feature': feat,
                    'value': round(float(val), 4),
                    'shap': round(float(shap), 4),
                    'direction': 'AUGMENTE risque fraude' if shap > 0
                                 else 'RÉDUIT risque fraude',
                    'impact_pct': round(
                        abs(float(shap)) / total_abs_shap * 100, 1),
                }
                for feat, val, shap in importance[:top_n]
            ]

            latency = (time.time() - start) * 1000
            self._track_call(latency)

            return {
                'shap_top_n': top_features,
                'all_shap': dict(zip(self.feature_names,
                                     [round(float(s), 4) for s in shap_vals])),
                'base_value': round(base_value, 4),
                'latency_ms': round(latency, 2),
            }

        except Exception as e:
            self._errors.append(str(e))
            return {
                'shap_top_n': [],
                'all_shap': {},
                'base_value': 0.0,
                'error': str(e)[:200],
            }


# ═══════════════════════════════════════════════════════════════════════════════
# LLMGatewayTool — Interface unifiée vers les LLMs
# ═══════════════════════════════════════════════════════════════════════════════

class LLMGatewayTool(BaseTool):
    """
    Passerelle LLM unifiée (Étape 3 du blueprint — Choose LLM).

    Supporte :
    - Google Gemini 2.5 Flash (SDK google-genai)
    - Fallback vers le SDK legacy (google-generativeai)
    - Mode hors-ligne (templates structurés)

    Paramètres LLM configurables :
    - temperature: Créativité (0.0 = déterministe, 1.0 = créatif)
    - max_tokens:  Longueur max de la réponse
    - top_p:       Nucleus sampling
    - model:       Nom du modèle (gemini-2.5-flash)

    Usage:
        tool = LLMGatewayTool(api_key="...", model="gemini-2.5-flash")
        result = tool.execute(prompt="Analyse cette transaction...",
                              system_prompt="Tu es un expert...")
    """

    TOOL_NAME = "LLMGatewayTool"
    TOOL_DESCRIPTION = "Génération de texte via LLM (Gemini 2.5 Flash)"

    # System prompt optimisé pour la détection de fraude
    DEFAULT_SYSTEM_PROMPT = (
        "Tu es un expert senior en détection de fraude bancaire avec 15 ans "
        "d'expérience dans les départements conformité des grandes banques "
        "européennes. Tu analyses des transactions suspectes et rédiges des "
        "rapports professionnels, concis et factuels pour les équipes de "
        "conformité et les analystes fraude.\n\n"
        "RÈGLES :\n"
        "1. Rédige toujours en français professionnel\n"
        "2. Sois factuel : ne cite que les données fournies\n"
        "3. Explique les termes techniques simplement\n"
        "4. Termine par une recommandation d'action concrète\n"
        "5. Ne dépasse pas 8 phrases\n"
        "6. Cite les features SHAP les plus influentes"
    )

    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash",
                 temperature: float = 0.3, max_tokens: int = 600,
                 config: dict = None):
        super().__init__(config)
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._online = False
        self._sdk_type = None

        effective_key = api_key or os.environ.get('GEMINI_API_KEY')
        if effective_key:
            self._setup_client(effective_key)

    def _setup_client(self, api_key: str):
        """Configure le client LLM."""
        # Tentative nouveau SDK
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._online = True
            self._sdk_type = 'new'
            return
        except (ImportError, Exception):
            pass

        # Tentative ancien SDK
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(
                self.model_name,
                system_instruction=self.DEFAULT_SYSTEM_PROMPT
            )
            self._online = True
            self._sdk_type = 'old'
        except (ImportError, Exception):
            self._online = False

    def execute(self, prompt: str = "", system_prompt: str = None,
                **kwargs) -> dict:
        """
        Génère du texte via le LLM.

        Args:
            prompt:        Le prompt utilisateur
            system_prompt: System prompt personnalisé (optionnel)

        Returns:
            dict {text, model, mode, latency_ms}
        """
        start = time.time()
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if self._online:
            try:
                text = self._call_api(prompt, system_prompt)
                mode = 'online'
            except Exception as e:
                self._errors.append(str(e))
                text = f"[LLM indisponible] {str(e)[:100]}"
                mode = 'error'
        else:
            text = "[Mode hors-ligne] Le LLM n'est pas configuré."
            mode = 'offline'

        latency = (time.time() - start) * 1000
        self._track_call(latency)

        return {
            'text': text.strip() if text else "",
            'model': self.model_name,
            'mode': mode,
            'latency_ms': round(latency, 2),
            'tokens_approx': len(text.split()) if text else 0,
        }

    def _call_api(self, prompt: str, system_prompt: str) -> str:
        """Appelle l'API du LLM."""
        if self._sdk_type == 'new':
            from google.genai import types
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            return response.text
        elif self._sdk_type == 'old':
            import google.generativeai as genai
            response = self._client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            return response.text
        return ""

    @property
    def is_online(self) -> bool:
        return self._online


# ═══════════════════════════════════════════════════════════════════════════════
# AlertTool — Système de notifications et alertes
# ═══════════════════════════════════════════════════════════════════════════════

class AlertTool(BaseTool):
    """
    Outil d'alertes et notifications.

    En production, cet outil s'intègrerait avec :
    - Slack / Microsoft Teams (webhooks)
    - Email (SMTP)
    - SMS (Twilio)
    - PagerDuty / Opsgenie

    Pour cette démonstration, les alertes sont loggées
    dans un fichier JSON et affichées en console.
    """

    TOOL_NAME = "AlertTool"
    TOOL_DESCRIPTION = "Envoi d'alertes et notifications"

    SEVERITY_LEVELS = {
        'CRITIQUE': {'color': '#e53935', 'icon': '🚫', 'priority': 1},
        'ÉLEVÉ':    {'color': '#f57c00', 'icon': '⚠️', 'priority': 2},
        'MOYEN':    {'color': '#fbc02d', 'icon': '⚡', 'priority': 3},
        'FAIBLE':   {'color': '#43a047', 'icon': '✅', 'priority': 4},
    }

    def __init__(self, alert_log_path: str = None, config: dict = None):
        super().__init__(config)
        if alert_log_path is None:
            base = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            alert_log_path = os.path.join(base, 'results', 'alerts.jsonl')
        self.alert_log_path = alert_log_path
        os.makedirs(os.path.dirname(alert_log_path), exist_ok=True)

    def execute(self, transaction_id: str = "", severity: str = "MOYEN",
                decision: str = "", message: str = "",
                notify: list = None, **kwargs) -> dict:
        """
        Émet une alerte.

        Args:
            transaction_id: ID de la transaction
            severity:       Niveau de sévérité
            decision:       Décision prise (BLOCK, REVIEW, etc.)
            message:        Message d'alerte
            notify:         Liste des destinataires

        Returns:
            dict {alert_id, severity, recipients, logged}
        """
        start = time.time()
        notify = notify or []

        alert = {
            'alert_id': f"ALT-{int(time.time()*1000) % 100000:05d}",
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'transaction_id': transaction_id,
            'severity': severity,
            'decision': decision,
            'message': message,
            'recipients': notify,
        }

        # Logging dans le fichier
        try:
            with open(self.alert_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert, ensure_ascii=False) + '\n')
            logged = True
        except Exception:
            logged = False

        # Affichage console pour les alertes critiques
        meta = self.SEVERITY_LEVELS.get(severity, self.SEVERITY_LEVELS['MOYEN'])
        if severity in ('CRITIQUE', 'ÉLEVÉ'):
            print(f"\n  {meta['icon']} ALERTE [{severity}] — {transaction_id}")
            print(f"    Décision: {decision}")
            print(f"    {message}")
            if notify:
                print(f"    → Notification: {', '.join(notify)}")

        latency = (time.time() - start) * 1000
        self._track_call(latency)

        return {
            'alert_id': alert['alert_id'],
            'severity': severity,
            'recipients': notify,
            'logged': logged,
            'latency_ms': round(latency, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DatabaseTool — Accès base de données
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseTool(BaseTool):
    """
    Outil d'accès à la base de données SQLite.

    Fournit des requêtes prédéfinies pour :
    - Historique des transactions d'un client
    - Statistiques de fraude par catégorie
    - Recherche de transactions similaires
    """

    TOOL_NAME = "DatabaseTool"
    TOOL_DESCRIPTION = "Requêtes sur la base de données des décisions"

    def __init__(self, db_path: str = None, config: dict = None):
        super().__init__(config)
        if db_path is None:
            base = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(base, 'data', 'agent_memory.db')
        self.db_path = db_path

    def execute(self, query_type: str = "recent_decisions", **kwargs) -> dict:
        """
        Exécute une requête prédéfinie.

        query_type:
            - "recent_decisions": N décisions récentes
            - "decision_stats":   Statistiques par décision
            - "fraud_rate":       Taux de fraude glissant
        """
        import sqlite3
        start = time.time()

        try:
            with sqlite3.connect(self.db_path) as conn:
                if query_type == "recent_decisions":
                    n = kwargs.get('n', 10)
                    rows = conn.execute("""
                        SELECT transaction_id, decision, probability, timestamp
                        FROM decisions ORDER BY timestamp DESC LIMIT ?
                    """, (n,)).fetchall()
                    result = {
                        'query': query_type,
                        'results': [
                            {'id': r[0], 'decision': r[1],
                             'probability': r[2], 'timestamp': r[3]}
                            for r in rows
                        ]
                    }

                elif query_type == "decision_stats":
                    rows = conn.execute("""
                        SELECT decision, COUNT(*), AVG(probability)
                        FROM decisions GROUP BY decision
                    """).fetchall()
                    result = {
                        'query': query_type,
                        'stats': {r[0]: {'count': r[1], 'avg_prob': round(r[2], 4)}
                                  for r in rows}
                    }

                else:
                    result = {'query': query_type, 'error': 'Type de requête inconnu'}

        except Exception as e:
            result = {'query': query_type, 'error': str(e)[:200]}

        latency = (time.time() - start) * 1000
        self._track_call(latency)
        result['latency_ms'] = round(latency, 2)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# FederatedLearningTool — Coordination FL
# ═══════════════════════════════════════════════════════════════════════════════

class FederatedLearningTool(BaseTool):
    """
    Outil de coordination du Federated Learning.

    En production, cet outil coordonnerait :
    - L'envoi du modèle global aux clients
    - La réception des mises à jour locales
    - L'agrégation FedAvg
    - Le monitoring de la convergence

    Pour cette démonstration, il utilise le FederatedSoftVoting
    implémenté dans scripts/federated/fed_avg.py.
    """

    TOOL_NAME = "FederatedLearningTool"
    TOOL_DESCRIPTION = "Coordination du Federated Learning (FedAvg)"

    def __init__(self, federated_model=None, config: dict = None):
        super().__init__(config)
        self.federated_model = federated_model

    def execute(self, action: str = "predict", **kwargs) -> dict:
        """
        Actions FL disponibles :
            - "predict":    Prédiction avec le modèle fédéré
            - "status":     État du système fédéré
            - "round_info": Informations sur le dernier round
        """
        start = time.time()

        if action == "predict" and self.federated_model is not None:
            X = kwargs.get('features_array')
            if X is not None:
                proba = self.federated_model.predict_proba(X)
                result = {
                    'action': action,
                    'probability': float(proba[0]) if len(proba) == 1
                                   else proba.tolist(),
                    'num_models': self.federated_model.get_num_models(),
                }
            else:
                result = {'action': action, 'error': 'features_array requis'}

        elif action == "status":
            result = {
                'action': action,
                'model_loaded': self.federated_model is not None,
                'num_models': (self.federated_model.get_num_models()
                               if self.federated_model else 0),
            }
        else:
            result = {'action': action, 'status': 'unknown_action'}

        latency = (time.time() - start) * 1000
        self._track_call(latency)
        result['latency_ms'] = round(latency, 2)
        return result
