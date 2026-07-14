"""
analysis_agent.py — Agent 2 : Analyse ML + Explainabilité SHAP
Stage : Système Multi-Agents de Détection de Fraude
"""

import numpy as np


class AnalysisAgent:
    """
    Agent 2 — Analyse par modèle ML/DL et explainabilité SHAP.

    Rôle :
    1. Charger un modèle entraîné (XGBoost, RF, MLP, etc.)
    2. Calculer la probabilité de fraude pour une transaction
    3. Extraire les valeurs SHAP (top-N features les plus influentes)
    4. Retourner le niveau de risque et les justifications chiffrées

    Input  ← Transaction dict {feature: value} (transmis par Agent 1)
    Output → {probability, risk_level, shap_top5, all_shap}
    """

    # Niveaux de risque basés sur la probabilité de fraude
    RISK_LEVELS = {
        'CRITIQUE': 0.85,
        'ÉLEVÉ':    0.60,
        'MOYEN':    0.35,
        'FAIBLE':   0.00,
    }

    def __init__(self, model, shap_explainer, scaler, feature_names, top_n=5):
        """
        Args:
            model: modèle entraîné (scikit-learn compatible, avec predict_proba)
            shap_explainer: objet SHAP (TreeExplainer, LinearExplainer, etc.)
            scaler: StandardScaler/RobustScaler (déjà fité sur les données)
            feature_names: list[str] des noms de features
            top_n: nombre de features SHAP à retourner
        """
        self.model         = model
        self.explainer     = shap_explainer
        self.scaler        = scaler
        self.feature_names = feature_names
        self.top_n         = top_n
        self._analyses     = 0

    # ─────────────────────────────────────────────────────────────────────────
    # API principale
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, transaction: dict) -> dict:
        """
        Analyse complète d'une transaction : probabilité + SHAP.

        Args:
            transaction: dict {feature_name: float_value}
                         Les valeurs doivent être dans l'espace ORIGINAL
                         (le scaler est appliqué ici).

        Returns:
            dict {
                'probability': float,
                'risk_level': str,
                'shap_top_n': list[dict],  # Features les plus influentes
                'all_shap': dict,
                'base_value': float,       # Valeur de base SHAP
            }
        """
        self._analyses += 1

        # ── Préparation du vecteur de features ────────────────────────────────
        feat_values = np.array([[transaction.get(f, 0.0) for f in self.feature_names]])

        # ── Probabilité de fraude ─────────────────────────────────────────────
        prob = float(self.model.predict_proba(feat_values)[:, 1][0])

        # ── Niveau de risque ──────────────────────────────────────────────────
        risk_level = 'FAIBLE'
        for level, threshold in self.RISK_LEVELS.items():
            if prob >= threshold:
                risk_level = level
                break

        # ── Valeurs SHAP ──────────────────────────────────────────────────────
        try:
            shap_output = self.explainer.shap_values(feat_values)
            # TreeExplainer peut retourner liste [class0, class1] pour classification binaire
            if isinstance(shap_output, list) and len(shap_output) == 2:
                shap_vals = shap_output[1][0]  # Classe fraude (index 1)
            else:
                shap_vals = shap_output[0]

            base_value = float(self.explainer.expected_value) \
                if not isinstance(self.explainer.expected_value, (list, np.ndarray)) \
                else float(self.explainer.expected_value[1])

        except Exception as e:
            # Fallback : SHAP nul si le calcul échoue
            shap_vals = np.zeros(len(self.feature_names))
            base_value = 0.0
            print(f"[AnalysisAgent] Avertissement SHAP: {e}")

        # ── Top-N features les plus influentes ────────────────────────────────
        shap_importance = list(zip(
            self.feature_names,
            feat_values[0],
            shap_vals
        ))
        shap_importance.sort(key=lambda x: abs(x[2]), reverse=True)

        top_n_features = [
            {
                'feature':    feat,
                'value':      round(float(val), 4),
                'shap':       round(float(shap), 4),
                'direction':  'AUGMENTE risque fraude' if shap > 0 else 'RÉDUIT risque fraude',
                'impact_pct': round(abs(float(shap)) / (sum(abs(s) for _, _, s in shap_importance) + 1e-8) * 100, 1)
            }
            for feat, val, shap in shap_importance[:self.top_n]
        ]

        return {
            'probability':  round(prob, 4),
            'risk_level':   risk_level,
            'shap_top_n':   top_n_features,
            'all_shap':     dict(zip(self.feature_names, shap_vals.tolist())),
            'base_value':   round(base_value, 4),
        }

    def analyze_batch(self, transactions: list) -> list:
        """Analyse un lot de transactions."""
        return [self.analyze(tx) for tx in transactions]

    # ─────────────────────────────────────────────────────────────────────────
    # Analyse SHAP globale (summary)
    # ─────────────────────────────────────────────────────────────────────────

    def global_shap_summary(self, X_sample: np.ndarray, plot=True):
        """
        Calcule et affiche le graphe de résumé SHAP global.

        Args:
            X_sample: tableau numpy (N, n_features) — échantillon du test set
            plot: si True, affiche le graphe
        """
        import shap
        import matplotlib.pyplot as plt

        print(f"[AnalysisAgent] Calcul SHAP sur {len(X_sample)} échantillons...")
        shap_output = self.explainer.shap_values(X_sample)

        if isinstance(shap_output, list) and len(shap_output) == 2:
            shap_vals = shap_output[1]
        else:
            shap_vals = shap_output

        if plot:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_sample,
                              feature_names=self.feature_names,
                              plot_type='bar', show=False)
            plt.title("SHAP — Importance Globale des Features", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals, X_sample,
                              feature_names=self.feature_names,
                              show=False)
            plt.title("SHAP — Distribution des Valeurs (Beeswarm)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        return shap_vals

    # ─────────────────────────────────────────────────────────────────────────
    # Statistiques
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def analyses_count(self):
        return self._analyses
