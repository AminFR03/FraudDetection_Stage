"""
surveillance_agent.py — Agent 1 : Surveillance et filtrage rapide
Stage : Système Multi-Agents de Détection de Fraude
"""

import numpy as np


class SurveillanceAgent:
    """
    Agent 1 — Surveillance et filtrage par règles métier.

    Rôle : Analyser chaque transaction entrante par des règles simples et rapides.
    - Si clairement normale → décision ALLOW immédiate (économie de ressources).
    - Si suspecte → transmettre à l'Agent 2 (analyse ML + SHAP).

    Ce filtrage réduit la charge computationnelle en évitant d'invoquer
    le modèle ML pour les transactions manifestement légitimes.
    """

    # ── Seuils ULB Dataset (features PCA standardisées) ──────────────────────
    ULB_THRESHOLDS = {
        'Amount_high':     2.0,    # Amount standardisé > 2 écarts-types
        'V14_low':        -2.5,    # V14 fortement corrélé avec la fraude
        'V12_low':        -2.0,
        'V10_low':        -2.0,
        'V4_high':         2.5,
        'V3_extreme':      3.5,    # V3 extrêmement anormal
    }

    # ── Seuils Synthetic Dataset (features lisibles) ──────────────────────────
    SYNTHETIC_THRESHOLDS = {
        'amt_high':          1500,    # Montant > 1500$
        'hour_unusual_min':  0,       # Heure de nuit (0h–4h)
        'hour_unusual_max':  4,
        'city_pop_low':      500,     # Ville très peu peuplée (risque géographique)
    }

    # ── Catégories à risque élevé (dataset synthétique) ──────────────────────
    HIGH_RISK_CATEGORIES = {
        'travel', 'online_retail', 'shopping_net', 'misc_net',
        'home', 'personal_care'
    }

    def __init__(self, dataset_type='ulb', custom_thresholds=None):
        """
        Args:
            dataset_type: 'ulb' ou 'synthetic'
            custom_thresholds: dict pour surcharger les seuils par défaut
        """
        self.dataset_type = dataset_type
        if custom_thresholds:
            if dataset_type == 'ulb':
                self.ULB_THRESHOLDS.update(custom_thresholds)
            else:
                self.SYNTHETIC_THRESHOLDS.update(custom_thresholds)

        # Statistiques de fonctionnement
        self._stats = {
            'total':    0,
            'flagged':  0,
            'cleared':  0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # API principale
    # ─────────────────────────────────────────────────────────────────────────

    def screen(self, transaction: dict) -> dict:
        """
        Analyse une transaction par règles métier.

        Args:
            transaction: dict {feature_name: value}

        Returns:
            dict {
                'suspicious': bool,
                'reasons': list[str],
                'fast_decision': str or None,  # 'ALLOW' si clairement normale
                'risk_score': int,             # Nombre d'alertes déclenchées
            }
        """
        self._stats['total'] += 1

        if self.dataset_type == 'ulb':
            reasons = self._screen_ulb(transaction)
        else:
            reasons = self._screen_synthetic(transaction)

        suspicious = len(reasons) > 0

        if suspicious:
            self._stats['flagged'] += 1
            return {
                'suspicious':    True,
                'reasons':       reasons,
                'fast_decision': None,
                'risk_score':    len(reasons),
            }
        else:
            self._stats['cleared'] += 1
            return {
                'suspicious':    False,
                'reasons':       [],
                'fast_decision': 'ALLOW',
                'risk_score':    0,
            }

    def screen_batch(self, transactions: list) -> list:
        """Analyse un lot de transactions."""
        return [self.screen(tx) for tx in transactions]

    # ─────────────────────────────────────────────────────────────────────────
    # Règles ULB (features PCA anonymisées)
    # ─────────────────────────────────────────────────────────────────────────

    def _screen_ulb(self, tx: dict) -> list:
        reasons = []
        t = self.ULB_THRESHOLDS

        # Montant élevé
        amount = tx.get('Amount', tx.get('amount', 0))
        if amount > t['Amount_high']:
            reasons.append(f"Montant anormalement élevé (Amount={amount:.2f} σ)")

        # V14 — meilleur séparateur fraude (corrélation négative)
        v14 = tx.get('V14', 0)
        if v14 < t['V14_low']:
            reasons.append(f"V14 anormal: {v14:.3f} < {t['V14_low']} "
                           f"(fort indicateur de fraude)")

        # V12 — second meilleur séparateur
        v12 = tx.get('V12', 0)
        if v12 < t['V12_low']:
            reasons.append(f"V12 anormal: {v12:.3f} < {t['V12_low']}")

        # V10
        v10 = tx.get('V10', 0)
        if v10 < t['V10_low']:
            reasons.append(f"V10 anormal: {v10:.3f} < {t['V10_low']}")

        # V4 — corrélation positive avec fraude
        v4 = tx.get('V4', 0)
        if v4 > t['V4_high']:
            reasons.append(f"V4 élevé: {v4:.3f} > {t['V4_high']}")

        # V3 extrêmement anormal
        v3 = tx.get('V3', 0)
        if abs(v3) > t['V3_extreme']:
            reasons.append(f"V3 extrême: {v3:.3f} (|V3| > {t['V3_extreme']})")

        # Combinaison d'anomalies mineures (règle composite)
        minor = sum([
            abs(tx.get('V7', 0)) > 2.5,
            abs(tx.get('V11', 0)) > 2.5,
            abs(tx.get('V16', 0)) > 2.5,
            abs(tx.get('V17', 0)) > 3.0,
        ])
        if minor >= 2:
            reasons.append(f"{minor} anomalies V-features mineures simultanées")

        return reasons

    # ─────────────────────────────────────────────────────────────────────────
    # Règles Synthétique (features lisibles)
    # ─────────────────────────────────────────────────────────────────────────

    def _screen_synthetic(self, tx: dict) -> list:
        reasons = []
        t = self.SYNTHETIC_THRESHOLDS

        # Montant élevé
        amt = tx.get('amt', tx.get('amount', 0))
        if amt > t['amt_high']:
            reasons.append(f"Montant élevé: ${amt:.2f} > ${t['amt_high']}")

        # Heure inhabituelle (nuit profonde)
        hour = tx.get('hour', None)
        if hour is None:
            tx_time = tx.get('Transaction_Time', '')
            if tx_time and ':' in str(tx_time):
                try:
                    hour = int(str(tx_time).split(':')[0])
                except ValueError:
                    hour = None

        if hour is not None and t['hour_unusual_min'] <= hour <= t['hour_unusual_max']:
            reasons.append(f"Transaction nocturne: {hour}h (entre minuit et 4h)")

        # Catégorie à risque
        category = tx.get('category', '').lower()
        if any(cat in category for cat in self.HIGH_RISK_CATEGORIES):
            reasons.append(f"Catégorie à risque élevé: '{category}'")

        # Population de la ville faible (risque géographique)
        city_pop = tx.get('city_pop', 9999)
        if city_pop < t['city_pop_low']:
            reasons.append(f"Ville peu peuplée: {city_pop:,} habitants (risque géo)")

        return reasons

    # ─────────────────────────────────────────────────────────────────────────
    # Statistiques
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total = self._stats['total']
        return {
            'Transactions analysées':     total,
            'Transactions suspectes':     self._stats['flagged'],
            'Transactions autorisées':    self._stats['cleared'],
            'Taux de suspicion':         f"{self._stats['flagged']/max(1,total)*100:.1f}%",
        }

    def reset_stats(self):
        self._stats = {'total': 0, 'flagged': 0, 'cleared': 0}
