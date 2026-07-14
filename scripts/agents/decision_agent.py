"""
decision_agent.py — Agent 3 : Décision basée sur la probabilité de fraude
Stage : Système Multi-Agents de Détection de Fraude
"""


class DecisionAgent:
    """
    Agent 3 — Prise de décision automatique basée sur la probabilité.

    4 décisions possibles :
    ┌─────────────────────────────────────────────────────────────────┐
    │  BLOCK   (prob ≥ 85%) — Fraude quasi-certaine → blocage auto   │
    │  REVIEW  (60–85%)     → Vérification humaine requise           │
    │  ALERT   (35–60%)     → Alerte + transaction autorisée         │
    │  ALLOW   (< 35%)      → Transaction approuvée normalement      │
    └─────────────────────────────────────────────────────────────────┘

    Les seuils sont configurables selon la politique de risque de la banque.
    """

    DEFAULT_THRESHOLDS = {
        'BLOCK':  0.85,
        'REVIEW': 0.60,
        'ALERT':  0.35,
        'ALLOW':  0.00,
    }

    DECISION_METADATA = {
        'BLOCK': {
            'severity': 'CRITIQUE',
            'action':   'Transaction refusée automatiquement. Dossier fraude ouvert. Client notifié.',
            'sla_min':  0,   # Décision immédiate
            'notify':   ['equipe_fraude', 'client', 'conformite', 'direction_risques'],
            'color':    '#e53935',
            'icon':     '🚫',
        },
        'REVIEW': {
            'severity': 'ÉLEVÉ',
            'action':   'Transaction suspendue. Analyste humain assigné (délai max: 15 min).',
            'sla_min':  15,
            'notify':   ['analyste_fraude', 'client'],
            'color':    '#f57c00',
            'icon':     '⚠️',
        },
        'ALERT': {
            'severity': 'MOYEN',
            'action':   'Transaction autorisée avec surveillance renforcée. Journalisation complète.',
            'sla_min':  60,
            'notify':   ['equipe_surveillance'],
            'color':    '#fbc02d',
            'icon':     '⚡',
        },
        'ALLOW': {
            'severity': 'FAIBLE',
            'action':   'Transaction approuvée. Aucune action requise.',
            'sla_min':  None,
            'notify':   [],
            'color':    '#43a047',
            'icon':     '✅',
        },
    }

    def __init__(self, thresholds=None):
        """
        Args:
            thresholds: dict optionnel pour surcharger les seuils par défaut.
                        ex: {'BLOCK': 0.90, 'REVIEW': 0.70, 'ALERT': 0.40, 'ALLOW': 0.00}
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self._counts    = {'BLOCK': 0, 'REVIEW': 0, 'ALERT': 0, 'ALLOW': 0}

    # ─────────────────────────────────────────────────────────────────────────
    # API principale
    # ─────────────────────────────────────────────────────────────────────────

    def decide(self, probability: float) -> dict:
        """
        Applique les règles métier et retourne la décision complète.

        Args:
            probability: float [0, 1] — probabilité de fraude (sortie de l'Agent 2)

        Returns:
            dict {
                'decision': str,      # 'BLOCK', 'REVIEW', 'ALERT', 'ALLOW'
                'probability': float,
                'threshold_used': float,
                'severity': str,
                'action': str,
                'notify': list[str],
                'sla_min': int or None,
            }
        """
        # Appliquer les seuils dans l'ordre décroissant
        decision = 'ALLOW'
        for action in ['BLOCK', 'REVIEW', 'ALERT', 'ALLOW']:
            if probability >= self.thresholds[action]:
                decision = action
                break

        self._counts[decision] += 1
        meta = self.DECISION_METADATA[decision]

        return {
            'decision':       decision,
            'probability':    round(probability, 4),
            'threshold_used': self.thresholds[decision],
            'severity':       meta['severity'],
            'action':         meta['action'],
            'notify':         meta['notify'],
            'sla_min':        meta['sla_min'],
            'color':          meta['color'],
            'icon':           meta['icon'],
        }

    def decide_batch(self, probabilities: list) -> list:
        """Prend des décisions pour un lot de probabilités."""
        return [self.decide(p) for p in probabilities]

    # ─────────────────────────────────────────────────────────────────────────
    # Statistiques et visualisations
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total = sum(self._counts.values())
        return {
            'Total décisions': total,
            **{f'{k} ({v} tx)': f"{v/max(1,total)*100:.1f}%"
               for k, v in self._counts.items() if v > 0}
        }

    def plot_decision_distribution(self, figsize=(7, 5)):
        """Graphe en camembert des décisions prises."""
        import matplotlib.pyplot as plt

        counts = {k: v for k, v in self._counts.items() if v > 0}
        if not counts:
            print("[DecisionAgent] Aucune décision prise pour l'instant.")
            return

        colors = [self.DECISION_METADATA[k]['color'] for k in counts]
        labels = [f"{self.DECISION_METADATA[k]['icon']} {k} ({v})"
                  for k, v in counts.items()]

        fig, ax = plt.subplots(figsize=figsize)
        pie_result = ax.pie(
            counts.values(), labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
        )
        
        if len(pie_result) == 3:
            autotexts = pie_result[2]
            for at in autotexts:
                at.set_fontsize(10)
                at.set_fontweight('bold')

        ax.set_title("Répartition des Décisions — Agent 3", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def reset(self):
        self._counts = {'BLOCK': 0, 'REVIEW': 0, 'ALERT': 0, 'ALLOW': 0}

    @property
    def decision_counts(self):
        return dict(self._counts)
