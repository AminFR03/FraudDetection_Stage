"""
explanation_agent.py — Agent 4 : Génération d'explications en langage naturel (LLM)
Stage : Système Multi-Agents de Détection de Fraude

Utilise l'API Google Gemini pour produire des rapports professionnels en français.
La clé API est lue depuis la variable d'environnement GEMINI_API_KEY si non fournie.
Un mode fallback hors-ligne est disponible si l'API n'est pas configurée ou indisponible.
"""

import os
import time

class ExplanationAgent:
    """
    Agent 4 — Génération d'explications en langage naturel via LLM (Gemini).

    Rôle :
    - Recevoir les résultats des 3 agents précédents
    - Construire un prompt structuré avec toutes les informations disponibles
    - Appeler l'API Gemini pour produire un rapport professionnel en français
    - Retourner ce rapport aux analystes humains / aux clients

    En mode hors-ligne (pas de clé API), un rapport structuré est généré
    localement à partir des données SHAP (sans appel LLM).
    """

    SYSTEM_CONTEXT = """Tu es un expert senior en détection de fraude bancaire avec 15 ans d'expérience.
Tu analyses des transactions suspectes et rédiges des rapports clairs pour les équipes de conformité.
Tes rapports sont toujours professionnels, factuels, concis et compréhensibles par des non-techniciens."""

    def __init__(self, api_key=None, model_name='gemini-1.5-flash',
                 temperature=0.4, max_tokens=500):
        """
        Args:
            api_key: clé API Gemini. Si None, tente de lire la variable
                     d'environnement GEMINI_API_KEY automatiquement.
            model_name: modèle Gemini à utiliser (ex: gemini-1.5-flash)
            temperature: créativité du modèle (0=déterministe, 1=créatif)
            max_tokens: longueur max de la réponse
        """
        self.model_name   = model_name
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self._gemini      = None
        self._online_mode = False
        self._count       = 0

        # Auto-lecture de la clé API depuis l'environnement si non fournie
        effective_key = api_key or os.environ.get('GEMINI_API_KEY')

        if effective_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=effective_key)
                self._gemini = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=self.SYSTEM_CONTEXT
                )
                self._online_mode = True
                print(f"[ExplanationAgent] Gemini ({model_name}) configuré — Mode en ligne")
                source = 'variable env GEMINI_API_KEY' if not api_key else 'clé fournie'
                print(f"[ExplanationAgent] Clé API chargée depuis : {source}")
            except ImportError:
                print("[ExplanationAgent] Package 'google-generativeai' non installé. Lancez : pip install google-generativeai")
                print("[ExplanationAgent] Mode fallback hors-ligne activé")
            except Exception as e:
                print(f"[ExplanationAgent] Gemini non disponible: {e}")
                print("[ExplanationAgent] Mode fallback hors-ligne activé")
        else:
            print("[ExplanationAgent] Mode hors-ligne (GEMINI_API_KEY non définie)")

    @property
    def is_online(self):
        return self._online_mode

    @property
    def explanations_count(self):
        return self._count

    # ─────────────────────────────────────────────────────────────────────────
    # API principale
    # ─────────────────────────────────────────────────────────────────────────

    def explain(self, transaction: dict,
                surveillance_result: dict,
                analysis_result: dict,
                decision_result: dict,
                transaction_id: str = None) -> str:
        """
        Génère une explication en langage naturel pour la décision prise.

        Args:
            transaction: dict de la transaction originale
            surveillance_result: sortie de l'Agent 1
            analysis_result: sortie de l'Agent 2 (prob, SHAP, niveau risque)
            decision_result: sortie de l'Agent 3 (BLOCK/REVIEW/ALERT/ALLOW)
            transaction_id: identifiant optionnel de la transaction

        Returns:
            str: rapport professionnel en français
        """
        self._count += 1
        tx_id = transaction_id or f"TXN-{self._count:04d}"

        if self._online_mode and self._gemini:
            return self._explain_online(
                tx_id, transaction, surveillance_result, analysis_result, decision_result)
        else:
            return self._explain_offline(
                tx_id, transaction, surveillance_result, analysis_result, decision_result)

    # ─────────────────────────────────────────────────────────────────────────
    # Mode en ligne (API Gemini)
    # ─────────────────────────────────────────────────────────────────────────

    def _explain_online(self, tx_id, transaction, surv, analysis, decision) -> str:
        prompt = self._build_prompt(tx_id, transaction, surv, analysis, decision)

        try:
            import google.generativeai as genai
            response = self._gemini.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            return response.text.strip()

        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'rate_limit' in err_str.lower() or 'quota' in err_str.lower():
                print(f"[ExplanationAgent] Quota/rate-limit Gemini (429) — Fallback hors-ligne")
            elif '401' in err_str or 'unauthorized' in err_str.lower() or 'invalid authentication' in err_str.lower():
                print(f"[ExplanationAgent] Clé API invalide (401) — Fallback hors-ligne")
            elif '403' in err_str or 'permission' in err_str.lower():
                print(f"[ExplanationAgent] Accès refusé (403) — Fallback hors-ligne")
            elif '404' in err_str or 'not found' in err_str.lower():
                print(f"[ExplanationAgent] Modèle introuvable (404) — Fallback hors-ligne")
            else:
                print(f"[ExplanationAgent] Erreur API Gemini: {e} — Fallback hors-ligne")
            return self._explain_offline(tx_id, transaction, surv, analysis, decision)

    def _build_prompt(self, tx_id, transaction, surv, analysis, decision) -> str:
        """Construit le prompt structuré pour le LLM."""

        # Règles de surveillance déclenchées
        if surv.get('reasons'):
            surv_text = '\n'.join(f"  • {r}" for r in surv['reasons'])
        else:
            surv_text = "  • Aucune anomalie de règle détectée"

        # Top features SHAP
        shap_key = 'shap_top5' if 'shap_top5' in analysis else 'shap_top_n'
        shap_items = analysis.get(shap_key, [])
        shap_text = ""
        for item in shap_items:
            sign = "+" if item['shap'] > 0 else ""
            shap_text += (
                f"  • {item['feature']} = {item['value']:.3f} "
                f"(SHAP: {sign}{item['shap']:.3f} → {item['direction']})\n"
            )

        # Détails transaction (features lisibles si disponibles)
        tx_details = ""
        for key in ['amt', 'Amount', 'category', 'merchant', 'Transaction_Time',
                    'job', 'Payment_Method', 'city', 'state']:
            if key in transaction:
                tx_details += f"  • {key}: {transaction[key]}\n"

        prompt = f"""═══════════════════════════════════════════════════════
RAPPORT D'ANALYSE — Transaction {tx_id}
═══════════════════════════════════════════════════════

[AGENT 1 — Surveillance]
Alertes règles métier :
{surv_text}

[AGENT 2 — Analyse ML]
Probabilité de fraude : {analysis['probability']*100:.1f}%
Niveau de risque      : {analysis['risk_level']}
Top features influentes (valeurs SHAP) :
{shap_text}

[AGENT 3 — Décision]
Décision prise  : {decision['decision']}
Sévérité        : {decision['severity']}
Action requise  : {decision['action']}
Parties alertées: {', '.join(decision['notify']) if decision['notify'] else 'Aucune'}

Informations transaction :
{tx_details if tx_details else "  • Données anonymisées (PCA)"}

═══════════════════════════════════════════════════════
INSTRUCTIONS DE RÉDACTION :
1. Rédige un rapport de 4 à 6 phrases en français professionnel.
2. Explique POURQUOI cette transaction a été {decision['decision']}.
3. Cite les 2-3 features SHAP les plus déterminantes et leur signification.
4. Adapte le niveau de langage à un analyste bancaire non-technicien.
5. Termine par une recommandation d'action concrète.
6. Sois factuel, précis, et n'invente aucune information.

Rapport professionnel :"""

        return prompt

    # ─────────────────────────────────────────────────────────────────────────
    # Mode hors-ligne (génération locale sans LLM)
    # ─────────────────────────────────────────────────────────────────────────

    def _explain_offline(self, tx_id, transaction, surv, analysis, decision) -> str:
        """Génère un rapport structuré sans appel LLM."""

        prob_pct  = analysis['probability'] * 100
        dec       = decision['decision']
        risk      = analysis['risk_level']
        action    = decision['action']

        shap_key  = 'shap_top5' if 'shap_top5' in analysis else 'shap_top_n'
        shap_list = analysis.get(shap_key, [])

        # Phrase d'ouverture selon la décision
        opening = {
            'BLOCK':  f"La transaction {tx_id} a été BLOQUÉE automatiquement",
            'REVIEW': f"La transaction {tx_id} a été mise en ATTENTE de vérification",
            'ALERT':  f"Une ALERTE a été générée pour la transaction {tx_id}",
            'ALLOW':  f"La transaction {tx_id} a été AUTORISÉE",
        }[dec]

        # Rapport structuré
        lines = [
            f"═══ RAPPORT DE DÉTECTION DE FRAUDE — {tx_id} ═══",
            "",
            f"[Décision] {opening} avec une probabilité de fraude de {prob_pct:.1f}% "
            f"(niveau de risque : {risk}).",
            "",
        ]

        # Explication SHAP
        if shap_list:
            top1 = shap_list[0]
            top2 = shap_list[1] if len(shap_list) > 1 else None
            top3 = shap_list[2] if len(shap_list) > 2 else None

            shap_desc = (
                f"[Facteurs] La feature la plus déterminante est "
                f"**{top1['feature']}** (valeur={top1['value']:.3f}, "
                f"impact SHAP={top1['shap']:+.3f}), "
                f"qui {top1['direction'].lower()}."
            )
            if top2:
                shap_desc += (
                    f" En second lieu, **{top2['feature']}** "
                    f"(SHAP={top2['shap']:+.3f}) {top2['direction'].lower()}."
                )
            if top3:
                shap_desc += (
                    f" Enfin, **{top3['feature']}** "
                    f"(SHAP={top3['shap']:+.3f}) {top3['direction'].lower()}."
                )
            lines.append(shap_desc)
            lines.append("")

        # Alertes surveillance
        if surv.get('reasons'):
            rules_text = "; ".join(surv['reasons'][:3])
            lines.append(f"[Règles] L'agent de surveillance a détecté : {rules_text}.")
            lines.append("")

        # Action
        lines.append(f"[Action] {action}")
        if decision.get('notify'):
            lines.append(f"[Notification] Parties alertées: {', '.join(decision['notify'])}.")

        lines.append("")
        lines.append("[Note] Rapport généré en mode hors-ligne (sans API LLM).")

        return '\n'.join(lines)
