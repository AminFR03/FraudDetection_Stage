"""
explanation_agent.py — Agent 4 : Génération d'explications en langage naturel (LLM)
Stage : Système Multi-Agents de Détection de Fraude

Utilise la nouvelle API Google Gemini (package google-genai) pour produire des
rapports professionnels en français avec gemini-2.5-flash.

La clé API est lue depuis la variable d'environnement GEMINI_API_KEY si non fournie.
Un mode fallback hors-ligne est disponible si l'API n'est pas configurée ou indisponible.

Installation requise : pip install google-genai
"""

import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# Tentative d'import du nouveau SDK google-genai (recommandé pour gemini-2.5-flash)
# Fallback sur google-generativeai (ancien SDK) si le nouveau n'est pas installé
# ─────────────────────────────────────────────────────────────────────────────
try:
    from google import genai as _new_genai
    from google.genai import types as _new_types
    _NEW_SDK = True
except ImportError:
    _NEW_SDK = False

try:
    import google.generativeai as _old_genai
    _OLD_SDK = True
except ImportError:
    _OLD_SDK = False


class ExplanationAgent:
    """
    Agent 4 — Génération d'explications en langage naturel via LLM (Gemini 2.5 Flash).

    Rôle :
    - Recevoir les résultats des 3 agents précédents
    - Construire un prompt structuré avec toutes les informations disponibles
    - Appeler l'API Gemini pour produire un rapport professionnel en français
    - Retourner ce rapport aux analystes humains / aux clients

    En mode hors-ligne (pas de clé API valide), un rapport structuré est généré
    localement à partir des données SHAP (sans appel LLM).

    Priorité SDK : google-genai (nouveau, compatible gemini-2.5-flash)
                   → google-generativeai (ancien, fallback)
    """

    SYSTEM_CONTEXT = (
        "Tu es un expert senior en détection de fraude bancaire avec 15 ans d'expérience. "
        "Tu analyses des transactions suspectes et rédiges des rapports clairs pour les équipes de conformité. "
        "Tes rapports sont toujours professionnels, factuels, concis et compréhensibles par des non-techniciens."
    )

    def __init__(self, api_key=None, model_name='gemini-2.5-flash',
                 temperature=0.4, max_tokens=500):
        """
        Args:
            api_key    : clé API Gemini. Si None, lit GEMINI_API_KEY depuis l'environnement.
                         Obtenez votre clé sur https://aistudio.google.com/apikey
                         (les clés valides commencent par 'AIza...')
            model_name : modèle Gemini à utiliser (défaut : gemini-2.5-flash)
            temperature: créativité du modèle (0=déterministe, 1=créatif)
            max_tokens : longueur max de la réponse en tokens
        """
        self.model_name   = model_name
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self._client      = None   # client new SDK
        self._gemini      = None   # model object old SDK
        self._online_mode = False
        self._sdk_type    = None   # 'new' | 'old'
        self._count       = 0

        # Auto-lecture depuis l'environnement si aucune clé fournie
        effective_key = api_key or os.environ.get('GEMINI_API_KEY')

        if not effective_key:
            print("[ExplanationAgent] ⚠️  GEMINI_API_KEY non définie → Mode hors-ligne activé")
            print("[ExplanationAgent]    Obtenez une clé sur : https://aistudio.google.com/apikey")
            return

        # ── Tentative avec le nouveau SDK (google-genai) ───────────────────
        if _NEW_SDK:
            try:
                client = _new_genai.Client(api_key=effective_key)
                # Pas de test de connexion au __init__ : la clé est supposée valide.
                # Les erreurs 403/429 seront capturées dans _explain_online() avec fallback.
                self._client      = client
                self._online_mode = True
                self._sdk_type    = 'new'
                src = 'GEMINI_API_KEY (env)' if not api_key else 'clé fournie'
                print(f"[ExplanationAgent] ✅ Gemini {model_name} configuré — Mode en ligne")
                print(f"[ExplanationAgent]    SDK : google-genai (nouveau) | Clé : {src}")
                return
            except Exception as e:
                print(f"[ExplanationAgent] ⚠️  Erreur new SDK : {self._fmt_error(e)}")

        # ── Fallback sur l'ancien SDK (google-generativeai) ───────────────
        if _OLD_SDK:
            try:
                _old_genai.configure(api_key=effective_key)
                model_obj = _old_genai.GenerativeModel(
                    model_name,
                    system_instruction=self.SYSTEM_CONTEXT
                )
                self._gemini      = model_obj
                self._online_mode = True
                self._sdk_type    = 'old'
                src = 'GEMINI_API_KEY (env)' if not api_key else 'clé fournie'
                print(f"[ExplanationAgent] ✅ Gemini {model_name} configuré — Mode en ligne (SDK legacy)")
                print(f"[ExplanationAgent]    SDK : google-generativeai (ancien) | Clé : {src}")
                return
            except Exception as e:
                print(f"[ExplanationAgent] ⚠️  Erreur old SDK : {self._fmt_error(e)}")

        if not _NEW_SDK and not _OLD_SDK:
            print("[ExplanationAgent] ❌ Aucun SDK Gemini installé.")
            print("[ExplanationAgent]    Lancez : pip install google-genai")

        print("[ExplanationAgent] Mode hors-ligne activé (API indisponible)")

    # ─────────────────────────────────────────────────────────────────────────
    # Propriétés
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_online(self):
        """True si l'agent est configuré et a une connexion API valide."""
        return self._online_mode

    # Alias pour compatibilité avec le code notebook qui utilise ._online
    @property
    def _online(self):
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
            transaction        : dict des features de la transaction originale
            surveillance_result: sortie de l'Agent 1 (rules, suspicious, reasons)
            analysis_result    : sortie de l'Agent 2 (probability, shap_top5, risk_level)
            decision_result    : sortie de l'Agent 3 (decision, action, severity, notify)
            transaction_id     : identifiant optionnel affiché dans le rapport

        Returns:
            str: rapport professionnel en français (LLM ou template hors-ligne)
        """
        self._count += 1
        tx_id = transaction_id or f"TXN-{self._count:04d}"

        if self._online_mode:
            return self._explain_online(
                tx_id, transaction, surveillance_result, analysis_result, decision_result)
        else:
            return self._explain_offline(
                tx_id, transaction, surveillance_result, analysis_result, decision_result)

    # ─────────────────────────────────────────────────────────────────────────
    # Appel API (online)
    # ─────────────────────────────────────────────────────────────────────────

    def _explain_online(self, tx_id, transaction, surv, analysis, decision) -> str:
        """Appelle l'API Gemini et retourne le texte généré (avec fallback offline)."""
        prompt = self._build_prompt(tx_id, transaction, surv, analysis, decision)

        try:
            if self._sdk_type == 'new':
                return self._call_new_sdk(prompt)
            else:
                return self._call_old_sdk(prompt)

        except Exception as e:
            print(f"[ExplanationAgent] ⚠️  Erreur lors de la génération : {self._fmt_error(e)}")
            print("[ExplanationAgent]    Fallback vers le mode hors-ligne")
            return self._explain_offline(tx_id, transaction, surv, analysis, decision)

    def _call_new_sdk(self, prompt: str) -> str:
        """Appel avec le nouveau SDK google-genai."""
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=_new_types.GenerateContentConfig(
                system_instruction=self.SYSTEM_CONTEXT,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        )
        return response.text.strip()

    def _call_old_sdk(self, prompt: str) -> str:
        """Appel avec l'ancien SDK google-generativeai."""
        response = self._gemini.generate_content(
            prompt,
            generation_config=_old_genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )
        return response.text.strip()

    # ─────────────────────────────────────────────────────────────────────────
    # Construction du prompt
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(self, tx_id, transaction, surv, analysis, decision) -> str:
        """Construit le prompt structuré envoyé au LLM."""

        # Alertes de surveillance
        if surv.get('reasons'):
            surv_text = '\n'.join(f"  • {r}" for r in surv['reasons'])
        else:
            surv_text = "  • Aucune anomalie de règle détectée"

        # Top features SHAP
        shap_key   = 'shap_top5' if 'shap_top5' in analysis else 'shap_top_n'
        shap_items = analysis.get(shap_key, [])
        shap_text  = ""
        for item in shap_items:
            sign = "+" if item['shap'] > 0 else ""
            shap_text += (
                f"  • {item['feature']} = {item['value']:.3f} "
                f"(SHAP: {sign}{item['shap']:.3f} → {item['direction']})\n"
            )

        # Détails lisibles de la transaction (si dataset synthétique)
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
{shap_text if shap_text else "  • Données SHAP non disponibles"}

[AGENT 3 — Décision]
Décision prise  : {decision['decision']}
Sévérité        : {decision['severity']}
Action requise  : {decision['action']}
Parties alertées: {', '.join(decision['notify']) if decision.get('notify') else 'Aucune'}

Informations transaction :
{tx_details if tx_details else "  • Données anonymisées (features PCA V1-V28)"}

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
    # Mode hors-ligne (template structuré, sans LLM)
    # ─────────────────────────────────────────────────────────────────────────

    def _explain_offline(self, tx_id, transaction, surv, analysis, decision) -> str:
        """Génère un rapport structuré sans appel LLM (toujours disponible)."""

        prob_pct  = analysis['probability'] * 100
        dec       = decision['decision']
        risk      = analysis['risk_level']
        action    = decision['action']

        shap_key  = 'shap_top5' if 'shap_top5' in analysis else 'shap_top_n'
        shap_list = analysis.get(shap_key, [])

        opening = {
            'BLOCK':  f"La transaction {tx_id} a été BLOQUÉE automatiquement",
            'REVIEW': f"La transaction {tx_id} a été mise en ATTENTE de vérification",
            'ALERT':  f"Une ALERTE a été générée pour la transaction {tx_id}",
            'ALLOW':  f"La transaction {tx_id} a été AUTORISÉE",
        }.get(dec, f"La transaction {tx_id} a reçu la décision : {dec}")

        lines = [
            f"═══ RAPPORT DE DÉTECTION DE FRAUDE — {tx_id} ═══",
            "",
            f"[Décision] {opening} avec une probabilité de fraude de {prob_pct:.1f}% "
            f"(niveau de risque : {risk}).",
            "",
        ]

        # Explication des features SHAP
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

        # Alertes de surveillance
        if surv.get('reasons'):
            rules_text = "; ".join(surv['reasons'][:3])
            lines.append(f"[Règles] L'agent de surveillance a détecté : {rules_text}.")
            lines.append("")

        # Action
        lines.append(f"[Action] {action}")
        if decision.get('notify'):
            lines.append(f"[Notification] Parties alertées : {', '.join(decision['notify'])}.")

        lines.append("")
        lines.append("[Note] Rapport généré en mode hors-ligne (sans API LLM).")

        return '\n'.join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitaire
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_error(e: Exception) -> str:
        """Formate un message d'erreur clair selon le code HTTP."""
        err = str(e)
        if '403' in err or 'denied' in err.lower() or 'permission' in err.lower():
            return (
                "403 Accès refusé — clé API invalide ou projet non autorisé.\n"
                "    → Vérifiez votre clé sur https://aistudio.google.com/apikey\n"
                "    → Les nouvelles clés AI Studio commencent par 'AQ.' (SDK google-genai)"
            )
        if '401' in err or 'unauthorized' in err.lower():
            return "401 Non autorisé — clé API incorrecte ou expirée."
        if '429' in err or 'quota' in err.lower() or 'rate_limit' in err.lower():
            return "429 Quota dépassé — attendez quelques secondes et réessayez."
        if '404' in err or 'not found' in err.lower():
            return f"404 Modèle introuvable : vérifiez le nom du modèle ({err[:120]})"
        return err[:200]
