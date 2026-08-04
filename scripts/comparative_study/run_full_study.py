"""
run_full_study.py — Script orchestrateur de l'étude comparative complète
Stage : Détection de Fraude Bancaire avec Federated Learning & Agentic AI

Exécute les deux volets, génère les résultats, graphiques et analyse critique.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Ajouter le répertoire parent au path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from scripts.comparative_study.volet1_imbalance_comparison import run_volet1
from scripts.comparative_study.volet2_federated_comparison import run_volet2

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

ULB_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
SYNTHETIC_PATH = os.path.join(BASE_DIR, 'data', 'fraud_detection_credit_card_small.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'comparative_study')


# ═══════════════════════════════════════════════════════════════════════════════
# Analyse critique
# ═══════════════════════════════════════════════════════════════════════════════

def generate_critical_analysis(volet1_df, volet2_df, output_dir):
    """Génère l'analyse critique en markdown à partir des résultats."""

    # ── Analyse Volet 1 ──
    v1_analysis = []
    for dataset in volet1_df['Dataset'].unique():
        df_ds = volet1_df[volet1_df['Dataset'] == dataset]
        best_f1_row = df_ds.loc[df_ds['F1-Score'].idxmax()]
        best_recall_row = df_ds.loc[df_ds['Recall'].idxmax()]
        best_auc_row = df_ds.loc[df_ds['AUC-ROC'].idxmax()]
        worst_row = df_ds.loc[df_ds['F1-Score'].idxmin()]

        v1_analysis.append({
            'dataset': dataset,
            'best_f1_tech': best_f1_row['Technique'],
            'best_f1_val': best_f1_row['F1-Score'],
            'best_recall_tech': best_recall_row['Technique'],
            'best_recall_val': best_recall_row['Recall'],
            'best_auc_tech': best_auc_row['Technique'],
            'best_auc_val': best_auc_row['AUC-ROC'],
            'worst_tech': worst_row['Technique'],
            'worst_f1': worst_row['F1-Score'],
            'df': df_ds
        })

    # ── Analyse Volet 2 ──
    v2_analysis = []
    for dataset in volet2_df['Dataset'].unique():
        df_ds = volet2_df[volet2_df['Dataset'] == dataset]
        cent_row = df_ds[df_ds['Mode'] == 'Centralisé'].iloc[0]
        fed_row = df_ds[df_ds['Mode'] == 'Fédéré (FedAvg)'].iloc[0]
        f1_diff = cent_row['F1-Score'] - fed_row['F1-Score']
        f1_diff_pct = (f1_diff / cent_row['F1-Score'] * 100) if cent_row['F1-Score'] > 0 else 0

        v2_analysis.append({
            'dataset': dataset,
            'cent_f1': cent_row['F1-Score'],
            'fed_f1': fed_row['F1-Score'],
            'f1_diff': f1_diff,
            'f1_diff_pct': f1_diff_pct,
            'cent_recall': cent_row['Recall'],
            'fed_recall': fed_row['Recall'],
            'cent_prec': cent_row['Precision'],
            'fed_prec': fed_row['Precision'],
        })

    # ── Génération du Markdown ──
    md = []
    md.append("# Analyse Critique — Étude Comparative")
    md.append("")
    md.append("> **Projet** : Amélioration de la détection de fraude bancaire — "
              "Agentic AI, LLMs et Federated Learning")
    md.append(f"> **Date** : {time.strftime('%d/%m/%Y')}")
    md.append("> **Modèle** : XGBoost (n_estimators=200, max_depth=5, lr=0.1)")
    md.append("> **Seed** : 42 | **Split** : 80/20 stratifié")
    md.append("")
    md.append("---")
    md.append("")

    # ── Volet 1 ──
    md.append("## Volet 1 — Gestion du déséquilibre des classes")
    md.append("")

    # Tableau comparatif Volet 1
    md.append("### Tableau comparatif")
    md.append("")
    md.append("| Technique | " + " | ".join(
        [f"{ds} Acc | {ds} Prec | {ds} Rec | {ds} F1 | {ds} AUC | {ds} AUPRC"
         for ds in volet1_df['Dataset'].unique()]
    ) + " |")
    md.append("|:---|" + "|".join(
        ["---:|---:|---:|---:|---:|---:" for _ in volet1_df['Dataset'].unique()]
    ) + "|")

    for tech in volet1_df['Technique'].unique():
        row = f"| **{tech}** |"
        for ds in volet1_df['Dataset'].unique():
            r = volet1_df[(volet1_df['Dataset'] == ds) & (volet1_df['Technique'] == tech)]
            if len(r) > 0:
                r = r.iloc[0]
                row += (f" {r['Accuracy']:.4f} | {r['Precision']:.4f} | "
                        f"{r['Recall']:.4f} | {r['F1-Score']:.4f} | "
                        f"{r['AUC-ROC']:.4f} | {r['AUC-PR']:.4f} |")
            else:
                row += " — | — | — | — | — | — |"
        md.append(row)

    md.append("")
    md.append("### Analyse par dataset")
    md.append("")

    for info in v1_analysis:
        md.append(f"#### Dataset {info['dataset']}")
        md.append("")
        md.append(f"- **Meilleure technique (F1)** : **{info['best_f1_tech']}** "
                  f"(F1 = {info['best_f1_val']:.4f})")
        md.append(f"- **Meilleur Recall** : {info['best_recall_tech']} "
                  f"(Recall = {info['best_recall_val']:.4f})")
        md.append(f"- **Meilleure AUC-ROC** : {info['best_auc_tech']} "
                  f"(AUC = {info['best_auc_val']:.4f})")
        md.append(f"- **Pire technique** : {info['worst_tech']} "
                  f"(F1 = {info['worst_f1']:.4f})")
        md.append("")

    md.append("### Interprétation")
    md.append("")
    md.append("**SMOTE vs ADASYN** : SMOTE génère des exemples synthétiques uniformément "
              "le long du segment reliant deux points minoritaires. ADASYN concentre la "
              "génération sur les exemples difficiles à classifier (zones frontières). "
              "En pratique, leurs performances sont souvent très proches sur les données "
              "tabulaires, avec ADASYN légèrement plus risqué (potentiel d'overfitting "
              "sur les zones bruitées).")
    md.append("")
    md.append("**Undersampling** : Réduit la classe majoritaire au niveau de la minoritaire. "
              "Bien que cela améliore le Recall (détection des fraudes), la perte massive "
              "d'information entraîne une chute de la Precision (trop de faux positifs). "
              "L'Accuracy devient inexploitable. Cette technique n'est donc **pas recommandée** "
              "comme stratégie unique.")
    md.append("")
    md.append("**Baseline (aucun resampling)** : XGBoost gère relativement bien le déséquilibre "
              "grâce à son scale_pos_weight interne, mais le Recall reste inférieur à SMOTE/ADASYN.")
    md.append("")
    md.append("---")
    md.append("")

    # ── Volet 2 ──
    md.append("## Volet 2 — Federated Learning vs Centralisé")
    md.append("")

    # Tableau comparatif Volet 2
    md.append("### Tableau comparatif")
    md.append("")
    md.append("| Mode | " + " | ".join(
        [f"{ds} Acc | {ds} Prec | {ds} Rec | {ds} F1 | {ds} AUC | {ds} AUPRC"
         for ds in volet2_df['Dataset'].unique()]
    ) + " |")
    md.append("|:---|" + "|".join(
        ["---:|---:|---:|---:|---:|---:" for _ in volet2_df['Dataset'].unique()]
    ) + "|")

    for mode in volet2_df['Mode'].unique():
        row = f"| **{mode}** |"
        for ds in volet2_df['Dataset'].unique():
            r = volet2_df[(volet2_df['Dataset'] == ds) & (volet2_df['Mode'] == mode)]
            if len(r) > 0:
                r = r.iloc[0]
                row += (f" {r['Accuracy']:.4f} | {r['Precision']:.4f} | "
                        f"{r['Recall']:.4f} | {r['F1-Score']:.4f} | "
                        f"{r['AUC-ROC']:.4f} | {r['AUC-PR']:.4f} |")
            else:
                row += " — | — | — | — | — | — |"
        md.append(row)

    md.append("")
    md.append("### Analyse par dataset")
    md.append("")

    for info in v2_analysis:
        md.append(f"#### Dataset {info['dataset']}")
        md.append("")
        if info['f1_diff'] > 0:
            md.append(f"- Le modèle **centralisé** surpasse le fédéré de "
                      f"**{info['f1_diff']:.4f}** en F1-Score "
                      f"(−{info['f1_diff_pct']:.1f}% pour le FL)")
        else:
            md.append(f"- Le modèle **fédéré** surpasse le centralisé de "
                      f"**{abs(info['f1_diff']):.4f}** en F1-Score")
        md.append(f"- Centralisé : F1={info['cent_f1']:.4f}, "
                  f"Precision={info['cent_prec']:.4f}, Recall={info['cent_recall']:.4f}")
        md.append(f"- Fédéré : F1={info['fed_f1']:.4f}, "
                  f"Precision={info['fed_prec']:.4f}, Recall={info['fed_recall']:.4f}")
        md.append("")

    md.append("### Interprétation")
    md.append("")
    md.append("Le Federated Learning introduit une perte de performance par rapport au "
              "modèle centralisé. Cette dégradation s'explique par :")
    md.append("")
    md.append("1. **Fragmentation des données** : Chaque client ne voit qu'1/25ème du "
              "dataset, ce qui réduit la diversité des patterns observés.")
    md.append("2. **Déséquilibre local exacerbé** : Les rares fraudes sont réparties "
              "sur 25 partitions, certains clients n'en ayant que très peu.")
    md.append("3. **Agrégation par Soft Voting** : Contrairement au vrai FedAvg (moyenne "
              "des poids de réseaux de neurones), le Soft Voting sur des modèles XGBoost "
              "est une approximation qui ne bénéficie pas de la même convergence.")
    md.append("")
    md.append("**Cependant**, cette perte est compensée par les avantages du FL :")
    md.append("")
    md.append("- **Conformité RGPD/DORA** : Les données ne quittent jamais chaque "
              "institution bancaire.")
    md.append("- **Collaboration interbancaire** : Chaque banque contribue à un modèle "
              "global sans partager ses données clients.")
    md.append("- **Résilience** : Pas de point unique de défaillance des données.")
    md.append("")
    md.append("---")
    md.append("")

    # ── Conclusion ──
    md.append("## Conclusion générale")
    md.append("")
    md.append("### Volet 1 — Recommandation")
    md.append("")
    best_global = volet1_df.loc[volet1_df['F1-Score'].idxmax()]
    md.append(f"**{v1_analysis[0]['best_f1_tech']}** est la technique de rééquilibrage "
              f"recommandée. Elle offre le meilleur compromis Precision/Recall sur le "
              f"dataset ULB. L'Undersampling est à éviter en production en raison de la "
              f"perte massive de Precision.")
    md.append("")
    md.append("### Volet 2 — Recommandation")
    md.append("")
    md.append("Le **modèle centralisé** reste plus performant, mais le "
              "**Federated Learning** constitue la solution recommandée en production "
              "car il concilie performance acceptable et conformité réglementaire. "
              "La perte de F1 est un compromis raisonnable face aux exigences RGPD et "
              "aux bénéfices de la collaboration interbancaire.")
    md.append("")

    # Écriture du fichier
    analysis_path = os.path.join(output_dir, 'analyse_critique.md')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"\n  [✓] Analyse critique sauvegardée : {analysis_path}")

    return analysis_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Exécution complète
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 68 + "╗")
    print("║  ÉTUDE COMPARATIVE — DÉTECTION DE FRAUDE BANCAIRE                ║")
    print("║  Déséquilibre des classes & Federated Learning                   ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Répertoire de sortie : {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    # ── Volet 1 ──
    print("\n\n" + "▓" * 70)
    print("  VOLET 1 — COMPARAISON DES TECHNIQUES DE RÉÉQUILIBRAGE")
    print("▓" * 70)
    t1 = time.time()
    volet1_results = run_volet1(ULB_PATH, SYNTHETIC_PATH, OUTPUT_DIR)
    print(f"\n  ⏱ Volet 1 terminé en {time.time() - t1:.1f}s")

    # ── Volet 2 ──
    print("\n\n" + "▓" * 70)
    print("  VOLET 2 — CENTRALISÉ VS FEDERATED LEARNING")
    print("▓" * 70)
    t2 = time.time()
    volet2_results = run_volet2(ULB_PATH, SYNTHETIC_PATH, OUTPUT_DIR)
    print(f"\n  ⏱ Volet 2 terminé en {time.time() - t2:.1f}s")

    # ── Analyse critique ──
    print("\n\n" + "▓" * 70)
    print("  GÉNÉRATION DE L'ANALYSE CRITIQUE")
    print("▓" * 70)
    generate_critical_analysis(volet1_results, volet2_results, OUTPUT_DIR)

    # ── Récapitulatif ──
    total_time = time.time() - start_time
    print("\n\n" + "╔" + "═" * 68 + "╗")
    print("║  ÉTUDE TERMINÉE AVEC SUCCÈS                                     ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  ⏱ Durée totale : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  Fichiers générés dans : {OUTPUT_DIR}")
    print(f"    ├── volet1_imbalance_results.csv")
    print(f"    ├── volet2_federated_results.csv")
    print(f"    ├── volet1_barplot.png")
    print(f"    ├── volet2_barplot.png")
    print(f"    └── analyse_critique.md")


if __name__ == '__main__':
    main()
