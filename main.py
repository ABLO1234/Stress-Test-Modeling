"""
=============================================================================
PIPELINE PRINCIPAL — Stress Test de Liquidité Bancaire
=============================================================================
Auteur    : Stress Test Liquidité
Description : Orchestre l'exécution complète du pipeline :
              Données → ML → Scénarios → Monte-Carlo → Analyse avancée → Rapport

Usage :
    python main.py

Sorties générées dans /tmp/stress_test_output/ :
    - 01_data_overview.png
    - 02_ml_results.png
    - 03_scenarios.png
    - 04_monte_carlo.png
    - 05_advanced_analysis.png
    - risk_measures.csv
    - model_validation.json
    - rapport_risques.txt
=============================================================================
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
sys.path.insert(0, 'C:/Users/X1 Carbon/Desktop/Stress Testing Projet/stress_test_liquidite')

# Répertoire de sortie
OUTPUT_DIR = 'C:/Users/X1 Carbon/Desktop/Stress Testing Projet/Out_puts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  STRESS TEST DE LIQUIDITÉ BANCAIRE")
print("  Projet complet — Économie Quantitative & Calculable")
print("=" * 70)

# =============================================================================
# ÉTAPE 1 : DONNÉES
# =============================================================================
print("\n[1/6] Génération de la base de données bancaire...")
t0 = time.time()

from simulate_data import create_full_dataset
df = create_full_dataset()
df.to_csv(f'{OUTPUT_DIR}/data_bancaire.csv')

# Graphique aperçu des données
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Aperçu des données bancaires simulées', fontsize=14, fontweight='bold')

plots = [
    ('pib_growth', 'Croissance PIB (%/an)', '#1a9641'),
    ('inflation',  'Inflation (%)',          '#d7191c'),
    ('depots_vue', 'Dépôts à vue (Mds XOF)', '#2c7bb6'),
    ('hqla',       'HQLA (Mds XOF)',          '#7b3294'),
    ('flux_nets_30j', 'Flux nets 30j (Mds)', '#f46d43'),
    ('ratio_transformation', 'Ratio transformation', '#fdae61'),
]

for ax, (col, title, color) in zip(axes.flatten(), plots):
    if col in df.columns:
        ax.plot(df.index, df[col], color=color, linewidth=2)
        ax.fill_between(df.index, df[col].min(), df[col], alpha=0.15, color=color)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Données générées ({len(df)} obs × {len(df.columns)} variables)")
print(f"  ✓ Temps : {time.time()-t0:.1f}s")

# =============================================================================
# ÉTAPE 2 : MACHINE LEARNING
# =============================================================================
print("\n[2/6] Entraînement des modèles Machine Learning...")
t0 = time.time()

from ml_models import (prepare_features, evaluate_models,
                               train_best_model, compute_feature_importance,
                               plot_model_results)

X, y, feature_names = prepare_features(df)
results_df = evaluate_models(X, y, feature_names, n_splits=4)
model, scaler, model_name = train_best_model(X, y,results_df)
importance_df = compute_feature_importance(model, X, y, feature_names, scaler)

fig_ml = plot_model_results(df, model, scaler, feature_names,
                             importance_df, results_df,
                             save_path=f'{OUTPUT_DIR}/02_ml_results.png')
plt.close()
print(f"  ✓ Meilleur modèle : {model_name}")
print(f"  ✓ Temps : {time.time()-t0:.1f}s")

# =============================================================================
# ÉTAPE 3 : SCÉNARIOS DE STRESS
# =============================================================================
print("\n[3/6] Construction des scénarios de stress...")
t0 = time.time()

from stress_scenarios import (define_scenarios, apply_all_scenarios,
                                         print_scenario_summary, plot_scenarios)

scenarios = define_scenarios()
summary_df = print_scenario_summary()
scenario_results_df = apply_all_scenarios(df)
scenario_results_df.to_csv(f'{OUTPUT_DIR}/scenario_results.csv', index=False)

fig_scen = plot_scenarios(save_path=f'{OUTPUT_DIR}/03_scenarios.png')
plt.close()
print(f"  ✓ {len(scenarios)} scénarios définis")
print(f"  ✓ Temps : {time.time()-t0:.1f}s")

# =============================================================================
# ÉTAPE 4 : SIMULATION MONTE-CARLO
# =============================================================================
print("\n[4/6] Simulation Monte-Carlo (10 000 scénarios × 4 stress)...")
t0 = time.time()

from monte_carlo import (run_monte_carlo, compute_risk_measures,
                                     plot_monte_carlo, interpret_results)

mc_results = run_monte_carlo(
    df=df,
    model=model,
    scaler=scaler,
    feature_names=feature_names,
    scenarios=scenarios,
    n_simulations=10_000,
    use_student_t=True,
    df_t=5.0,
)

risk_measures = compute_risk_measures(mc_results)
risk_measures.to_csv(f'{OUTPUT_DIR}/risk_measures.csv', index=False)

rapport_texte = interpret_results(risk_measures)
with open(f'{OUTPUT_DIR}/rapport_risques.txt', 'w', encoding='utf-8') as f:
    f.write(rapport_texte)

fig_mc = plot_monte_carlo(mc_results, risk_measures,
                           save_path=f'{OUTPUT_DIR}/04_monte_carlo.png')
plt.close()
print(f"  ✓ {10_000 * len(scenarios):,} simulations effectuées")
print(f"  ✓ Temps : {time.time()-t0:.1f}s")

# =============================================================================
# ÉTAPE 5 : ANALYSES AVANCÉES
# =============================================================================
print("\n[5/6] Analyses avancées (sensibilité, reverse stress, validation)...")
t0 = time.time()

from advanced_analysis import (sensitivity_analysis_v2,
                                       reverse_stress_test,
                                       validate_model,
                                       plot_advanced_analysis)

sensitivity   = sensitivity_analysis_v2(df, model, scaler, feature_names)
critical_lvls = reverse_stress_test(df, model, scaler, feature_names, lcr_threshold=80.0)
validation    = validate_model(df, X, y, feature_names)

# Sauvegarde JSON de la validation
val_json = {k: float(v) if isinstance(v, (np.floating, float)) else
            (bool(v) if isinstance(v, (np.bool_, bool)) else
             [float(x) for x in v] if isinstance(v, list) else v)
            for k, v in validation.items()}
with open(f'{OUTPUT_DIR}/model_validation.json', 'w') as f:
    json.dump(val_json, f, indent=2)

fig_adv = plot_advanced_analysis(sensitivity, critical_lvls, validation,
                                  save_path=f'{OUTPUT_DIR}/05_advanced_analysis.png')
plt.close()
print(f"  ✓ Temps : {time.time()-t0:.1f}s")

# =============================================================================
# ÉTAPE 6 : RAPPORT DE SYNTHÈSE
# =============================================================================
print("\n[6/6] Génération du rapport de synthèse...")

report_lines = []
report_lines.append("="*70)
report_lines.append("RAPPORT DE STRESS TEST DE LIQUIDITÉ BANCAIRE")
report_lines.append("Conforme Bâle III / BCEAO / EBA")
report_lines.append("="*70)

report_lines.append(f"\nDATE D'EXÉCUTION : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
report_lines.append(f"MODÈLE ML UTILISÉ : {model_name}")
report_lines.append(f"VALIDITÉ DU MODÈLE : R²={validation['mean_r2']:.4f}  RMSE={validation['mean_rmse']:.3f} Mds XOF")
report_lines.append(f"NOMBRE DE SIMULATIONS : 10 000 par scénario")

report_lines.append("\n" + "="*70)
report_lines.append("RÉSULTATS PAR SCÉNARIO")
report_lines.append("="*70)

for _, row in risk_measures.iterrows():
    report_lines.append(f"\n{row['Scénario'].upper()} :")
    report_lines.append(f"  LCR médian    : {row['LCR médian (%)']:.1f}%")
    report_lines.append(f"  VaR 99%       : {row['VaR 99% (Mds)']:.2f} Mds XOF")
    report_lines.append(f"  ES 99%        : {row['ES 99% (Mds)']:.2f} Mds XOF")
    report_lines.append(f"  P(LCR < 100%) : {row['P(LCR<100%) %']:.2f}%")

report_lines.append("\n" + "="*70)
report_lines.append("FICHIERS PRODUITS")
report_lines.append("="*70)

files_produced = [
    'data_bancaire.csv         — Dataset bancaire (60 obs × variables)',
    '01_data_overview.png      — Aperçu données historiques',
    '02_ml_results.png         — Résultats et validation des modèles ML',
    '03_scenarios.png          — Hypothèses macroéconomiques par scénario',
    '04_monte_carlo.png        — Dashboard Monte-Carlo complet',
    '05_advanced_analysis.png  — Sensibilité, Reverse Stress, Validation',
    'scenario_results.csv      — Résultats bilan par scénario',
    'risk_measures.csv         — Mesures de risque (VaR, ES, P(breach))',
    'model_validation.json     — Métriques de validation technique',
    'rapport_risques.txt       — Commentaire économique automatique',
]

for f in files_produced:
    report_lines.append(f"  {f}")

full_report = "\n".join(report_lines)
print(full_report)

with open(f'{OUTPUT_DIR}/rapport_synthese.txt', 'w', encoding='utf-8') as f:
    f.write(full_report)

print("\n" + "="*70)
print("  PIPELINE COMPLET EXÉCUTÉ AVEC SUCCÈS")
print(f"  Sorties disponibles dans : {OUTPUT_DIR}/")
print("="*70)
