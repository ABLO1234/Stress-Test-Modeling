"""
=============================================================================
MODULE 3 : Construction des scénarios de stress macroéconomique
=============================================================================
Auteur    : Stress Test Liquidité
Description : Définition et calibration de 4 scénarios de stress conformes
              aux meilleures pratiques EBA/BCE/FMI.

Méthodologie de calibration :
- Scénario normal       : baseline + légères fluctuations cycliques
- Scénario adverse      : calibré sur crises régionales (Mali 2012, UEMOA)
- Scénario sévère       : calibré sur crises systémiques (Afrique 2015-16)
- Scénario crise syst.  : calibré sur Lehman 2008 adapté UEMOA / COVID 2020

Chaque scénario spécifie :
1. Les chocs sur variables macroéconomiques (écarts par rapport au baseline)
2. Les taux de fuite des dépôts (comportement des déposants)
3. Les facteurs multiplicateurs sur les sorties de liquidité
4. La durée estimée du choc (en mois)
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. STRUCTURE D'UN SCÉNARIO
# =============================================================================

@dataclass
class StressScenario:
    """
    Représente un scénario de stress complet avec ses hypothèses macro
    et comportementales.
    
    Les chocs sont exprimés comme écarts absolus ou multiplicateurs
    par rapport au scénario baseline.
    """
    name: str
    label: str                          # Identifiant court (ex: 'adverse')
    color: str                          # Couleur pour les graphiques
    description: str
    
    # --- Chocs macroéconomiques (valeurs absolues ciblées) ---
    pib_growth_target:  float           # Croissance PIB cible (%)
    inflation_target:   float           # Inflation cible (%)
    chomage_target:     float           # Chômage cible (%)
    taux_court_target:  float           # Taux court terme cible (%)
    taux_long_target:   float           # Taux long terme cible (%)
    taux_change_shock:  float           # Choc sur taux de change (% de dépréciation)
    
    # --- Comportement des déposants (taux de fuite Bâle III) ---
    fuite_depots_vue:   float           # % des dépôts à vue fuyant en 30 jours
    fuite_depots_terme: float           # % des dépôts à terme fuyant en 30 jours
    fuite_refinancement: float          # % du refinancement non renouvelé
    
    # --- Décotes sur actifs (haircuts HQLA) ---
    haircut_souverains: float = 0.00    # Décote supplémentaire sur titres souverains
    haircut_prives:     float = 0.00    # Décote supplémentaire sur titres privés
    
    # --- Paramètres de durée ---
    duree_choc_mois:    int = 3         # Durée estimée du stress (mois)
    
    # --- Probabilité subjective d'occurrence ---
    proba_occurrence:   float = 0.10    # Probabilité annuelle estimée
    
    def to_dict(self) -> dict:
        return {
            'Scénario':          self.name,
            'PIB (%/an)':        self.pib_growth_target,
            'Inflation (%)':     self.inflation_target,
            'Chômage (%)':       self.chomage_target,
            'Taux court (%)':    self.taux_court_target,
            'Taux long (%)':     self.taux_long_target,
            'Dépréc. change (%)': self.taux_change_shock,
            'Fuite DàV (%)':     self.fuite_depots_vue * 100,
            'Fuite DàT (%)':     self.fuite_depots_terme * 100,
            'Fuite refi (%)':    self.fuite_refinancement * 100,
            'Décote souv. (%)':  self.haircut_souverains * 100,
            'Durée (mois)':      self.duree_choc_mois,
            'Proba occ. (%)':    self.proba_occurrence * 100,
        }


# =============================================================================
# 2. DÉFINITION DES 4 SCÉNARIOS
# =============================================================================

def define_scenarios() -> Dict[str, StressScenario]:
    """
    Définit les 4 scénarios de stress.
    
    Calibration basée sur :
    - Données historiques BCEAO / BCRG / BEAC
    - EBA Adverse Scenario methodology
    - IMF Stress Testing Manual (Li & Zanforlin 2020)
    - Crises spécifiques UEMOA : Mali 2012, Côte d'Ivoire 2011, COVID 2020
    
    Returns:
        Dictionnaire {label: StressScenario}
    """
    scenarios = {}
    
    # -------------------------------------------------------------------------
    # SCÉNARIO 0 : NORMAL (Baseline)
    # Conditions macroéconomiques favorables, croissance soutenue UEMOA
    # Référence : moyenne historique BCEAO 2017-2019 (avant chocs)
    # -------------------------------------------------------------------------
    scenarios['normal'] = StressScenario(
        name='Scénario normal',
        label='normal',
        color='#1a9641',
        description=(
            "Conditions macroéconomiques conformes aux projections de base BCEAO. "
            "Croissance inclusive, inflation maîtrisée sous le seuil de 3%, "
            "stabilité financière régionale. Comportement des déposants normal."
        ),
        pib_growth_target   = 5.5,
        inflation_target    = 2.2,
        chomage_target      = 7.5,
        taux_court_target   = 3.5,
        taux_long_target    = 5.0,
        taux_change_shock   = 0.0,
        fuite_depots_vue    = 0.05,     # Taux Bâle III standard retail
        fuite_depots_terme  = 0.02,
        fuite_refinancement = 0.25,
        haircut_souverains  = 0.00,
        duree_choc_mois     = 1,
        proba_occurrence    = 0.70,     # Scenario le plus probable
    )
    
    # -------------------------------------------------------------------------
    # SCÉNARIO 1 : ADVERSE (Ralentissement modéré)
    # Calibré sur : ralentissement économique Mali/UEMOA 2016-2017
    # Chocs : -2pp PIB, +1.5pp inflation, resserrement taux directeur
    # -------------------------------------------------------------------------
    scenarios['adverse'] = StressScenario(
        name='Scénario adverse',
        label='adverse',
        color='#fdae61',
        description=(
            "Ralentissement économique significatif : chute des cours des matières "
            "premières (or, coton), contraction budgétaire, resserrement monétaire. "
            "Début de tensions sur les dépôts, pression sur le refinancement. "
            "Référence historique : UEMOA 2016-2017, chocs termes de l'échange."
        ),
        pib_growth_target   = 3.0,      # -2.5pp vs baseline
        inflation_target    = 4.5,      # +2.3pp (choc offre)
        chomage_target      = 10.0,     # +2.5pp
        taux_court_target   = 5.0,      # +1.5pp (resserrement BCEAO)
        taux_long_target    = 7.0,      # +2pp (prime de risque)
        taux_change_shock   = 5.0,      # -5% XOF (dépréciation EUR)
        fuite_depots_vue    = 0.10,     # 2× le taux normal
        fuite_depots_terme  = 0.05,
        fuite_refinancement = 0.40,
        haircut_souverains  = 0.00,
        duree_choc_mois     = 3,
        proba_occurrence    = 0.20,
    )
    
    # -------------------------------------------------------------------------
    # SCÉNARIO 2 : SÉVÈRE (Crise de confiance + chocs externes)
    # Calibré sur : crise socio-politique Mali 2012, COVID West Africa 2020
    # Chocs : -5pp PIB, tensions bancaires, flight-to-safety
    # -------------------------------------------------------------------------
    scenarios['severe'] = StressScenario(
        name='Scénario sévère',
        label='severe',
        color='#f46d43',
        description=(
            "Crise de confiance combinant choc politique et choc financier externe. "
            "Crise socio-politique déclenchant une ruée bancaire partielle, combinée "
            "à une dégradation des finances publiques et une montée des NPL. "
            "Référence : Mali mars 2012, COVID-19 impact UEMOA Q2 2020."
        ),
        pib_growth_target   = 0.5,      # Quasi-stagnation
        inflation_target    = 8.0,      # Choc majeur des prix alimentaires
        chomage_target      = 14.0,
        taux_court_target   = 7.0,
        taux_long_target    = 10.0,
        taux_change_shock   = 15.0,     # Pression sur ancrage CFA
        fuite_depots_vue    = 0.20,     # 4× le taux normal — début de bank run
        fuite_depots_terme  = 0.12,
        fuite_refinancement = 0.65,
        haircut_souverains  = 0.05,     # Décote sur dettes souveraines
        haircut_prives      = 0.10,
        duree_choc_mois     = 6,
        proba_occurrence    = 0.08,
    )
    
    # -------------------------------------------------------------------------
    # SCÉNARIO 3 : CRISE SYSTÉMIQUE (Stress test extrême)
    # Calibré sur : Lehman Brothers 2008 adapté UEMOA + contagion régionale
    # Scénario de "queue de distribution" — probabilité faible, impact maximal
    # -------------------------------------------------------------------------
    scenarios['systemic'] = StressScenario(
        name='Crise systémique',
        label='systemic',
        color='#d73027',
        description=(
            "Scénario de queue extrême : effondrement de la confiance dans le "
            "système bancaire régional, contagion inter-bancaire, assèchement "
            "du marché monétaire UMOA, intervention d'urgence BCEAO. "
            "Analogue à Lehman 2008 calibré sur la structure financière UEMOA. "
            "Ce scénario teste la résilience maximale du bilan."
        ),
        pib_growth_target   = -3.0,     # Récession profonde
        inflation_target    = 14.0,     # Hyperinflation importée + offre
        chomage_target      = 22.0,
        taux_court_target   = 12.0,     # Crise de liquidité interbancaire
        taux_long_target    = 18.0,
        taux_change_shock   = 35.0,     # Dévaluation de fait / spéculation
        fuite_depots_vue    = 0.40,     # Bank run massif
        fuite_depots_terme  = 0.25,
        fuite_refinancement = 0.90,     # Marché monétaire asséché
        haircut_souverains  = 0.15,     # Défaut partiel souverain
        haircut_prives      = 0.30,
        duree_choc_mois     = 12,
        proba_occurrence    = 0.02,
    )
    
    return scenarios


# =============================================================================
# 3. APPLICATION DES SCÉNARIOS AU BILAN
# =============================================================================

def apply_scenario_to_balance_sheet(bilan_row: pd.Series,
                                     scenario: StressScenario) -> dict:
    """
    Applique un scénario de stress à une ligne du bilan bancaire.
    
    Calcule les flux sous stress et les indicateurs réglementaires LCR/NSFR.
    
    Args:
        bilan_row  : Une ligne du DataFrame bilan (situation initiale)
        scenario   : Le scénario de stress à appliquer
    
    Returns:
        Dictionnaire avec les indicateurs sous stress
    """
    # Flux sortants sous stress
    sorties_dav   = bilan_row['depots_vue']    * scenario.fuite_depots_vue
    sorties_dat   = bilan_row['depots_terme']  * scenario.fuite_depots_terme
    sorties_refi  = bilan_row['refinancement'] * scenario.fuite_refinancement
    sorties_hors_bilan = bilan_row.get('credits', 400) * 0.03  # appels de garanties
    
    total_sorties = sorties_dav + sorties_dat + sorties_refi + sorties_hors_bilan
    
    # Entrées sous stress (plafonnées à 75% des sorties — règle Bâle III)
    entrees_base = bilan_row.get('entrees_30j', 0)
    entrees_stress = entrees_base * 0.5  # Les remboursements ralentissent
    entrees_plafonnees = min(entrees_stress, 0.75 * total_sorties)
    
    flux_nets_stress = total_sorties - entrees_plafonnees
    
    # HQLA sous stress (avec décotes supplémentaires)
    hqla_stress = (
        bilan_row.get('reserves_bc', 25) * 1.00                                  # toujours 100%
        + bilan_row.get('titres_souverains', 80) * (1.00 - scenario.haircut_souverains)
        + bilan_row.get('titres_prives', 30) * (0.85 - scenario.haircut_prives)
    )
    hqla_stress = max(hqla_stress, 0)
    
    # LCR sous stress
    lcr_stress = (hqla_stress / flux_nets_stress * 100) if flux_nets_stress > 0 else 999.0
    
    # NSFR simplifié (financement stable disponible vs requis)
    # FSD = capital + dépôts stables + dettes LT
    capital_tier1 = bilan_row.get('hqla', 100) * 0.12  # approximation
    fsd = (capital_tier1 * 1.00
           + bilan_row['depots_terme'] * (1 - scenario.fuite_depots_terme) * 0.90
           + bilan_row['depots_vue'] * (1 - scenario.fuite_depots_vue) * 0.50)
    
    # FSR = actifs iliquides pondérés
    fsr = bilan_row.get('credits', 400) * 0.65 + bilan_row.get('titres_prives', 30) * 0.50
    nsfr_stress = (fsd / fsr * 100) if fsr > 0 else 999.0
    
    return {
        'scenario':           scenario.label,
        'scenario_name':      scenario.name,
        'total_sorties_30j':  total_sorties,
        'entrees_30j':        entrees_plafonnees,
        'flux_nets_stress':   flux_nets_stress,
        'hqla_stress':        hqla_stress,
        'lcr_stress':         lcr_stress,
        'nsfr_stress':        nsfr_stress,
        'lcr_conforme':       lcr_stress >= 100.0,
        'nsfr_conforme':      nsfr_stress >= 100.0,
    }


def apply_all_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les 4 scénarios à chaque période du dataset.
    
    Returns:
        DataFrame long avec scénario × date × indicateurs
    """
    scenarios = define_scenarios()
    results = []
    
    # On utilise la dernière observation comme situation bilancielle initiale
    last_row = df.iloc[-1]
    
    print("\n" + "="*70)
    print("RÉSULTATS PAR SCÉNARIO DE STRESS (situation au", df.index[-1].strftime('%Y-%m'), ")")
    print("="*70)
    
    for label, scen in scenarios.items():
        res = apply_scenario_to_balance_sheet(last_row, scen)
        results.append(res)
        status_lcr  = "✓ CONFORME" if res['lcr_conforme']  else "✗ BREACH"
        status_nsfr = "✓ CONFORME" if res['nsfr_conforme'] else "✗ BREACH"
        print(f"\n{scen.name.upper()}")
        print(f"  Sorties 30j      : {res['total_sorties_30j']:.1f} Mds XOF")
        print(f"  HQLA sous stress : {res['hqla_stress']:.1f} Mds XOF")
        print(f"  LCR              : {res['lcr_stress']:.1f}%  {status_lcr}")
        print(f"  NSFR             : {res['nsfr_stress']:.1f}%  {status_nsfr}")
    
    return pd.DataFrame(results)


# =============================================================================
# 4. TABLEAU COMPARATIF DES SCÉNARIOS
# =============================================================================

def print_scenario_summary():
    """Affiche le tableau comparatif de tous les scénarios."""
    scenarios = define_scenarios()
    rows = [s.to_dict() for s in scenarios.values()]
    df_summary = pd.DataFrame(rows)
    
    print("\nTABLEAU COMPARATIF DES SCÉNARIOS DE STRESS")
    print("="*90)
    print(df_summary.to_string(index=False))
    return df_summary


# =============================================================================
# 5. VISUALISATION DES SCÉNARIOS
# =============================================================================

def plot_scenarios(save_path: str = None):
    """
    Visualise les chocs macroéconomiques par scénario sous forme de radar chart
    et de tableau comparatif.
    """
    scenarios = define_scenarios()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scénarios de Stress — Hypothèses Macroéconomiques',
                 fontsize=15, fontweight='bold')
    
    scenario_list = list(scenarios.values())
    variables  = ['PIB (%)', 'Inflation (%)', 'Chômage (%)',
                  'Taux court (%)', 'Taux long (%)', 'Fuite DàV (%)']
    
    for idx, (ax, scen) in enumerate(zip(axes.flatten(), scenario_list)):
        vals = [
            scen.pib_growth_target,
            scen.inflation_target,
            scen.chomage_target,
            scen.taux_court_target,
            scen.taux_long_target,
            scen.fuite_depots_vue * 100,
        ]
        bars = ax.barh(variables, vals, color=scen.color, alpha=0.80, edgecolor='white')
        
        # Valeurs baseline (scénario normal)
        baseline_scen = scenarios['normal']
        baselines = [
            baseline_scen.pib_growth_target,
            baseline_scen.inflation_target,
            baseline_scen.chomage_target,
            baseline_scen.taux_court_target,
            baseline_scen.taux_long_target,
            baseline_scen.fuite_depots_vue * 100,
        ]
        if idx > 0:  # Pas de ligne baseline pour le scénario normal lui-même
            ax.barh(variables, baselines, color='#cccccc', alpha=0.4,
                    edgecolor='white', label='Baseline')
        
        ax.set_title(f'{scen.name}\n({scen.proba_occurrence*100:.0f}% prob. occurrence)',
                     fontsize=11, fontweight='bold', color=scen.color)
        ax.set_xlabel('Valeur (%)')
        
        # Annotations
        for bar, val in zip(bars, vals):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', ha='left', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, max(vals) * 1.2 + 2)
        
        if idx > 0:
            ax.legend(fontsize=8, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graphique sauvegardé : {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'C:/Users/X1 Carbon/Desktop/Stress Testing Projet/stress_test_liquidite')
    from simulate_data import create_full_dataset
    
    df = create_full_dataset()
    
    print("\nDéfinition et application des scénarios...")
    summary_df = print_scenario_summary()
    results_df = apply_all_scenarios(df)
    
    print("\nGénération des visualisations...")
    plot_scenarios(save_path='C:/Users/X1 Carbon/Desktop/Stress Testing Projet/scenarios.png')
    
    print("\n[TERMINÉ] Module Scénarios exécuté avec succès.")
