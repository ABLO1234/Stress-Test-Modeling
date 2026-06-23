"""
=============================================================================
MODULE 1 : Simulation de la base de données bancaire
=============================================================================
Description : Génération d'une base de données bancaire synthétique réaliste
              couvrant les variables bilancielles et macroéconomiques sur
              60 périodes mensuelles (5 ans).
=============================================================================

"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Reproductibilité
# -----------------------------------------------------------------------------
np.random.seed(42)


def simulate_macro_variables(n_periods: int = 600) -> pd.DataFrame:
    """
    Simule les variables macroéconomiques avec autocorrélation (processus AR(1)).
    
    Hypothèses calibrées sur les données BCEAO / UMOA zone UEMOA :
    - Croissance du PIB : moyenne 5.5%, volatilité modérée
    - Inflation (IHPC) : moyenne 2.5%, objectif BCEAO < 3%
    - Taux de chômage : structurellement élevé en Afrique subsaharienne
    - Taux d'intérêt : Taux Directeur BCEAO (3.5% en 2023)
    - Taux de change XOF/USD : ancrage au franc CFA via EUR/USD
    
    Returns:
        DataFrame avec index temporel mensuel
    """
    dates = pd.date_range(start='2019-01-01', periods=n_periods, freq='ME')
    
    def ar1_process(mu, sigma, phi, n):
        """Processus AR(1) : X_t = mu(1-phi) + phi*X_{t-1} + sigma*epsilon_t"""
        x = np.zeros(n)
        x[0] = mu
        for i in range(1, n):
            x[i] = mu * (1 - phi) + phi * x[i-1] + sigma * np.random.normal()
        return x
    
    pib_growth  = ar1_process(mu=5.5,   sigma=1.2,  phi=0.7,  n=n_periods)
    inflation   = ar1_process(mu=2.5,   sigma=0.6,  phi=0.8,  n=n_periods)
    chomage     = ar1_process(mu=7.5,   sigma=0.4,  phi=0.9,  n=n_periods)
    taux_court  = ar1_process(mu=3.5,   sigma=0.3,  phi=0.85, n=n_periods)
    taux_long   = taux_court + ar1_process(mu=1.5, sigma=0.2, phi=0.8, n=n_periods)
    taux_change = ar1_process(mu=655.957, sigma=8.0,  phi=0.95, n=n_periods)  # XOF/USD
    
    # Contraintes : pas de valeurs négatives
    inflation   = np.clip(inflation, 0.1, 15.0)
    chomage     = np.clip(chomage,   2.0, 25.0)
    taux_court  = np.clip(taux_court, 0.5, 10.0)
    taux_long   = np.clip(taux_long,  1.0, 15.0)
    taux_change = np.clip(taux_change, 580.0, 740.0)
    
    return pd.DataFrame({
        'date':         dates,
        'pib_growth':   pib_growth,
        'inflation':    inflation,
        'chomage':      chomage,
        'taux_court':   taux_court,
        'taux_long':    taux_long,
        'taux_change':  taux_change,
    }).set_index('date')


def simulate_balance_sheet(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simule le bilan bancaire avec dépendance aux conditions macroéconomiques.
    
    La structure reproduit le bilan d'une banque commerciale de taille moyenne
    dans la zone UEMOA (environ 500 Mds XOF d'actifs).
    
    Variables clés (en milliards XOF) :
    - Dépôts à vue         : passif court terme, très volatils
    - Dépôts à terme       : passif moyen terme, plus stables
    - Encours de crédits   : actif principal, illiquide
    - Réserves liquides    : HQLA niveau 1 (caisse, BC)
    - Titres souverains    : HQLA niveau 1 (bons du Trésor UMOA)
    - Titres privés        : HQLA niveau 2A
    - Flux entrants        : remboursements, intérêts reçus
    - Flux sortants nets   : retraits, décaissements, refinancements
    
    La sensibilité aux chocs macro est modélisée via des élasticités.
    """
    n = len(macro_df)
    dates = macro_df.index
    
    # -------------------------------------------------------------------------
    # Actifs liquides (HQLA)
    # -------------------------------------------------------------------------
    # Réserves BC (caisse + dépôts obligatoires + excédents)
    reserves_bc = 25 + np.cumsum(np.random.normal(0.5, 1.5, n))
    reserves_bc = np.clip(reserves_bc, 8, 60)
    
    # Titres souverains UEMOA (niveau 1 HQLA)
    titres_souverains = 80 + np.cumsum(np.random.normal(0.3, 2.0, n))
    titres_souverains = np.clip(titres_souverains, 40, 130)
    
    # Titres privés investment grade (niveau 2A, décote 15%)
    titres_prives = 30 + np.cumsum(np.random.normal(0.1, 1.2, n))
    titres_prives = np.clip(titres_prives, 10, 60)
    
    # -------------------------------------------------------------------------
    # Passifs (dépôts)
    # -------------------------------------------------------------------------
    # Les dépôts réagissent positivement à la croissance du PIB
    # et négativement à l'inflation (effet richesse réel)
    pib_effect = 0.5 * (macro_df['pib_growth'].values - 5.5)
    inf_effect = -0.3 * (macro_df['inflation'].values - 2.5)
    
    depots_vue = (150 + np.cumsum(np.random.normal(1.0, 4.0, n))
                  + 2.0 * pib_effect + 1.5 * inf_effect)
    depots_vue = np.clip(depots_vue, 80, 280)
    
    depots_terme = (200 + np.cumsum(np.random.normal(0.8, 3.0, n))
                    + 1.0 * pib_effect - 0.5 * (macro_df['taux_court'].values - 3.5))
    depots_terme = np.clip(depots_terme, 100, 380)
    
    # Refinancement marché (certificats de dépôt, emprunts interbancaires)
    refinancement = (60 + np.cumsum(np.random.normal(0.2, 2.5, n))
                     + 1.5 * (macro_df['taux_court'].values - 3.5))
    refinancement = np.clip(refinancement, 15, 120)
    
    # -------------------------------------------------------------------------
    # Actifs illiquides (crédits)
    # -------------------------------------------------------------------------
    credits = (350 + np.cumsum(np.random.normal(1.5, 3.5, n))
               + 3.0 * pib_effect - 2.0 * (macro_df['taux_long'].values - 5.0))
    credits = np.clip(credits, 200, 550)
    
    # -------------------------------------------------------------------------
    # Flux de trésorerie sur 30 jours (modèle LCR)
    # -------------------------------------------------------------------------
    # Sorties contractuelles (Bâle III taux de fuite standards)
    sorties_depots_vue   = depots_vue * 0.05   # 5% fuite sur dépôts retail
    sorties_depots_terme = depots_terme * 0.02  # 2% fuite sur dépôts stables
    sorties_refin        = refinancement * 0.25 # 25% non-renouvellement
    sorties_autres       = 8 + np.random.normal(0, 1.5, n)  # décaissements divers
    sorties_autres       = np.clip(sorties_autres, 2, 20)
    
    sorties_nettes_30j = (sorties_depots_vue + sorties_depots_terme
                          + sorties_refin + sorties_autres)
    
    # Entrées contractuelles (plafonné à 75% des sorties — règle Bâle III)
    entrees_credits = credits * 0.015  # 1.5% remboursements mensuels
    entrees_interets = (credits * macro_df['taux_long'].values / 100 / 12)
    entrees_brutes = entrees_credits + entrees_interets
    entrees_30j = np.minimum(entrees_brutes, 0.75 * sorties_nettes_30j)
    
    flux_nets_30j = sorties_nettes_30j - entrees_30j
    
    # -------------------------------------------------------------------------
    # HQLA calculés (avec décotes Bâle III)
    # -------------------------------------------------------------------------
    hqla = (reserves_bc * 1.00          # niveau 1 : 100%
            + titres_souverains * 1.00   # niveau 1 : 100%
            + titres_prives * 0.85)      # niveau 2A : 85%
    
    # -------------------------------------------------------------------------
    # Variable cible : ratio sorties / actifs liquides (stress proxy)
    # -------------------------------------------------------------------------
    ratio_stress = flux_nets_30j / (hqla + 1e-6)
    
    return pd.DataFrame({
        # Actifs liquides
        'reserves_bc':         reserves_bc,
        'titres_souverains':   titres_souverains,
        'titres_prives':       titres_prives,
        'hqla':                hqla,
        # Passifs
        'depots_vue':          depots_vue,
        'depots_terme':        depots_terme,
        'refinancement':       refinancement,
        # Actifs illiquides
        'credits':             credits,
        # Flux
        'sorties_nettes_30j':  sorties_nettes_30j,
        'entrees_30j':         entrees_30j,
        'flux_nets_30j':       flux_nets_30j,
        # Indicateurs
        'hqla_total':          hqla,
        'ratio_stress':        ratio_stress,
    }, index=dates)


def create_full_dataset() -> pd.DataFrame:
    """
    Assemble le dataset complet (macro + bilan) en un seul DataFrame.
    
    Returns:
        DataFrame de 600 lignes × 23 colonnes, prêt pour la modélisation ML.
    """
    macro = simulate_macro_variables(n_periods=600)
    bilan = simulate_balance_sheet(macro)
    df = pd.concat([macro, bilan], axis=1)
    
    # Variables lag (valeurs retardées — important pour la prédiction)
    for col in ['pib_growth', 'inflation', 'taux_court', 'depots_vue']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
    
    # Variation mensuelle des dépôts (signal de fuite)
    df['delta_depots_vue']   = df['depots_vue'].pct_change() * 100
    df['delta_depots_terme'] = df['depots_terme'].pct_change() * 100
    df['delta_refinancement'] = df['refinancement'].pct_change() * 100
    
    # Ratio de transformation (crédits / dépôts)
    df['ratio_transformation'] = df['credits'] / (df['depots_vue'] + df['depots_terme'])
    
    # Ratio de liquidité immédiate
    df['ratio_liquidite_immediate'] = df['hqla'] / df['depots_vue']
    
    df = df.dropna()
    
    print(f"Dataset créé : {df.shape[0]} observations × {df.shape[1]} variables")
    print(f"Période : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"\nStatistiques descriptives (variables clés) :")
    cols_display = ['pib_growth', 'inflation', 'depots_vue', 'hqla', 
                    'flux_nets_30j', 'ratio_stress']
    print(df[cols_display].describe().round(2))
    
    return df
    


if __name__ == "__main__":
    df = create_full_dataset()
    df.to_csv('C:/Users/X1 Carbon/Desktop/Stress Testing Projet/stress_test_data.csv')
    print("\nDataset sauvegardé dans C:/Users/X1 Carbon/Desktop/Stress Testing Projet/tmp/stress_test_data.csv")
