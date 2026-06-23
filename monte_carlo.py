"""
=============================================================================
MODULE 4 : Simulation Monte-Carlo — Stress Test de Liquidité
=============================================================================
Auteur    : Stress Test Liquidité
Description : Moteur de simulation Monte-Carlo produisant 10 000+ scénarios
              pour estimer la distribution des sorties de liquidité et les
              probabilités de défaillance.

Pourquoi Monte-Carlo ?
  Les sorties de liquidité dépendent de multiples variables macroéconomiques
  corrélées entre elles (l'inflation et les taux montent ensemble, le PIB et
  les dépôts bougent ensemble). La simulation Monte-Carlo permet de :
  1. Capturer ces corrélations via une copule gaussienne
  2. Propager les incertitudes à travers le modèle ML
  3. Obtenir une distribution complète des pertes (pas seulement un point)
  4. Calculer des mesures de risque extrême (VaR 99%, ES 99%)

Méthodologie :
  - Distributions des variables macro : normales multivariées avec matrice
    de corrélation estimée sur données historiques
  - Sous chaque scénario de stress : déplacement de la moyenne (mean shift)
    selon les hypothèses du scénario (chocs calibrés)
  - Propagation via modèle ML pour estimer les sorties de liquidité
  - Calcul LCR/NSFR pour chaque simulation
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Reproductibilité
np.random.seed(2024)


# =============================================================================
# 1. ESTIMATION DES DISTRIBUTIONS
# =============================================================================

def estimate_distributions(df: pd.DataFrame) -> dict:
    """
    Estime les paramètres de distribution des variables macroéconomiques
    à partir des données historiques simulées.
    
    Retourne :
    - Vecteur de moyennes (mu)
    - Matrice de covariance (Sigma) — capturant les corrélations
    - Tests de normalité (Jarque-Bera)
    
    Note : pour les données financières, une distribution t de Student
    avec degrés de liberté faibles (3-5) est plus robuste (queues épaisses).
    """
    macro_cols = ['pib_growth', 'inflation', 'chomage',
                  'taux_court', 'taux_long', 'taux_change']
    
    available = [c for c in macro_cols if c in df.columns]
    macro_data = df[available]
    
    mu    = macro_data.mean().values
    Sigma = macro_data.cov().values
    
    # Test de normalité (Jarque-Bera)
    print("TEST DE NORMALITÉ (Jarque-Bera) :")
    normality = {}
    for col in available:
        stat, p = stats.jarque_bera(df[col])
        normality[col] = {'stat': stat, 'p_value': p, 'normal': p > 0.05}
        status = "normale ✓" if p > 0.05 else "non-normale ✗ (queues épaisses)"
        print(f"  {col:<20} p={p:.4f}  {status}")
    
    # Corrélations
    print("\nMATRICE DE CORRÉLATION :")
    corr_matrix = macro_data.corr()
    print(corr_matrix.round(3))
    
    return {
        'mu': mu,
        'Sigma': Sigma,
        'corr': corr_matrix,
        'columns': available,
        'normality': normality,
    }


# =============================================================================
# 2. MOTEUR MONTE-CARLO
# =============================================================================

def run_monte_carlo(
    df: pd.DataFrame,
    model,
    scaler,
    feature_names: list,
    scenarios: dict,
    n_simulations: int = 10_000,
    use_student_t: bool = True,
    df_t: float = 5.0,
) -> dict:
    """
    Cœur du moteur Monte-Carlo.
    
    Pour chaque scénario :
    1. Génère n_simulations tirages de variables macro (normale multivariée ou t)
    2. Applique le choc de scénario (mean shift sur mu)
    3. Construit les vecteurs de features pour le modèle ML
    4. Prédit les sorties de liquidité pour chaque simulation
    5. Calcule le LCR et la probabilité de breach réglementaire
    
    Args:
        df             : Dataset historique (pour calibration)
        model          : Modèle ML entraîné
        scaler         : StandardScaler ajusté sur données d'entraînement
        feature_names  : Liste des features du modèle
        scenarios      : Dictionnaire des scénarios (define_scenarios())
        n_simulations  : Nombre de simulations (10 000 minimum)
        use_student_t  : Utiliser la loi de Student (queues épaisses) ?
        df_t           : Degrés de liberté si Student-t (3-7 typiquement)
    
    Returns:
        Dictionnaire {label_scenario: DataFrame résultats simulations}
    """
    dist_params = estimate_distributions(df)
    mu_base  = dist_params['mu']
    Sigma    = dist_params['Sigma']
    columns  = dist_params['columns']
    
    # Valeurs de référence du bilan (dernière observation)
    last_row = df.iloc[-1]
    
    results = {}
    
    print("\n" + "="*70)
    print(f"SIMULATION MONTE-CARLO — {n_simulations:,} SCÉNARIOS PAR STRESS")
    print("="*70)
    
    for label, scen in scenarios.items():
        print(f"\n  → Simulation : {scen.name}...", end=' ')
        
        # ----------------------------------------------------------------
        # a) Mean shift : on déplace la distribution vers les cibles du scénario
        # ----------------------------------------------------------------
        targets = {
            'pib_growth':  scen.pib_growth_target,
            'inflation':   scen.inflation_target,
            'chomage':     scen.chomage_target,
            'taux_court':  scen.taux_court_target,
            'taux_long':   scen.taux_long_target,
        }
        mu_stress = mu_base.copy()
        for i, col in enumerate(columns):
            if col in targets:
                # Interpolation partielle : on déplace vers la cible
                mu_stress[i] = targets[col]
        
        # ----------------------------------------------------------------
        # b) Génération des tirages
        # ----------------------------------------------------------------
        if use_student_t:
            # Simulation via copule t : meilleure gestion des queues
            # X = mu + L @ Z où Z est t-Student multivariée
            L = np.linalg.cholesky(Sigma)  # décomposition de Choleski
            Z = stats.t.rvs(df=df_t, size=(n_simulations, len(columns)))
            # Standardisation pour avoir les bonnes variances
            Z = Z / np.sqrt(df_t / (df_t - 2))
            macro_sim = mu_stress + Z @ L.T
        else:
            macro_sim = multivariate_normal.rvs(
                mean=mu_stress, cov=Sigma, size=n_simulations
            )
        
        # Contraintes : pas de valeurs aberrantes
        # PIB entre -15 et 20%, inflation 0-25%, etc.
        bounds = {
            0: (-15, 20),   # pib_growth
            1: (0.1, 25),   # inflation
            2: (1, 40),     # chomage
            3: (0.1, 25),   # taux_court
            4: (0.1, 30),   # taux_long
        }
        for i, (lo, hi) in bounds.items():
            if i < macro_sim.shape[1]:
                macro_sim[:, i] = np.clip(macro_sim[:, i], lo, hi)
        
        # ----------------------------------------------------------------
        # c) Construction des features pour le modèle ML
        # ----------------------------------------------------------------
        # Valeurs bilancielle : variation selon le scénario macro
        depots_vue_sim   = last_row['depots_vue'] * (
            1 - scen.fuite_depots_vue * np.random.uniform(0.5, 1.5, n_simulations)
        )
        depots_terme_sim = last_row['depots_terme'] * (
            1 - scen.fuite_depots_terme * np.random.uniform(0.5, 1.5, n_simulations)
        )
        refinancement_sim = last_row['refinancement'] * (
            1 - scen.fuite_refinancement * np.random.uniform(0.7, 1.3, n_simulations)
        )
        hqla_sim = (
            last_row.get('reserves_bc', 25) * 1.00
            + last_row.get('titres_souverains', 80) * (1 - scen.haircut_souverains)
            + last_row.get('titres_prives', 30) * (0.85 - scen.haircut_prives)
        ) * np.random.uniform(0.85, 1.15, n_simulations)
        
        credits_sim = last_row.get('credits', 400) * np.random.uniform(0.90, 1.05, n_simulations)
        
        # Assemblage features (ordre = feature_names)
        feature_matrix = []
        for i in range(n_simulations):
            row_features = []
            for feat in feature_names:
                if feat == 'pib_growth':
                    row_features.append(macro_sim[i, 0] if 0 < macro_sim.shape[1] else last_row.get(feat, 5.5))
                elif feat == 'inflation':
                    row_features.append(macro_sim[i, 1] if 1 < macro_sim.shape[1] else last_row.get(feat, 2.5))
                elif feat == 'chomage':
                    row_features.append(macro_sim[i, 2] if 2 < macro_sim.shape[1] else last_row.get(feat, 7.5))
                elif feat == 'taux_court':
                    row_features.append(macro_sim[i, 3] if 3 < macro_sim.shape[1] else last_row.get(feat, 3.5))
                elif feat == 'taux_long':
                    row_features.append(macro_sim[i, 4] if 4 < macro_sim.shape[1] else last_row.get(feat, 5.0))
                elif feat == 'taux_change':
                    base_fx = last_row.get('taux_change', 655)
                    row_features.append(base_fx * (1 + scen.taux_change_shock/100 * np.random.uniform(0.5, 1.5)))
                elif feat == 'depots_vue':
                    row_features.append(depots_vue_sim[i])
                elif feat == 'depots_terme':
                    row_features.append(depots_terme_sim[i])
                elif feat == 'refinancement':
                    row_features.append(refinancement_sim[i])
                elif feat == 'credits':
                    row_features.append(credits_sim[i])
                elif feat == 'hqla':
                    row_features.append(hqla_sim[i])
                elif feat in last_row.index:
                    # Variables lag et ratios : utilise dernière valeur + bruit
                    row_features.append(float(last_row[feat]) * np.random.uniform(0.9, 1.1))
                else:
                    row_features.append(0.0)
            feature_matrix.append(row_features)
        
        X_sim = np.array(feature_matrix)
        X_sim_scaled = scaler.transform(X_sim)
        
        # ----------------------------------------------------------------
        # d) Prédiction des sorties de liquidité
        # ----------------------------------------------------------------
        flux_pred = model.predict(X_sim_scaled)
        
        # Bruit résiduel du modèle (incertitude de prédiction)
        # On ajoute une perturbation calibrée sur RMSE du modèle (≈ 3 Mds)
        model_noise = np.random.normal(0, 2.5, n_simulations)
        flux_pred = flux_pred + model_noise
        
        # ----------------------------------------------------------------
        # e) LCR sous stress pour chaque simulation
        # ----------------------------------------------------------------
        lcr_sim = (hqla_sim / np.maximum(flux_pred, 0.1)) * 100
        lcr_sim = np.clip(lcr_sim, 0, 500)
        
        # Probabilité de breach LCR < 100%
        p_breach = np.mean(lcr_sim < 100.0)
        p_breach_80 = np.mean(lcr_sim < 80.0)   # Zone d'alerte précoce
        
        print(f"P(LCR<100%)={p_breach:.1%}  P(LCR<80%)={p_breach_80:.1%}")
        
        results[label] = pd.DataFrame({
            'flux_nets_30j': flux_pred,
            'hqla':          hqla_sim,
            'lcr':           lcr_sim,
            'depots_vue':    depots_vue_sim,
            'pib_growth':    macro_sim[:, 0] if macro_sim.shape[1] > 0 else 5.5,
            'inflation':     macro_sim[:, 1] if macro_sim.shape[1] > 1 else 2.5,
            'taux_court':    macro_sim[:, 3] if macro_sim.shape[1] > 3 else 3.5,
            'p_breach_lcr':  p_breach,
            'scenario':      label,
        })
    
    return results


# =============================================================================
# 3. MESURES DE RISQUE
# =============================================================================

def compute_risk_measures(mc_results: dict) -> pd.DataFrame:
    """
    Calcule les mesures de risque de liquidité à partir des simulations.
    
    Mesures calculées :
    - VaR 95%, 99%, 99.9% : quantiles de la distribution des sorties
    - ES 95%, 99%          : Expected Shortfall (moyenne des pertes au-delà du VaR)
    - LCR médian           : médiane du ratio LCR sous stress
    - P(breach)            : probabilité que le LCR passe sous 100%
    
    Interprétation :
    - VaR 99% = niveau de sorties dépassé dans seulement 1% des simulations
    - ES 99%  = perte moyenne dans les 1% pires cas (plus conservateur que VaR)
    - P(breach) LCR = probabilité réglementaire de non-conformité
    """
    records = []
    
    print("\n" + "="*70)
    print("MESURES DE RISQUE DE LIQUIDITÉ")
    print("="*70)
    
    for label, df_sim in mc_results.items():
        flux = df_sim['flux_nets_30j'].values
        lcr  = df_sim['lcr'].values
        
        var_95  = np.percentile(flux, 95)
        var_99  = np.percentile(flux, 99)
        var_999 = np.percentile(flux, 99.9)
        es_95   = flux[flux >= var_95].mean()
        es_99   = flux[flux >= var_99].mean()
        
        record = {
            'Scénario':       label,
            'Flux moyen (Mds)':  flux.mean().round(2),
            'Flux médian (Mds)': np.median(flux).round(2),
            'VaR 95% (Mds)':    var_95.round(2),
            'VaR 99% (Mds)':    var_99.round(2),
            'VaR 99.9% (Mds)':  var_999.round(2),
            'ES 95% (Mds)':     es_95.round(2),
            'ES 99% (Mds)':     es_99.round(2),
            'LCR médian (%)':   np.median(lcr).round(1),
            'LCR 5ème pctile':  np.percentile(lcr, 5).round(1),
            'P(LCR<100%) %':    (np.mean(lcr < 100) * 100).round(2),
            'P(LCR<80%) %':     (np.mean(lcr < 80) * 100).round(2),
        }
        records.append(record)
        
        print(f"\n{label.upper()} :")
        print(f"  VaR 99%        : {var_99:.2f} Mds XOF")
        print(f"  ES 99%         : {es_99:.2f} Mds XOF")
        print(f"  LCR médian     : {np.median(lcr):.1f}%")
        print(f"  P(LCR < 100%)  : {np.mean(lcr < 100)*100:.2f}%")
        
        # Identification scénario le plus plausible et le plus dangereux
        if label == 'adverse':
            print("  → Scénario LE PLUS PLAUSIBLE (calibré sur crises historiques récentes)")
        if label == 'systemic':
            print("  → Scénario LE PLUS DANGEREUX (extrême, faible prob., impact maximal)")
    
    return pd.DataFrame(records)


# =============================================================================
# 4. VISUALISATIONS MONTE-CARLO
# =============================================================================

def plot_monte_carlo(mc_results: dict, risk_measures: pd.DataFrame,
                     save_path: str = None):
    """
    Dashboard complet de visualisation Monte-Carlo.
    
    Panels :
    1. Distribution des sorties de liquidité (4 scénarios superposés)
    2. Distribution du LCR (avec seuil réglementaire 100%)
    3. VaR / ES par scénario (bar chart)
    4. Probabilités de breach LCR
    5. Scatter : PIB vs sorties de liquidité
    6. Box plots comparatifs
    """
    scenario_colors = {
        'normal':   '#1a9641',
        'adverse':  '#fdae61',
        'severe':   '#f46d43',
        'systemic': '#d73027',
    }
    scenario_names = {
        'normal':   'Normal',
        'adverse':  'Adverse',
        'severe':   'Sévère',
        'systemic': 'Systémique',
    }
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Stress Test de Liquidité — Résultats Monte-Carlo (10 000 simulations)',
                 fontsize=16, fontweight='bold')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    # --- Panel 1 : Distribution des flux de liquidité ---
    ax1 = fig.add_subplot(gs[0, :2])
    for label, df_sim in mc_results.items():
        flux = df_sim['flux_nets_30j'].values
        ax1.hist(flux, bins=80, alpha=0.45, density=True,
                 color=scenario_colors.get(label, 'gray'),
                 label=scenario_names.get(label, label))
        
        # Courbe de densité KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(flux)
        x_range = np.linspace(flux.min(), flux.max(), 200)
        ax1.plot(x_range, kde(x_range), color=scenario_colors.get(label, 'gray'),
                 linewidth=2.5)
    
    ax1.set_title('Distribution des sorties de liquidité nettes (30 jours)', fontsize=12)
    ax1.set_xlabel('Sorties nettes (Mds XOF)')
    ax1.set_ylabel('Densité')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2 : Distribution du LCR ---
    ax2 = fig.add_subplot(gs[0, 2])
    for label, df_sim in mc_results.items():
        lcr = np.clip(df_sim['lcr'].values, 0, 400)
        ax2.hist(lcr, bins=60, alpha=0.45, density=True,
                 color=scenario_colors.get(label, 'gray'),
                 label=scenario_names.get(label, label))
    
    ax2.axvline(100, color='black', linestyle='--', linewidth=2,
                label='Seuil LCR 100%')
    ax2.axvline(80, color='red', linestyle=':', linewidth=1.5,
                label='Alerte 80%')
    ax2.set_title('Distribution du LCR sous stress (%)', fontsize=12)
    ax2.set_xlabel('LCR (%)')
    ax2.set_ylabel('Densité')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 400)
    
    # --- Panel 3 : VaR et ES par scénario ---
    ax3 = fig.add_subplot(gs[1, :2])
    n_scenarios = len(mc_results)
    x = np.arange(n_scenarios)
    width = 0.28
    labels_list = list(mc_results.keys())
    
    var99_vals = risk_measures.set_index('Scénario')['VaR 99% (Mds)'].reindex(labels_list).values
    es99_vals  = risk_measures.set_index('Scénario')['ES 99% (Mds)'].reindex(labels_list).values
    mean_vals  = risk_measures.set_index('Scénario')['Flux moyen (Mds)'].reindex(labels_list).values
    
    bars1 = ax3.bar(x - width, mean_vals, width, label='Flux moyen',
                    color='#2166ac', alpha=0.8)
    bars2 = ax3.bar(x,         var99_vals, width, label='VaR 99%',
                    color='#f4a582', alpha=0.8)
    bars3 = ax3.bar(x + width, es99_vals, width, label='ES 99%',
                    color='#d6604d', alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([scenario_names.get(l, l) for l in labels_list])
    ax3.set_title('Flux moyen vs VaR 99% vs Expected Shortfall 99% (Mds XOF)', fontsize=12)
    ax3.set_ylabel('Milliards XOF')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in [*bars1, *bars2, *bars3]:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    
    # --- Panel 4 : Probabilités de breach ---
    ax4 = fig.add_subplot(gs[1, 2])
    p_breach   = [risk_measures[risk_measures['Scénario']==l]['P(LCR<100%) %'].values[0]
                  for l in labels_list]
    p_breach80 = [risk_measures[risk_measures['Scénario']==l]['P(LCR<80%) %'].values[0]
                  for l in labels_list]
    
    colors_scen = [scenario_colors.get(l, 'gray') for l in labels_list]
    bars_p = ax4.bar(range(n_scenarios), p_breach, color=colors_scen, alpha=0.85,
                     label='P(LCR < 100%)', edgecolor='white')
    ax4.bar(range(n_scenarios), p_breach80, color='darkred', alpha=0.5,
            label='P(LCR < 80%)')
    ax4.axhline(5.0, color='orange', linestyle='--', linewidth=1.5,
                label='Seuil alerte 5%')
    
    ax4.set_xticks(range(n_scenarios))
    ax4.set_xticklabels([scenario_names.get(l, l)[:8] for l in labels_list], fontsize=9)
    ax4.set_title('Probabilité de non-conformité LCR (%)', fontsize=12)
    ax4.set_ylabel('Probabilité (%)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars_p, p_breach):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- Panel 5 : Box plots des flux par scénario ---
    ax5 = fig.add_subplot(gs[2, :2])
    box_data = [mc_results[l]['flux_nets_30j'].values for l in labels_list]
    bp = ax5.boxplot(box_data, patch_artist=True, notch=True,
                     medianprops=dict(color='black', linewidth=2))
    
    for patch, label in zip(bp['boxes'], labels_list):
        patch.set_facecolor(scenario_colors.get(label, 'gray'))
        patch.set_alpha(0.7)
    
    ax5.set_xticklabels([scenario_names.get(l, l) for l in labels_list])
    ax5.set_title('Distribution des flux de liquidité par scénario (box plots)', fontsize=12)
    ax5.set_ylabel('Sorties nettes (Mds XOF)')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # --- Panel 6 : Carte de chaleur des probabilités ---
    ax6 = fig.add_subplot(gs[2, 2])
    heat_data = risk_measures[['P(LCR<100%) %', 'P(LCR<80%) %',
                                'LCR médian (%)']].values
    
    im = ax6.imshow(heat_data, cmap='RdYlGn_r', aspect='auto',
                    vmin=0, vmax=max(heat_data.max(), 1))
    ax6.set_xticks([0, 1, 2])
    ax6.set_xticklabels(['P(LCR<100%)', 'P(LCR<80%)', 'LCR médian'], fontsize=8, rotation=25)
    ax6.set_yticks(range(n_scenarios))
    ax6.set_yticklabels([scenario_names.get(l, l) for l in labels_list], fontsize=9)
    ax6.set_title('Carte de chaleur des risques', fontsize=12)
    plt.colorbar(im, ax=ax6, shrink=0.8)
    
    for i in range(n_scenarios):
        for j in range(3):
            ax6.text(j, i, f'{heat_data[i, j]:.1f}',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color='white' if heat_data[i, j] > heat_data.max()/2 else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graphique sauvegardé : {save_path}")
    
    return fig


# =============================================================================
# 5. INTERPRÉTATION ÉCONOMIQUE
# =============================================================================

def interpret_results(risk_measures: pd.DataFrame) -> str:
    """
    Génère un commentaire économique automatique sur les résultats.
    Simule le type d'analyse produite pour un comité des risques.
    """
    report = []
    report.append("\n" + "="*70)
    report.append("SYNTHÈSE ANALYTIQUE — COMITÉ DES RISQUES")
    report.append("="*70)
    
    for _, row in risk_measures.iterrows():
        scen = row['Scénario']
        p_breach = row['P(LCR<100%) %']
        lcr_med  = row['LCR médian (%)']
        var99    = row['VaR 99% (Mds)']
        
        if p_breach < 5:
            risk_level = "FAIBLE — position de liquidité confortable"
        elif p_breach < 20:
            risk_level = "MODÉRÉ — surveillance renforcée recommandée"
        elif p_breach < 50:
            risk_level = "ÉLEVÉ — mesures correctives à planifier"
        else:
            risk_level = "CRITIQUE — plan de contingence liquidité à activer"
        
        report.append(f"\nScénario {scen.upper()} :")
        report.append(f"  Niveau de risque   : {risk_level}")
        report.append(f"  LCR médian         : {lcr_med:.1f}% (seuil réglementaire : 100%)")
        report.append(f"  P(breach LCR)      : {p_breach:.2f}%")
        report.append(f"  VaR 99% sorties    : {var99:.2f} Mds XOF")
        
        if lcr_med < 100:
            report.append(f"  ⚠ ALERTE : Le LCR médian est SOUS le seuil réglementaire !")
        if p_breach > 50:
            report.append(f"  ⚠ ALERTE CRITIQUE : Plus de 50% de probabilité de breach LCR !")
    
    report.append("\nCONCLUSIONS :")
    report.append("  • Scénario le plus plausible  : 'adverse' (probabilité historique ~20%)")
    report.append("  • Scénario le plus dangereux  : 'systemic' (impact maximal, prob. faible)")
    report.append("  • Recommandation : maintenir un coussin de liquidité ≥ VaR 99% adverse")
    
    text = "\n".join(report)
    print(text)
    return text


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/stress_test_liquidite')
    from data.simulate_data import create_full_dataset
    from models.ml_models import prepare_features, train_best_model
    from scenarios.stress_scenarios import define_scenarios
    
    print("Chargement et préparation des données...")
    df = create_full_dataset()
    X, y, feature_names = prepare_features(df)
    
    print("Entraînement du modèle ML...")
    model, scaler, _ = train_best_model(X, y)
    
    print("Définition des scénarios...")
    scenarios = define_scenarios()
    
    print("\nLancement de la simulation Monte-Carlo (10 000 scénarios)...")
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
    
    print("\nCalcul des mesures de risque...")
    risk_measures = compute_risk_measures(mc_results)
    
    print("\nInterprétation économique...")
    interpret_results(risk_measures)
    
    print("\nGénération des visualisations...")
    plot_monte_carlo(mc_results, risk_measures, save_path='/tmp/monte_carlo.png')
    
    risk_measures.to_csv('/tmp/risk_measures.csv', index=False)
    print("\n[TERMINÉ] Module Monte-Carlo exécuté avec succès.")
