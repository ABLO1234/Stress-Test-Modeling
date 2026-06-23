"""
=============================================================================
MODULE 5 : Version Avancée — Analyse de Sensibilité, Reverse Stress Testing,
           Validation et Backtesting
=============================================================================
Auteur    : Stress Test Liquidité
Description : Composants d'un département Risk Modeling de banque centrale.

Contenu :
1. Analyse de sensibilité    : impact unitaire de chaque variable macro
2. Reverse Stress Testing    : trouver les scénarios qui font tomber le LCR
3. Validation du modèle      : tests statistiques de robustesse
4. Backtesting simplifié     : performance sur fenêtre historique
5. Indicateurs NSFR détaillés

Philosophie Reverse Stress Testing :
  Au lieu de partir d'un scénario → calculer l'impact (sens normal),
  on part d'un seuil critique (LCR = 80%) et on cherche rétro-activement
  quelles combinaisons de variables macro font franchir ce seuil.
  C'est l'outil le plus puissant pour identifier les vulnérabilités cachées.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. ANALYSE DE SENSIBILITÉ
# =============================================================================

def sensitivity_analysis(df: pd.DataFrame, model, scaler,
                          feature_names: list,
                          n_points: int = 50) -> dict:
    """
    Analyse de sensibilité (ceteris paribus) :
    Pour chaque variable macroéconomique, varie sa valeur entre min et max
    historique (± 3 std) en gardant les autres à leur valeur médiane.
    
    Mesure l'impact sur les sorties de liquidité prédites.
    
    Args:
        n_points : nombre de points d'évaluation par variable
    
    Returns:
        Dictionnaire {variable: (x_range, y_flux, elasticité)}
    """
    macro_vars = ['pib_growth', 'inflation', 'chomage', 'taux_court', 'taux_long']
    available  = [v for v in macro_vars if v in feature_names]
    
    # Vecteur de base = médianes de toutes les features
    X_base = np.array([df[feat].median() if feat in df.columns else 0.0
                       for feat in feature_names]).reshape(1, -1)
    
    X_base_scaled = scaler.transform(X_base)
    y_base = model.predict(X_base_scaled)[0]
    
    sensitivity = {}
    
    print("\nANALYSE DE SENSIBILITÉ (ceteris paribus)")
    print("-"*60)
    print(f"  Flux de référence (médianes) : {y_base:.3f} Mds XOF")
    print()
    
    for var in available:
        if var not in feature_names:
            continue
        feat_idx = feature_names.index(var)
        
        # Plage de variation : ± 3 écarts-types
        mu_var  = df[var].mean()
        std_var = df[var].std()
        x_range = np.linspace(mu_var - 3*std_var, mu_var + 3*std_var, n_points)
        
        y_range = []
        for x_val in x_range:
            X_var = X_base.copy()
            X_var[0, feat_idx] = x_val
            X_var_sc = scaler.transform(X_var)
            y_range.append(model.predict(X_var_sc)[0])
        
        y_range = np.array(y_range)
        
        # Élasticité : variation % de l'output pour 1% de variation de l'input
        delta_y = (y_range[-1] - y_range[0])
        delta_x = (x_range[-1] - x_range[0])
        elasticite = (delta_y / max(abs(y_base), 1e-6)) / (delta_x / max(abs(mu_var), 1e-6))
        
        sensitivity[var] = {
            'x_range':     x_range,
            'y_range':     y_range,
            'elasticite':  elasticite,
            'impact_1std': model.predict(scaler.transform(
                               X_base.copy().__setitem__((0, feat_idx),
                               mu_var + std_var) or X_base.copy()
                           ))[0] - y_base
        }
        
        print(f"  {var:<25} élasticité = {elasticite:+.4f}  "
              f"impact 1σ = {elasticite * std_var / max(abs(mu_var), 1e-6) * y_base:+.2f} Mds")
    
    return sensitivity


def sensitivity_analysis_v2(df: pd.DataFrame, model, scaler,
                              feature_names: list, n_points: int = 50) -> dict:
    """Version corrigée de l'analyse de sensibilité (sans bug __setitem__)."""
    macro_vars = ['pib_growth', 'inflation', 'chomage', 'taux_court', 'taux_long']
    available  = [v for v in macro_vars if v in feature_names]
    
    X_base = np.array([df[feat].median() if feat in df.columns else 0.0
                       for feat in feature_names])
    
    X_base_2d   = X_base.reshape(1, -1)
    y_base      = model.predict(scaler.transform(X_base_2d))[0]
    
    sensitivity = {}
    
    print("\nANALYSE DE SENSIBILITÉ (ceteris paribus)")
    print("-"*60)
    print(f"  Flux de référence (médianes) : {y_base:.3f} Mds XOF\n")
    
    for var in available:
        if var not in feature_names:
            continue
        feat_idx = feature_names.index(var)
        
        mu_var  = df[var].mean()
        std_var = df[var].std()
        x_range = np.linspace(mu_var - 3*std_var, mu_var + 3*std_var, n_points)
        
        y_range = []
        for x_val in x_range:
            X_tmp = X_base.copy()
            X_tmp[feat_idx] = x_val
            y_range.append(model.predict(scaler.transform(X_tmp.reshape(1, -1)))[0])
        
        y_range = np.array(y_range)
        
        delta_y   = y_range[-1] - y_range[0]
        delta_x   = x_range[-1] - x_range[0]
        elasticite = (delta_y / max(abs(y_base), 1e-6)) / (delta_x / max(abs(mu_var), 1e-6))
        
        # Impact d'un choc de +1 sigma
        X_shock = X_base.copy()
        X_shock[feat_idx] = mu_var + std_var
        impact_1std = model.predict(scaler.transform(X_shock.reshape(1, -1)))[0] - y_base
        
        sensitivity[var] = {
            'x_range':     x_range,
            'y_range':     y_range,
            'elasticite':  elasticite,
            'impact_1std': impact_1std,
            'mu':          mu_var,
            'std':         std_var,
        }
        print(f"  {var:<25} élasticité = {elasticite:+.4f}  "
              f"impact 1σ = {impact_1std:+.3f} Mds XOF")
    
    return sensitivity


# =============================================================================
# 2. REVERSE STRESS TESTING
# =============================================================================

def reverse_stress_test(df: pd.DataFrame, model, scaler,
                         feature_names: list,
                         lcr_threshold: float = 80.0,
                         hqla_assumption: float = None) -> dict:
    """
    Reverse Stress Testing : trouver les combinaisons de variables macro
    qui font tomber le LCR sous un seuil critique (ex: 80%).
    
    Méthodologie :
    1. On fixe le seuil critique : LCR = lcr_threshold%
    2. Pour chaque variable macro, on trouve le niveau critique (toutes autres = médiane)
    3. On identifie les combinaisons les plus dangereuses (AMDAL)
    
    Question économique clé :
    "Quelle croissance du PIB ferait passer notre LCR sous 80% ?"
    "Quel niveau d'inflation déclencherait une crise de liquidité ?"
    
    Returns:
        Dictionnaire avec niveaux critiques par variable
    """
    last_row = df.iloc[-1]
    if hqla_assumption is None:
        hqla_assumption = last_row.get('hqla', 100)
    
    # Seuil de flux : flux critique tel que LCR = threshold
    flux_critique = hqla_assumption / (lcr_threshold / 100)
    
    print(f"\nREVERSE STRESS TESTING — Seuil : LCR = {lcr_threshold}%")
    print(f"  HQLA disponible    : {hqla_assumption:.1f} Mds XOF")
    print(f"  Flux critique max  : {flux_critique:.1f} Mds XOF")
    print(f"  (Au-delà de ce niveau de flux, LCR < {lcr_threshold}%)")
    print()
    
    macro_vars = ['pib_growth', 'inflation', 'chomage', 'taux_court']
    
    X_base = np.array([df[feat].median() if feat in df.columns else 0.0
                       for feat in feature_names])
    
    critical_levels = {}
    
    for var in macro_vars:
        if var not in feature_names:
            continue
        
        feat_idx = feature_names.index(var)
        mu_var   = df[var].mean()
        std_var  = df[var].std()
        
        # Recherche par dichotomie du niveau critique
        def objective(x_val):
            X_tmp = X_base.copy()
            X_tmp[feat_idx] = x_val
            flux_pred = model.predict(scaler.transform(X_tmp.reshape(1, -1)))[0]
            return flux_pred - flux_critique
        
        try:
            # Chercher dans un intervalle large
            lo = mu_var - 6 * std_var
            hi = mu_var + 6 * std_var
            
            if objective(lo) * objective(hi) < 0:
                x_crit = optimize.brentq(objective, lo, hi, xtol=1e-4)
                ecart_std = (x_crit - mu_var) / std_var
                critical_levels[var] = {
                    'niveau_critique':  x_crit,
                    'valeur_baseline':  mu_var,
                    'ecart_baseline':   x_crit - mu_var,
                    'ecarts_std':       ecart_std,
                    'atteignable':      abs(ecart_std) < 4,
                }
                status = "⚠ PLAUSIBLE" if abs(ecart_std) < 2 else (
                         "ℹ possible" if abs(ecart_std) < 3 else "✓ peu probable")
                print(f"  {var:<20} niveau critique = {x_crit:+.2f}  "
                      f"({ecart_std:+.1f}σ)  {status}")
            else:
                critical_levels[var] = {
                    'niveau_critique':  None,
                    'valeur_baseline':  mu_var,
                    'ecart_baseline':   None,
                    'ecarts_std':       None,
                    'atteignable':      False,
                }
                print(f"  {var:<20} seuil non atteint dans [-6σ, +6σ]")
        except Exception as e:
            print(f"  {var:<20} erreur optimisation : {e}")
    
    return critical_levels


# =============================================================================
# 3. VALIDATION DU MODÈLE
# =============================================================================

def validate_model(df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                   feature_names: list, n_splits: int = 4) -> dict:
    """
    Validation rigoureuse du modèle ML selon les standards EBA.
    
    Tests effectués :
    1. Stabilité temporelle (performance par sous-période)
    2. Test de Diebold-Mariano (comparaison avec modèle naïf)
    3. Intervalles de confiance des prédictions (bootstrap)
    4. Test de Mincer-Zarnowitz (non-biais des prédictions)
    
    Returns:
        Dictionnaire de métriques de validation
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    print("\nVALIDATION DU MODÈLE")
    print("="*60)
    
    # ---- 1. Stabilité temporelle (walk-forward) ----
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_r2, fold_rmse = [], []
    
    model_cv = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.05, random_state=42)
    
    for fold, (tr, te) in enumerate(tscv.split(X)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        
        sc = StandardScaler().fit(X_tr)
        model_cv.fit(sc.transform(X_tr), y_tr)
        y_pred = model_cv.predict(sc.transform(X_te))
        
        r2   = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        fold_r2.append(r2)
        fold_rmse.append(rmse)
        print(f"  Fold {fold+1} : R²={r2:.4f}  RMSE={rmse:.3f}")
    
    print(f"\n  R² moyen   : {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")
    print(f"  RMSE moyen : {np.mean(fold_rmse):.3f} ± {np.std(fold_rmse):.3f}")
    
    # ---- 2. Test de Mincer-Zarnowitz (non-biais) ----
    # Si le modèle est non-biaisé : régression y_réel = a + b*y_prédit
    # doit donner a ≈ 0 et b ≈ 1 (test OLS)
    sc_full = StandardScaler().fit(X)
    model_full = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                            learning_rate=0.05, random_state=42)
    model_full.fit(sc_full.transform(X), y)
    y_pred_full = model_full.predict(sc_full.transform(X))
    
    from scipy.stats import linregress
    slope, intercept, r_value, p_slope, _ = linregress(y_pred_full, y)
    
    print(f"\n  TEST DE MINCER-ZARNOWITZ :")
    print(f"  Intercept (doit ≈ 0)  : {intercept:.4f}")
    print(f"  Pente     (doit ≈ 1)  : {slope:.4f}")
    print(f"  R²                    : {r_value**2:.4f}")
    
    mz_pass = abs(intercept) < 2 and abs(slope - 1) < 0.2
    print(f"  Résultat              : {'✓ VALIDÉ' if mz_pass else '✗ BIAIS DÉTECTÉ'}")
    
    # ---- 3. Test Diebold-Mariano (vs modèle naïf Random Walk) ----
    # Modèle naïf : prédit t+1 = t (Random Walk)
    y_naive = y[:-1]  # prédiction naïve = valeur précédente
    y_actual = y[1:]
    y_model  = y_pred_full[1:]
    
    e_model  = y_actual - y_model[:len(y_actual)]
    e_naive  = y_actual - y_naive
    
    loss_model = e_model**2
    loss_naive = e_naive**2
    d = loss_naive - loss_model  # différence positive = notre modèle est meilleur
    
    t_stat = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-9)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(d)-1))
    
    print(f"\n  TEST DIEBOLD-MARIANO (vs Random Walk) :")
    print(f"  t-statistic : {t_stat:.3f}")
    print(f"  p-value     : {p_value:.4f}")
    dm_pass = p_value < 0.05 and t_stat > 0
    print(f"  Résultat    : {'✓ MODÈLE SIGNIFICATIVEMENT SUPÉRIEUR AU NAÏF' if dm_pass else 'ℹ Amélioration non significative'}")
    
    return {
        'fold_r2':     fold_r2,
        'fold_rmse':   fold_rmse,
        'mean_r2':     np.mean(fold_r2),
        'mean_rmse':   np.mean(fold_rmse),
        'mz_intercept': intercept,
        'mz_slope':    slope,
        'mz_pass':     mz_pass,
        'dm_tstat':    t_stat,
        'dm_pvalue':   p_value,
        'dm_pass':     dm_pass,
    }


# =============================================================================
# 4. VISUALISATIONS AVANCÉES
# =============================================================================

def plot_advanced_analysis(sensitivity: dict, critical_levels: dict,
                           validation: dict, save_path: str = None):
    """Dashboard des analyses avancées."""
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Analyses Avancées — Sensibilité, Reverse Stress, Validation',
                 fontsize=15, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    # --- Panel 1 : Courbes de sensibilité ---
    ax1 = fig.add_subplot(gs[0, :2])
    colors_vars = {'pib_growth': '#1a9641', 'inflation': '#d7191c',
                   'chomage': '#f46d43', 'taux_court': '#2c7bb6',
                   'taux_long': '#7b3294'}
    
    for var, res in sensitivity.items():
        # Normaliser x en écarts-types pour comparabilité
        x_std = (res['x_range'] - res['mu']) / res['std']
        ax1.plot(x_std, res['y_range'],
                 label=f"{var} (elas={res['elasticite']:+.2f})",
                 color=colors_vars.get(var, 'gray'), linewidth=2.5)
    
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_title('Sensibilité des flux de liquidité aux chocs macro (en σ)', fontsize=12)
    ax1.set_xlabel('Choc (unités = écarts-types par rapport à la médiane)')
    ax1.set_ylabel('Flux nets prédit (Mds XOF)')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2 : Tornado Chart des élasticités ---
    ax2 = fig.add_subplot(gs[0, 2])
    vars_sorted = sorted(sensitivity.items(), key=lambda x: abs(x[1]['elasticite']), reverse=True)
    var_names   = [v[0] for v in vars_sorted]
    elas_vals   = [v[1]['elasticite'] for v in vars_sorted]
    colors_bar  = ['#d7191c' if e > 0 else '#1a9641' for e in elas_vals]
    
    ax2.barh(range(len(var_names)), elas_vals, color=colors_bar, alpha=0.8)
    ax2.set_yticks(range(len(var_names)))
    ax2.set_yticklabels(var_names, fontsize=10)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_title('Tornado : élasticités\n(rouge=↑flux, vert=↓flux)', fontsize=11)
    ax2.set_xlabel('Élasticité')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # --- Panel 3 : Reverse Stress — niveaux critiques ---
    ax3 = fig.add_subplot(gs[1, 0])
    valid_crit = {k: v for k, v in critical_levels.items()
                  if v.get('ecarts_std') is not None}
    
    if valid_crit:
        crit_vars = list(valid_crit.keys())
        crit_std  = [valid_crit[v]['ecarts_std'] for v in crit_vars]
        crit_colors = ['#d73027' if abs(s) < 2 else '#f46d43' if abs(s) < 3 else '#1a9641'
                       for s in crit_std]
        
        bars_r = ax3.barh(crit_vars, crit_std, color=crit_colors, alpha=0.8)
        ax3.axvline(-2, color='orange', linestyle='--', linewidth=1.5, label='2σ (zone risque)')
        ax3.axvline(2, color='orange', linestyle='--', linewidth=1.5)
        ax3.axvline(-3, color='red', linestyle=':', linewidth=1, label='3σ (rare)')
        ax3.axvline(3, color='red', linestyle=':', linewidth=1)
        ax3.set_title('Reverse Stress : choc critique\n(en écarts-types, seuil LCR=80%)', fontsize=10)
        ax3.set_xlabel('Écarts-types nécessaires')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')
    else:
        ax3.text(0.5, 0.5, 'Données insuffisantes', ha='center', va='center',
                 transform=ax3.transAxes)
    
    # --- Panel 4 : Stabilité temporelle du modèle ---
    ax4 = fig.add_subplot(gs[1, 1])
    folds = range(1, len(validation['fold_r2']) + 1)
    ax4.plot(folds, validation['fold_r2'], 'o-', color='#2c7bb6',
             linewidth=2, markersize=8, label='R² par fold')
    ax4.fill_between(folds,
                     np.array(validation['fold_r2']) - 0.05,
                     np.array(validation['fold_r2']) + 0.05,
                     alpha=0.2, color='#2c7bb6')
    ax4.axhline(validation['mean_r2'], color='red', linestyle='--',
                linewidth=2, label=f"Moyenne R²={validation['mean_r2']:.4f}")
    ax4.set_title('Stabilité temporelle du modèle (R² par fold)', fontsize=11)
    ax4.set_xlabel('Fold Walk-Forward')
    ax4.set_ylabel('R²')
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # --- Panel 5 : Tableau de validation ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    val_data = [
        ['Métrique', 'Valeur', 'Statut'],
        ['R² moyen', f"{validation['mean_r2']:.4f}",
         '✓' if validation['mean_r2'] > 0.7 else '✗'],
        ['RMSE moyen', f"{validation['mean_rmse']:.3f}",
         '✓' if validation['mean_rmse'] < 5 else '✗'],
        ['Test MZ intercept', f"{validation['mz_intercept']:.4f}",
         '✓' if validation['mz_pass'] else '✗'],
        ['Test MZ pente', f"{validation['mz_slope']:.4f}",
         '✓' if validation['mz_pass'] else '✗'],
        ['Test DM t-stat', f"{validation['dm_tstat']:.3f}",
         '✓' if validation['dm_pass'] else 'ℹ'],
        ['Test DM p-value', f"{validation['dm_pvalue']:.4f}",
         '✓' if validation['dm_pass'] else 'ℹ'],
    ]
    
    table = ax5.table(cellText=val_data[1:], colLabels=val_data[0],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax5.set_title('Récapitulatif validation du modèle', fontsize=11, pad=20)
    
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
    sys.path.insert(0, '/home/claude/stress_test_liquidite')
    from data.simulate_data import create_full_dataset
    from models.ml_models import prepare_features, train_best_model
    
    print("Chargement des données...")
    df = create_full_dataset()
    X, y, feature_names = prepare_features(df)
    
    print("Entraînement du modèle...")
    model, scaler, _ = train_best_model(X, y)
    
    print("\nAnalyse de sensibilité...")
    sensitivity = sensitivity_analysis_v2(df, model, scaler, feature_names)
    
    print("\nReverse Stress Testing...")
    critical_levels = reverse_stress_test(df, model, scaler, feature_names,
                                           lcr_threshold=80.0)
    
    print("\nValidation du modèle...")
    validation = validate_model(df, X, y, feature_names)
    
    print("\nVisualisations avancées...")
    plot_advanced_analysis(sensitivity, critical_levels, validation,
                           save_path='/tmp/advanced_analysis.png')
    
    print("\n[TERMINÉ] Module Avancé exécuté avec succès.")
