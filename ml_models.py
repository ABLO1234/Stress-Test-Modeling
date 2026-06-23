"""
=============================================================================
MODULE 2 : Modèles Machine Learning — Prédiction des sorties de liquidité
=============================================================================
Auteur    : Stress Test Liquidité
Description : Entraînement, évaluation et comparaison de 5 modèles ML pour
              prédire les flux nets de trésorerie (variable cible).

Variable cible : flux_nets_30j (sorties de liquidité nettes sur 30 jours)
                 exprimée en milliards XOF.

Méthodologie : Walk-Forward Validation (time-series split) pour respecter
               la chronologie des données — pas de data leakage.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Import conditionnel XGBoost / LightGBM
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] XGBoost non disponible — utilisation GradientBoosting sklearn")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[INFO] LightGBM non disponible — sera ignoré")
    

try:
    from sklearn.neural_network import MLPRegressor
    HAS_MLP = True
except ImportError:
    HAS_MLP = False


# =============================================================================
# 1. FEATURE ENGINEERING
# =============================================================================

FEATURES = [
    # Macro variables (contemporaines)
    'pib_growth', 'inflation', 'chomage',
    'taux_court', 'taux_long', 'taux_change',
    # Macro variables (retardées)
    'pib_growth_lag1', 'inflation_lag1', 'taux_court_lag1',
    'pib_growth_lag3', 'inflation_lag3',
    # Variables bilancielles
    'depots_vue', 'depots_terme', 'refinancement', 'credits', 'hqla',
    # Variations (signaux de tensions)
    'delta_depots_vue', 'delta_depots_terme', 'delta_refinancement',
    # Ratios structurels
    'ratio_transformation', 'ratio_liquidite_immediate',
]

TARGET = 'flux_nets_30j'


def prepare_features(df: pd.DataFrame):
    """
    Prépare X et y pour la modélisation.
    
    Returns:
        X (features), y (target), feature_names
    """
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].values
    y = df[TARGET].values
    return X, y, available


# =============================================================================
# 2. DÉFINITION DES MODÈLES
# =============================================================================

def get_models() -> dict:
    """
    Retourne un dictionnaire de modèles avec leurs hyperparamètres calibrés.
    
    Notes sur les choix :
    - Ridge : régularisation L2 pour éviter la multicolinéarité (macro variables corrélées)
    - RandomForest : robuste aux outliers, capture les non-linéarités
    - XGBoost : meilleure performance sur données tabulaires financières
    - LightGBM : plus rapide que XGBoost, utile pour Monte-Carlo
    - MLP : architecture légère (2 couches cachées) — les données sont limitées (56 obs)
    """
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest':    RandomForestRegressor(
                                n_estimators=200,
                                max_depth=5,
                                min_samples_leaf=3,
                                random_state=42,
                                n_jobs=-1
                            ),
        'Gradient Boosting': GradientBoostingRegressor(
                                n_estimators=200,
                                max_depth=3,
                                learning_rate=0.05,
                                subsample=0.8,
                                random_state=42
                             ),
    }
    
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
    
    if HAS_LGB:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    
    if HAS_MLP:
        models['Réseau de neurones'] = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )
    
    return models


# =============================================================================
# 3. ÉVALUATION — WALK-FORWARD CROSS-VALIDATION
# =============================================================================

def evaluate_models(X: np.ndarray, y: np.ndarray,
                    feature_names: list, n_splits: int = 5) -> pd.DataFrame:
    """
    Évalue chaque modèle via Time-Series Cross-Validation (walk-forward).
    
    Walk-forward signifie que chaque fold entraîne sur le passé et prédit le futur —
    jamais l'inverse. C'est la seule méthode valide en séries temporelles.
    
    Métriques calculées :
    - RMSE : Root Mean Squared Error (sensible aux outliers — important en risque)
    - MAE  : Mean Absolute Error (interprétable en milliards XOF)
    - MAPE : Mean Absolute Percentage Error (erreur relative)
    - R²   : Coefficient de détermination
    
    Returns:
        DataFrame de résultats comparatifs
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    models = get_models()
    results = []
    
    print("="*70)
    print("ÉVALUATION DES MODÈLES — WALK-FORWARD CROSS-VALIDATION")
    print("="*70)
    
    for name, model in models.items():
        rmse_list, mae_list, mape_list, r2_list = [], [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Normalisation (ajustée sur train uniquement)
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)
            
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            
            rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            mape_list.append(mean_absolute_percentage_error(y_test, y_pred) * 100)
            r2_list.append(r2_score(y_test, y_pred))
        
        avg = {
            'Modèle':     name,
            'RMSE (Mds)': np.mean(rmse_list).round(3),
            'MAE (Mds)':  np.mean(mae_list).round(3),
            'MAPE (%)':   np.mean(mape_list).round(2),
            'R²':         np.mean(r2_list).round(4),
            'RMSE std':   np.std(rmse_list).round(3),
        }
        results.append(avg)
        print(f"  {name:<25} RMSE={avg['RMSE (Mds)']:.3f}  "
              f"MAE={avg['MAE (Mds)']:.3f}  R²={avg['R²']:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('RMSE (Mds)')
    print("\nCLASSEMENT PAR RMSE (meilleur modèle en premier) :")
    print(results_df[['Modèle', 'RMSE (Mds)', 'MAE (Mds)', 'MAPE (%)', 'R²']].to_string(index=False))
    
    return results_df


# =============================================================================
# 4. ENTRAÎNEMENT DU MEILLEUR MODÈLE (FULL DATASET)
# =============================================================================

def train_best_model(X: np.ndarray, y: np.ndarray, results_df: pd.DataFrame):
    """
    Entraîne le modèle ayant le plus petit RMSE sur le dataset complet.
    Ce modèle est sélectionné à partir de l'évaluation walk-forward.
    
    Returns:
        (modèle entraîné, scaler ajusté, modèle sélectionné)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model_name = results_df.sort_values('RMSE (Mds)').iloc[0]['Modèle']
    models = get_models()
    model = models.get(best_model_name)

    if model is None:
        print(f"[WARN] Modèle '{best_model_name}' introuvable, fallback vers Gradient Boosting.")
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        best_model_name = 'Gradient Boosting'

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    print(f"\nModèle final entraîné : {best_model_name}")
    print(f"  R² (train) = {r2_score(y, y_pred):.4f}")
    print(f"  RMSE (train) = {np.sqrt(mean_squared_error(y, y_pred)):.3f} Mds XOF")

    return model, scaler, best_model_name


# =============================================================================
# 5. IMPORTANCE DES VARIABLES
# =============================================================================

def compute_feature_importance(model, X: np.ndarray, y: np.ndarray,
                                feature_names: list, scaler) -> pd.DataFrame:
    """
    Calcule l'importance des variables via permutation importance.
    La permutation importance est model-agnostique et évite le biais
    des importances de Gini pour les variables continues.
    """
    X_scaled = scaler.transform(X)
    result = permutation_importance(model, X_scaled, y,
                                    n_repeats=30, random_state=42, n_jobs=-1)
    
    importance_df = pd.DataFrame({
        'Feature':    feature_names,
        'Importance': result.importances_mean,
        'Std':        result.importances_std,
    }).sort_values('Importance', ascending=False)
    
    print("\nTOP 10 VARIABLES LES PLUS IMPORTANTES :")
    print(importance_df.head(10).to_string(index=False))
    
    return importance_df


# =============================================================================
# 6. VISUALISATIONS
# =============================================================================

def plot_model_results(df: pd.DataFrame, model, scaler,
                       feature_names: list, importance_df: pd.DataFrame,
                       results_df: pd.DataFrame, save_path: str = None):
    """
    Produit un dashboard complet de visualisation des résultats ML.
    """
    X, y, _ = prepare_features(df)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Analyse des Modèles ML — Prédiction des Sorties de Liquidité',
                 fontsize=16, fontweight='bold', y=1.01)
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    # --- Plot 1 : Prédictions vs Réalisations ---
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df.index, y, label='Réalisé', color='#2c7bb6', linewidth=2)
    ax1.plot(df.index, y_pred, label='Prédit', color='#d7191c',
             linewidth=2, linestyle='--')
    ax1.fill_between(df.index, y, y_pred, alpha=0.15, color='orange')
    ax1.set_title('Flux nets 30j — Réalisé vs Prédit (Mds XOF)', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Milliards XOF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2 : Scatter résidus ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(y_pred, residuals, alpha=0.6, color='#1a9641', s=40)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_title('Résidus vs Prédictions', fontsize=12)
    ax2.set_xlabel('Prédictions (Mds XOF)')
    ax2.set_ylabel('Résidus')
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3 : Importance des variables ---
    ax3 = fig.add_subplot(gs[1, :2])
    top10 = importance_df.head(10)
    colors = ['#d7191c' if i < 3 else '#2c7bb6' if i < 7 else '#ffffbf'
              for i in range(len(top10))]
    bars = ax3.barh(top10['Feature'][::-1], top10['Importance'][::-1],
                    xerr=top10['Std'][::-1], color=colors[::-1],
                    capsize=3, alpha=0.85)
    ax3.set_title('Importance des variables (Permutation Importance)', fontsize=12)
    ax3.set_xlabel('Importance moyenne')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # --- Plot 4 : Comparaison modèles ---
    ax4 = fig.add_subplot(gs[1, 2])
    models_names = results_df['Modèle'].values
    rmse_vals = results_df['RMSE (Mds)'].values
    colors_m = ['#1a9641' if i == 0 else '#2c7bb6' for i in range(len(models_names))]
    ax4.barh(range(len(models_names)), rmse_vals, color=colors_m, alpha=0.8)
    ax4.set_yticks(range(len(models_names)))
    ax4.set_yticklabels([m[:20] for m in models_names], fontsize=9)
    ax4.set_title('RMSE par modèle (↓ meilleur)', fontsize=12)
    ax4.set_xlabel('RMSE (Mds XOF)')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # --- Plot 5 : Distribution des résidus ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(residuals, bins=20, color='#7b2d8b', alpha=0.75, edgecolor='white')
    ax5.axvline(0, color='red', linestyle='--')
    ax5.set_title('Distribution des résidus', fontsize=12)
    ax5.set_xlabel('Résidu (Mds XOF)')
    ax5.grid(True, alpha=0.3)
    
    # --- Plot 6 : Q-Q plot des résidus ---
    from scipy import stats
    ax6 = fig.add_subplot(gs[2, 1])
    stats.probplot(residuals, dist='norm', plot=ax6)
    ax6.set_title('Q-Q plot des résidus (normalité)', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # --- Plot 7 : Tableau des métriques ---
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    table_data = results_df[['Modèle', 'RMSE (Mds)', 'R²']].values
    table = ax7.table(
        cellText=[[str(v)[:18] if isinstance(v, str) else f'{v:.3f}' for v in row]
                  for row in table_data],
        colLabels=['Modèle', 'RMSE', 'R²'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax7.set_title('Tableau récapitulatif', fontsize=12, pad=20)
    
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
    
    print("Chargement des données...")
    df = create_full_dataset()
    
    print("\nPréparation des features...")
    X, y, feature_names = prepare_features(df)
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Features : {feature_names}")
    
    print("\nÉvaluation des modèles...")
    results_df = evaluate_models(X, y, feature_names, n_splits=4)
    
    print("\nEntraînement du modèle final...")
    model, scaler, model_name = train_best_model(X, y)
    
    print("\nImportance des variables...")
    importance_df = compute_feature_importance(model, X, y, feature_names, scaler)
    
    print("\nGénération des visualisations...")
    plot_model_results(df, model, scaler, feature_names,
                      importance_df, results_df,
                      save_path='C:/Users/X1 Carbon/Desktop/Stress Testing Projet/ml_results.png')
    
    print("\n[TERMINÉ] Module ML exécuté avec succès.")
