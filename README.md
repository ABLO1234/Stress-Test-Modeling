# Stress Test de Liquidité Bancaire — Projet Complet

> Modélisation avancée du risque de liquidité bancaire conformément aux standards **Bâle III / BCEAO / EBA**.

## Structure du projet

```
stress_test_liquidite/
│
├── data/
│   └── simulate_data.py         # Simulation base de données bancaire
│
├── models/
│   ├── ml_models.py             # 5 modèles ML (Ridge, RF, XGBoost, LightGBM, MLP)
│   └── advanced_analysis.py     # Sensibilité, Reverse Stress, Validation
│
├── scenarios/
│   └── stress_scenarios.py      # 4 scénarios (normal, adverse, sévère, systémique)
│
├── montecarlo/
│   └── monte_carlo.py           # Moteur Monte-Carlo (10 000+ simulations)
│
├── main.py                      # Pipeline complet
└── README.md
```

## Fonctionnalités

| Module | Contenu |
|---|---|
| **Données** | 60 observations mensuelles, 23 variables (macro + bilan) avec corrélations réalistes |
| **ML** | 5 modèles comparés via walk-forward CV, importance des variables |
| **Scénarios** | Normal / Adverse / Sévère / Crise systémique — calibrés sur données UEMOA |
| **Monte-Carlo** | Copule t-Student, 10 000 simulations/scénario, VaR/ES/P(breach) |
| **Avancé** | Analyse de sensibilité, Reverse Stress Testing, test Mincer-Zarnowitz, Diebold-Mariano |

## Indicateurs réglementaires calculés

- **LCR** (Liquidity Coverage Ratio) — seuil Bâle III : ≥ 100%
- **NSFR** (Net Stable Funding Ratio) — seuil Bâle III : ≥ 100%
- **VaR de liquidité** 95%, 99%, 99.9%
- **Expected Shortfall** 95%, 99%
- **Probabilité de défaillance** P(LCR < 100%)

## Utilisation

```bash
# Installation des dépendances
pip install numpy pandas scipy scikit-learn matplotlib xgboost lightgbm

# Exécution du pipeline complet
cd stress_test_liquidite
python main.py
```

## Calibration des scénarios

| Scénario | PIB (%/an) | Inflation (%) | Fuite DàV (%) | Durée (mois) |
|---|---|---|---|---|
| Normal | 5.5 | 2.2 | 5% | 1 |
| Adverse | 3.0 | 4.5 | 10% | 3 |
| Sévère | 0.5 | 8.0 | 20% | 6 |
| Crise systémique | -3.0 | 14.0 | 40% | 12 |

## Références

- Basel Committee on Banking Supervision (2013). *Basel III: The Liquidity Coverage Ratio*
- BCEAO (2023). *Rapport sur la Stabilité Financière dans l'UMOA*
- EBA (2022). *EU-Wide Stress Testing Methodology*
- McNeil, A., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management*
- IMF (2020). *Stress Testing Handbook — Systemic Risk Assessment*

## Auteur

Projet réalisé par Abdoulaye TANGARA
