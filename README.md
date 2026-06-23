# Stress Test de Liquidité Bancaire — Projet Complet 🚀

Une solution robuste et automatisée pour évaluer la résilience, la stabilité et les capacités de récupération de vos applications sous des charges de travail extrêmes. Ce framework permet de pousser le système au-delà de ses limites nominales afin d'identifier les goulets d'étranglement et de prévenir les pannes en production.

> Modélisation avancée du risque de liquidité bancaire conformément aux standards **Bâle III**.


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


---

## 📌 Fonctionnalités Clés

- **Simulation de Charge Massive :** Génération de requêtes simultanées injectant une charge supérieure aux limites attendues du système.
- **Analyse des Points de Rupture :** Identification précise du moment exact et des conditions sous lesquelles l'application échoue (fuites de mémoire, crashs).
- **Mesure de Récupération (Recoverability) :** Analyse du comportement du système après une panne pour vérifier s'il redémarre proprement.
- **Rapports de Performance :** Génération automatique de graphiques détaillant le temps de réponse et l'utilisation des ressources (CPU, RAM).

## 🛠️ Technologies Utilisées

- **Langage Principal :** Python / Node.js
- **Outil de Charge :** [Locust](https://locust.io) ou [K6](https://k6.io) *(à adapter selon votre outil)*
- **Monitoring :** Prometheus & Grafana

## 🚀 Démarrage Rapide

### Prérequis
Assurez-vous d'avoir installé la version 3.10+ de Python.

### Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com
   cd stress-testing
   ```

2. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```

### Exécution du Test
Lancez le script principal pour démarrer l'injection de charge :
```bash
python main.py --users 5000 --spawn-rate 100 --duration 10m
```

## 📊 Structure des Scénarios de Test

Le framework propose trois approches méthodologiques distinctes :
1. **Analyse de Scénarios :** Simulation d'événements critiques spécifiques (ex: pics de trafic).
2. **Sensibilité :** Modification de variables système sans narration explicite pour observer les réactions de l'infrastructure.
3. **Stress Inverse (Reverse Stress) :** Partir d'un état de panne critique défini pour identifier la charge exacte qui le déclenche.

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour proposer des modifications :
1. Créez un **Fork** du projet.
2. Créez votre branche de fonctionnalité (`git checkout -b feature/AmazingFeature`).
3. Validez vos modifications (`git commit -m 'Add some AmazingFeature'`).
4. Poussez la branche (`git push origin feature/AmazingFeature`).
5. Ouvrez une **Pull Request**.

## 📄 Licence

Distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

---
👨‍💻 **Auteur :** [tangaraabdoulaye7222](https://github.com) — Juin 2026

