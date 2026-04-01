# Résultats SEIRDV (France) à partir des graphiques

## 1. Périmètre et objectif

Ce document synthétise les résultats obtenus avec le pipeline SEIRDV sur la France (fenêtre `2020-01-05` à `2023-07-01`, 1274 jours). L’objectif est de présenter les paramètres dynamiques estimés, d’expliquer la méthode d’estimation, puis d’interpréter les variations observées d’un point de vue épidémiologique.

Les figures principales présentées sont `beta(t)`, `mu(t)`, `R_eff(t)` et `nu(t)`.

## 2. Rappel de la méthode d’estimation

Le modèle utilisé est :

```text
dS/dt = -beta(t) * S * I / N - nu(t) * S
dV/dt =  nu(t) * S - (1 - epsilon_v) * beta(t) * V * I / N
dE/dt =  beta(t) * S * I / N + (1 - epsilon_v) * beta(t) * V * I / N - sigma * E
dI/dt =  sigma * E - gamma * I - mu(t) * I
dR/dt =  gamma * I
dD/dt =  mu(t) * I
```

Configuration principale utilisée :

- `latent_period_days = 5` donc `sigma = 0.2`
- `infectious_period_days = 14` donc `gamma = 0.07142857`
- `death_delay_days = 14`
- `epsilon_v = 0.6` (efficacité vaccinale fixée)
- dérivée numérique `gradient`
- lissage par moyenne glissante (`window = 7`)
- exclusion des zones instables par règle robuste médiane + MAD

Formules de calcul effectivement employées dans le code :

```text
beta(t) ~= [dE/dt + sigma * E_t] / [((S_t + (1 - epsilon_v) * V_t) * I_t) / N]
mu(t) ~= Delta_D_window(t) / Sum_window(I_lagged)
Reff(t) ~= [beta(t) / (gamma + mu(t))] * [S_t + (1 - epsilon_v) * V_t] / N
nu(t) ~= dV/dt (approché par différence finie, puis lissé)
```

## 3. Qualité d’ajustement observée

Sur cette exécution, les performances de reconstruction sont élevées :

- Cas : `R2 = 0.9808`, `RMSE = 7429.7`, `MAE = 2278.8`
- Décès : `R2 = 0.9333`, `RMSE = 42.25`, `MAE = 21.97`

Ces scores soutiennent une bonne cohérence entre signaux observés et flux reconstruits, en particulier pour l’analyse des tendances de paramètres.

## 4. Valeurs estimées des paramètres

### 4.1 Valeurs numériques (séries lissées)

| Paramètre | Moyenne | Médiane | P05 | P95 | Min (date) | Max (date) |
|---|---:|---:|---:|---:|---|---|
| `beta_smoothed` | 0.1588 | 0.1469 | 0.0603 | 0.3116 | 0.0374 (2020-11-21) | 0.3627 (2022-09-23) |
| `mu_smoothed` | 0.000568 | 0.000323 | 0.000067 | 0.002051 | 0.000051 (2022-08-08) | 0.003022 (2020-12-31) |
| `R_eff_proxy_smoothed` | 1.0160 | 0.9878 | 0.5401 | 1.5385 | 0.3941 (2023-01-08) | 2.1936 (2021-07-30) |
| `nu_flow_smoothed` (pers/jour) | 41,752 | 656 | 0 | 261,543 | 0 (2020-01-05) | 428,123 (2021-07-01) |

Informations complémentaires utiles :

- `R_eff > 1` pendant 513 jours et `< 1` pendant 547 jours (sur 1060 jours valides).
- Le pic de vaccination journalière est cohérent avec la phase d’accélération de la campagne en 2021.

### 4.2 Évolution temporelle (moyennes annuelles)

- `beta_smoothed` : 2020 = 0.082, 2021 = 0.116, 2022 = 0.228, 2023 = 0.198
- `mu_smoothed` : 2020 = 0.00134, 2021 = 0.000836, 2022 = 0.000117, 2023 = 0.000306
- `R_eff_smoothed` : 2020 = 1.146, 2021 = 1.038, 2022 = 1.016, 2023 = 0.866
- `nu_smoothed` (pers/jour) : 2020 ~= 0, 2021 ~= 135,490, 2022 ~= 10,120, 2023 ~= 244

Lecture épidémiologique : la mortalité conditionnelle `mu(t)` baisse fortement après 2020, la vaccination est maximale en 2021, et `R_eff` revient progressivement vers des niveaux plus contrôlés en 2023.

## 5. Figures principales

Note GitHub : pour éviter les problèmes d’affichage relatifs, les images sont référencées via URL brute (`raw.githubusercontent.com`) et un lien fichier est ajouté.

### 5.1 `beta(t)`

![SEIRDV beta(t)](https://raw.githubusercontent.com/delabrov/COVID-epidemic-prediction-ML/main/outputs/figures/seirdv/covid_france_seirdv_beta_estimates.png)

Fichier : [outputs/figures/seirdv/covid_france_seirdv_beta_estimates.png](../outputs/figures/seirdv/covid_france_seirdv_beta_estimates.png)

Interprétation : la variabilité de `beta(t)` suit les changements de transmissibilité effective (comportements, variants, saisonnalité, mesures). Les pics correspondent à des phases de reprise épidémique.

### 5.2 `mu(t)`

![SEIRDV mu(t)](https://raw.githubusercontent.com/delabrov/COVID-epidemic-prediction-ML/main/outputs/figures/seirdv/covid_france_seirdv_mu_estimates.png)

Fichier : [outputs/figures/seirdv/covid_france_seirdv_mu_estimates.png](../outputs/figures/seirdv/covid_france_seirdv_mu_estimates.png)

Interprétation : les pics de `mu(t)` apparaissent surtout au début des vagues les plus sévères, puis la tendance centrale baisse avec l’amélioration de la prise en charge et l’immunisation.

### 5.3 `R_eff(t)`

![SEIRDV R_eff(t)](https://raw.githubusercontent.com/delabrov/COVID-epidemic-prediction-ML/main/outputs/figures/seirdv/covid_france_seirdv_reff_proxy.png)

Fichier : [outputs/figures/seirdv/covid_france_seirdv_reff_proxy.png](../outputs/figures/seirdv/covid_france_seirdv_reff_proxy.png)

Interprétation : quand `R_eff > 1`, la dynamique est expansive; quand `R_eff < 1`, la dynamique est régressive. La série oscille autour de 1, ce qui est cohérent avec des alternances de reprise et de contrôle.

### 5.4 `nu(t)`

![SEIRDV nu(t)](https://raw.githubusercontent.com/delabrov/COVID-epidemic-prediction-ML/main/outputs/figures/seirdv/covid_france_seirdv_nu_flow.png)

Fichier : [outputs/figures/seirdv/covid_france_seirdv_nu_flow.png](../outputs/figures/seirdv/covid_france_seirdv_nu_flow.png)

Interprétation : `nu(t)` reflète la dynamique de campagne vaccinale. La montée rapide en 2021 puis la décroissance en 2022-2023 sont cohérentes avec une phase d’extension de couverture puis un régime de rappels.

## 6. Comparaison avec la littérature scientifique

Le tableau ci-dessous compare les ordres de grandeur du modèle aux valeurs généralement rapportées dans la littérature COVID-19.

| Quantité | Estimation SEIRDV (ce projet) | Ordres de grandeur dans la littérature | Cohérence |
|---|---|---|---|
| `beta(t)` journalier | médiane `0.147`, P05-P95 `0.060-0.312` | souvent de l’ordre `0.1-0.4 / jour` dans des SEIR calibrés (fortement dépendant du modèle, de la période et de la définition des compartiments) | Oui, ordre de grandeur plausible |
| `R_eff(t)` | médiane `0.988`, P95 `1.539`, max `2.194` | phases de contrôle proches de 1, avec dépassements >1 pendant les vagues ; `R0` initial souvent `~2-4` au début de la pandémie | Oui |
| `mu(t)` (taux dynamique) | médiane `3.23e-4 / jour`, P95 `2.05e-3 / jour` | non directement comparable à l’IFR, mais la dynamique décroissante post-2020 est cohérente avec la baisse de sévérité observée | Oui (avec prudence) |
| Proxy fatalité `mu/(gamma+mu)` | médiane `0.45%`, P95 `2.8%` | IFR globale initiale souvent estimée autour de `~0.5-1%`, très hétérogène selon âge, période et immunité | Globalement cohérent |
| `epsilon_v` (fixé) | `0.6` | VE contre l’infection : élevée au début contre souches historiques, plus modérée/variable contre Omicron avec décroissance dans le temps | Plausible comme moyenne agrégée |

Points de vigilance :

- `beta` est structurellement dépendant du modèle et de la reconstruction des compartiments ; la comparaison inter-études doit rester qualitative.
- `mu(t)` n’est pas une IFR démographique directe. C’est un paramètre effectif conditionné par les choix de reconstruction (`I_lagged`, fenêtre glissante, délai décès).
- `epsilon_v` constant simplifie fortement la réalité (waning, rappels, variants).

## 7. Références (littérature)

1. Liu Y et al. *The reproductive number of COVID-19 is higher compared to SARS coronavirus* (J Travel Med, 2020). PubMed: https://pubmed.ncbi.nlm.nih.gov/32052846/
2. Meyerowitz-Katz G, Merone L. *A systematic review and meta-analysis of published research data on COVID-19 infection fatality rates* (2020). PubMed: https://pubmed.ncbi.nlm.nih.gov/33007452/
3. O'Driscoll M et al. *Age-specific mortality and immunity patterns of SARS-CoV-2* (Nature, 2021). PubMed: https://pubmed.ncbi.nlm.nih.gov/33137809/
4. Lu L et al. *Effectiveness of COVID-19 Vaccines against SARS-CoV-2 Omicron Variant: A Systematic Review and Meta-Analysis* (Vaccines, 2023). PubMed: https://pubmed.ncbi.nlm.nih.gov/36560590/
5. Hu J et al. *Real-World Effectiveness of COVID-19 Vaccines against Omicron* (Vaccines, 2023). PubMed: https://pubmed.ncbi.nlm.nih.gov/36851102/
