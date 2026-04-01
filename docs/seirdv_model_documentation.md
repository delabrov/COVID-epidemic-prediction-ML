# Documentation du modèle SEIRDV pour la dynamique COVID-19 en France

## 1. Introduction

La pandémie de COVID-19 a mis en évidence la nécessité de disposer d’outils capables de relier des observations (cas confirmés, décès rapportés, couverture vaccinale) à des mécanismes dynamiques de transmission. Les modèles compartimentaux sont adaptés à cet objectif, car ils représentent explicitement les flux entre états épidémiologiques et permettent d’interpréter les paramètres estimés.

Dans ce projet, l’analyse est menée à l’échelle nationale (France) à partir de séries journalières lissées sur 7 jours pour atténuer les effets de reporting hebdomadaire. L’approche progresse d’un cadre SIR vers SEIRD, puis SEIRDV avec vaccination explicite. L’objectif est de reconstruire les compartiments latents `S, E, I, R, D, V`, d’estimer des paramètres dépendants du temps (`beta(t)`, `mu(t)`), puis d’analyser leur dynamique.

## 2. Description du modèle SEIRDV

Le modèle considère six compartiments : `S` (susceptibles), `E` (exposés non encore infectieux), `I` (infectieux), `R` (retirés/guéris), `D` (décès), `V` (vaccinés). La population totale `N` est supposée constante à l’échelle d’étude.

Le système dynamique est :

```text
dS/dt = -beta(t) * S * I / N - nu(t) * S
dV/dt =  nu(t) * S - (1 - epsilon_v) * beta(t) * V * I / N
dE/dt =  beta(t) * S * I / N + (1 - epsilon_v) * beta(t) * V * I / N - sigma * E
dI/dt =  sigma * E - gamma * I - mu(t) * I
dR/dt =  gamma * I
dD/dt =  mu(t) * I
```

Ici, `epsilon_v` est l’efficacité vaccinale contre l’infection (constante dans la version actuelle). Le terme `(1 - epsilon_v)` modélise le risque résiduel d’infection chez les vaccinés.

## 3. Reconstruction des états

Les séries utilisées sont `new_cases_7d_avg`, `new_deaths_7d_avg`, `people_fully_vaccinated` (fallback `people_vaccinated`) et `population`.

`E(t)` est reconstruit par convolution de l’incidence avec un noyau de latence (uniforme dans la configuration standard). `I(t)` est reconstruit par convolution avec un profil d’infectivité (gamma dans la configuration standard). `D(t)` provient des décès observés (cumulés dans le pipeline d’estimation).

`R(t)` est reconstruit par bilan de masse à partir des infections cumulées, puis `S(t)` est obtenu par conservation :

```text
S(t) = N - E(t) - I(t) - R(t) - D(t)
```

`V(t)` vient directement des données vaccinales, et `nu(t)` est approché par dérivation numérique de `V(t)`.

## 4. Estimation des paramètres

Les paramètres fixes sont :

```text
sigma = 1 / latent_period_days
gamma = 1 / infectious_period_days
```

La dérivée `dE/dt` est estimée numériquement (gradient ou différence finie selon la configuration), puis :

```text
beta(t) ~= [dE/dt + sigma * E_t] / [((S_t + (1 - epsilon_v) * V_t) * I_t) / N]
```

Cette forme est exactement celle implémentée dans `estimate_seirdv_parameters.py`.

Pour `mu(t)`, le pipeline n’utilise pas un ratio point par point, mais une estimation intégrée glissante avec délai décès `tau_d` :

```text
I_lagged(t) = I(t - tau_d)
mu(t) ~= Delta_D_window(t) / Sum_window(I_lagged)
```

avec :

```text
Delta_D_window(t) = D(t) - D(t - w)
Sum_window(I_lagged) = somme des I_lagged sur les w derniers jours
```

Cette formulation est plus stable que `DeltaD / I_lagged` point par point.

Le proxy utilisé pour le nombre de reproduction effectif est :

```text
Reff(t) ~= [beta(t) / (gamma + mu(t))] * [S_t + (1 - epsilon_v) * V_t] / N
```

Les séries `beta` et `mu` sont ensuite lissées par moyenne glissante. Les périodes initiales instables sont gérées par une détection robuste fondée sur médiane + MAD.

