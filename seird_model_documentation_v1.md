# Documentation du modèle SEIRD (v1)

## Introduction

Le modèle SEIRD est une extension du modèle SIR pour représenter plus finement la dynamique d’une maladie infectieuse. La population totale `N` (supposée constante à l’échelle étudiée) est répartie en cinq compartiments : `S(t)` (susceptibles), `E(t)` (exposés non encore infectieux), `I(t)` (infectieux), `R(t)` (retirés/guéris) et `D(t)` (décès).

## Équations du modèle

Le système dynamique est :

```text
dS/dt = -beta(t) * S * I / N
dE/dt =  beta(t) * S * I / N - sigma * E
dI/dt =  sigma * E - gamma * I - mu(t) * I
dR/dt =  gamma * I
dD/dt =  mu(t) * I
```

`beta(t)` est le taux de transmission effectif. `sigma` est le taux de passage `E -> I` (souvent `1 / latent_period_days`). `gamma` est le taux de sortie `I -> R` (souvent `1 / infectious_period_days`). `mu(t)` est un taux de décès associé aux infectieux.

## Construction du projet

Le pipeline ne se limite pas à simuler les équations. Il reconstruit d’abord les états latents (`S, E, I, R, D`) à partir de données observées (cas, décès), puis estime les paramètres dynamiques.

Les signaux observés sont lissés sur 7 jours pour réduire le bruit hebdomadaire de notification.

## Reconstruction des compartiments

`I(t)` est reconstruit à partir des nouveaux cas via un profil d’infectivité (souvent gamma). `E(t)` est reconstruit via un profil de latence (souvent uniforme). `D(t)` provient des décès cumulés/observés selon le pipeline.

`R(t)` est reconstruit par bilan de masse à partir des infections cumulées, puis :

```text
S(t) = N - E(t) - I(t) - R(t) - D(t)
```

## Estimation des paramètres

Les paramètres fixes sont :

```text
sigma = 1 / latent_period_days
gamma = 1 / infectious_period_days
```

Dans l’implémentation actuelle, `beta(t)` est estimé via l’équation de `E` (et non via `dS/dt`) :

```text
beta(t) ~= [dE/dt + sigma * E_t] / [(S_t * I_t) / N]
```

`mu(t)` est estimé à partir des décès avec délai `tau_d`, sous forme intégrée glissante pour stabiliser le ratio :

```text
I_lagged(t) = I(t - tau_d)
mu(t) ~= Delta_D_window(t) / Sum_window(I_lagged)
```

avec :

```text
Delta_D_window(t) = D(t) - D(t - w)
Sum_window(I_lagged) = somme des I_lagged sur les w derniers jours
```

Cette forme est plus robuste qu’une division point par point.

## Hypothèses et approximations

Le modèle suppose un mélange homogène de la population, des durées moyennes de latence et d’infectiosité fixes, et une représentativité acceptable des données observées après lissage.

Les paramètres estimés (`beta`, `mu`) doivent être lus comme des paramètres effectifs agrégés.

## Gestion des instabilités initiales

Les premiers jours peuvent produire des estimations instables (faibles dénominateurs, sous-détection, retards de notification). Une exclusion robuste est appliquée à partir de critères médiane + MAD afin de réduire l’influence des valeurs aberrantes initiales.

## Conclusion

Le cadre SEIRD fournit une base interprétable pour reconstruire les états latents et suivre la dynamique temporelle des paramètres à partir de données observées. Cette base est ensuite prolongée par le modèle SEIRDV qui intègre explicitement la vaccination.
