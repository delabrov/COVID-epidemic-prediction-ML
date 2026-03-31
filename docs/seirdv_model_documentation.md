# Documentation du modèle SEIRDV pour la dynamique COVID-19 en France

## 1. Introduction

La pandémie de COVID-19 a mis en évidence la nécessité de disposer d’outils capables de relier des observations (cas confirmés, décès rapportés, couverture vaccinale) à des mécanismes dynamiques sous-jacents de transmission. Les modèles compartimentaux constituent un cadre particulièrement adapté à cet objectif, parce qu'ils permettent de représenter explicitement les flux entre états épidémiologiques et d’interpréter les paramètres estimés en termes biologiques et populationnels.

Dans ce projet, l’analyse est menée à l’échelle nationale (France) à partir de séries journalières lissées sur 7 jours, afin de réduire les effets de reporting hebdomadaire (rattrapages des détections chaque lundi). L’approche a progressé d’un cadre SIR vers un modèle SEIRDV intégrant explicitement la vaccination. L’ambition n’est pas seulement de produire un ajustement descriptif des données observées, mais de reconstruire les compartiments latents S, E, I, R, D, V, d’estimer des paramètres dépendants du temps beta(t), mu(t), et de caractériser leur évolution épidémiologique au cours du temps.

L’objectif central du modèle SEIRDV est donc double. D’une part, il s’agit de fournir une reconstruction cohérente des états non observés à partir de signaux partiels et bruités. D’autre part, il s’agit d’obtenir des indicateurs dynamiques interprétables, notamment la transmissibilité effective et la létalité apparente des infections, en tenant compte de la réduction de susceptibilité induite par la vaccination.

## 2. Description du modèle SEIRDV

Le modèle SEIRDV considère six compartiments : les susceptibles (S), les exposés non encore infectieux (E), les infectieux (I), les retirés (R), les décès (D), et les vaccinés (V). La population totale (N) est supposée constante à l’échelle de temps étudiée, en négligeant migrations et variations démographiques lentes.

Le système différentiel continu s’écrit :

dS/dt = -beta(t) x S I / N - nu(t) x S

dV/dt = nu(t) x S - (1-epsilon) x beta(t) x V I / N

dE/dt = beta(t) x S I / N + (1-epsilon) x beta(t) x V I / N - sigma x E

dI/dt = sigma x E - gamma x I - mu(t) x I

dR/dt = gamma x I

dD/dt = mu(t) x I

Dans ce cadre, beta(t) représente l’intensité de transmission effective, sigma le taux de sortie du compartiment exposé (inverse de la période de latence), gamma le taux de guérison/sortie infectieuse (inverse de la période infectieuse), mu(t) le taux de décès associé à l’état infectieux, nu(t) le flux de vaccination, et varepsilon l’efficacité vaccinale contre l’infection (supposée constante dans la version actuelle).

L’expression (1-epsilon) x beta x V I / N traduit le fait qu’un individu vacciné n’est pas nécessairement totalement protégé : sa contribution au risque d’infection est réduite d’un facteur (1-epsilon), mais reste positive si epsilon < 1.

## 3. Reconstruction des états

La reconstruction des compartiments repose sur les séries observées \(`new_cases_7d_avg`, `new_deaths_7d_avg`, `people_fully_vaccinated`.

Le compartiment exposé E(t) est estimé par convolution de l’incidence observée avec un noyau de latence uniforme. Cette opération redistribue les nouveaux cas détectés sur une fenêtre de latence, ce qui revient à approximer la mémoire du processus d’infection avant l’entrée en infectiosité. Le compartiment infectieux I(t) est reconstruit via une convolution avec un noyau gamma d’infectivité, mieux adapté qu’un noyau uniforme pour représenter une probabilité de présence infectieuse asymétrique dans le temps.

Le compartiment des décès D(t) est obtenu à partir des décès observés (cumulés ou flux selon le pipeline), tandis que R(t) est reconstruit par bilan de masse à partir des infections cumulées, en retirant les contributions de E, I et D. Le compartiment susceptible est ensuite obtenu comme résidu :

S(t) = N - E(t) - I(t) - R(t) - D(t)

Dans l’extension SEIRDV, V(t) provient directement des données vaccinales. Le flux nu(t) est estimé par dérivation numérique de V(t). Cette étape relie explicitement les données de couverture vaccinale à la dynamique des flux inter-compartimentaux.

Cette reconstruction repose sur plusieurs hypothèses structurantes : homogénéité de mélange à l’échelle nationale, délais biologiques moyens stationnaires pour sigma et gamma, qualité suffisante du lissage 7 jours pour réduire le bruit de reporting, et interprétation de la vaccination par une réduction de susceptibilité moyenne unique epsilon.

## 4. Estimation des paramètres

L’estimation est effectuée point par point, puis stabilisée par lissage temporel. Les paramètres fixes sont :

sigma = 1 / latent_period_days
gamma = 1 / infectious_period_days

Le paramètre beta(t) est identifié à partir de l’équation de E. En notant Delta E_t une approximation de dE/dt par différence finie, on obtient :

beta(t) ~ Delta E_t + sigma x E_t x (S_t + (1-epsilon) x V_t) x I_t / N

Cette forme montre que la vaccination agit via une population effectivement susceptible S_t + (1-epsilon) x V_t, et non via S_t seul.

Le paramètre mu(t) est estimé à partir du flux de décès, en intégrant un délai moyen infection-décès \(`death_delay_days`\). Une approximation usuelle est :

mu(t) ~ \Delta D_t x I_{t - tau_d}

avec tau_d\ le délai de décès et Delta D_t la différence finie des décès cumulés (ou directement les décès journaliers selon la convention du pipeline). Cette quantité doit être interprétée comme un proxy dynamique de létalité conditionnelle à l’état infectieux reconstruit, et non comme une IFR strictement démographique.

Le nombre de reproduction effectif est défini comme un proxy mécaniste :

R_eff(t) ~ beta(t) / gamma + mu(t) x (S_t + (1-epsilon) x V_t) / N

Les estimateurs bruts étant sensibles au bruit (dérivées numériques et divisions par des quantités parfois faibles), un lissage par moyenne glissante est appliqué aux séries paramétriques. Ce choix réduit la variance locale tout en préservant les tendances de moyen terme utiles à l’interprétation.

Les zones instables, en particulier en début de série lorsque les compartiments reconstruits sont fragiles, sont traitées par une méthode fondée sur la médiane et la MAD (Median Absolute Deviation). Concrètement, les valeurs aberrantes au-delà d’un seuil robuste autour de la médiane sont exclues de l’analyse ou neutralisées. Ce choix est justifié par la non-gaussianité fréquente des erreurs en contexte épidémiologique et par la sensibilité des ratios dynamiques aux petits dénominateurs.

## 6. Discussion scientifique

La quantité \(\beta(t)\) doit être interprétée comme une transmissibilité effective agrégée, qui condense à la fois des facteurs biologiques (propriétés des variants), comportementaux (contacts, adhésion aux mesures), environnementaux et institutionnels. Ce paramètre n’est donc pas une constante intrinsèque du pathogène, mais un indicateur composite dépendant du contexte spatio-temporel.

Le paramètre \(\mu(t)\) représente une intensité de mortalité conditionnelle à l’état infectieux reconstruit. Il capture simultanément la sévérité clinique moyenne, la structure des cas, la pression hospitalière et les délais de notification. Son interprétation doit rester prudente : une baisse de \(\mu(t)\) peut refléter une amélioration de la prise en charge, une modification de l’âge moyen des infectés, un effet de vaccination contre les formes graves, ou un mélange de ces mécanismes.

L’intégration explicite de la vaccination est un apport majeur du modèle SEIRDV. En introduisant \(V\), \(\nu(t)\) et \(\varepsilon\), le modèle distingue mieux l’érosion du réservoir réellement susceptible de la dynamique infectieuse pure. Cette structure améliore l’interprétation de \(R_{\mathrm{eff}}\), notamment dans les périodes de forte montée de couverture vaccinale.

Les limites principales demeurent toutefois importantes. L’hypothèse \(\varepsilon\) constante est forte alors que l’efficacité varie selon le temps, les doses, les variants et l’échappement immunitaire. Le modèle reste agrégé sans stratification par âge, territoire ou statut immunitaire détaillé. Enfin, les dérivées numériques et ratios introduisent mécaniquement du bruit, même après lissage.

