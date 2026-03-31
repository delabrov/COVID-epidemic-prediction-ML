# Modélisation COVID-19 en France (SIR / SEIRD / SEIRDV)

Ce projet propose un pipeline de modélisation épidémiologique du COVID-19 en France, basé sur des modèles mathématiques compartimentaux et des séries temporelles journalières lissées disponibles sur internet.

L’objectif est de : 
1) reconstruire les compartiments \(S, E, I, R, D, V\) à partir des données observées,
2) estimer les paramètres dépendants du temps (notamment \(\beta(t)\), \(\mu(t)\), \(R_{\mathrm{eff}}(t)\)),
3) analyser l’évolution de la dynamique épidémique et l’impact de la vaccination.

Les principales données utilisées incluent :
- `new_cases_7d_avg`
- `new_deaths_7d_avg`
- `people_fully_vaccinated` (ou `people_vaccinated` en fallback)
- `population`

## Données utilisées et provenance

Les données proviennent du jeu de données **Our World in Data (OWID) COVID-19**, téléchargé automatiquement par le pipeline depuis :

- `https://covid.ourworldindata.org/data/owid-covid-data.csv`
- URL de secours : `https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv`

Le pipeline filtre ensuite le pays d’intérêt (France par défaut), applique des règles de nettoyage, puis construit des signaux lissés sur 7 jours pour limiter les effets de reporting hebdomadaire.

Les variables principales mobilisées pour la modélisation SEIRDV sont les suivantes :

- `new_cases_7d_avg` : incidence journalière lissée sur 7 jours, utilisée pour reconstruire les flux d’infection.
- `new_deaths_7d_avg` : mortalité journalière lissée sur 7 jours, utilisée pour reconstruire le flux vers le compartiment des décès.
- `people_fully_vaccinated` (fallback `people_vaccinated`) : stock cumulé de vaccinés, utilisé pour construire \(V(t)\).
- `population` : taille de la population \(N\), utilisée dans les termes d’interaction et les normalisations.

## Modèle SEIRDV

Le modèle SEIRDV étend SEIRD en ajoutant un compartiment de vaccination. Les compartiments sont :
\(S\) (susceptibles), \(E\) (exposés), \(I\) (infectieux), \(R\) (retirés/guéris), \(D\) (décès), \(V\) (vaccinés).

Le système dynamique continu est :

\[
\frac{dS}{dt} = -\beta(t)\frac{SI}{N} - \nu(t)S
\]
\[
\frac{dV}{dt} = \nu(t)S - (1-\varepsilon)\beta(t)\frac{VI}{N}
\]
\[
\frac{dE}{dt} = \beta(t)\frac{SI}{N} + (1-\varepsilon)\beta(t)\frac{VI}{N} - \sigma E
\]
\[
\frac{dI}{dt} = \sigma E - \gamma I - \mu(t)I
\]
\[
\frac{dR}{dt} = \gamma I
\]
\[
\frac{dD}{dt} = \mu(t)I
\]

## Paramètres et interprétation

Les paramètres du modèle sont définis comme suit :

- \(\beta(t)\) : taux de transmission effectif, dépendant du temps. Il agrège les effets de contact, comportements, mesures sanitaires et propriétés des variants.
- \(\sigma\) : taux de progression \(E \rightarrow I\), généralement fixé à \(1/\text{latent\_period\_days}\).
- \(\gamma\) : taux de sortie \(I \rightarrow R\), généralement fixé à \(1/\text{infectious\_period\_days}\).
- \(\mu(t)\) : taux de décès parmi les infectieux, estimé dynamiquement à partir des décès observés (avec délai).
- \(\nu(t)\) : flux de vaccination, dérivé de \(V(t)\).
- \(\varepsilon\) : efficacité vaccinale contre l’infection (constante dans la version actuelle). Le terme \((1-\varepsilon)\) traduit le risque résiduel d’infection chez les vaccinés.

Un proxy du nombre de reproduction effectif est utilisé :

\[
R_{\mathrm{eff}}(t) \approx \frac{\beta(t)}{\gamma + \mu(t)} \times \frac{S(t) + (1-\varepsilon)V(t)}{N}
\]

Cette expression permet de relier la transmission, la dynamique de sortie des infectieux et la réduction du réservoir effectivement susceptible via la vaccination.

## Documentation complète

La documentation scientifique détaillée du modèle (reconstruction des états, estimation paramétrique, hypothèses, limites et discussion) est disponible dans :

- [docs/seirdv_model_documentation.md](docs/seirdv_model_documentation.md)
