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
