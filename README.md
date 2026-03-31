# Modélisation COVID-19 en France (SIR / SEIRD / SEIRDV)

Ce projet propose un pipeline de modélisation épidémiologique du COVID-19 en France, basé sur des modèles compartimentaux et des séries temporelles journalières lissées.

L’objectif est de :
- reconstruire les compartiments latents \(S, E, I, R, D, V\) à partir des données observées,
- estimer des paramètres dynamiques dépendants du temps (notamment \(\beta(t)\), \(\mu(t)\), \(R_{\mathrm{eff}}(t)\)),
- analyser l’évolution de la dynamique épidémique et l’impact de la vaccination.

Les principales données utilisées incluent :
- `new_cases_7d_avg`
- `new_deaths_7d_avg`
- `people_fully_vaccinated` (ou `people_vaccinated` en fallback)
- `population`

## Documentation scientifique

La documentation complète du modèle SEIRDV (équations, reconstruction des états, estimation des paramètres, hypothèses, limites, résultats et discussion scientifique) est disponible ici :

- [Documentation détaillée SEIRDV](docs/seirdv_model_documentation.md)

## Exécution rapide

Installer les dépendances :

```bash
python -m pip install -r requirements.txt
```

Lancer le pipeline principal :

```bash
python main.py --country France
```
