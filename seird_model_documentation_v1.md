# SEIRD Model Documentation (v1)

## Introduction

Le modèle SEIRD est une extension du modèle épidémiologique classique SIR, conçue pour représenter de manière plus réaliste la dynamique de propagation d’une maladie infectieuse en intégrant explicitement les phases d’incubation et les décès. Il repose sur une décomposition de la population totale \( N \), supposée constante à l’échelle temporelle considérée, en cinq compartiments dynamiques : les individus susceptibles \( S(t) \), les individus exposés mais non encore infectieux \( E(t) \), les individus infectieux \( I(t) \), les individus retirés (guéris ou immunisés) \( R(t) \), et les individus décédés \( D(t) \).

---

## Équations du modèle

Le système dynamique est régi par les équations différentielles suivantes :

\[
\frac{dS}{dt} = -\beta(t) \frac{S(t) I(t)}{N}
\]

\[
\frac{dE}{dt} = \beta(t) \frac{S(t) I(t)}{N} - \sigma E(t)
\]

\[
\frac{dI}{dt} = \sigma E(t) - \gamma I(t) - \mu(t) I(t)
\]

\[
\frac{dR}{dt} = \gamma I(t)
\]

\[
\frac{dD}{dt} = \mu(t) I(t)
\]

Dans ce cadre, le paramètre \( \beta(t) \) représente le taux de transmission, c’est-à-dire le nombre moyen de contacts infectieux par unité de temps, et peut varier au cours du temps pour refléter les changements de comportement, les politiques sanitaires ou l’émergence de nouveaux variants.

Le paramètre \( \sigma \) correspond à l’inverse de la durée moyenne d’incubation, définissant la vitesse de passage du compartiment exposé au compartiment infectieux. Le paramètre \( \gamma \) représente le taux de guérison, soit l’inverse de la durée moyenne d’infectiosité. Enfin, \( \mu(t) \) correspond au taux de mortalité instantané parmi les individus infectieux, et peut également varier dans le temps.

---

## Construction du projet

Le projet repose sur une approche inverse consistant non pas à simuler directement ces équations, mais à reconstruire les états du système à partir de données observées, puis à estimer les paramètres du modèle à partir de ces états.

Les données utilisées sont principalement des séries temporelles de nouveaux cas et de nouveaux décès, lissées par moyenne mobile sur sept jours afin de réduire les effets de bruit et de reporting irrégulier.

---

## Reconstruction des compartiments

La reconstruction des compartiments repose sur plusieurs approximations structurantes.

Le nombre d’individus infectieux \( I(t) \) est approximé comme la somme glissante des nouveaux cas sur une fenêtre correspondant à la durée moyenne d’infectiosité. Cela revient à considérer que chaque individu reste infectieux pendant un nombre fixe de jours.

Le compartiment exposé \( E(t) \) est reconstruit de manière analogue, en utilisant une version décalée et cumulée des nouveaux cas, correspondant à la durée d’incubation.

Le compartiment des décès \( D(t) \) est obtenu à partir des décès cumulés observés ou reconstruits.

Le compartiment des retirés \( R(t) \) est estimé comme la différence entre les infections cumulées et les individus encore présents dans les compartiments \( E \), \( I \) et \( D \).

Le compartiment des susceptibles \( S(t) \) est alors déduit par conservation de la population totale, selon la relation :

\[
S(t) = N - E(t) - I(t) - R(t) - D(t)
\]

---

## Estimation des paramètres

L’estimation des paramètres repose sur l’inversion directe des équations différentielles.

Le paramètre de transmission \( \beta(t) \) est obtenu en réarrangeant l’équation de \( dS/dt \), ce qui permet de l’exprimer en fonction de la dérivée de \( S(t) \), de \( S(t) \), de \( I(t) \) et de \( N \).

Le paramètre de mortalité \( \mu(t) \) est estimé à partir de l’équation des décès, en reliant le flux de décès au nombre d’individus infectieux, avec prise en compte d’un délai entre infection et décès.

Afin de rendre cette estimation plus robuste, une approche intégrée est utilisée, consistant à lisser les flux de décès et à les comparer à une somme glissante du nombre d’infectés retardés. Cette méthode permet de réduire la sensibilité aux fluctuations locales et au bruit des données.

---

## Hypothèses et approximations

Plusieurs hypothèses importantes structurent ce modèle.

La population est considérée comme homogène et bien mélangée, ce qui signifie que tous les individus ont la même probabilité d’entrer en contact les uns avec les autres.

Les durées d’incubation et d’infectiosité sont supposées constantes, ce qui simplifie la dynamique mais ignore la variabilité individuelle.

Les données observées sont supposées être proportionnelles aux flux réels, bien que l’on sache que le nombre de cas est généralement sous-estimé, en particulier au début de l’épidémie.

Les délais entre infection et décès sont modélisés de manière déterministe, alors qu’ils sont en réalité distribués.

---

## Gestion des instabilités initiales

Une difficulté majeure du modèle réside dans la qualité des données initiales, en particulier lors des premières phases de l’épidémie, où les cas sont fortement sous-déclarés et les décès peuvent être rapportés de manière irrégulière.

Cela conduit à des estimations instables des paramètres, notamment du taux de mortalité \( \mu(t) \), lorsque le nombre d’infectés estimés est très faible.

Afin de corriger cet effet, une procédure robuste d’exclusion de la phase initiale a été introduite, basée sur une détection automatique des valeurs aberrantes de \( \mu(t) \) à l’aide de statistiques robustes (médiane et écart absolu médian).

Cette approche permet d’identifier et d’exclure la région temporelle où les estimations ne sont pas fiables, sans recourir à des seuils arbitraires.

---

## Conclusion

Ce travail constitue une première étape visant à estimer de manière cohérente les paramètres d’un modèle épidémiologique à partir de données observées.

L’objectif n’est pas de produire une simulation fidèle de l’épidémie, mais de reconstruire des quantités latentes et d’obtenir des paramètres interprétables.

Ces résultats pourront ensuite être utilisés dans des approches plus avancées, notamment des modèles d’apprentissage automatique destinés à reproduire ou prévoir les dynamiques observées.