# PESTO_Laplace_Beltrami_project

Les apports du projet sont contenus essentiellement dans :
- les modules
- les notebooks
- le dossier data

## Les modules

Nous avons modifié la classe Decomposer pour effectuer une décomposition uniquement sur le rayon, et pas sur les 3 coordonnées x, y et z. La nouvelle classe est "RadialDecomposer".

Nous avons également réécrit toutes les méthodes pour qu'elles ne soient plus encapsulées dans une classe. Ces fonctions, ainsi que d'autres outils pratiques, sont dans le module "MainFunctions".

Quelques méthodes utiles pour la SVD sont contenues dans "SVD".

Idem pour la génération de formes avec "ShapeGenerator".

## Les notebooks

Les notebooks sont :

- notebook_test_MF : il s'agit d'un tutoriel pour mettre en application les fonctions du modules "MainFunctions". Pour la suite, il faut lancer dans le notebook la commande **MF.compute_RY(M=100, Lmax=18, saveRY=True, savePath='data/precomputedRY/', returnRY=False, barycentre=None, verbose=True)**, cela peut prendre un peu de temps (quelques minutes) mais elle permet de calculer et de sauvegarde une bonne fois pour toute la matrice RY (4 ou 5 Go) utilisée pour les formes dans la suite.

- notebook_test_performance : il permet de mettre en évidence le gain d'efficacité avec l'algorithme optimisé de décomposition / reconstruction. La fin du notebook calcule le spectre des formes de L. Leprince et les sauvegarde.

- data_augmentation : il permet d'augmenter les donner de L. Leprince.

- shape_generator : il génère des formes de type union de pavés ou de type polynômes trigonométriques.

- Autoencodeur : une architecture type d'autoencodeur qu'on a utilisé. Il faut jouer un peu avec ! On présente au cours du notebook les différents outils qu'on a utilisés pour analyser les données.

- comparaison_SVD_autoencodeur : calcul du spectre issu de la SVD et issu de la troncature du spectre. On compare les résultats de ces méthodes avec ceux obtenus avec nos autoencodeurs.

## Les data

On a essayé d'un peu alléger. Il y a :

- Les formes de L. Leprince dans 3DGeneratedForms (et leur spectre dans 3DGeneratedSpectra).

- Les formes ci-dessus augmentées dans AugmentedLPForms (et leur spectre dans AugmentedLPSpectra).

- Les unions de pavés et leur spectre dans 3DGeneratedCuboids.

- Les formes et spectres issus de polynômes trigonométriques dans 3DGeneratedRadius.

- Les temps mesurés dans les tests de performance dans Performances.

- Les matrices RY sont habituellement stockées dans precomptutedRY (il faut relancer les calculs pour remplir le dossier, voir la partie notebooks, car les matrices sont très lourdes).