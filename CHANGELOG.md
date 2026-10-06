# [1.0.0-rc.4](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/compare/v1.0.0-rc.3...v1.0.0-rc.4) (2026-10-06)


### Bug Fixes

* **resultats:** supprimer le paramètre limit de l'historique et de l'annuaire ([dd027ff](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/dd027ff550408cb474d593c467c6892f8360d95c))

# [1.0.0-rc.3](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/compare/v1.0.0-rc.2...v1.0.0-rc.3) (2026-10-05)


### Bug Fixes

* **tests:** aligner test_couverture sur l'authentification ([b82dc41](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/b82dc41f5ffcd18d2086fc0cfc6c5fd186577de7))


### Features

* **auth:** lier les prédictions à un compte utilisateur ([c2a2d8b](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/c2a2d8bc24495314448c6324ecd4563dc30e817e))

# [1.0.0-rc.2](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/compare/v1.0.0-rc.1...v1.0.0-rc.2) (2026-10-01)


### Features

* ajout de la partie ancien employé, depuis y.csv ([b583915](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/b583915c284faaa85d6435ec4fc5f4aa5032fa12))

# 1.0.0-rc.1 (2026-09-28)


### Bug Fixes

* add ajout d'un job de test des dépendances ([f4b292d](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/f4b292d6595d6f0566b2df652a321c0f477ec4cf))
* add httpx dependency for FastAPI tests ([6000f9d](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/6000f9dbb501a4968b7082f34bbb819acd967e8f))
* **exploration,etude:** lisibilité des graphes et SHAP local multi-modèles ([18d6268](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/18d6268714cfc1755a0a50f36e5a0c65c4dd1841))
* resolution des conflits de merge avec Main ([c89237c](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/c89237c7fb0d245d7b40c1c45708ea6381bcdb2a))
* **script ini-db:** création des tables si elles n'existent pas ([d861dd8](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/d861dd8b60f25822084204df886582e718696ca3))
* **shap:** séparer le local du recalcul global et annoter les matrices ([208b6f5](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/208b6f55ff6b02541ebf9e208783431c3a7b2090))
* sync Data and models then surface compose logs ([ae28739](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/ae28739662a9227776bc7730e60754347a3c98e2))


### Features

* add release ([1322581](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/1322581289663ca19db1ff22e08f9266c51f2de2))
* ajout d'un golden path ([30bd680](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/30bd68086085e9bf3597fd7cd2a86a4b96107cd2))
* ajout de l'analyse shap pour modèle non entrainé automatiquement ([6b6d9b4](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/6b6d9b43a453d516f1f5bf290116953bedc4fad0))
* ajout de la fonction de calcule du seuil pour les modeles qui peuvent l'utiliser ([fd7022b](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/fd7022ba05f1725dde40939e7e7de43c77e9cb51))
* **attrition:** livrer L2 calibré avec son seuil de décision ([8deca26](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/8deca266b6788c06d26ca71dadde0fcc64b4da5a))
* **cv:** rendre dynamique le random_state de StratifiedKFold ([bbcd3a4](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/bbcd3a44c81e0e750a12e19843839e91289e4bdf))
* **docker:** containerize FastAPI app and add docker-compose with CSV-to-SQL export script ([3480d44](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/3480d447250ec9240a423d9656a5c831314260ab))
* **hyperparameters_interface:** limiter la sélection des modèles aux modèles déjà entraînés ([468caaf](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/468caaff95e30e440425fdfbc5af03adb10d1cba))
* **interface:** ajoute le tri dynamique et la sélection de colonnes pour les résultats + enregsitrement en local de deux modèles hyperparamêtrés ([71cbb4d](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/71cbb4d24c9b8baa9fb422a8d9499a2a2f4ab7b8))
* **model:** integrate interactive model selector and dynamic metrics ([003ad73](https://github.com/Aurelien77/P5-Deployez-un-modele-de-Machine-Learning/commit/003ad73c3a21bc7f17fc2a2620ec51cecc097071))
