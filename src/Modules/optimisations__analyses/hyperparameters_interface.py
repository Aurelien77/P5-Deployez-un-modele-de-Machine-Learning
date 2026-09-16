# ============================================================
# hyperparameters_interface.py
# ============================================================
#
# Interface graphique pour :
#   - sélectionner les modèles
#   - choisir automatique / manuel
#   - lancer GridSearchCV
#   - modifier le seuil de classification
#   - afficher les résultats
#   - trier les résultats
#   - sélectionner plusieurs colonnes à afficher
#   - analyser les features
#   - sauvegarder les meilleurs modèles
#
# ============================================================


import sys
import traceback
import time
from pathlib import Path
from collections import OrderedDict

import joblib
import pandas as pd
import numpy as np
import ipywidgets as widgets

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from IPython.display import (
    display,
    clear_output,
    HTML
)


# ============================================================
# 1. GRILLES AUTOMATIQUES
# ============================================================

def obtenir_grilles_hyperparametres():

    grilles = {

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        "Dummy_Stratified": {},


        # ----------------------------------------------------
        # MODELES LINEAIRES
        # ----------------------------------------------------

        "LogisticRegression_None": {},

        "LogisticRegression_L1": {
            "classifier__C": [
                0.001,
                0.01,
                0.1,
                1,
                10
            ]
        },

        "LogisticRegression_L2": {
            "classifier__C": [
                0.001,
                0.01,
                0.1,
                1,
                10
            ]
        },

        "Lasso_LogReg": {
            "classifier__C": [
                0.001,
                0.01,
                0.1,
                1,
                10
            ]
        },

        "ElasticNet_LogReg": {
            "classifier__C": [
                0.001,
                0.01,
                0.1,
                1,
                10
            ],

            "classifier__l1_ratio": [
                0.1,
                0.3,
                0.5,
                0.7,
                0.9
            ]
        },

        "Ridge": {
            "classifier__alpha": [
                0.001,
                0.01,
                0.1,
                1,
                10,
                100
            ]
        },


        # ----------------------------------------------------
        # DECISION TREE
        # ----------------------------------------------------

        "DecisionTree": {
            "classifier__max_depth": [
                None,
                3,
                5,
                8,
                10,
                15,
                20
            ],

            "classifier__min_samples_split": [
                2,
                5,
                10,
                20
            ],

            "classifier__min_samples_leaf": [
                1,
                2,
                5,
                10
            ]
        },


        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        "RandomForest": {
            "classifier__n_estimators": [
                50,
                100,
                200,
                300,
                500
            ],

            "classifier__max_depth": [
                None,
                5,
                10,
                20,
                30
            ],

            "classifier__min_samples_split": [
                2,
                5,
                10,
                20
            ]
        },


        # ----------------------------------------------------
        # GRADIENT BOOSTING
        # ----------------------------------------------------

        "GradientBoosting": {
            "classifier__n_estimators": [
                50,
                100,
                200,
                300
            ],

            "classifier__learning_rate": [
                0.01,
                0.05,
                0.1,
                0.2
            ],

            "classifier__max_depth": [
                3,
                4,
                5,
                7
            ]
        },


        # ----------------------------------------------------
        # ADABOOST
        # ----------------------------------------------------

        "AdaBoost": {
            "classifier__n_estimators": [
                50,
                100,
                200
            ],

            "classifier__learning_rate": [
                0.01,
                0.1,
                1
            ]
        },


        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        "XGBoost": {
            "classifier__n_estimators": [
                50,
                100,
                200,
                300
            ],

            "classifier__learning_rate": [
                0.01,
                0.05,
                0.1,
                0.2
            ],

            "classifier__max_depth": [
                3,
                4,
                5,
                6
            ],

            "classifier__subsample": [
                0.7,
                0.85,
                1.0
            ]
        },


        # ----------------------------------------------------
        # SVC
        # ----------------------------------------------------

        "SVC_Prob": {
            "classifier__C": [
                0.1,
                1,
                10
            ],

            "classifier__gamma": [
                "scale",
                "auto",
                0.01,
                0.1
            ]
        },

        "SVC_RBF": {
            "classifier__C": [
                0.1,
                1,
                10
            ],

            "classifier__gamma": [
                "scale",
                "auto",
                0.01,
                0.1
            ]
        },

        "SVC_Linear": {
            "classifier__C": [
                0.1,
                1,
                10
            ]
        },

        "SVC_Poly": {
            "classifier__C": [
                0.1,
                1
            ],

            "classifier__degree": [
                2,
                3,
                4,
                5
            ]
        },


        # ----------------------------------------------------
        # SVR
        # ----------------------------------------------------

        "SVR_Linear": {
            "classifier__C": [
                0.1,
                1,
                10
            ]
        },


        # ----------------------------------------------------
        # KNN
        # ----------------------------------------------------

        "KNN": {
            "classifier__n_neighbors": [
                3,
                5,
                7,
                9,
                11
            ],

            "classifier__weights": [
                "uniform",
                "distance"
            ]
        }
    }

    return grilles


# ============================================================
# 2. RECUPERATION DES SCORES DU MODELE
# ============================================================

def _obtenir_scores_modele(model, X):

    # --------------------------------------------------------
    # predict_proba
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        proba = model.predict_proba(X)

        proba = np.asarray(proba)

        if proba.ndim == 2:

            if proba.shape[1] >= 2:
                return proba[:, 1]

            return proba[:, 0]

        return proba.ravel()


    # --------------------------------------------------------
    # decision_function
    # --------------------------------------------------------

    if hasattr(model, "decision_function"):

        scores = model.decision_function(X)

        scores = np.asarray(scores)

        # Classification binaire
        if scores.ndim == 1:

            scores = np.clip(
                scores,
                -500,
                500
            )

            return 1 / (
                1 + np.exp(-scores)
            )

        # Eventuellement deux colonnes
        if (
            scores.ndim == 2
            and scores.shape[1] >= 2
        ):

            scores = scores[:, 1]

            scores = np.clip(
                scores,
                -500,
                500
            )

            return 1 / (
                1 + np.exp(-scores)
            )


    # --------------------------------------------------------
    # Dernier recours
    # --------------------------------------------------------

    predictions = model.predict(X)

    return np.asarray(
        predictions,
        dtype=float
    )


# ============================================================
# 3. OPTIMISATION DES MODELES
# ============================================================

def optimiser_modeles(
    modeles_selectionnes,
    dict_modeles,
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor,
    cv,
    grilles=None,
    threshold=0.5
):

    if grilles is None:
        grilles = (
            obtenir_grilles_hyperparametres()
        )

    resultats = []

    best_pipelines = {}

    for nom_modele in modeles_selectionnes:

        print()
        print("=" * 70)
        print(
            f"Optimisation : {nom_modele}"
        )
        print("=" * 70)

        if nom_modele not in dict_modeles:

            print(
                f"⚠️ Modèle '{nom_modele}' "
                f"absent de dict_modeles."
            )

            continue

        try:

            model = dict_modeles[
                nom_modele
            ]

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),

                    (
                        "classifier",
                        model
                    )
                ]
            )

            grille = grilles.get(
                nom_modele,
                {}
            )

            # ------------------------------------------------
            # GRIDSEARCH
            # ------------------------------------------------

            debut = time.time()

            grid = GridSearchCV(

                estimator=pipeline,

                param_grid=grille,

                cv=cv,

                scoring="roc_auc",

                n_jobs=-1,

                refit=True
            )

            grid.fit(
                X_train,
                y_train
            )

            duree = (
                time.time()
                - debut
            )

            best_model = (
                grid.best_estimator_
            )

            best_pipelines[
                nom_modele
            ] = best_model

            # ------------------------------------------------
            # SCORES
            # ------------------------------------------------

            y_train_score = (
                _obtenir_scores_modele(
                    best_model,
                    X_train
                )
            )

            y_test_score = (
                _obtenir_scores_modele(
                    best_model,
                    X_test
                )
            )

            # ------------------------------------------------
            # PREDICTIONS SELON LE SEUIL
            # ------------------------------------------------

            y_train_pred = (
                y_train_score >= threshold
            ).astype(int)

            y_test_pred = (
                y_test_score >= threshold
            ).astype(int)

            # ------------------------------------------------
            # METRIQUES
            # ------------------------------------------------

            roc_train = roc_auc_score(
                y_train,
                y_train_score
            )

            roc_test = roc_auc_score(
                y_test,
                y_test_score
            )

            accuracy_test = (
                accuracy_score(
                    y_test,
                    y_test_pred
                )
            )

            precision_test = (
                precision_score(
                    y_test,
                    y_test_pred,
                    zero_division=0
                )
            )

            recall_train = (
                recall_score(
                    y_train,
                    y_train_pred,
                    zero_division=0
                )
            )

            recall_test = (
                recall_score(
                    y_test,
                    y_test_pred,
                    zero_division=0
                )
            )

            f1_test = (
                f1_score(
                    y_test,
                    y_test_pred,
                    zero_division=0
                )
            )

            # ------------------------------------------------
            # RESULTAT
            # ------------------------------------------------

            resultats.append({

                "Model":
                    nom_modele,

                "Meilleurs paramètres":
                    grid.best_params_,

                "ROC_AUC_cv":
                    grid.best_score_,

                "ROC_AUC_train":
                    roc_train,

                "Recall_train":
                    recall_train,

                "ROC_AUC_test":
                    roc_test,

                "Accuracy_test":
                    accuracy_test,

                "Precision_test":
                    precision_test,

                "Recall_test":
                    recall_test,

                "F1_test":
                    f1_test,

                "Duration_s":
                    duree
            })

            print(
                f"ROC AUC CV    : "
                f"{grid.best_score_:.3f}"
            )

            print(
                f"ROC AUC train : "
                f"{roc_train:.3f}"
            )

            print(
                f"ROC AUC test  : "
                f"{roc_test:.3f}"
            )

            print(
                f"Temps         : "
                f"{duree:.2f} s"
            )

            print(
                "Meilleurs paramètres : "
                f"{grid.best_params_}"
            )

        except Exception as e:

            print(
                f"❌ Erreur avec "
                f"{nom_modele} : {e}"
            )

            traceback.print_exc()

    df_res = pd.DataFrame(
        resultats
    )

    return (
        df_res,
        best_pipelines
    )


# ============================================================
# 4. FORMATAGE DES PARAMETRES
# ============================================================

def _formater_parametres(
    params,
    longueur_max=75
):

    if not isinstance(
        params,
        dict
    ):
        return str(params)

    texte = ", ".join(

        f"{k.split('__')[-1]}={v}"

        for k, v in params.items()
    )

    if len(texte) > longueur_max:

        texte = (
            texte[:longueur_max - 3]
            + "..."
        )

    return texte


# ============================================================
# 5. AFFICHAGE DES RESULTATS
# ============================================================

def afficher_resultats_tuning(
    df_res,
    colonne_tri="ROC_AUC_cv",
    ordre_decroissant=True,
    colonnes_visibles=None
):

    if (
        df_res is None
        or df_res.empty
    ):

        display(
            HTML(
                "<b>⚠️ Aucun résultat "
                "à afficher.</b>"
            )
        )

        return

    df_display = df_res.copy()

    # ========================================================
    # ECART TRAIN / CV
    # ========================================================

    if (
        "ROC_AUC_train"
        in df_display.columns
        and
        "ROC_AUC_cv"
        in df_display.columns
    ):

        df_display[
            "Ecart_Train_CV"
        ] = (
            df_display[
                "ROC_AUC_train"
            ]
            -
            df_display[
                "ROC_AUC_cv"
            ]
        )

    # ========================================================
    # ECART CV / TEST
    # ========================================================

    if (
        "ROC_AUC_cv"
        in df_display.columns
        and
        "ROC_AUC_test"
        in df_display.columns
    ):

        df_display[
            "Ecart_CV_Test"
        ] = (
            df_display[
                "ROC_AUC_cv"
            ]
            -
            df_display[
                "ROC_AUC_test"
            ]
        )

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    diagnostics = []

    for _, row in df_display.iterrows():

        train = row.get(
            "ROC_AUC_train",
            np.nan
        )

        cv_score = row.get(
            "ROC_AUC_cv",
            np.nan
        )

        ecart = (
            train - cv_score
            if pd.notna(train)
            and pd.notna(cv_score)
            else np.nan
        )

        if (
            pd.notna(train)
            and pd.notna(cv_score)
            and train < 0.65
            and cv_score < 0.65
        ):

            diagnostic = (
                "⚠️ Sous-apprentissage"
            )

        elif (
            pd.notna(ecart)
            and ecart > 0.05
        ):

            diagnostic = (
                "🔴 Overfitting possible"
            )

        elif (
            pd.notna(ecart)
            and ecart < -0.05
        ):

            diagnostic = (
                "🟣 CV > Train"
            )

        else:

            diagnostic = (
                "🟢 Cohérent"
            )

        diagnostics.append(
            diagnostic
        )

    df_display[
        "Diagnostic"
    ] = diagnostics

    # ========================================================
    # PARAMETRES COMPACTS
    # ========================================================

    if (
        "Meilleurs paramètres"
        in df_display.columns
    ):

        df_display[
            "Meilleurs paramètres"
        ] = (
            df_display[
                "Meilleurs paramètres"
            ]
            .apply(
                _formater_parametres
            )
        )

    # ========================================================
    # COLONNES NUMERIQUES DISPONIBLES POUR LE TRI
    # ========================================================

    colonnes_numeriques = (
        df_display
        .select_dtypes(
            include=[
                "number",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    # ========================================================
    # SECURITE DU TRI
    # ========================================================

    if (
        colonne_tri
        not in colonnes_numeriques
    ):

        if (
            "ROC_AUC_cv"
            in colonnes_numeriques
        ):

            colonne_tri = (
                "ROC_AUC_cv"
            )

        elif colonnes_numeriques:

            colonne_tri = (
                colonnes_numeriques[0]
            )

        else:

            colonne_tri = None

    # ========================================================
    # TRI
    # ========================================================

    if colonne_tri is not None:

        df_display = (
            df_display
            .sort_values(
                by=colonne_tri,
                ascending=(
                    not ordre_decroissant
                ),
                kind="mergesort"
            )
        )

    # ========================================================
    # COLONNES PAR DEFAUT
    # ========================================================

    colonnes_defaut = [

        "Model",

        "ROC_AUC_cv",

        "ROC_AUC_train",

        "ROC_AUC_test",

        "Recall_train",

        "Recall_test",

        "Accuracy_test",

        "Precision_test",

        "F1_test",

        "Ecart_Train_CV",

        "Ecart_CV_Test",

        "Diagnostic",

        "Meilleurs paramètres",

        "Duration_s"
    ]

    colonnes_defaut = [

        col

        for col in colonnes_defaut

        if col in df_display.columns
    ]

    if colonnes_visibles is None:

        colonnes_visibles = (
            colonnes_defaut
        )

    else:

        colonnes_visibles = [

            col

            for col in colonnes_visibles

            if col in df_display.columns
        ]

    # Sécurité si l'utilisateur
    # désélectionne tout.

    if not colonnes_visibles:

        colonnes_visibles = [
            "Model"
        ]

        if (
            colonne_tri
            in df_display.columns
        ):

            colonnes_visibles.append(
                colonne_tri
            )

    # ========================================================
    # DATAFRAME A AFFICHER
    # ========================================================

    df_affiche = (
        df_display[
            colonnes_visibles
        ]
        .copy()
    )

    # ========================================================
    # STYLER
    # ========================================================

    styler = (
        df_affiche
        .style
    )

    # ========================================================
    # FORMAT NOMBRES
    # ========================================================

    colonnes_scores = [

        "ROC_AUC_cv",

        "ROC_AUC_train",

        "ROC_AUC_test",

        "Recall_train",

        "Recall_test",

        "Accuracy_test",

        "Precision_test",

        "F1_test",

        "Ecart_Train_CV",

        "Ecart_CV_Test"
    ]

    colonnes_scores = [

        col

        for col in colonnes_scores

        if col in df_affiche.columns
    ]

    if colonnes_scores:

        styler = styler.format(

            {
                col: "{:.3f}"

                for col
                in colonnes_scores
            }
        )

    if (
        "Duration_s"
        in df_affiche.columns
    ):

        styler = styler.format(
            {
                "Duration_s":
                    "{:.2f}"
            }
        )

    # ========================================================
    # UNE SEULE COLONNE VERTE
    # ========================================================
    #
    # IMPORTANT :
    # La coloration est appliquée UNIQUEMENT
    # à la colonne choisie dans "Trier par".
    #
    # ========================================================

    if (
        colonne_tri
        in df_affiche.columns
        and
        pd.api.types.is_numeric_dtype(
            df_affiche[
                colonne_tri
            ]
        )
    ):

        styler = (
            styler
            .background_gradient(
                subset=[
                    colonne_tri
                ],
                cmap="Greens"
            )
        )

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    if (
        "Diagnostic"
        in df_affiche.columns
    ):

        def couleur_diagnostic(
            value
        ):

            texte = str(value)

            if "Overfitting" in texte:
                return (
                    "background-color: #f8d7da"
                )

            if "Sous-apprentissage" in texte:
                return (
                    "background-color: #ffe5b4"
                )

            if "CV > Train" in texte:
                return (
                    "background-color: #e5d4f7"
                )

            if "Cohérent" in texte:
                return (
                    "background-color: #d8f3dc"
                )

            return ""

        styler = (
            styler
            .map(
                couleur_diagnostic,
                subset=[
                    "Diagnostic"
                ]
            )
        )

    # ========================================================
    # CSS DU TABLEAU
    # ========================================================

    styler = (
        styler
        .set_properties(
            **{

                "font-size":
                    "11px",

                "padding":
                    "4px",

                "text-align":
                    "center",

                "vertical-align":
                    "middle",

                "white-space":
                    "normal",

                "word-wrap":
                    "break-word",

                "overflow-wrap":
                    "break-word"
            }
        )
    )

    styler = (
        styler
        .set_table_styles([

            {
                "selector":
                    "table",

                "props": [

                    (
                        "width",
                        "100%"
                    ),

                    (
                        "max-width",
                        "100%"
                    ),

                    (
                        "table-layout",
                        "fixed"
                    ),

                    (
                        "border-collapse",
                        "collapse"
                    )
                ]
            },

            {
                "selector":
                    "th",

                "props": [

                    (
                        "font-size",
                        "10px"
                    ),

                    (
                        "font-weight",
                        "bold"
                    ),

                    (
                        "white-space",
                        "normal"
                    ),

                    (
                        "word-wrap",
                        "break-word"
                    ),

                    (
                        "overflow-wrap",
                        "break-word"
                    ),

                    (
                        "text-align",
                        "center"
                    ),

                    (
                        "vertical-align",
                        "middle"
                    ),

                    (
                        "padding",
                        "4px"
                    )
                ]
            },

            {
                "selector":
                    "td",

                "props": [

                    (
                        "max-width",
                        "110px"
                    ),

                    (
                        "overflow-wrap",
                        "break-word"
                    ),

                    (
                        "white-space",
                        "normal"
                    )
                ]
            }
        ])
    )

    # ========================================================
    # AFFICHAGE
    # ========================================================

    display(
        HTML(
            f"""
            <div style="
                width:100%;
                max-width:100%;
                overflow-x:hidden;
                margin-top:10px;
            ">
                {styler.to_html()}
            </div>
            """
        )
    )

    # ========================================================
    # INFORMATIONS
    # ========================================================

    print(
        f"🔎 Tri actuel : {colonne_tri} "
        f"({'décroissant'
           if ordre_decroissant
           else 'croissant'})"
    )

    print(
        "🟢 La colonne verte correspond "
        "uniquement au critère de tri."
    )


# ============================================================
# 6. INTERFACE PRINCIPALE
# ============================================================

def interface_tuning(
    dict_modeles_base,
    categories_modeles
):

    # ========================================================
    # NAMESPACE DU NOTEBOOK
    # ========================================================

    namespace = (
        sys.modules[
            "__main__"
        ].__dict__
    )

    variables_requises = [

        "X_train",
        "y_train",

        "X_test",
        "y_test",

        "preprocessor",
        "cv"
    ]

    # ========================================================
    # HYPERPARAMETRES MANUELS
    # ========================================================

    profondeur = widgets.Dropdown(

        options=[
            "None",
            "3",
            "5",
            "8",
            "10",
            "15",
            "20",
            "30"
        ],

        value="8",

        description="Max depth :",

        layout=widgets.Layout(
            width="220px"
        )
    )

    n_estimators = widgets.Dropdown(

        options=[
            "50",
            "100",
            "200",
            "300",
            "500"
        ],

        value="100",

        description="N estimators :",

        layout=widgets.Layout(
            width="220px"
        )
    )

    C = widgets.Dropdown(

        options=[
            "0.001",
            "0.01",
            "0.1",
            "1",
            "10",
            "100",
            "1000"
        ],

        value="0.1",

        description="C :",

        layout=widgets.Layout(
            width="220px"
        )
    )

    alpha = widgets.Dropdown(

        options=[
            "0.0001",
            "0.001",
            "0.01",
            "0.1",
            "1.0",
            "10.0"
        ],

        value="1.0",

        description="Alpha :",

        layout=widgets.Layout(
            width="220px"
        )
    )

    learning_rate = widgets.Dropdown(

        options=[
            "0.01",
            "0.03",
            "0.05",
            "0.1",
            "0.2",
            "0.3"
        ],

        value="0.1",

        description="Learning rate :",

        layout=widgets.Layout(
            width="220px"
        )
    )

    threshold = widgets.FloatSlider(

        value=0.35,

        min=0.1,

        max=0.9,

        step=0.05,

        description="Seuil :",

        continuous_update=False,

        readout_format=".2f",

        layout=widgets.Layout(
            width="350px"
        )
    )

    # ========================================================
    # SELECTION DES MODELES
    # ========================================================
    #
    # Seuls les modeles deja entraines (presents dans
    # fitted_pipelines_widget, cf. etape de modelisation) sont
    # proposes ici : un modele non entraine n'a pas de resultats
    # de validation croisee sur lesquels s'appuyer, et lancer une
    # optimisation d'hyperparametres pour lui n'aurait donc aucun
    # sens. Un bouton "Actualiser" permet de rafraichir la liste
    # si l'entrainement est relance/modifie apres l'ouverture de
    # cette interface (meme logique que l'etape d'etude des
    # modeles).

    boutons_modeles = {}

    sortie = widgets.Output()

    zone_selection_modeles = widgets.VBox()

    def _modeles_entraines():
        """Noms des modeles reellement entraines (= disponibles ici)."""

        fitted = namespace.get(
            "fitted_pipelines_widget"
        )

        return (
            list(fitted.keys())
            if fitted else []
        )

    def _filtrer_categories_par_modeles(
        categories,
        noms_disponibles
    ):
        """Ne conserve, dans chaque categorie, que les modeles
        entraines. Les categories sans modele disponible sont
        retirees ; les modeles entraines absents de `categories`
        sont regroupes dans une categorie "Autres"."""

        noms_set = set(
            noms_disponibles
        )

        resultat = OrderedDict()
        deja_places = set()

        for categorie, modeles in (
            categories.items()
        ):

            presents = [
                m for m in modeles
                if m in noms_set
            ]

            if presents:
                resultat[categorie] = presents
                deja_places.update(presents)

        non_classes = [
            m for m in noms_disponibles
            if m not in deja_places
        ]

        if non_classes:
            resultat["Autres"] = non_classes

        return resultat

    tout_selectionner_modeles = (
        widgets.ToggleButton(

            value=False,

            description=(
                "☑️ Sélectionner tous"
            ),

            button_style="info",

            layout=widgets.Layout(
                width="220px"
            )
        )
    )

    def selection_globale_modeles(
        change
    ):

        if change["name"] != "value":
            return

        valeur = change["new"]

        for bouton in (
            boutons_modeles.values()
        ):

            bouton.value = valeur

        if valeur:

            tout_selectionner_modeles.description = (
                "☑️ Tous sélectionnés"
            )

        else:

            tout_selectionner_modeles.description = (
                "☑️ Sélectionner tous"
            )

    tout_selectionner_modeles.observe(
        selection_globale_modeles,
        names="value"
    )

    def _construire_selection_modeles(b=None):
        """(Re)construit les cases a cocher a partir des modeles
        actuellement entraines (fitted_pipelines_widget)."""

        boutons_modeles.clear()

        noms_disponibles = _modeles_entraines()

        if not noms_disponibles:

            zone_selection_modeles.children = [
                widgets.HTML(
                    "<i>⚠️ Aucun modèle entraîné pour l'instant. "
                    "Lancez d'abord l'entraînement (étape "
                    "précédente), puis cliquez sur 🔄 pour "
                    "rafraîchir cette liste.</i>"
                )
            ]

            tout_selectionner_modeles.disabled = True

            return

        tout_selectionner_modeles.disabled = False

        categories_disponibles = (
            _filtrer_categories_par_modeles(
                categories_modeles,
                noms_disponibles
            )
        )

        accordions_categories_locales = []

        for categorie, modeles in (
            categories_disponibles.items()
        ):

            boutons = []

            for nom_modele in modeles:

                bouton = widgets.ToggleButton(

                    value=False,

                    description=(
                        f"OFF : {nom_modele}"
                    ),

                    button_style="",

                    layout=widgets.Layout(
                        width="260px"
                    )
                )

                boutons_modeles[
                    nom_modele
                ] = bouton

                def changement(
                    change,
                    bouton=bouton,
                    nom_modele=nom_modele
                ):

                    if change["name"] != "value":
                        return

                    if change["new"]:

                        bouton.description = (
                            f"ON : {nom_modele}"
                        )

                        bouton.button_style = (
                            "success"
                        )

                    else:

                        bouton.description = (
                            f"OFF : {nom_modele}"
                        )

                        bouton.button_style = ""

                bouton.observe(
                    changement,
                    names="value"
                )

                boutons.append(
                    bouton
                )

            # ----------------------------------------------------
            # BOUTON "TOUT SELECTIONNER" PROPRE A LA CATEGORIE
            # ----------------------------------------------------
            # Evite d'avoir a cliquer sur chaque modele un par un
            # quand on veut activer/desactiver une categorie entiere.

            bouton_cat_tous = widgets.ToggleButton(

                value=False,

                description="☑️ Tout activer",

                button_style="info",

                layout=widgets.Layout(
                    width="260px"
                )
            )

            def changement_categorie(
                change,
                boutons=boutons,
                bouton_cat_tous=bouton_cat_tous
            ):

                if change["name"] != "value":
                    return

                for bouton in boutons:
                    bouton.value = change["new"]

                bouton_cat_tous.description = (
                    "☑️ Tout desactiver"
                    if change["new"]
                    else "☑️ Tout activer"
                )

            bouton_cat_tous.observe(
                changement_categorie,
                names="value"
            )

            bloc_categorie = widgets.VBox(
                [bouton_cat_tous] + boutons
            )

            accordion = widgets.Accordion(
                children=[
                    bloc_categorie
                ]
            )

            accordion.set_title(
                0,
                f"{categorie} ({len(modeles)})"
            )

            accordions_categories_locales.append(
                accordion
            )

        zone_selection_modeles.children = (
            accordions_categories_locales
        )

    bouton_rafraichir_modeles = widgets.Button(

        description=(
            "🔄 Actualiser la liste des modèles entraînés"
        ),

        layout=widgets.Layout(
            width="320px"
        )
    )

    bouton_rafraichir_modeles.on_click(
        _construire_selection_modeles
    )

    _construire_selection_modeles()  # construction initiale, selon l'etat actuel de l'entrainement

    # ========================================================
    # BOUTONS OPTIMISATION
    # ========================================================

    bouton_manuel = widgets.Button(

        description=(
            "⚙️ Optimisation manuelle"
        ),

        button_style="warning",

        layout=widgets.Layout(
            width="230px"
        )
    )

    bouton_auto = widgets.Button(

        description=(
            "🚀 Optimisation automatique"
        ),

        button_style="danger",

        layout=widgets.Layout(
            width="230px"
        )
    )

    bouton_features = widgets.Button(

        description=(
            "🔎 Analyse des features"
        ),

        button_style="info",

        disabled=True,

        tooltip=(
            "Disponible une fois qu'une "
            "optimisation a été lancée"
        ),

        layout=widgets.Layout(
            width="230px"
        )
    )

    # ========================================================
    # TOP N
    # ========================================================

    metric_top = widgets.Dropdown(

        options=[
            "ROC_AUC_test",
            "ROC_AUC_cv",
            "ROC_AUC_train",
            "Recall_test",
            "Precision_test",
            "F1_test",
            "Accuracy_test"
        ],

        value="ROC_AUC_test",

        description="Top N selon :",

        layout=widgets.Layout(
            width="300px"
        )
    )

    top_n = widgets.IntText(

        value=5,

        min=1,

        description="Top N :",

        layout=widgets.Layout(
            width="180px"
        )
    )

    bouton_top = widgets.Button(

        description=(
            "💾 Sauver Top N"
        ),

        button_style="success",

        layout=widgets.Layout(
            width="180px"
        )
    )

    # ========================================================
    # TRI
    # ========================================================

    tri_colonne = widgets.Dropdown(

        options=[

            "ROC_AUC_cv",

            "ROC_AUC_train",

            "ROC_AUC_test",

            "Recall_train",

            "Recall_test",

            "Accuracy_test",

            "Precision_test",

            "F1_test",

            "Ecart_Train_CV",

            "Ecart_CV_Test"
        ],

        value="ROC_AUC_cv",

        description="Trier par :",

        layout=widgets.Layout(
            width="300px"
        )
    )

    tri_decroissant = widgets.Checkbox(

        value=True,

        description="Décroissant"
    )

    # ========================================================
    # COLONNES DISPONIBLES
    # ========================================================

    colonnes_disponibles = [

        "Model",

        "ROC_AUC_cv",

        "ROC_AUC_train",

        "ROC_AUC_test",

        "Recall_train",

        "Recall_test",

        "Accuracy_test",

        "Precision_test",

        "F1_test",

        "Ecart_Train_CV",

        "Ecart_CV_Test",

        "Diagnostic",

        "Meilleurs paramètres",

        "Duration_s"
    ]

    # ========================================================
    # CASES A COCHER
    # ========================================================
    #
    # Contrairement à SelectMultiple :
    #
    #   clic 1 -> ajoute
    #   clic 2 -> ajoute
    #   clic 3 -> ajoute
    #
    # Aucun Ctrl nécessaire.
    #
    # ========================================================

    checkboxes_colonnes = {}

    for colonne in (
        colonnes_disponibles
    ):

        checkbox = widgets.Checkbox(

            value=True,

            description=colonne,

            indent=False,

            layout=widgets.Layout(
                width="280px"
            )
        )

        checkboxes_colonnes[
            colonne
        ] = checkbox

    # ========================================================
    # CONTENEUR DES COLONNES
    # ========================================================

    # Grille à 2 colonnes plutôt qu'une longue liste verticale
    # à faire défiler : toutes les cases sont visibles d'un
    # coup, sans scroll.
    liste_checkboxes = widgets.GridBox(

        list(
            checkboxes_colonnes.values()
        ),

        layout=widgets.Layout(

            grid_template_columns="repeat(2, 260px)",

            width="530px",

            border="1px solid #ccc",

            padding="5px"
        )
    )

    # ========================================================
    # BOUTONS COLONNES
    # ========================================================

    bouton_tout_colonnes = widgets.Button(

        description=(
            "☑️ Tout sélectionner"
        ),

        button_style="info",

        layout=widgets.Layout(
            width="180px"
        )
    )

    bouton_aucune_colonne = widgets.Button(

        description=(
            "☐ Tout désélectionner"
        ),

        button_style="",

        layout=widgets.Layout(
            width="180px"
        )
    )

    # ========================================================
    # RECUPERATION DES COLONNES COCHEES
    # ========================================================

    def obtenir_colonnes_visibles():

        return [

            colonne

            for colonne, checkbox
            in checkboxes_colonnes.items()

            if checkbox.value
        ]

    # ========================================================
    # TOUT SELECTIONNER
    # ========================================================

    def toutes_colonnes(_):

        for checkbox in (
            checkboxes_colonnes.values()
        ):

            checkbox.value = True

    bouton_tout_colonnes.on_click(
        toutes_colonnes
    )

    # ========================================================
    # TOUT DESELECTIONNER
    # ========================================================

    def aucune_colonne(_):

        for checkbox in (
            checkboxes_colonnes.values()
        ):

            checkbox.value = False

    bouton_aucune_colonne.on_click(
        aucune_colonne
    )

    # ========================================================
    # FONCTION DE RAFRAICHISSEMENT
    # ========================================================

    def rafraichir_affichage(
        _=None
    ):

        with sortie:

            clear_output()

            df_res = namespace.get(
                "df_tuned_res"
            )

            if (
                df_res is None
                or df_res.empty
            ):

                print(
                    "ℹ️ Aucun résultat "
                    "à afficher."
                )

                return

            afficher_resultats_tuning(

                df_res,

                colonne_tri=(
                    tri_colonne.value
                ),

                ordre_decroissant=(
                    tri_decroissant.value
                ),

                colonnes_visibles=(
                    obtenir_colonnes_visibles()
                )
            )

    # ========================================================
    # RAFRAICHIR LORS D'UN CLIC SUR UNE COLONNE
    # ========================================================

    for checkbox in (
        checkboxes_colonnes.values()
    ):

        checkbox.observe(
            rafraichir_affichage,
            names="value"
        )

    # ========================================================
    # RAFRAICHIR LORSQUE LE TRI CHANGE
    # ========================================================

    tri_colonne.observe(
        rafraichir_affichage,
        names="value"
    )

    tri_decroissant.observe(
        rafraichir_affichage,
        names="value"
    )

    # ========================================================
    # LANCEMENT DE L'OPTIMISATION
    # ========================================================

    def lancer_optimisation(
        manuel=False
    ):

        with sortie:

            clear_output()

            # ------------------------------------------------
            # VERIFICATION VARIABLES
            # ------------------------------------------------

            manquantes = [

                variable

                for variable
                in variables_requises

                if variable
                not in namespace
            ]

            if manquantes:

                print(
                    "❌ Variables absentes "
                    "du notebook :"
                )

                for variable in manquantes:

                    print(
                        f"   - {variable}"
                    )

                return

            # ------------------------------------------------
            # VARIABLES
            # ------------------------------------------------

            X_train = namespace[
                "X_train"
            ]

            y_train = namespace[
                "y_train"
            ]

            X_test = namespace[
                "X_test"
            ]

            y_test = namespace[
                "y_test"
            ]

            preprocessor = namespace[
                "preprocessor"
            ]

            cv = namespace[
                "cv"
            ]

            # ------------------------------------------------
            # MODELES SELECTIONNES
            # ------------------------------------------------

            modeles_selectionnes = [

                nom

                for nom, bouton
                in boutons_modeles.items()

                if bouton.value
            ]

            if not modeles_selectionnes:

                print(
                    "⚠️ Aucun modèle "
                    "sélectionné."
                )

                return

            print(
                "Modèles sélectionnés :"
            )

            for modele in (
                modeles_selectionnes
            ):

                print(
                    f"  • {modele}"
                )

            print()

            # ------------------------------------------------
            # SEUIL
            # ------------------------------------------------

            namespace[
                "current_threshold"
            ] = threshold.value

            # ------------------------------------------------
            # GRILLES AUTOMATIQUES
            # ------------------------------------------------

            grilles = (
                obtenir_grilles_hyperparametres()
            )

            # =================================================
            # MODE MANUEL
            # =================================================

            if manuel:

                depth_value = (

                    None

                    if profondeur.value
                    == "None"

                    else int(
                        profondeur.value
                    )
                )

                n_est_value = int(
                    n_estimators.value
                )

                C_value = float(
                    C.value
                )

                alpha_value = float(
                    alpha.value
                )

                lr_value = float(
                    learning_rate.value
                )

                grilles_manuelles = {}

                for nom in (
                    modeles_selectionnes
                ):

                    # ----------------------------------------
                    # LOGISTIC
                    # ----------------------------------------

                    if nom in [

                        "LogisticRegression_L1",

                        "LogisticRegression_L2",

                        "Lasso_LogReg"
                    ]:

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__C": [
                                C_value
                            ]
                        }

                    # ----------------------------------------
                    # ELASTIC NET
                    # ----------------------------------------

                    elif (
                        nom
                        == "ElasticNet_LogReg"
                    ):

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__C": [
                                C_value
                            ],

                            "classifier__l1_ratio": [
                                0.1,
                                0.5,
                                0.9
                            ]
                        }

                    # ----------------------------------------
                    # RIDGE
                    # ----------------------------------------

                    elif nom == "Ridge":

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__alpha": [
                                alpha_value
                            ]
                        }

                    # ----------------------------------------
                    # DECISION TREE
                    # ----------------------------------------

                    elif (
                        nom
                        == "DecisionTree"
                    ):

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__max_depth": [
                                depth_value
                            ],

                            "classifier__min_samples_split": [
                                2,
                                5,
                                10
                            ]
                        }

                    # ----------------------------------------
                    # RANDOM FOREST
                    # ----------------------------------------

                    elif (
                        nom
                        == "RandomForest"
                    ):

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__n_estimators": [
                                n_est_value
                            ],

                            "classifier__max_depth": [
                                depth_value
                            ],

                            "classifier__min_samples_split": [
                                2,
                                5,
                                10
                            ]
                        }

                    # ----------------------------------------
                    # GRADIENT BOOSTING
                    # ----------------------------------------

                    elif (
                        nom
                        == "GradientBoosting"
                    ):

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__n_estimators": [
                                n_est_value
                            ],

                            "classifier__max_depth": [
                                3,
                                5
                            ],

                            "classifier__learning_rate": [
                                lr_value
                            ]
                        }

                    # ----------------------------------------
                    # XGBOOST
                    # ----------------------------------------

                    elif nom == "XGBoost":

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__n_estimators": [
                                n_est_value
                            ],

                            "classifier__max_depth": [
                                3,
                                5
                            ],

                            "classifier__learning_rate": [
                                lr_value
                            ]
                        }

                    # ----------------------------------------
                    # SVC LINEAR
                    # ----------------------------------------

                    elif (
                        nom
                        == "SVC_Linear"
                    ):

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__C": [
                                C_value
                            ]
                        }

                    # ----------------------------------------
                    # SVC RBF
                    # ----------------------------------------

                    elif nom in [

                        "SVC_RBF",

                        "SVC_Prob"
                    ]:

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__C": [
                                C_value
                            ]
                        }

                    # ----------------------------------------
                    # SVC POLY
                    # ----------------------------------------

                    elif (
                        nom
                        == "SVC_Poly"
                    ):

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__C": [
                                C_value
                            ],

                            "classifier__degree": [
                                2,
                                3,
                                4
                            ]
                        }

                    # ----------------------------------------
                    # SVR
                    # ----------------------------------------

                    elif (
                        nom
                        == "SVR_Linear"
                    ):

                        grilles_manuelles[
                            nom
                        ] = {

                            "classifier__C": [
                                C_value
                            ]
                        }

                    # ----------------------------------------
                    # AUTRES MODELES
                    # ----------------------------------------

                    else:

                        grilles_manuelles[
                            nom
                        ] = grilles.get(
                            nom,
                            {}
                        )

                grilles = (
                    grilles_manuelles
                )

            # =================================================
            # OPTIMISATION
            # =================================================

            (
                df_res,
                best_pipelines
            ) = optimiser_modeles(

                modeles_selectionnes,

                dict_modeles_base,

                X_train,
                y_train,

                X_test,
                y_test,

                preprocessor,

                cv,

                grilles=grilles,

                threshold=threshold.value
            )

            # ------------------------------------------------
            # STOCKAGE
            # ------------------------------------------------

            namespace[
                "df_tuned_res"
            ] = df_res

            namespace[
                "best_pipelines"
            ] = best_pipelines

            # ------------------------------------------------
            # REVELATION DES BLOCS "RESULTATS"
            # ------------------------------------------------
            # Ces blocs n'ont de sens qu'une fois qu'on a des
            # résultats : on les affiche seulement maintenant.

            bloc_affichage.layout.display = ""

            bloc_top.layout.display = ""

            bouton_features.disabled = False

            # ------------------------------------------------
            # AFFICHAGE
            # ------------------------------------------------

            print()
            print(
                "📊 Résultats de l'optimisation"
            )

            print()

            afficher_resultats_tuning(

                df_res,

                colonne_tri=(
                    tri_colonne.value
                ),

                ordre_decroissant=(
                    tri_decroissant.value
                ),

                colonnes_visibles=(
                    obtenir_colonnes_visibles()
                )
            )

    # ========================================================
    # CALLBACK AUTO
    # ========================================================

    def lancer_auto(_):

        lancer_optimisation(
            manuel=False
        )

    bouton_auto.on_click(
        lancer_auto
    )

    # ========================================================
    # CALLBACK MANUEL
    # ========================================================

    def lancer_manuel(_):

        lancer_optimisation(
            manuel=True
        )

    bouton_manuel.on_click(
        lancer_manuel
    )

    # ========================================================
    # ANALYSE DES FEATURES
    # ========================================================

    def analyser_features(_):

        with sortie:

            clear_output()

            best_pipelines = (
                namespace.get(
                    "best_pipelines"
                )
            )

            if not best_pipelines:

                print(
                    "⚠️ Aucun modèle "
                    "optimisé."
                )

                return

            print(
                "🔎 Analyse des features"
            )

            print()

            for (
                nom_modele,
                pipeline
            ) in best_pipelines.items():

                print()
                print(
                    "=" * 60
                )

                print(
                    nom_modele
                )

                print(
                    "=" * 60
                )

                classifier = (
                    pipeline
                    .named_steps
                    .get(
                        "classifier"
                    )
                )

                preproc = (
                    pipeline
                    .named_steps
                    .get(
                        "preprocessor"
                    )
                )

                # --------------------------------------------
                # COEFFICIENTS
                # --------------------------------------------

                if hasattr(
                    classifier,
                    "coef_"
                ):

                    importance = (
                        np.abs(
                            classifier.coef_
                        )
                    )

                    if importance.ndim > 1:

                        importance = (
                            importance[0]
                        )

                # --------------------------------------------
                # IMPORTANCE FEATURES
                # --------------------------------------------

                elif hasattr(
                    classifier,
                    "feature_importances_"
                ):

                    importance = (
                        classifier
                        .feature_importances_
                    )

                else:

                    print(
                        "ℹ️ Ce modèle ne possède "
                        "ni coef_ ni "
                        "feature_importances_."
                    )

                    continue

                # --------------------------------------------
                # NOMS DES FEATURES
                # --------------------------------------------

                try:

                    noms_features = (
                        preproc
                        .get_feature_names_out()
                    )

                except Exception:

                    noms_features = [

                        f"Feature_{i}"

                        for i in range(
                            len(importance)
                        )
                    ]

                # --------------------------------------------
                # ALIGNEMENT
                # --------------------------------------------

                longueur = min(

                    len(
                        noms_features
                    ),

                    len(
                        importance
                    )
                )

                df_features = pd.DataFrame({

                    "Feature":
                        noms_features[
                            :longueur
                        ],

                    "Importance":
                        importance[
                            :longueur
                        ]
                })

                df_features = (

                    df_features

                    .sort_values(
                        "Importance",
                        ascending=False
                    )

                    .head(20)
                )

                display(

                    df_features
                    .style

                    .background_gradient(
                        subset=[
                            "Importance"
                        ],
                        cmap="Greens"
                    )

                    .format(
                        {
                            "Importance":
                                "{:.5f}"
                        }
                    )
                )

    bouton_features.on_click(
        analyser_features
    )

    # ========================================================
    # SAUVEGARDE TOP N
    # ========================================================

    def sauvegarder_top(_):

        with sortie:

            clear_output()

            df_res = namespace.get(
                "df_tuned_res"
            )

            best_pipelines = (
                namespace.get(
                    "best_pipelines"
                )
            )

            if (
                df_res is None
                or df_res.empty
            ):

                print(
                    "⚠️ Aucun résultat "
                    "disponible."
                )

                return

            metric = (
                metric_top.value
            )

            if (
                metric
                not in df_res.columns
            ):

                print(
                    f"❌ Colonne absente : "
                    f"{metric}"
                )

                return

            n = max(
                1,
                int(top_n.value)
            )

            df_top = (

                df_res

                .sort_values(
                    metric,
                    ascending=False
                )

                .head(n)
            )

            # ------------------------------------------------
            # RACINE PROJET
            # ------------------------------------------------

            try:

                fichier_module = (
                    Path(__file__)
                    .resolve()
                )

                racine = (
                    fichier_module.parent
                )

                while (

                    racine != racine.parent

                    and
                    racine.name != "src"
                ):

                    racine = (
                        racine.parent
                    )

                if (
                    racine.name
                    == "src"
                ):

                    racine_projet = (
                        racine.parent
                    )

                else:

                    racine_projet = (
                        fichier_module.parent
                    )

            except Exception:

                racine_projet = (
                    Path.cwd()
                )

            dossier_modeles = (

                racine_projet
                / "models"
            )

            dossier_modeles.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # SAUVEGARDE
            # ------------------------------------------------

            for (
                rang,
                (_, row)
            ) in enumerate(
                df_top.iterrows(),
                start=1
            ):

                nom_modele = (
                    row["Model"]
                )

                if (
                    nom_modele
                    not in best_pipelines
                ):

                    continue

                modele = (
                    best_pipelines[
                        nom_modele
                    ]
                )

                nom_fichier = (

                    f"modele_top"
                    f"{rang}_"
                    f"{nom_modele}_"
                    f"{metric}.pkl"
                )

                chemin = (
                    dossier_modeles
                    / nom_fichier
                )

                joblib.dump(
                    modele,
                    chemin
                )

                print(
                    f"✅ Sauvegardé : "
                    f"{chemin}"
                )

    bouton_top.on_click(
        sauvegarder_top
    )

    # ========================================================
    # ORGANISATION INTERFACE
    # ========================================================

    bloc_hyperparametres_contenu = widgets.VBox([

        widgets.HTML(
            "<i>Utilisés uniquement par le bouton "
            "« ⚙️ Optimisation manuelle ». Ignorés par "
            "« 🚀 Optimisation automatique » (qui teste "
            "une grille de valeurs).</i>"
        ),

        widgets.HBox([
            profondeur,
            n_estimators
        ]),

        widgets.HBox([
            C,
            alpha
        ]),

        widgets.HBox([
            learning_rate,
            threshold
        ])
    ])

    # Replié par défaut : ces réglages ne concernent que le
    # mode manuel, ils n'ont pas besoin d'être visibles avant
    # que l'utilisateur choisisse ce mode.
    bloc_hyperparametres = widgets.Accordion(
        children=[
            bloc_hyperparametres_contenu
        ],
        selected_index=None
    )

    bloc_hyperparametres.set_title(
        0,
        "⚙️ Hyperparamètres manuels (optionnel)"
    )

    bloc_selection_modeles = widgets.VBox([

        widgets.HTML(
            "<h4>1️⃣ Sélection des modèles</h4>"
        ),

        widgets.HTML(
            "<i>Seuls les modèles déjà entraînés (étape de "
            "modélisation) sont proposés ci-dessous.</i>"
        ),

        widgets.HBox([
            tout_selectionner_modeles,
            bouton_rafraichir_modeles
        ]),

        zone_selection_modeles
    ])

    bloc_commandes = widgets.VBox([

        widgets.HTML(
            "<h4>2️⃣ Lancer l'optimisation</h4>"
        ),

        widgets.HBox([
            bouton_auto,
            bouton_manuel
        ]),

        widgets.HBox([
            bouton_features
        ])
    ])

    # ========================================================
    # BLOC AFFICHAGE
    # ========================================================

    bloc_affichage = widgets.VBox(
        [
            widgets.HTML(
                "<h4>📊 Affichage des "
                "résultats</h4>"
            ),

            widgets.HBox([

                tri_colonne,

                tri_decroissant
            ]),

            widgets.HBox([

                bouton_tout_colonnes,

                bouton_aucune_colonne
            ]),

            widgets.HTML(
                "<b>Colonnes à afficher :</b>"
            ),

            liste_checkboxes
        ],

        # Masqué tant qu'aucune optimisation n'a été lancée :
        # ces réglages n'ont d'effet qu'une fois des résultats
        # disponibles.
        layout=widgets.Layout(
            display="none"
        )
    )

    # ========================================================
    # BLOC TOP N
    # ========================================================

    bloc_top = widgets.VBox(
        [
            widgets.HTML(
                "<h4>💾 Sauvegarde des "
                "meilleurs modèles</h4>"
            ),

            widgets.HBox([

                metric_top,

                top_n,

                bouton_top
            ])
        ],

        # Masqué tant qu'aucun résultat n'existe (rien à
        # sauvegarder avant d'avoir lancé une optimisation).
        layout=widgets.Layout(
            display="none"
        )
    )

    # ========================================================
    # INTERFACE FINALE
    # ========================================================

    interface = widgets.VBox([

        bloc_selection_modeles,

        widgets.HTML(
            "<hr>"
        ),

        bloc_hyperparametres,

        widgets.HTML(
            "<hr>"
        ),

        bloc_commandes,

        widgets.HTML(
            "<hr>"
        ),

        widgets.HTML(
            "<h4>3️⃣ Résultats</h4>"
        ),

        bloc_affichage,

        bloc_top,

        sortie
    ])

    return interface