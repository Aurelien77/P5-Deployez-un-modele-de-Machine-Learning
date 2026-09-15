"""
Module regroupant les fonctions de modélisation pour la prédiction de
l'attrition (a_quitte_l_entreprise).
"""

import time
import pandas as pd
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, RepeatedStratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier
import xgboost as xgb


# ----------------------------------------------------------------------
# 1. Sélection des variables selon la configuration
# ----------------------------------------------------------------------
def selectionner_features(
    colonnes_lineaires,
    colonnes_non_lineaires,
    cols_qualitatives,
    cols_booleennes,
    cols_ratios,
    inclure_lineaires=True,
    inclure_non_lineaires=True,
    inclure_qualitatives=True,
    inclure_booleennes=True,
    inclure_ratios=True,
):
    features_selectionnees = []
    if inclure_lineaires:
        features_selectionnees.extend(colonnes_lineaires)
    if inclure_non_lineaires:
        features_selectionnees.extend(colonnes_non_lineaires)
    if inclure_qualitatives:
        features_selectionnees.extend(cols_qualitatives)
    if inclure_booleennes:
        features_selectionnees.extend(cols_booleennes)
    if inclure_ratios:
        features_selectionnees.extend(cols_ratios)
    return features_selectionnees


# ----------------------------------------------------------------------
# 2. Construction du préprocesseur
# ----------------------------------------------------------------------
def construire_preprocessor(
    colonnes_lineaires,
    colonnes_non_lineaires,
    cols_qualitatives,
    cols_booleennes,
    cols_ratios,
    inclure_lineaires=True,
    inclure_non_lineaires=True,
    inclure_qualitatives=True,
    inclure_booleennes=True,
    inclure_ratios=True,
):
    transformers = []
    if inclure_lineaires and colonnes_lineaires:
        transformers.append(('num_lin', StandardScaler(), colonnes_lineaires))
    if inclure_non_lineaires and colonnes_non_lineaires:
        transformers.append(('num_nonlin', StandardScaler(), colonnes_non_lineaires))
    if inclure_qualitatives and cols_qualitatives:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cols_qualitatives))
    if inclure_booleennes and cols_booleennes:
        transformers.append(('bool', 'passthrough', cols_booleennes))
    if inclure_ratios and cols_ratios:
        transformers.append(('ratios', StandardScaler(), cols_ratios))

    return ColumnTransformer(transformers=transformers, remainder='drop')


# ----------------------------------------------------------------------
# 3. Dictionnaire des modèles à comparer
# ----------------------------------------------------------------------
def obtenir_modeles(random_state=42):
    return {
        "Dummy_Stratified": DummyClassifier(strategy='stratified', random_state=random_state),
        "LogisticRegression_None": LogisticRegression(
            penalty=None, solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=random_state
        ),
        "LogisticRegression_L1": LogisticRegression(
            penalty='l1', solver='liblinear', max_iter=1000,
            class_weight='balanced', random_state=random_state
        ),
        "LogisticRegression_L2": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=random_state
        ),
        "Ridge": RidgeClassifier(random_state=random_state),
        "ElasticNet_LogReg": LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000,
            class_weight='balanced', random_state=random_state
        ),
        "DecisionTree": DecisionTreeClassifier(class_weight='balanced', random_state=random_state),
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "XGBoost": xgb.XGBClassifier(random_state=random_state),
        "SVC_Prob": SVC(probability=True, class_weight='balanced', random_state=random_state),
        "SVC_RBF": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=random_state),
        "SVC_Linear": SVC(kernel='linear', probability=True, class_weight='balanced', random_state=random_state),
        "SVC_Poly": SVC(kernel='poly', degree=3, probability=True, class_weight='balanced', random_state=random_state),
        "SVR_Linear": LinearSVC(
            class_weight='balanced',
            random_state=random_state,
            max_iter=20000,
            dual=False,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


# ----------------------------------------------------------------------
# 4. Boucle d'évaluation (Train, CV, Test)
# ----------------------------------------------------------------------
def evaluer_modeles(X_train, y_train, X_test, y_test, preprocessor, models=None, cv=None, verbose=True):
    if models is None:
        models = obtenir_modeles()
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    fitted_pipelines = {}

    for name, clf in models.items():
        if verbose:
            print(f"Évaluation en cours : {name}...")

        pipeline = Pipeline(steps=[
            ('preprocessor', clone(preprocessor)),
            ('classifier', clone(clf)),
        ])

        start_time = time.time()
        
        # 1. Validation croisée sur le train
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')

        # 2. Entraînement complet sur tout le X_train
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        duration = time.time() - start_time

        # --- Calcul des probas et prédictions sur TRAIN ---
        if hasattr(pipeline, "predict_proba"):
            y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        elif hasattr(pipeline.named_steps['classifier'], "decision_function"):
            decisions_train = pipeline.decision_function(X_train)
            import numpy as np
            y_train_proba = 1 / (1 + np.exp(-decisions_train))
        else:
            y_train_proba = [0] * len(y_train)

        y_train_pred = pipeline.predict(X_train)

        # --- Calcul des probas et prédictions sur TEST ---
        if hasattr(pipeline, "predict_proba"):
            y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline.named_steps['classifier'], "decision_function"):
            decisions_test = pipeline.decision_function(X_test)
            import numpy as np
            y_test_proba = 1 / (1 + np.exp(-decisions_test))
        else:
            y_test_proba = [0] * len(y_test)

        y_test_pred = pipeline.predict(X_test)

        # --- Stockage des résultats Train & Test ---
        results.append({
            "Model": name,
            # Métriques CV
            "ROC_AUC_cv": cv_scores.mean(),
            # Métriques TRAIN (pour détecter l'overfitting)
            "ROC_AUC_train": roc_auc_score(y_train, y_train_proba) if len(set(y_train)) > 1 else 0.5,
            "Recall_train": recall_score(y_train, y_train_pred, zero_division=0),
            # Métriques TEST
            "ROC_AUC_test": roc_auc_score(y_test, y_test_proba) if len(set(y_test)) > 1 else 0.5,
            "Accuracy_test": accuracy_score(y_test, y_test_pred),
            "Precision_test": precision_score(y_test, y_test_pred, zero_division=0),
            "Recall_test": recall_score(y_test, y_test_pred, zero_division=0),
            "F1_test": f1_score(y_test, y_test_pred, zero_division=0),
            "Duration_s": duration,
        })

    df_res = pd.DataFrame(results).set_index("Model").sort_values("ROC_AUC_cv", ascending=False)
    return df_res, fitted_pipelines


# ----------------------------------------------------------------------
# 5. Affichage HTML stylé
# ----------------------------------------------------------------------
def afficher_resultats(df_res, titre="📊 Comparatif des performances des modèles (Train vs CV vs Test)"):
    display(HTML(f"<h3>{titre}</h3>"))
    display(
        df_res.style
        .format(precision=3)
        .background_gradient(subset=["ROC_AUC_cv", "ROC_AUC_train", "ROC_AUC_test", "F1_test"], cmap="Greens")
    )


# ----------------------------------------------------------------------
# 6. Widget UI pour l'étape du préprocesseur
# ----------------------------------------------------------------------
def bouton_preprocesseur_etape(get_X_train_test_callback):
    btn_prep = widgets.Button(
        description="Lancer le préprocesseur",
        button_style="warning",
        layout=widgets.Layout(width="300px", height="40px"),
    )
    out_prep = widgets.Output()

    def on_prep(b):
        with out_prep:
            clear_output()
            try:
                X_train, X_test = get_X_train_test_callback()
            except Exception:
                print("Erreur : Impossible de récupérer X_train et X_test.")
                return

            import sys
            main_ns = sys.modules['__main__'].__dict__
            if "var_buttons" not in main_ns:
                print("Le panneau de sélection des variables n'a pas été initialisé.")
                return
            var_buttons_local = main_ns["var_buttons"]

            from features_selection import obtenir_variables_selectionnees
            selection_par_groupe = obtenir_variables_selectionnees(var_buttons_local, par_groupe=True)
            
            prep = construire_preprocessor(
                colonnes_lineaires=selection_par_groupe.get("lineaires", []),
                colonnes_non_lineaires=selection_par_groupe.get("non_lineaires", []),
                cols_qualitatives=selection_par_groupe.get("qualitatives", []),
                cols_booleennes=selection_par_groupe.get("booleennes", []),
                cols_ratios=selection_par_groupe.get("nouvelles_features", []),
                inclure_lineaires=True,
                inclure_non_lineaires=True,
                inclure_qualitatives=True,
                inclure_booleennes=True,
                inclure_ratios=True,
            )
            
            prep.fit(X_train)
            X_train_prep_local = prep.transform(X_train)
            X_test_prep_local = prep.transform(X_test)
            
            main_ns["preprocessor"] = prep
            main_ns["X_train_prep"] = X_train_prep_local
            main_ns["X_test_prep"] = X_test_prep_local
            
            print("Fit uniquement sur X_train (pas de fuite du test).")
            print(f"X_train transformé : {getattr(X_train_prep_local, 'shape', type(X_train_prep_local))}")
            print(f"X_test transformé  : {getattr(X_test_prep_local, 'shape', type(X_test_prep_local))}")
            
            display(HTML("""
                <div style="background-color: #d4edda; color: #155724; padding: 12px; border-radius: 6px; border: 1px solid #c3e6cb; margin-top: 10px; font-weight: bold; font-size: 14px;">
                    ✅ Étape 2 OK → Vous pouvez maintenant passer à la cellule de modélisation !
                </div>
            """))

    btn_prep.on_click(on_prep)
    return widgets.VBox([btn_prep, out_prep])


# ----------------------------------------------------------------------
# 7. Widget UI pour l'étape de modélisation
# ----------------------------------------------------------------------
def interface_modelisation_etape():
    dict_modeles_base = obtenir_modeles()

    dropdown_cv_methode = widgets.Dropdown(
        options=[
            ("Stratified K-Fold (recommandé)", "StratifiedKFold"),
            ("Repeated Stratified K-Fold", "RepeatedStratifiedKFold"),
            ("K-Fold standard", "KFold"),
        ],
        value="StratifiedKFold",
        description="Méthode CV :",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
    )
    slider_folds = widgets.IntSlider(
        value=5, min=3, max=10, step=1,
        description="Folds :",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
    )
    slider_repeats = widgets.IntSlider(
        value=3, min=1, max=5, step=1,
        description="Répétitions :",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
    )

    categories_modeles = {
        "Référence (Baseline)": ["Dummy_Stratified"],
        "Linéaires & régularisés": [
            "LogisticRegression_None", "LogisticRegression_L1", "LogisticRegression_L2",
            "Ridge", "ElasticNet_LogReg",
        ],
        "Arbres & boosting": [
            "DecisionTree", "RandomForest", "GradientBoosting", "AdaBoost", "XGBoost",
        ],
        "Autres (SVM, KNN)": [
            "SVC_Prob", "SVC_RBF", "SVC_Linear", "SVC_Poly", "SVR_Linear", "KNN",
        ],
    }

    model_var_buttons, model_ui_blocks = {}, []
    for cat_name, model_list in categories_modeles.items():
        models_in_cat = [m for m in model_list if m in dict_modeles_base]
        if not models_in_cat:
            continue
        grp_chk = widgets.Checkbox(
            value=True,
            description=f"Activer tout : {cat_name} ({len(models_in_cat)})",
            style={"description_width": "initial"},
        )
        btn_list = []
        model_var_buttons[cat_name] = []
        for model_name in models_in_cat:
            t_btn = widgets.ToggleButton(
                value=True, description=f"ON : {model_name}", button_style="success",
                layout=widgets.Layout(width="auto", margin="2px"),
            )
            def make_toggle_observer(btn, name):
                def on_change(change):
                    btn.description = f"{'ON' if change['new'] else 'OFF'} : {name}"
                    btn.button_style = "success" if change["new"] else ""
                return on_change
            t_btn.observe(make_toggle_observer(t_btn, model_name), names="value")
            model_var_buttons[cat_name].append((model_name, t_btn))
            btn_list.append(t_btn)

        def make_group_observer(btn_list_inner):
            def on_change(change):
                if change["name"] == "value":
                    for btn in btn_list_inner:
                        btn.value = change["new"]
            return on_change
        grp_chk.observe(make_group_observer(btn_list), names="value")

        model_ui_blocks.append(widgets.VBox(
            [grp_chk, widgets.HBox(btn_list, layout=widgets.Layout(flex_flow="wrap", margin="5px 0 10px 20px"))],
            layout=widgets.Layout(border="solid 1px #ddd", padding="10px", margin="5px 0"),
        ))

    btn_models = widgets.Button(
        description="Lancer la comparaison des modèles",
        button_style="success",
        layout=widgets.Layout(width="320px", height="45px"),
    )
    out_models = widgets.Output()

    def on_models(b):
        with out_models:
            clear_output()
            import sys
            main_ns = sys.modules['__main__'].__dict__
            
            if "preprocessor" not in main_ns or "X_train" not in main_ns:
                print("Lance d’abord la cellule 1 puis la cellule 2.")
                return

            X_train = main_ns["X_train"]
            y_train = main_ns["y_train"]
            X_test = main_ns["X_test"]
            y_test = main_ns["y_test"]
            preprocessor = main_ns["preprocessor"]

            models_a_tester = {
                name: dict_modeles_base[name]
                for btn_list in model_var_buttons.values()
                for name, btn in btn_list
                if btn.value
            }
            if not models_a_tester:
                print("Sélectionne au moins un modèle.")
                return

            methode = dropdown_cv_methode.value
            n_splits = slider_folds.value
            if methode == "StratifiedKFold":
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            elif methode == "RepeatedStratifiedKFold":
                cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=slider_repeats.value, random_state=42)
            else:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            print(f"{methode} | {n_splits} folds | {len(models_a_tester)} modèle(s)\n")
            df_res_widget, fitted_pipelines_widget = evaluer_modeles(
                X_train, y_train, X_test, y_test, preprocessor,
                models=models_a_tester, cv=cv, verbose=True
            )
            
            main_ns["models"] = models_a_tester
            main_ns["cv"] = cv
            main_ns["df_res_widget"] = df_res_widget
            main_ns["fitted_pipelines_widget"] = fitted_pipelines_widget

            afficher_resultats(df_res_widget)
            print("\nModélisation terminée.")

    btn_models.on_click(on_models)

    return widgets.VBox([
        dropdown_cv_methode, slider_folds, slider_repeats,
        *model_ui_blocks, btn_models, out_models
    ])