import sys
import traceback
import importlib
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import numpy as np

def obtenir_grilles_hyperparametres():
    """
    Renvoie le dictionnaire des grilles d'hyperparamètres pour le GridSearch automatique.
    """
    return {
        "Dummy_Stratified": {},
        "LogisticRegression_None": {},
        "LogisticRegression_L1": {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        },
        "LogisticRegression_L2": {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        },
        "Lasso_LogReg": {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        },
        "ElasticNet_LogReg": {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "Ridge": {
            'classifier__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "DecisionTree": {
            'classifier__max_depth': [None, 3, 5, 8, 10, 15, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
        },
        "RandomForest": {
            'classifier__n_estimators': [50, 100, 200, 300, 500],
            'classifier__max_depth': [None, 5, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
        },
        "GradientBoosting": {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 4, 5, 7],
        },
        "AdaBoost": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1.0],
        },
        "XGBoost": {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 4, 5, 6],
            'classifier__subsample': [0.7, 0.85, 1.0],
        },
        "SVC_Prob": {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
        },
        "SVC_RBF": {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
        },
        "SVC_Linear": {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        },
        "SVC_Poly": {
            'classifier__C': [0.1, 1, 10],
            'classifier__degree': [2, 3],
        },
        "SVR_Linear": {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        },
        "KNN": {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
        }
    }

def optimiser_modeles(models, preprocessor, cv, X_train, y_train, X_test, y_test, param_grids, scoring_grid="roc_auc", tri_par="ROC_AUC_test", verbose=True, threshold=0.5):
    results = []
    best_pipelines = {}
    
    for name, model in models.items():
        if verbose:
            print(f"Optimisation de {name}...")
        
        start_time = time.time()
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        grid_params = param_grids.get(name, {})
        
        grid_search = GridSearchCV(pipeline, grid_params, cv=cv, scoring=scoring_grid, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        duration = time.time() - start_time
        best_model = grid_search.best_estimator_
        best_pipelines[name] = best_model
        
        # Récupération des probabilités / scores de décision
        if hasattr(best_model, "predict_proba"):
            y_train_proba = best_model.predict_proba(X_train)[:, 1]
            y_test_proba = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_train_proba = 1 / (1 + np.exp(-best_model.decision_function(X_train)))
            y_test_proba = 1 / (1 + np.exp(-best_model.decision_function(X_test)))
        else:
            y_train_proba = best_model.predict(X_train).astype(float)
            y_test_proba = best_model.predict(X_test).astype(float)
            
        # Prédictions binaires basées sur le seuil
        y_train_pred = (y_train_proba >= threshold).astype(int)
        y_test_pred = (y_test_proba >= threshold).astype(int)
        
        # Calcul de toutes les métriques détaillées
        try:
            roc_auc_train = roc_auc_score(y_train, y_train_proba)
        except Exception:
            roc_auc_train = 0.0
            
        try:
            roc_auc_test = roc_auc_score(y_test, y_test_proba)
        except Exception:
            roc_auc_test = 0.0
            
        results.append({
            "Model": name,
            "Meilleurs paramètres": grid_search.best_params_,
            "ROC_AUC_cv": round(grid_search.best_score_, 3),
            "ROC_AUC_train": round(roc_auc_train, 3),
            "Recall_train": round(recall_score(y_train, y_train_pred, zero_division=0), 3),
            "ROC_AUC_test": round(roc_auc_test, 3),
            "Accuracy_test": round(accuracy_score(y_test, y_test_pred), 3),
            "Precision_test": round(precision_score(y_test, y_test_pred, zero_division=0), 3),
            "Recall_test": round(recall_score(y_test, y_test_pred, zero_division=0), 3),
            "F1_test": round(f1_score(y_test, y_test_pred, zero_division=0), 3),
            "Duration_s": round(duration, 3)
        })
        
    df_res = pd.DataFrame(results)
    if not df_res.empty and tri_par in df_res.columns:
        df_res = df_res.sort_values(by=tri_par, ascending=False).reset_index(drop=True)
    return df_res, best_pipelines

def afficher_resultats_tuning(df_tuned_res, titre=""):
    print(f"\n--- {titre} ---")
    
    # Affichage avec style visuel semblable à une heatmap verte si supporté, sinon affichage classique
    try:
        numeric_cols = df_tuned_res.select_dtypes(include=[np.number]).columns
        styled_df = df_tuned_res.style.background_gradient(cmap="Greens", subset=numeric_cols)
        display(styled_df)
    except Exception:
        display(df_tuned_res)
    
    print("\n📋 Meilleurs paramètres détaillés par modèle :")
    for _, row in df_tuned_res.iterrows():
        print(f"👉 {row['Model']} :")
        print(f"    {row['Meilleurs paramètres']}\n")

def interface_tuning(dict_modeles_base, categories_modeles):
    select_max_depth = widgets.Dropdown(
        options=["None", "3", "5", "8", "10", "15", "20", "30"],
        value="8",
        description="Prof. Arbres",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="260px"),
    )
    desc_depth = widgets.HTML("<div style='font-size:10px;color:#7f8c8d;margin-left:95px;'>DecisionTree / Forêts</div>")

    select_n_estimators = widgets.Dropdown(
        options=["50", "100", "200", "300", "500"],
        value="100",
        description="N Estimators",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="260px"),
    )
    desc_nest = widgets.HTML("<div style='font-size:10px;color:#7f8c8d;margin-left:95px;'>Forêts / Boosting</div>")

    select_C_values = widgets.Dropdown(
        options=["0.001", "0.01", "0.1", "1", "10", "100", "1000"],
        value="0.1",
        description="Valeur de C",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="260px"),
    )
    desc_c = widgets.HTML("<div style='font-size:10px;color:#7f8c8d;margin-left:95px;'>LogReg L1/L2 & SVM</div>")

    select_alpha = widgets.Dropdown(
        options=["0.0001", "0.001", "0.01", "0.1", "1.0", "10.0"],
        value="1.0",
        description="Alpha",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="260px"),
    )
    desc_alpha = widgets.HTML("<div style='font-size:10px;color:#7f8c8d;margin-left:95px;'>Ridge / Lasso</div>")

    select_learning_rate = widgets.Dropdown(
        options=["0.01", "0.03", "0.05", "0.1", "0.2", "0.3"],
        value="0.1",
        description="Learn. Rate",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="260px"),
    )
    desc_lr = widgets.HTML("<div style='font-size:10px;color:#7f8c8d;margin-left:95px;'>Boosting</div>")

    slider_decision_threshold = widgets.FloatSlider(
        value=0.35, min=0.1, max=0.9, step=0.05,
        description="Seuil Décision",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="260px"),
    )
    desc_threshold = widgets.HTML("<div style='font-size:10px;color:#7f8c8d;margin-left:95px;'>Rappel vs précision</div>")

    content_grid = widgets.VBox([
        widgets.HTML("<div style='font-size:11px;font-weight:bold;margin-bottom:8px;'>Paramètres pour le réglage manuel :</div>"),
        widgets.HBox([
            widgets.VBox([select_max_depth, desc_depth]),
            widgets.VBox([select_n_estimators, desc_nest]),
            widgets.VBox([select_C_values, desc_c]),
        ], layout=widgets.Layout(justify_content="space-around")),
        widgets.HBox([
            widgets.VBox([select_alpha, desc_alpha]),
            widgets.VBox([select_learning_rate, desc_lr]),
            widgets.VBox([slider_decision_threshold, desc_threshold]),
        ], layout=widgets.Layout(justify_content="space-around")),
    ])
    panel_options_grid = widgets.Accordion(children=[content_grid])
    panel_options_grid.set_title(0, "Réglage fin & seuil de décision")
    panel_options_grid.selected_index = 0

    chk_all_models = widgets.Checkbox(
        value=False,
        description=f"Activer / désactiver tous les modèles ({len(dict_modeles_base)})",
        style={"description_width": "initial"},
    )

    tuning_model_buttons, btn_model_list, category_boxes = [], [], []

    def make_toggle_observer(btn, model_name):
        def on_change(change):
            btn.description = f"{'ON' if change['new'] else 'OFF'} : {model_name}"
            btn.button_style = "success" if change["new"] else ""
        return on_change

    for cat_name, model_names in categories_modeles.items():
        cat_buttons = []
        for name in model_names:
            if name not in dict_modeles_base:
                continue
            t_btn = widgets.ToggleButton(
                value=False,
                description=f"OFF : {name}",
                button_style="",
                layout=widgets.Layout(width="auto", margin="2px"),
            )
            t_btn.observe(make_toggle_observer(t_btn, name), names="value")
            tuning_model_buttons.append((name, t_btn))
            btn_model_list.append(t_btn)
            cat_buttons.append(t_btn)
        if cat_buttons:
            category_boxes.append(widgets.VBox([
                widgets.HTML(f"<div style='margin-top:5px;'><b>{cat_name}</b></div>"),
                widgets.HBox(cat_buttons, layout=widgets.Layout(flex_flow="wrap")),
            ]))

    def on_global_models_change(change):
        if change["name"] == "value":
            for btn in btn_model_list:
                btn.value = change["new"]

    chk_all_models.observe(on_global_models_change, names="value")

    panel_models_box = widgets.VBox(
        [chk_all_models] + category_boxes,
        layout=widgets.Layout(border="1px solid #e1e8ed", border_radius="8px", padding="15px", margin="10px 0"),
    )

    btn_tuning = widgets.Button(
        description="Réglages manuels (1 valeur)",
        button_style="warning",
        layout=widgets.Layout(width="49%", height="45px"),
    )
    btn_auto = widgets.Button(
        description="Vrai GridSearch (recherche auto)",
        button_style="danger",
        layout=widgets.Layout(width="49%", height="45px"),
    )
    btn_features = widgets.Button(
        description="🔍 Analyser les features (coefficients / importances)",
        button_style="info",
        layout=widgets.Layout(width="100%", height="40px", margin="8px 0 0 0"),
    )
    output_tuning = widgets.Output()

    def modeles_coches():
        return {
            name: dict_modeles_base[name]
            for name, btn in tuning_model_buttons if btn.value
        }

    def grilles_manuelles():
        depth_val = None if select_max_depth.value == "None" else int(select_max_depth.value)
        n_est_val = int(select_n_estimators.value)
        c_val = float(select_C_values.value)
        alpha_val = float(select_alpha.value)
        lr_val = float(select_learning_rate.value)
        return {
            "LogisticRegression_L1": {"classifier__C": [c_val]},
            "LogisticRegression_L2": {"classifier__C": [c_val]},
            "Lasso_LogReg": {"classifier__C": [c_val]},
            "ElasticNet_LogReg": {"classifier__C": [c_val], "classifier__l1_ratio": [0.1, 0.5, 0.9]},
            "Ridge": {"classifier__alpha": [alpha_val]},
            "DecisionTree": {"classifier__max_depth": [depth_val], "classifier__min_samples_split": [2, 5]},
            "RandomForest": {"classifier__n_estimators": [n_est_val], "classifier__max_depth": [depth_val]},
            "GradientBoosting": {
                "classifier__n_estimators": [n_est_val],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [lr_val],
            },
            "XGBoost": {
                "classifier__n_estimators": [n_est_val],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [lr_val],
            },
            "SVC_Linear": {"classifier__C": [c_val]},
            "SVC_RBF": {"classifier__C": [c_val]},
            "SVC_Poly": {"classifier__C": [c_val]},
            "SVR_Linear": {"classifier__C": [c_val]},
        }

    def lancer_optimisation(param_grids, titre):
        models_actives = modeles_coches()
        if not models_actives:
            print("Coche au moins un modèle.")
            return
        
        main_ns = sys.modules['__main__'].__dict__
        if "X_train" not in main_ns or "preprocessor" not in main_ns:
            print("❌ Erreur : Les données d'entraînement ne sont pas chargées.")
            return

        current_threshold = slider_decision_threshold.value
        print(f"Seuil de décision appliqué : {current_threshold}")
        print(f"Modèles : {list(models_actives)}\n")
        
        df_tuned_res, best_pipelines = optimiser_modeles(
            models=models_actives,
            preprocessor=main_ns["preprocessor"],
            cv=main_ns["cv"],
            X_train=main_ns["X_train"],
            y_train=main_ns["y_train"],
            X_test=main_ns["X_test"],
            y_test=main_ns["y_test"],
            param_grids=param_grids,
            scoring_grid="roc_auc",
            tri_par="ROC_AUC_test",
            verbose=True,
            threshold=current_threshold
        )
        
        main_ns["current_threshold"] = current_threshold
        main_ns["df_tuned_res"] = df_tuned_res
        main_ns["best_pipelines"] = best_pipelines

        print("\nTerminé.")
        afficher_resultats_tuning(df_tuned_res, titre=titre)

    def on_tuning_clicked(b):
        with output_tuning:
            clear_output()
            try:
                print("Mode manuel : 1 valeur par paramètre.")
                lancer_optimisation(grilles_manuelles(), "GridSearch manuel (valeurs du panneau)")
            except Exception:
                traceback.print_exc()

    def on_auto_clicked(b):
        with output_tuning:
            clear_output()
            try:
                print("Mode auto : recherche de plages automatiques.")
                lancer_optimisation(
                    obtenir_grilles_hyperparametres(),
                    "GridSearch automatique (plages du module)",
                )
            except Exception:
                traceback.print_exc()

    def on_features_clicked(b):
        with output_tuning:
            try:
                main_ns = sys.modules['__main__'].__dict__
                if "best_pipelines" not in main_ns or not main_ns["best_pipelines"]:
                    print("❌ Aucun modèle optimisé trouvé. Lancez d'abord un GridSearch (bouton orange ou rouge).")
                    return
                
                print("\n------------------------------------------------------------")
                print("📊 Analyse des features pour les modèles entraînés :\n")
                best_pipelines = main_ns["best_pipelines"]
                
                for name, pipeline in best_pipelines.items():
                    print(f"\n🔹 Modèle : {name}")
                    preprocessor = pipeline.named_steps.get('preprocessor')
                    classifier = pipeline.named_steps.get('classifier')
                    
                    try:
                        feature_names = preprocessor.get_feature_names_out()
                    except Exception:
                        feature_names = [f"feat_{i}" for i in range(100)]
                    
                    if hasattr(classifier, "coef_"):
                        coefs = classifier.coef_
                        if coefs.ndim > 1:
                            coefs = coefs[0]
                        df_feat = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
                        df_feat["Valeur Absolue"] = df_feat["Coefficient"].abs()
                        df_feat = df_feat.sort_values(by="Valeur Absolue", ascending=False).reset_index(drop=True)
                        display(df_feat.head(15))
                    elif hasattr(classifier, "feature_importances_"):
                        importances = classifier.feature_importances_
                        df_feat = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                        df_feat = df_feat.sort_values(by="Importance", ascending=False).reset_index(drop=True)
                        display(df_feat.head(15))
                    else:
                        print(f"⚠️ Ce type de modèle ({type(classifier).__name__}) ne expose ni coef_ ni feature_importances_.")
            except Exception:
                traceback.print_exc()

    btn_tuning.on_click(on_tuning_clicked)
    btn_auto.on_click(on_auto_clicked)
    btn_features.on_click(on_features_clicked)

    return widgets.VBox([
        widgets.HTML("<h2>Centre de contrôle du GridSearch</h2>"),
        panel_options_grid,
        panel_models_box,
        widgets.HBox([btn_tuning, btn_auto]),
        btn_features,
        output_tuning,
    ], layout=widgets.Layout(padding="10px", max_width="950px"))