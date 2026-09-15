"""
Module regroupant les fonctions de modélisation pour la prédiction de
l'attrition (a_quitte_l_entreprise).

MISES A JOUR :
- Intégration du panneau interactif de sélection par catégorie de modèles (ON/OFF).
- Ajout de la métrique PR_AUC_test (Precision-Recall AUC).
- Sélecteur interactif d'objectif métier.
- Paramétrage dynamique du nombre de folds, des répétitions et du seuil.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    cross_val_score, train_test_split, StratifiedKFold, 
    RepeatedStratifiedKFold, KFold, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)
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
# 4. Boucle d'évaluation avec scores par Fold, Moyenne, Écart-type & Anti-Fuite
# ----------------------------------------------------------------------
def evaluer_modeles(X_train, y_train, X_test, y_test, preprocessor, models=None, cv=None, n_splits=5, n_repeats=None, verbose=True):
    if models is None:
        models = obtenir_modeles()
     
    if cv is None:
        if n_repeats and n_repeats > 1:
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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

        cv_results = cross_validate(
            pipeline, 
            X_train, 
            y_train, 
            cv=cv, 
            scoring=['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'average_precision'],
            return_train_score=True,
            n_jobs=-1
        )

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        duration = time.time() - start_time

        if hasattr(pipeline, "predict_proba"):
            y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        elif hasattr(pipeline.named_steps['classifier'], "decision_function"):
            decisions_train = pipeline.decision_function(X_train)
            y_train_proba = 1 / (1 + np.exp(-decisions_train))
        else:
            y_train_proba = [0] * len(y_train)

        y_train_pred = pipeline.predict(X_train)

        if hasattr(pipeline, "predict_proba"):
            y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline.named_steps['classifier'], "decision_function"):
            decisions_test = pipeline.decision_function(X_test)
            y_test_proba = 1 / (1 + np.exp(-decisions_test))
        else:
            y_test_proba = [0] * len(y_test)

        y_test_pred = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "ROC_AUC_cv": cv_results['test_roc_auc'].mean(),
            "ROC_AUC_cv_std": cv_results['test_roc_auc'].std(),
            "ROC_AUC_folds": cv_results['test_roc_auc'].tolist(),
            
            "F1_cv": cv_results['test_f1'].mean(),
            "F1_cv_std": cv_results['test_f1'].std(),
            
            "ROC_AUC_train": roc_auc_score(y_train, y_train_proba) if len(set(y_train)) > 1 else 0.5,
            "ROC_AUC_train_cv_mean": cv_results['train_roc_auc'].mean(),
            "ROC_AUC_train_cv_std": cv_results['train_roc_auc'].std(),
            
            "ROC_AUC_test": roc_auc_score(y_test, y_test_proba) if len(set(y_test)) > 1 else 0.5,
            "PR_AUC_test": average_precision_score(y_test, y_test_proba) if len(set(y_test)) > 1 else 0.0,
            "Accuracy_test": accuracy_score(y_test, y_test_pred),
            "Precision_test": precision_score(y_test, y_test_pred, zero_division=0),
            "Recall_test": recall_score(y_test, y_test_pred, zero_division=0),
            "F1_test": f1_score(y_test, y_test_pred, zero_division=0),
            "Duration_s": duration,
        })

    df_res = pd.DataFrame(results).set_index("Model")
    
    seuil_overfitting = 0.05
    seuil_score_faible = 0.65

    df_res['Ecart_Train_CV'] = df_res['ROC_AUC_train_cv_mean'] - df_res['ROC_AUC_cv']
    df_res['Ecart_CV_Test'] = df_res['ROC_AUC_cv'] - df_res['ROC_AUC_test']

    def _diagnostic(row):
        if row.get('ROC_AUC_cv_std', 0) > 0.1:
            return "⚠️ Instable (Fort écart-type folds)"
        elif row['ROC_AUC_train_cv_mean'] < seuil_score_faible and row['ROC_AUC_cv'] < seuil_score_faible:
            return "🟠 Sous-apprentissage"
        elif row['Ecart_Train_CV'] > seuil_overfitting:
            return "🔴 Surapprentissage"
        elif row['Ecart_Train_CV'] < -seuil_overfitting:
            return "🟣 Atypique (CV > Train)"
        else:
            return "🟢 Bien équilibré"

    df_res['Diagnostic'] = df_res.apply(_diagnostic, axis=1)
    df_res = df_res.sort_values("ROC_AUC_cv", ascending=False)
    
    return df_res, fitted_pipelines


# ----------------------------------------------------------------------
# 5. Affichage HTML stylé
# ----------------------------------------------------------------------
def afficher_resultats(df_res, titre="📊 Comparatif des performances des modèles (Train vs CV vs Test)"):
    display(HTML(f"<h3>{titre}</h3>"))
    display(
        df_res.style
        .format(precision=3)
        .background_gradient(subset=["ROC_AUC_cv", "ROC_AUC_train", "ROC_AUC_test", "PR_AUC_test", "F1_test"], cmap="Greens")
    )


# ----------------------------------------------------------------------
# 5bis. Objectifs métier disponibles pour le sélecteur interactif
# ----------------------------------------------------------------------
def get_objectifs_disponibles():
    return {
        "recall_test": {
            "label": "🎯 Détecter un maximum de départs (Recall)",
            "description": "Priorise le Recall_test : on veut rater le moins de départs possible.",
            "colonne": "Recall_test",
        },
        "precision_test": {
            "label": "🎯 Limiter les fausses alertes (Precision)",
            "description": "Priorise la Precision_test : on ne déclenche une action que sur des cas vraiment à risque.",
            "colonne": "Precision_test",
        },
        "f1_test": {
            "label": "⚖️ Compromis équilibré (F1-score)",
            "description": "Priorise le F1_test : bon compromis par défaut entre Precision et Recall.",
            "colonne": "F1_test",
        },
        "roc_auc_test": {
            "label": "📈 Pouvoir discriminant global (ROC-AUC)",
            "description": "Priorise le ROC_AUC_test : capacité globale du modèle à classer les profils.",
            "colonne": "ROC_AUC_test",
        },
        "pr_auc_test": {
            "label": "📊 Discrimination sur la classe minoritaire (PR-AUC)",
            "description": "Priorise le PR_AUC_test : recommandé pour l'attrition (classe minoritaire).",
            "colonne": "PR_AUC_test",
        },
    }


# ----------------------------------------------------------------------
# 5ter. Stylisation dynamique
# ----------------------------------------------------------------------
def styliser_resultats(df_res, objectif_key="f1_test"):
    objectifs = get_objectifs_disponibles()
    objectif = objectifs.get(objectif_key, objectifs["f1_test"])
    colonne_cible = objectif["colonne"]

    df_trie = df_res.sort_values(colonne_cible, ascending=False)

    styled = (
        df_trie.style
        .format(precision=3)
        .background_gradient(subset=[colonne_cible], cmap="Greens")
    )
    
    if 'Ecart_Train_CV' in df_trie.columns:
        styled = styled.background_gradient(subset=['Ecart_Train_CV'], cmap='RdYlGn_r')
        
    return styled, objectif, df_trie


# ----------------------------------------------------------------------
# 5quater. Widget interactif de sélection d'objectif métier
# ----------------------------------------------------------------------
def interface_choix_objectif(df_res):
    objectifs = get_objectifs_disponibles()

    dropdown_objectif = widgets.Dropdown(
        options=[(v["label"], k) for k, v in objectifs.items()],
        value="f1_test",
        description="Objectif :",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )

    out = widgets.Output()

    def rafraichir(change=None):
        with out:
            clear_output()
            styled, objectif, df_trie = styliser_resultats(df_res, dropdown_objectif.value)
            display(HTML(
                f"<div style='background-color:#eaf4ea;padding:10px;border-radius:6px;"
                f"border:1px solid #c3e6cb;margin-bottom:8px;'>"
                f"<b>{objectif['label']}</b><br>{objectif['description']}<br>"
                f"<i>Tableau trié par {objectif['colonne']} décroissant. "
                f"Meilleur modèle : {df_trie.index[0]} "
                f"({objectif['colonne']} = {df_trie.iloc[0][objectif['colonne']]:.3f})</i>"
                f"</div>"
            ))
            display(styled)

    dropdown_objectif.observe(rafraichir, names="value")
    rafraichir()

    return widgets.VBox([dropdown_objectif, out])


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
            """ ""))

    btn_prep.on_click(on_prep)
    return widgets.VBox([btn_prep, out_prep])


# ----------------------------------------------------------------------
# Fonctions d'analyse des erreurs et visualisations
# ----------------------------------------------------------------------
def afficher_analyse_erreurs_detaillees(pipeline, X_eval, y_eval, seuil=0.5):
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X_eval)[:, 1]
    else:
        decisions = pipeline.decision_function(X_eval)
        probas = 1 / (1 + np.exp(-decisions))
        
    y_pred = (probas >= seuil).astype(int)
    
    df_diag = X_eval.copy()
    df_diag['Vraie_Classe'] = y_eval.values
    df_diag['Proba_Predite'] = probas
    df_diag['Classe_Predite'] = y_pred
    
    df_erreurs = df_diag[df_diag['Vraie_Classe'] != df_diag['Classe_Predite']].copy()
    
    def getTypeErreur(row):
        if row['Vraie_Classe'] == 0 and row['Classe_Predite'] == 1:
            return "Faux Positif (Fausse alerte)"
        else:
            return "Faux Négatif (Départ raté)"
            
    if not df_erreurs.empty:
        df_erreurs['Type_Erreur'] = df_erreurs.apply(getTypeErreur, axis=1)
        
    return df_erreurs


def tracer_sigmoide_et_erreurs(pipeline, X_eval, y_eval, seuil=0.5):
    plt.figure(figsize=(10, 5))
    
    if hasattr(pipeline, "named_steps") and hasattr(pipeline.named_steps.get('classifier', None), "decision_function"):
        scores = pipeline.decision_function(X_eval)
        probas = 1 / (1 + np.exp(-scores))
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_eval)
        probas = 1 / (1 + np.exp(-scores))
    else:
        probas = pipeline.predict_proba(X_eval)[:, 1]
        
    y_pred = (probas >= seuil).astype(int)
    corrects = (y_pred == y_eval)
    
    plt.scatter(range(len(probas))[corrects], probas[corrects], color='green', alpha=0.6, label='Bonne prédiction')
    plt.scatter(range(len(probas))[~corrects], probas[~corrects], color='red', alpha=0.8, label='Erreur')
    plt.axhline(y=seuil, color='orange', linestyle='--', linewidth=2, label=f'Seuil ({seuil})')
    
    plt.title("Courbe de décision, seuil et répartition des erreurs")
    plt.xlabel("Index des individus dans le jeu de données")
    plt.ylabel("Probabilité prédite")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


# ----------------------------------------------------------------------
# 6bis. Interface globale d'étude des modèles
# ----------------------------------------------------------------------
def interface_etude_modeles_etape():
    import sys
    main_ns = sys.modules['__main__'].__dict__
    
    all_models = list(main_ns.get("fitted_pipelines_widget", {}).keys())
    if not all_models:
        all_models = ["Dummy_Stratified", "LogisticRegression_L2", "RandomForest", "XGBoost"]

    select_models = widgets.SelectMultiple(
        options=all_models,
        value=all_models[:min(2, len(all_models))], 
        description='Modèles ciblés :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='320px', height='80px')
    )

    metrique_dropdown = widgets.Dropdown(
        options=['Recall', 'F1', 'ROC_AUC', 'Precision', 'Accuracy', 'PR_AUC'],
        value='Recall',
        description='Métrique Top N :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )

    top_n_models_slider = widgets.IntSlider(
        value=min(1, len(all_models)),
        min=1,
        max=max(1, len(all_models)),
        step=1,
        description='Top N :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    seuil_slider = widgets.FloatSlider(
        value=main_ns.get("seuil_personnalise", 0.5),
        min=0.05, max=0.95, step=0.05,
        description='Seuil de décision :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='260px'),
        readout_format='.2f'
    )

    dataset_dropdown = widgets.Dropdown(
        options=[('Jeu de Test', 'test'), ('Jeu d\'Entraînement (Train)', 'train')],
        value='test',
        description='Données :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )

    btn_eval = widgets.Button(
        description="🚀 Charger / Actualiser les analyses",
        button_style='primary',
        layout=widgets.Layout(width='320px', height='40px')
    )

    tab_contents = [widgets.Output() for _ in range(7)]
    tab = widgets.Tab()
    tab.children = tab_contents
    tab.set_title(0, "1. Probas & Erreurs")
    tab.set_title(1, "2. Résidus & CV")
    tab.set_title(2, "3. Importances")
    tab.set_title(3, "4. Confusion & Class.")
    tab.set_title(4, "5. Grille True vs Pred")
    tab.set_title(5, "6. Corrélations")
    tab.set_title(6, "7. ROC / PR / Seuil")

    def on_eval_clicked(b):
        import sys
        main_ns = sys.modules['__main__'].__dict__
        
        if "df_res_widget" not in main_ns or "fitted_pipelines_widget" not in main_ns:
            with tab.children[0]:
                clear_output()
                print("❌ Erreur : Veuillez d'abord exécuter l'entraînement des modèles !")
            return

        current_df_res = main_ns["df_res_widget"]
        current_fitted_pipelines = main_ns["fitted_pipelines_widget"]
        
        updated_models = list(current_fitted_pipelines.keys())
        select_models.options = updated_models
        if not select_models.value or any(m not in updated_models for m in select_models.value):
            select_models.value = updated_models[:min(2, len(updated_models))]

        X_train, y_train = main_ns["X_train"], main_ns["y_train"]
        X_test, y_test = main_ns["X_test"], main_ns["y_test"]

        selected_metric = metrique_dropdown.value.lower()
        current_top_n = top_n_models_slider.value
        use_train = (dataset_dropdown.value == 'train')
        seuil_actuel = seuil_slider.value
        
        X_eval, y_eval = (X_train, y_train) if use_train else (X_test, y_test)
        dataset_label = "Entraînement (Train)" if use_train else "Test"

        models_manuel = list(select_models.value)
        colonnes_possibles = [c for c in current_df_res.columns if selected_metric in c.lower()]
        colonne_selection = colonnes_possibles[0] if colonnes_possibles else current_df_res.columns[0]
        
        top_models_df = current_df_res.sort_values(by=colonne_selection, ascending=False).head(current_top_n)
        models_top_n = top_models_df.index.tolist()
        
        active_models = list(set(models_manuel + models_top_n))

        with tab.children[0]:
            clear_output()
            print(f"--- 1. Distribution des probabilités & Erreurs [{dataset_label}] (Seuil actif : {seuil_actuel}) ---")
            print(f"Modèles analysés : {active_models}")

        with tab.children[1]:
            clear_output()
            print(f"--- 2. Diagnostic global Overfitting ---")
            display(current_df_res.loc[active_models, ['ROC_AUC_cv', 'ROC_AUC_cv_std', 'ROC_AUC_test', 'Diagnostic']])

        for i in range(2, 4):
            with tab.children[i]:
                clear_output()
                print(f"Analyse prête pour les modèles : {active_models}")

        with tab.children[4]:
            clear_output()
            print(f"--- 5. Grille True vs Pred & Analyse détaillée des erreurs [{dataset_label}] (Seuil : {seuil_actuel}) ---")
            if active_models:
                nom_modele = active_models[0]
                pipeline_actif = current_fitted_pipelines.get(nom_modele)
                if pipeline_actif:
                    print(f"Modèle affiché : {nom_modele}")
                    try:
                        tracer_sigmoide_et_erreurs(pipeline_actif, X_eval, y_eval, seuil=seuil_actuel)
                    except Exception as e:
                        print(f"Erreur lors du tracé : {e}")
                    print("\n📋 Tableau détaillé des individus mal prédits :")
                    df_err = afficher_analyse_erreurs_detaillees(pipeline_actif, X_eval, y_eval, seuil=seuil_actuel)
                    if df_err.empty:
                        print("🎉 Aucune erreur sur ce jeu de données avec ce seuil !")
                    else:
                        display(df_err)

        for i in range(5, 7):
            with tab.children[i]:
                clear_output()
                print(f"Analyse prête pour les modèles : {active_models}")

    btn_eval.on_click(on_eval_clicked)

    return widgets.VBox([
        widgets.HBox([select_models, widgets.VBox([metrique_dropdown, top_n_models_slider])]),
        widgets.HBox([seuil_slider, dataset_dropdown, btn_eval]),
        tab
    ])


# ----------------------------------------------------------------------
# 7. Widget UI pour l'étape de modélisation globale (AVEC PANNEAU PAR CATÉGORIE)
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
    slider_seuil = widgets.FloatSlider(
        value=0.5, min=0.05, max=0.95, step=0.05,
        description="Seuil de décision :",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
        readout_format='.2f'
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
                print("Lance d'abord la cellule 1 puis la cellule 2.")
                return

            X_train = main_ns["X_train"]
            y_train = main_ns["y_train"]
            X_test = main_ns["X_test"]
            y_test = main_ns["y_test"]
            preprocessor = main_ns["preprocessor"]

            models_a_tester = {
                name: dict_modeles_base[name]
                for cat_name, btn_list in model_var_buttons.items()
                for name, btn in btn_list
                if btn.value
            }
            if not models_a_tester:
                print("Sélectionne au moins un modèle.")
                return

            methode = dropdown_cv_methode.value
            n_splits = slider_folds.value
            n_repeats = slider_repeats.value

            if methode == "StratifiedKFold":
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            elif methode == "RepeatedStratifiedKFold":
                cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
            else:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            print(f"{methode} | {n_splits} folds | {len(models_a_tester)} modèle(s)\n")
            df_res_widget, fitted_pipelines_widget = evaluer_modeles(
                X_train, y_train, X_test, y_test, preprocessor,
                models=models_a_tester, cv=cv, n_splits=n_splits, n_repeats=n_repeats, verbose=True
            )

            main_ns["seuil_personnalise"] = slider_seuil.value
            main_ns["models"] = models_a_tester
            main_ns["cv"] = cv
            main_ns["df_res_widget"] = df_res_widget
            main_ns["fitted_pipelines_widget"] = fitted_pipelines_widget

            print("\nModélisation terminée.\n")
            display(interface_choix_objectif(df_res_widget))

    btn_models.on_click(on_models)

    return widgets.VBox([
        dropdown_cv_methode, slider_folds, slider_repeats, slider_seuil,
        *model_ui_blocks, btn_models, out_models
    ])