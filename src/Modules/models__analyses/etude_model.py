"""
etude_model.py
===============
Module regroupant les analyses post-entraînement du modèle retenu pour la
prédiction de l'attrition (a_quitte_l_entreprise) :

- analyser_residus()                       : diagnostic des résidus (4 graphiques + tests statistiques)
- distribution_probabilites_par_type()     : distribution des probabilités prédites par modèle
- comparer_train_test()                    : graphique en barres horizontales Train (CV) vs Test
- importance_permutation()                 : classement des variables par importance (permutation importance)
- tracer_beeswarm_shap()                   : graphique Beeswarm SHAP (vue globale)
- tracer_shap_local()                      : graphique SHAP local (waterfall/force, une observation)
- analyser_distribution_erreurs()          : distribution globale des erreurs (histogramme & KDE)
- afficher_grille_true_vs_pred()           : grille intelligente de graphiques True vs Predicted
- importance_permutation_avec_erreur()     : permutation importance avec écart-type et affichage tabulaire
- afficher_matrice_confusion()             : matrice de confusion sous forme de heatmap
- analyser_correlations_features()         : matrices Pearson, Spearman, paires corrélées & pairplot
- tracer_courbe_roc()                      : courbes ROC superposées (plusieurs modèles)
- tracer_courbe_precision_recall()         : courbes Precision-Recall superposées (avec baseline)
- tracer_courbe_calibration()              : diagramme de fiabilité (probas prédites vs réelles)
- tracer_courbe_apprentissage()            : learning curve (score Train vs CV selon taille échantillon) + verdict auto
- analyser_seuil_optimal()                 : Precision / Recall / F1 en fonction du seuil de décision
- diagnostiquer_overfitting_tableau()      : tableau récapitulatif Overfitting / Underfitting / Bien équilibré, tous modèles
- interface_etude_modeles_etape()          : interface interactive à 8 onglets avec choix Train/Test

Toutes les fonctions prennent le pipeline/modèle en paramètre explicite
(jamais de variable globale implicite), pour éviter les erreurs de type
NameError liées à un objet non défini.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.model_selection import learning_curve, cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, precision_score, recall_score, f1_score,
)
from sklearn.calibration import calibration_curve
from IPython.display import display, clear_output
import ipywidgets as widgets
from collections import OrderedDict

# Importation de SHAP pour le graphique Beeswarm
import shap

# Réutilise la catégorisation des modèles définie dans model.py si le module
# est disponible dans le même package ; sinon, on retombe sur une copie locale
# autonome pour que ce module reste indépendant.
try:
    from model import obtenir_categories_modeles, creer_selecteur_modeles_categorise, obtenir_modeles
    _NOMS_MODELES_DEFAUT = list(obtenir_modeles().keys())
except ImportError:
    _NOMS_MODELES_DEFAUT = [
        "Dummy_Stratified", "LogisticRegression_None", "LogisticRegression_L1", "LogisticRegression_L2",
        "Ridge", "ElasticNet_LogReg", "DecisionTree", "RandomForest", "GradientBoosting", "AdaBoost",
        "XGBoost", "SVC_Prob", "SVC_RBF", "SVC_Linear", "SVC_Poly", "SVR_Linear", "KNN",
    ]

    CATEGORIES_MODELES = OrderedDict([
        ("Référence (Baseline)", ["Dummy_Stratified"]),
        ("Linéaires & Régularisés", [
            "LogisticRegression_None", "LogisticRegression_L1", "LogisticRegression_L2",
            "Ridge", "ElasticNet_LogReg", "SVR_Linear",
        ]),
        ("Arbres & Boosting", [
            "DecisionTree", "RandomForest", "GradientBoosting", "AdaBoost", "XGBoost",
        ]),
        ("Autres (SVM, KNN)", [
            "SVC_Prob", "SVC_RBF", "SVC_Linear", "SVC_Poly", "KNN",
        ]),
    ])

    def obtenir_categories_modeles(noms_modeles=None):
        if noms_modeles is None:
            noms_modeles = [m for modeles in CATEGORIES_MODELES.values() for m in modeles]
        noms_set = set(noms_modeles)
        resultat = OrderedDict()
        deja_places = set()
        for categorie, modeles in CATEGORIES_MODELES.items():
            presents = [m for m in modeles if m in noms_set]
            if presents:
                resultat[categorie] = presents
                deja_places.update(presents)
        non_classes = [m for m in noms_modeles if m not in deja_places]
        if non_classes:
            resultat["Autres"] = non_classes
        return resultat

    def creer_selecteur_modeles_categorise(noms_modeles, valeurs_par_defaut=None, categories=None):
        if categories is None:
            categories = obtenir_categories_modeles(noms_modeles)
        valeurs_par_defaut = set(noms_modeles) if valeurs_par_defaut is None else set(valeurs_par_defaut)

        checkboxes = OrderedDict()
        panneaux, titres = [], []

        def _handler_cocher(cases, valeur):
            def handler(b):
                for c in cases:
                    c.value = valeur
            return handler

        for categorie, modeles in categories.items():
            cases_categorie = []
            for nom in modeles:
                cb = widgets.Checkbox(
                    value=(nom in valeurs_par_defaut), description=nom,
                    indent=False, layout=widgets.Layout(width='auto'),
                )
                checkboxes[nom] = cb
                cases_categorie.append(cb)

            btn_tout = widgets.Button(description="Tout cocher", button_style='info',
                                       layout=widgets.Layout(width='110px'))
            btn_rien = widgets.Button(description="Tout décocher",
                                       layout=widgets.Layout(width='110px'))
            btn_tout.on_click(_handler_cocher(cases_categorie, True))
            btn_rien.on_click(_handler_cocher(cases_categorie, False))

            panneaux.append(widgets.VBox([widgets.HBox([btn_tout, btn_rien])] + cases_categorie))
            titres.append(f"{categorie} ({len(modeles)})")

        accordion = widgets.Accordion(children=panneaux)
        for i, titre in enumerate(titres):
            accordion.set_title(i, titre)
        if panneaux:
            accordion.selected_index = 0

        label_compteur = widgets.Label()

        def _maj_compteur(change=None):
            n = sum(cb.value for cb in checkboxes.values())
            label_compteur.value = f"{n} / {len(checkboxes)} modèle(s) sélectionné(s)"

        for cb in checkboxes.values():
            cb.observe(_maj_compteur, names='value')
        _maj_compteur()

        btn_tout_global = widgets.Button(description="✅ Tout sélectionner", button_style='success',
                                          layout=widgets.Layout(width='170px'))
        btn_rien_global = widgets.Button(description="❌ Tout désélectionner", button_style='danger',
                                          layout=widgets.Layout(width='170px'))
        btn_tout_global.on_click(_handler_cocher(list(checkboxes.values()), True))
        btn_rien_global.on_click(_handler_cocher(list(checkboxes.values()), False))

        conteneur = widgets.VBox([
            widgets.HBox([btn_tout_global, btn_rien_global, label_compteur]),
            accordion,
        ])

        def get_selection():
            return [nom for nom, cb in checkboxes.items() if cb.value]

        return conteneur, get_selection


# ----------------------------------------------------------------------
# 1. Analyse des résidus
# ----------------------------------------------------------------------
def analyser_residus(modele, X_test, y_test, nom_modele=None):
    if nom_modele:
        print(f"Modèle utilisé pour l'analyse des résidus : {nom_modele}")

    common_idx = X_test.index.intersection(y_test.index)
    X_test_aligned = X_test.loc[common_idx]
    y_test_aligned = y_test.loc[common_idx]

    if hasattr(modele, "predict_proba"):
        y_pred_proba = modele.predict_proba(X_test_aligned)[:, 1]
        label_y_axe = "Résidus (y_réel - proba)"
    elif hasattr(modele, "decision_function"):
        scores = modele.decision_function(X_test_aligned)
        y_pred_proba = 1 / (1 + np.exp(-scores))
        label_y_axe = "Résidus (y_réel - sigmoid(decision_function))"
    else:
        y_pred_proba = modele.predict(X_test_aligned).astype(float)
        label_y_axe = "Résidus (y_réel - prediction_brute)"

    residuals = y_test_aligned.values - y_pred_proba

    df_residuals = X_test_aligned.copy()
    df_residuals['Residuals'] = residuals
    df_residuals['Abs_Residuals'] = np.abs(residuals)
    df_residuals['Predicted_Proba'] = y_pred_proba
    df_residuals['Vraie_Classe'] = y_test_aligned.values

    correlation_matrix = df_residuals.corr(numeric_only=True)[['Residuals', 'Abs_Residuals']]
    display(
        correlation_matrix.sort_values(by='Abs_Residuals', ascending=False)
        .head(10)
        .style.background_gradient(cmap="coolwarm", subset=['Residuals', 'Abs_Residuals'])
        .format(precision=3)
    )

    labels_map = {0: 'Reste (y=0)', 1: 'Quitte (y=1)'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for classe, color in [(0, 'steelblue'), (1, 'darkorange')]:
        mask = df_residuals['Vraie_Classe'] == classe
        axes[0, 0].scatter(
            df_residuals.loc[mask, 'Predicted_Proba'],
            df_residuals.loc[mask, 'Residuals'],
            alpha=0.6, edgecolor='k', linewidth=0.3,
            color=color, label=labels_map[classe]
        )
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel("Score / Probabilité prédite")
    axes[0, 0].set_ylabel(label_y_axe)
    axes[0, 0].set_title("Résidus vs Prédictions")
    axes[0, 0].legend()

    for classe, color in [(0, 'steelblue'), (1, 'darkorange')]:
        mask = df_residuals['Vraie_Classe'] == classe
        sns.histplot(df_residuals.loc[mask, 'Residuals'], kde=True, ax=axes[0, 1],
                     color=color, label=labels_map[classe], alpha=0.5)
    axes[0, 1].axvline(x=0, color='red', linestyle='--')
    axes[0, 1].set_title("Distribution des résidus par vraie classe")
    axes[0, 1].legend()

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("QQ-Plot (indicatif — normalité non attendue en classification)")

    for classe, color in [(0, 'steelblue'), (1, 'darkorange')]:
        mask = df_residuals['Vraie_Classe'] == classe
        axes[1, 1].scatter(
            np.where(mask)[0],
            df_residuals.loc[mask, 'Residuals'],
            alpha=0.6, edgecolor='k', linewidth=0.3,
            color=color, label=labels_map[classe]
        )
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Résidus")
    axes[1, 1].set_title("Résidus vs Ordre des observations")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"Test de normalité (Shapiro-Wilk) : stat={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    print(f"\nMoyenne des résidus : {residuals.mean():.4f}")
    print(f"Écart-type des résidus : {residuals.std():.4f}")

    return df_residuals


# ----------------------------------------------------------------------
# 2. Distribution des probabilités par type de prédiction
# ----------------------------------------------------------------------
def distribution_probabilites_par_type(fitted_pipelines, df_res, X_train, y_train, X_test, y_test, threshold=0.5, nom_modele=None):
    if isinstance(nom_modele, str):
        nom_modele = [nom_modele]
    elif nom_modele is None:
        nom_modele = list(fitted_pipelines.keys())
        
    for name in nom_modele:
        if name not in fitted_pipelines:
            continue
            
        pipeline = fitted_pipelines[name]
        
        datasets = {
            "Entraînement (Train)": (X_train, y_train),
            "Test": (X_test, y_test)
        }
        
        for dataset_name, (X_data, y_data) in datasets.items():
            if X_data is None or y_data is None:
                continue
                
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_data)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                decisions = pipeline.decision_function(X_data)
                y_proba = 1 / (1 + np.exp(-decisions))
            else:
                y_proba = pipeline.predict(X_data).astype(float)
                
            y_pred = (y_proba >= threshold).astype(int)
            y_true = np.array(y_data)
            
            df_plot = pd.DataFrame({
                'Probabilite': y_proba,
                'Vrai_Label': y_true,
                'Pred_Label': y_pred
            })
            
            def get_category(row):
                if row['Vrai_Label'] == 0 and row['Pred_Label'] == 0:
                    return 'Vrai Négatif (TN)'
                elif row['Vrai_Label'] == 0 and row['Pred_Label'] == 1:
                    return 'Faux Positif (FP)'
                elif row['Vrai_Label'] == 1 and row['Pred_Label'] == 0:
                    return 'Faux Négatif (FN)'
                else:
                    return 'Vrai Positif (TP)'
                    
            df_plot['Categorie'] = df_plot.apply(get_category, axis=1)
            
            fig, ax = plt.subplots(figsize=(9, 4.5))
            
            palette_colors = {
                'Vrai Négatif (TN)': 'steelblue', 
                'Faux Positif (FP)': 'orange', 
                'Faux Négatif (FN)': 'indianred', 
                'Vrai Positif (TP)': 'forestgreen'
            }
            
            sns.histplot(
                data=df_plot, 
                x='Probabilite', 
                hue='Categorie', 
                bins=30, 
                multiple='stack',
                palette=palette_colors,
                ax=ax
            )
            
            ax.axvline(x=threshold, color='red', linestyle='--', label=f'Seuil ({threshold})')
            ax.set_title(f"Distribution des probabilités [{dataset_name}] — {name}")
            ax.set_xlabel("Probabilité prédite de la classe positive")
            ax.set_ylabel("Effectif")
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='steelblue', label='Vrai Négatif (TN)'),
                Patch(facecolor='orange', label='Faux Positif (FP)'),
                Patch(facecolor='indianred', label='Faux Négatif (FN)'),
                Patch(facecolor='forestgreen', label='Vrai Positif (TP)'),
                plt.Line2D([0], [0], color='red', linestyle='--', label=f'Seuil ({threshold})')
            ]
            
            ax.legend(handles=legend_elements, loc='upper right')
            plt.tight_layout()
            plt.show()


# ----------------------------------------------------------------------
# 3. Comparaison Train (CV) vs Test
# ----------------------------------------------------------------------
def comparer_train_test(
    df_res,
    top_model_names=None,
    col_train='ROC_AUC_train',
    col_cv='ROC_AUC_cv',
    col_test='ROC_AUC_test',
    titre="Train (résubstitution) vs CV vs Test — diagnostic overfitting / underfitting",
    xlabel="Score",
    xlim=(0.4, 1.05),
    seuil_overfitting=0.05,
    seuil_score_faible=0.65,
):
    if top_model_names is not None:
        df_plot = df_res.loc[df_res.index.intersection(top_model_names)]
        df_plot = df_plot.reindex([m for m in top_model_names if m in df_plot.index])
    else:
        df_plot = df_res

    modeles = df_plot.index.tolist()
    y_pos = np.arange(len(modeles))
    hauteur = 0.26

    fig, ax = plt.subplots(figsize=(10.5, max(len(modeles) * 0.7, 4)))

    ax.barh(y_pos + hauteur, df_plot[col_train], height=hauteur,
            color='skyblue', edgecolor='black', label='Train (résubstitution)')
    ax.barh(y_pos, df_plot[col_cv], height=hauteur,
            color='goldenrod', edgecolor='black', label='CV (validation croisée)')
    ax.barh(y_pos - hauteur, df_plot[col_test], height=hauteur,
            color='seagreen', edgecolor='black', label='Test')

    for i, modele in enumerate(modeles):
        train_score = df_plot.loc[modele, col_train]
        cv_score = df_plot.loc[modele, col_cv]
        test_score = df_plot.loc[modele, col_test]
        ecart_cv = train_score - cv_score
        ecart_test = train_score - test_score

        if train_score < seuil_score_faible and cv_score < seuil_score_faible:
            verdict, couleur = "Sous-apprentissage", "darkorange"
        elif ecart_cv > seuil_overfitting:
            verdict, couleur = "Surapprentissage", "crimson"
        elif ecart_cv < -seuil_overfitting:
            verdict, couleur = "Atypique (CV > train)", "purple"
        else:
            verdict, couleur = "Bien équilibré", "forestgreen"

        ax.text(
            max(train_score, cv_score, test_score) + 0.015, y_pos[i],
            f"ΔCV={ecart_cv:+.3f}  ΔTest={ecart_test:+.3f}  {verdict}",
            va='center', fontsize=8, color=couleur, fontweight='bold',
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(modeles)
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(titre)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, axis='x', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 4. Importance des variables (permutation importance)
# ----------------------------------------------------------------------
def importance_permutation(
    pipeline, X, y,
    n_repeats=10, random_state=42, n_jobs=-1, scoring='f1', top_n=15
):
    result = permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs, scoring=scoring,
    )

    importances_df = pd.DataFrame({
        'Variable': X.columns,
        'Importance_Moyenne': result.importances_mean,
    }).sort_values(by='Importance_Moyenne', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(importances_df['Variable'][-top_n:], importances_df['Importance_Moyenne'][-top_n:])
    plt.xlabel(f"Baisse de performance ({scoring}) si la variable est brouillée")
    plt.title(f"Top {top_n} des indicateurs sur lesquels le modèle s'accroche le plus")
    plt.tight_layout()
    plt.show()

    return importances_df.sort_values(by='Importance_Moyenne', ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# 5. Importance des variables — comparaison sur PLUSIEURS modèles
# ----------------------------------------------------------------------
def importance_permutation_tous_modeles(
    fitted_pipelines, X, y,
    models=None, n_repeats=10, random_state=42, n_jobs=-1, scoring='f1', verbose=True,
):
    if models is None:
        models = list(fitted_pipelines.keys())

    resultats = {}
    for nom_modele in models:
        if verbose:
            print(f"Calcul de l'importance par permutation : {nom_modele}...")
        pipeline = fitted_pipelines[nom_modele]
        result = permutation_importance(
            pipeline, X, y,
            n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs, scoring=scoring,
        )
        resultats[nom_modele] = pd.Series(result.importances_mean, index=X.columns)

    importances_pivot = pd.DataFrame(resultats)
    importances_pivot['Moyenne_tous_modeles'] = importances_pivot.mean(axis=1)
    importances_pivot = importances_pivot.sort_values(by='Moyenne_tous_modeles', ascending=False)

    return importances_pivot


def plot_importance_heatmap(importances_pivot, top_n=15, scoring='f1'):
    data = importances_pivot.drop(columns='Moyenne_tous_modeles', errors='ignore').head(top_n)

    plt.figure(figsize=(1.2 * data.shape[1] + 3, 0.45 * len(data) + 2))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="Blues", linewidths=0.5,
        cbar_kws={'label': f"Baisse de performance ({scoring})"}
    )
    plt.title(f"Importance des variables par permutation — comparaison des modèles (Top {top_n})",
              fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Modèle")
    plt.ylabel("Variable")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_importance_grille(importances_pivot, top_n=15, ncols=2, scoring='f1', taille_case=(5.5, 4)):
    modeles = [c for c in importances_pivot.columns if c != 'Moyenne_tous_modeles']
    n_modeles = len(modeles)

    nrows = int(np.ceil(n_modeles / ncols))
    figsize = (taille_case[0] * ncols, taille_case[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, nom_modele in enumerate(modeles):
        top_var = importances_pivot[nom_modele].sort_values(ascending=True).tail(top_n)
        axes[i].barh(top_var.index, top_var.values, color='steelblue')
        axes[i].set_title(nom_modele, fontsize=10, fontweight='bold')
        axes[i].set_xlabel(f"Baisse de performance ({scoring})", fontsize=8)
        axes[i].tick_params(labelsize=7)

    for j in range(n_modeles, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 6. Graphique Beeswarm SHAP (Nouveau)
# ----------------------------------------------------------------------
def _calculer_shap_values(pipeline, X_eval, nom_modele="Modèle"):
    """
    Fonction interne partagée : calcule les valeurs SHAP pour un pipeline donné.
    Retourne (shap_values, X_transforme, feature_names) en gérant le cas
    multi-classes (shape à 3 dimensions -> on garde la classe positive).
    Utilisée à la fois par tracer_beeswarm_shap() (vue globale) et
    tracer_shap_local() (vue locale, par observation).
    """
    print(f"Calcul des valeurs SHAP ({nom_modele})...")

    if hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get('preprocessor', None)
        model = pipeline.named_steps.get('classifier', pipeline.named_steps.get('model', list(pipeline.named_steps.values())[-1]))

        if preprocessor is not None:
            X_transforme = preprocessor.transform(X_eval)
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = [f"feat_{i}" for i in range(X_transforme.shape[1])]
        else:
            X_transforme = X_eval
            feature_names = X_eval.columns
    else:
        model = pipeline
        X_transforme = X_eval
        feature_names = X_eval.columns if hasattr(X_eval, "columns") else None

    background = shap.sample(X_transforme, min(100, X_transforme.shape[0])) if hasattr(X_transforme, "shape") else X_transforme[:100]

    if hasattr(model, "predict_proba"):
        # Modèles avec sortie probabiliste (arbres, régression logistique, SVC(probability=True), etc.)
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(X_transforme)
    elif hasattr(model, "feature_importances_") or type(model).__name__ in (
        "RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier",
        "LGBMClassifier", "CatBoostClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier",
    ):
        # Modèles à base d'arbres sans predict_proba exposée directement
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_transforme)
    elif hasattr(model, "coef_"):
        # Modèles linéaires sans predict_proba (ex: LinearSVC, LinearSVR, SGDClassifier sans "log_loss")
        try:
            explainer = shap.LinearExplainer(model, background)
        except Exception:
            explainer = shap.Explainer(model.decision_function if hasattr(model, "decision_function") else model.predict, background)
        shap_values = explainer(X_transforme)
    else:
        # Dernier recours : explication basée uniquement sur predict()
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(X_transforme)

    if len(shap_values.values.shape) == 3:
        # Cas multi-classes (ou proba à 2 colonnes) : on ne garde que la classe positive (1)
        shap_values = shap_values[..., 1]

    return shap_values, X_transforme, feature_names


def tracer_beeswarm_shap(pipeline, X_eval, nom_modele="Modèle", max_display=15):
    """
    Génère un graphique Beeswarm SHAP pour visualiser l'impact global
    de chaque variable sur les prédictions du modèle (vue d'ensemble,
    toutes observations superposées).
    """
    try:
        shap_values, X_transforme, feature_names = _calculer_shap_values(pipeline, X_eval, nom_modele)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values, X_transforme, feature_names=feature_names, max_display=max_display, show=False)

        plt.title(f"Graphique Beeswarm SHAP — {nom_modele}", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"❌ Échec de la génération SHAP ({nom_modele}) : {e}")


def tracer_shap_local(pipeline, X_eval, index=0, nom_modele="Modèle", max_display=15, type_graphique="waterfall"):
    """
    Génère une explication SHAP LOCALE, c'est-à-dire pour une seule observation
    (une ligne de X_eval), identifiée par sa position `index`.

    Complémentaire au beeswarm (vue globale) : ici on répond à la question
    "pourquoi le modèle a-t-il pris CETTE décision pour CET individu ?".

    Paramètres
    ----------
    index : int ou list[int]
        Position(s) de la ou des observation(s) à expliquer dans X_eval.
        Un seul index -> waterfall (ou force plot). Une liste -> force plot
        empilé sur plusieurs observations.
    type_graphique : "waterfall" ou "force"
    """
    try:
        shap_values, X_transforme, feature_names = _calculer_shap_values(pipeline, X_eval, nom_modele)

        if isinstance(index, (list, tuple, np.ndarray)):
            # Plusieurs observations à la fois -> force plot empilé
            shap.force_plot(
                shap_values.base_values[index[0]] if np.ndim(shap_values.base_values) > 0 else shap_values.base_values,
                shap_values.values[index],
                X_transforme[index] if hasattr(X_transforme, "__getitem__") else None,
                feature_names=feature_names,
                matplotlib=True,
                show=False,
            )
            plt.title(f"SHAP local (observations {list(index)}) — {nom_modele}", fontsize=12, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.show()
            return

        # Une seule observation
        obs = shap_values[index]

        plt.figure(figsize=(10, 6))
        if type_graphique == "force":
            shap.force_plot(obs.base_values, obs.values, obs.data, feature_names=feature_names, matplotlib=True, show=False)
        else:
            shap.plots.waterfall(obs, max_display=max_display, show=False)

        plt.title(f"SHAP local (observation n°{index}) — {nom_modele}", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"❌ Échec de la génération SHAP locale ({nom_modele}) : {e}")


# ----------------------------------------------------------------------
# 7. Distribution globale des erreurs
# ----------------------------------------------------------------------
def analyser_distribution_erreurs(modele, X_test, y_test, nom_modele=None):
    if nom_modele:
        print(f"\nAnalyse de la distribution des erreurs pour : {nom_modele}")

    if hasattr(modele, "predict_proba"):
        y_pred_proba = modele.predict_proba(X_test)[:, 1]
    elif hasattr(modele, "decision_function"):
        scores = modele.decision_function(X_test)
        y_pred_proba = 1 / (1 + np.exp(-scores))
    else:
        y_pred_proba = modele.predict(X_test).astype(float)
    
    df_analysis = X_test.copy()
    df_analysis['reel'] = y_test.values
    df_analysis['probabilite'] = y_pred_proba
    df_analysis['error'] = df_analysis['probabilite'] - df_analysis['reel']
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df_analysis['error'], kde=True, bins=30, color='purple')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label="Erreur nulle (0)")
    
    titre = f"Distribution globale des erreurs — {nom_modele}" if nom_modele else "Distribution globale des erreurs"
    plt.title(titre)
    plt.xlabel("Erreur (Score prédit - Réalité)")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    return df_analysis


# ----------------------------------------------------------------------
# 8. Importance des variables par permutation (jeu de test, avec écart-type)
# ----------------------------------------------------------------------
def importance_permutation_avec_erreur(
    pipeline, X_test, y_test, nom_modele=None,
    scoring='roc_auc', n_repeats=10, random_state=42, n_jobs=-1, top_n=15,
):
    if nom_modele:
        print(f"Modèle sélectionné pour les importances : {nom_modele}")

    result = permutation_importance(
        pipeline, X_test, y_test,
        scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs,
    )

    perm_sorted_idx = result.importances_mean.argsort()
    df_perm = pd.DataFrame({
        'Feature': X_test.columns[perm_sorted_idx],
        'Importance_Mean': result.importances_mean[perm_sorted_idx],
        'Importance_Std': result.importances_std[perm_sorted_idx],
    })

    print(f"\nTop {top_n} des variables les plus importantes :")
    display(df_perm.tail(top_n).iloc[::-1])

    plt.figure(figsize=(10, 8))
    plt.barh(
        df_perm['Feature'][-top_n:], df_perm['Importance_Mean'][-top_n:],
        xerr=df_perm['Importance_Std'][-top_n:],
        color='steelblue', edgecolor='black', capsize=4,
    )
    plt.xlabel(f"Baisse moyenne du score {scoring} lors de la permutation")
    titre_suffixe = f"— {nom_modele}" if nom_modele else ""
    plt.title(f"Permutation Importances (Top {top_n}) {titre_suffixe}")
    plt.tight_layout()
    plt.show()

    return df_perm.sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# 9. Matrice de confusion
# ----------------------------------------------------------------------
def afficher_matrice_confusion(y_true, y_pred, nom_modele="Modèle"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matrice de confusion — {nom_modele}")
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs réelles")
    plt.tight_layout()
    plt.show()  
    return cm


# ----------------------------------------------------------------------
# 10. Grille intelligente de probabilités prédites vs Réalité
# ----------------------------------------------------------------------
def afficher_grille_true_vs_pred(
    fitted_pipelines, top_model_names, X_test, y_test,
    threshold=0.5, ncols=2, figsize_par_case=(6, 5)
):
    n_plots = len(top_model_names)
    if n_plots == 0:
        print("Aucune donnée à afficher.")
        return

    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_par_case[0] * ncols, figsize_par_case[1] * nrows)
    )
    axes = np.atleast_1d(axes).ravel()

    for i, nom_modele in enumerate(top_model_names):
        ax = axes[i]
        pipeline = fitted_pipelines[nom_modele]

        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_proba = 1 / (1 + np.exp(-pipeline.decision_function(X_test)))
        else:
            y_proba = pipeline.predict(X_test).astype(float)

        yt = np.asarray(y_test)
        yhat = (y_proba >= threshold).astype(int)
        jitter = np.random.normal(0, 0.04, size=len(yt))

        ax.axhspan(threshold, 1.05, color="orange", alpha=0.08)
        ax.axhspan(-0.05, threshold, color="steelblue", alpha=0.08)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.4,
                   label=f"Seuil {threshold}  →  ≥ {threshold} = départ (1)")

        ok = yhat == yt
        ax.scatter(yt[ok] + jitter[ok], y_proba[ok],
                   c="forestgreen", alpha=0.55, edgecolor="k", linewidth=0.3, label="Bonne prédiction")
        ax.scatter(yt[~ok] + jitter[~ok], y_proba[~ok],
                   c="crimson", alpha=0.7, edgecolor="k", linewidth=0.3, label="Erreur")

        ax.set_title(f"Proba vs réel — {nom_modele}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Vraie classe (0 = reste, 1 = quitte)")
        ax.set_ylabel("Proba prédite de départ")
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.5)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 11. Analyse des corrélations (Pearson & Spearman) et Pairplot
# ----------------------------------------------------------------------
def analyser_correlations_features(X, seuil_pearson=0.85):
    print("--- 1. Analyse de la corrélation de Pearson (Linéaire) ---")
    corr_pearson = X.select_dtypes(include=[np.number]).corr(method='pearson')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_pearson, annot=False, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Matrice de corrélation de Pearson")
    plt.tight_layout()
    plt.show()

    hautes_corrs = []
    columns = corr_pearson.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            val = corr_pearson.iloc[i, j]
            if abs(val) >= seuil_pearson:
                hautes_corrs.append((columns[i], columns[j], val))
                
    if hautes_corrs:
        print(f"⚠️ Paires avec une corrélation linéaire |r| >= {seuil_pearson} :")
        for f1, f2, val in hautes_corrs:
            print(f"   - {f1} <--> {f2} : {val:.3f}")
    else:
        print(f"Aucune paire ne dépasse le seuil de corrélation linéaire de {seuil_pearson}.")

    print("\n--- 2. Analyse de la corrélation de Spearman (Non-linéaire) ---")
    corr_spearman = X.select_dtypes(include=[np.number]).corr(method='spearman')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_spearman, annot=False, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Matrice de corrélation de Spearman (Non-linéaire)")
    plt.tight_layout()
    plt.show()

    print("\n--- 3. Tracé du Pairplot ---")
    cols_a_tracer = X.select_dtypes(include=[np.number]).columns
    if len(cols_a_tracer) > 6:
        cols_a_tracer = cols_a_tracer[:6]

    sns.pairplot(X[cols_a_tracer], diag_kind='kde', corner=True)
    plt.suptitle("Pairplot des features", y=1.02)
    plt.show()

    return corr_pearson, corr_spearman


# ----------------------------------------------------------------------
# Utilitaire interne : extraction robuste des probabilités
# ----------------------------------------------------------------------
def _get_probas(pipeline, X):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        return 1 / (1 + np.exp(-pipeline.decision_function(X)))
    else:
        return pipeline.predict(X).astype(float)


# ----------------------------------------------------------------------
# 12. Courbe ROC
# ----------------------------------------------------------------------
def tracer_courbe_roc(fitted_pipelines, X, y, model_names=None, figsize=(7, 6)):
    if model_names is None:
        model_names = list(fitted_pipelines.keys())

    plt.figure(figsize=figsize)
    for name in model_names:
        pipeline = fitted_pipelines[name]
        y_proba = _get_probas(pipeline, X)
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Aléatoire (AUC=0.5)")
    plt.xlabel("Taux de faux positifs (1 - Spécificité)")
    plt.ylabel("Taux de vrais positifs (Recall)")
    plt.title("Courbes ROC")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 13. Courbe Precision-Recall
# ----------------------------------------------------------------------
def tracer_courbe_precision_recall(fitted_pipelines, X, y, model_names=None, figsize=(7, 6)):
    if model_names is None:
        model_names = list(fitted_pipelines.keys())

    plt.figure(figsize=figsize)
    for name in model_names:
        pipeline = fitted_pipelines[name]
        y_proba = _get_probas(pipeline, X)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (PR-AUC={pr_auc:.3f})")

    baseline = np.mean(np.asarray(y) == 1)
    plt.axhline(baseline, color='k', linestyle='--', alpha=0.5,
                label=f"Baseline (taux de départs = {baseline:.3f})")

    plt.xlabel("Recall (rappel)")
    plt.ylabel("Precision")
    plt.title("Courbes Precision-Recall")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 14. Courbe de calibration
# ----------------------------------------------------------------------
def tracer_courbe_calibration(fitted_pipelines, X, y, model_names=None, n_bins=10, figsize=(7, 6)):
    if model_names is None:
        model_names = list(fitted_pipelines.keys())

    plt.figure(figsize=figsize)
    for name in model_names:
        pipeline = fitted_pipelines[name]
        y_proba = _get_probas(pipeline, X)
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=n_bins, strategy='quantile')
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Calibration parfaite")
    plt.xlabel("Probabilité moyenne prédite")
    plt.ylabel("Fréquence réelle observée")
    plt.title("Courbe de calibration")
    plt.legend(fontsize=8, loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 15. Courbe d'apprentissage
# ----------------------------------------------------------------------
def tracer_courbe_apprentissage(
    pipeline, X, y, cv=5, scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 8), nom_modele=None, n_jobs=-1,
    seuil_overfitting=0.05, seuil_score_faible=0.65,
):
    train_sizes_abs, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=cv, scoring=scoring,
        train_sizes=train_sizes, n_jobs=n_jobs,
    )

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)
    ecart_final = train_mean[-1] - test_mean[-1]

    if train_mean[-1] < seuil_score_faible and test_mean[-1] < seuil_score_faible:
        verdict = "Sous-apprentissage"
        couleur_verdict = "darkorange"
    elif ecart_final > seuil_overfitting:
        verdict = "Surapprentissage"
        couleur_verdict = "crimson"
    else:
        verdict = "Bien équilibré"
        couleur_verdict = "forestgreen"

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='steelblue', label='Score Train')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                      alpha=0.15, color='steelblue')
    plt.plot(train_sizes_abs, test_mean, 'o-', color='darkorange', label='Score CV')
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                      alpha=0.15, color='darkorange')

    plt.vlines(train_sizes_abs[-1], test_mean[-1], train_mean[-1],
               color=couleur_verdict, linestyle=':', linewidth=2)
    
    titre = f"Courbe d'apprentissage — {nom_modele}" if nom_modele else "Courbe d'apprentissage"
    plt.title(titre)
    plt.xlabel("Taille de l'échantillon d'entraînement")
    plt.ylabel(f"Score ({scoring})")
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.figtext(0.5, -0.02, f"Diagnostic : {verdict}", ha='center', fontsize=10,
                color=couleur_verdict, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 16. Analyse du seuil optimal de décision
# ----------------------------------------------------------------------
def analyser_seuil_optimal(pipeline, X_test, y_test, nom_modele="Modèle"):
    y_proba = _get_probas(pipeline, X_test)
    precisions, rappels, seuils = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * rappels) / (precisions + rappels + 1e-10)
    
    seuils_complets = np.append(seuils, 1.0)
    best_idx = np.argmax(f1_scores)
    best_seuil = seuils_complets[best_idx]
    best_f1 = f1_scores[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(seuils_complets, precisions, label="Précision", color="blue")
    plt.plot(seuils_complets, rappels, label="Rappel", color="orange")
    plt.plot(seuils_complets, f1_scores, label="F1-score", color="green", linestyle="--")
    plt.axvline(best_seuil, color='red', linestyle=':', label=f"Seuil optimal (F1={best_f1:.3f}) : {best_seuil:.2f}")
    
    plt.xlabel("Seuil de décision")
    plt.ylabel("Score métrique")
    plt.title(f"Évolution des métriques selon le seuil — {nom_modele}")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return best_seuil


# ----------------------------------------------------------------------
# 17. Tableau récapitulatif du diagnostic d'Overfitting / Underfitting
# ----------------------------------------------------------------------
def diagnostiquer_overfitting_tableau(df_res, col_train='ROC_AUC_train', col_cv='ROC_AUC_cv', seuil_overfit=0.05):
    df_diag = df_res.copy()
    if col_train in df_diag.columns and col_cv in df_diag.columns:
        df_diag['Ecart_Train_CV'] = df_diag[col_train] - df_diag[col_cv]
        
        def attribuer_statut(ecart):
            if ecart > seuil_overfit:
                return "Surapprentissage (Overfitting)"
            elif ecart < -0.02:
                return "Atypique (CV > Train)"
            else:
                return "Bien équilibré"
                
        df_diag['Diagnostic'] = df_diag['Ecart_Train_CV'].apply(attribuer_statut)
        
    display(df_diag.style.background_gradient(cmap="coolwarm", subset=[col_train, col_cv] if col_train in df_diag else None))
    return df_diag


# ----------------------------------------------------------------------
# 18. Interface interactive globale (8 onglets incluant le Beeswarm SHAP)
# ----------------------------------------------------------------------
class _PipelineEvalCV:
    """
    Proxy « CV » : se comporte comme un pipeline entraîné (mêmes méthodes
    predict / predict_proba), mais renvoie des prédictions Out-Of-Fold
    (validation croisée, non biaisées) quand on l'appelle sur exactement
    X_train — le même objet que celui utilisé pour construire ce proxy.

    Pour toute autre matrice (ex: colonnes permutées par permutation_importance,
    ou données de test), il délègue simplement au pipeline réellement entraîné
    sur l'ensemble Train complet. Toute autre attribut (named_steps, coef_...)
    est aussi délégué au pipeline réel.
    """
    def __init__(self, pipeline_fit, X_train, y_train, cv=5, random_state=42):
        self._pipeline_fit = pipeline_fit
        self._X_train_ref = X_train
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        self._y_pred_oof = cross_val_predict(clone(pipeline_fit), X_train, y_train, cv=cv_split)

        self._y_proba_oof = None
        if hasattr(pipeline_fit, "predict_proba"):
            try:
                self._y_proba_oof = cross_val_predict(
                    clone(pipeline_fit), X_train, y_train, cv=cv_split, method="predict_proba"
                )
            except Exception:
                self._y_proba_oof = None

    def predict(self, X):
        if X is self._X_train_ref:
            return self._y_pred_oof
        return self._pipeline_fit.predict(X)

    def predict_proba(self, X):
        if X is self._X_train_ref and self._y_proba_oof is not None:
            return self._y_proba_oof
        return self._pipeline_fit.predict_proba(X)

    def score(self, X, y):
        return self._pipeline_fit.score(X, y)

    def __getattr__(self, name):
        # Délègue tout le reste (named_steps, coef_, get_params, etc.) au pipeline réel
        return getattr(self._pipeline_fit, name)


def interface_etude_modeles_etape():
    import sys
    main_ns = sys.modules['__main__'].__dict__

    try:
        max_models = len(main_ns.get("df_res_widget", [1, 2, 3, 4, 5]))
    except Exception:
        max_models = 5

    def _modeles_entraines():
        fitted = main_ns.get("fitted_pipelines_widget")
        return list(fitted.keys()) if fitted else []

    zone_selecteur = widgets.VBox()
    _get_modeles_coches_ref = {"fn": (lambda: [])}

    def _reconstruire_selecteur(b=None):
        noms = _modeles_entraines()
        if noms:
            conteneur, getter = creer_selecteur_modeles_categorise(noms, valeurs_par_defaut=[])
            zone_selecteur.children = [conteneur]
        else:
            getter = lambda: []
            zone_selecteur.children = [widgets.HTML(
                "<i>⚠️ Aucun modèle entraîné pour l'instant. Lancez d'abord l'entraînement...</i>"
            )]
        _get_modeles_coches_ref["fn"] = getter

    btn_refresh_modeles = widgets.Button(
        description="🔄 Actualiser la liste des modèles entraînés",
        layout=widgets.Layout(width='300px')
    )
    btn_refresh_modeles.on_click(_reconstruire_selecteur)
    _reconstruire_selecteur()

    metrique_dropdown = widgets.Dropdown(
        options=['Recall', 'F1', 'ROC_AUC', 'Precision', 'Accuracy', 'PR_AUC'],
        value='Recall',
        description='Métrique :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )

    top_n_models_slider = widgets.IntSlider(
        value=1, min=1, max=max_models, step=1,
        description='Top N :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    dataset_dropdown = widgets.Dropdown(
        options=[
            ('Jeu de Test', 'test'),
            ('Jeu d\'Entraînement (Train)', 'train'),
            ('Validation croisée (CV, 5 folds)', 'cv'),
        ],
        value='test',
        description='Données :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='260px')
    )

    btn_eval = widgets.Button(
        description="🚀 Charger / Actualiser les analyses",
        button_style='primary',
        layout=widgets.Layout(width='300px', height='40px')
    )

    # Création de 8 onglets au total
    tab_contents = [widgets.Output() for _ in range(8)]
    tab = widgets.Tab()
    tab.children = tab_contents
    tab.set_title(0, "1. Probas & Erreurs")
    tab.set_title(1, "2. Résidus & CV")
    tab.set_title(2, "3. Importances")
    tab.set_title(3, "4. Beeswarm SHAP")      # <-- Nouvel onglet ajouté
    tab.set_title(4, "5. Matrices de confusion")
    tab.set_title(5, "6. Grille True vs Pred")
    tab.set_title(6, "7. Corrélations & Pairplot")
    tab.set_title(7, "8. ROC / PR / Calibration / Seuil")

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
        X_train, y_train = main_ns["X_train"], main_ns["y_train"]
        X_test, y_test = main_ns["X_test"], main_ns["y_test"]

        selected_metric = metrique_dropdown.value.lower()
        current_top_n = top_n_models_slider.value
        mode_donnees = dataset_dropdown.value  # 'test' | 'train' | 'cv'

        if mode_donnees == 'train':
            X_eval, y_eval = X_train, y_train
            dataset_label = "Entraînement (Train)"
        elif mode_donnees == 'cv':
            X_eval, y_eval = X_train, y_train
            dataset_label = "Validation croisée (CV, 5 folds)"
        else:
            X_eval, y_eval = X_test, y_test
            dataset_label = "Test"

        colonnes_possibles = [c for c in current_df_res.columns if selected_metric in c.lower()]
        colonne_selection = colonnes_possibles[0] if colonnes_possibles else current_df_res.columns[0]
            
        top_models_df = current_df_res.sort_values(by=colonne_selection, ascending=False).head(current_top_n)
        top_model_names = top_models_df.index.tolist()

        modeles_coches = [m for m in _get_modeles_coches_ref["fn"]() if m in current_fitted_pipelines]
        noms_vus = set()
        top_model_names_fusionnes = []
        for m in modeles_coches + top_model_names:
            if m not in noms_vus:
                noms_vus.add(m)
                top_model_names_fusionnes.append(m)
        top_model_names = top_model_names_fusionnes

        if not top_model_names:
            with tab.children[0]:
                clear_output()
                print("❌ Aucun modèle à analyser.")
            return

        # current_fitted_pipelines_eval : dictionnaire utilisé pour l'AFFICHAGE
        # (probas, matrices de confusion, ROC, etc.). En mode CV, il s'agit de
        # proxies renvoyant des prédictions Out-Of-Fold. Le SHAP (onglet 4)
        # continue lui d'utiliser current_fitted_pipelines (le vrai modèle final)
        # car il n'existe pas de "SHAP par pli de CV".
        if mode_donnees == 'cv':
            print("⏳ Calcul des prédictions Out-Of-Fold (validation croisée, 5 folds) — cela peut prendre un instant...")
            current_fitted_pipelines_eval = {
                nom: _PipelineEvalCV(current_fitted_pipelines[nom], X_train, y_train, cv=5)
                for nom in top_model_names
            }
        else:
            current_fitted_pipelines_eval = current_fitted_pipelines

        # --- Onglet 0 : Probas & Erreurs ---
        with tab.children[0]:
            clear_output()
            print(f"--- 1. Distribution des probabilités & Erreurs [{dataset_label}] ---")
            distribution_probabilites_par_type(
                current_fitted_pipelines, current_df_res, X_train, y_train, X_test, y_test,
                threshold=0.5, nom_modele=top_model_names
            )
            for nom_modele in top_model_names:
                analyser_distribution_erreurs(current_fitted_pipelines_eval[nom_modele], X_eval, y_eval, nom_modele=f"{nom_modele} ({dataset_label})")

        # --- Onglet 1 : Résidus & CV ---
        with tab.children[1]:
            clear_output()
            print(f"--- 2. Diagnostic surapprentissage / sous-apprentissage ---")
            diagnostiquer_overfitting_tableau(current_df_res)
            print(f"\n--- 3. Train vs CV vs Test ---")
            comparer_train_test(current_df_res, top_model_names=top_model_names)
            print(f"\n--- 4. Analyse des résidus [{dataset_label}] ---")
            for nom_modele in top_model_names:
                analyser_residus(current_fitted_pipelines_eval[nom_modele], X_eval, y_eval, nom_modele=f"{nom_modele} ({dataset_label})")

        # --- Onglet 2 : Importances ---
        with tab.children[2]:
            clear_output()
            print(f"--- 4. Importances par permutation ({dataset_label}) ---")
            importances_pivot = importance_permutation_tous_modeles(
                current_fitted_pipelines_eval, X_eval, y_eval, models=top_model_names, scoring='f1'
            )
            plot_importance_heatmap(importances_pivot, top_n=10)
            plot_importance_grille(importances_pivot, top_n=10, ncols=2)
            
            print(f"\n--- 5. Permutation Importances détaillée avec écart-type ---")
            meilleur_modele = top_model_names[0]
            importance_permutation_avec_erreur(
                current_fitted_pipelines_eval[meilleur_modele], X_eval, y_eval, 
                nom_modele=f"{meilleur_modele} ({dataset_label})", scoring='roc_auc'
            )

        # --- Onglet 3 : Beeswarm SHAP (NOUVEL ONGLET) ---
        with tab.children[3]:
            clear_output()
            if mode_donnees == 'cv':
                print("ℹ️ SHAP n'est pas calculable par pli de CV (chaque pli entraîne un modèle "
                      "différent). Les graphiques ci-dessous expliquent le modèle final, entraîné "
                      "sur l'ensemble Train complet, évalué sur ce même Train.")
            print(f"--- Graphique Beeswarm SHAP (vue globale) [{dataset_label}] ---")
            for nom_modele in top_model_names:
                # SHAP utilise toujours le vrai pipeline entraîné (pas le proxy CV)
                tracer_beeswarm_shap(current_fitted_pipelines[nom_modele], X_eval, nom_modele=f"{nom_modele} ({dataset_label})")

            print(f"\n--- Graphique SHAP local (exemple sur la 1ère observation) [{dataset_label}] ---")
            meilleur_modele_shap = top_model_names[0]
            tracer_shap_local(
                current_fitted_pipelines[meilleur_modele_shap], X_eval, index=0,
                nom_modele=f"{meilleur_modele_shap} ({dataset_label})"
            )
            # Astuce : appelez tracer_shap_local(pipeline, X_eval, index=42, ...)
            # pour expliquer n'importe quelle autre observation précise.

        # --- Onglet 4 : Matrices de confusion ---
        with tab.children[4]:
            clear_output()
            print(f"--- 6. Matrices de confusion [{dataset_label}] ---")
            for nom_modele in top_model_names:
                pipeline = current_fitted_pipelines_eval[nom_modele]
                y_pred = pipeline.predict(X_eval)
                afficher_matrice_confusion(y_eval, y_pred, nom_modele=f"{nom_modele} ({dataset_label})")

        # --- Onglet 5 : Grille True vs Pred ---
        with tab.children[5]:
            clear_output()
            print(f"--- 7. Grille comparative Probabilités vs Réel [{dataset_label}] ---")
            afficher_grille_true_vs_pred(current_fitted_pipelines_eval, top_model_names, X_eval, y_eval, ncols=2)

        # --- Onglet 6 : Corrélations & Pairplot ---
        with tab.children[6]:
            clear_output()
            print(f"--- 8. Analyse des corrélations & Pairplot [{dataset_label}] ---")
            analyser_correlations_features(X_eval, seuil_pearson=0.85)

        # --- Onglet 7 : ROC / PR / Calibration / Seuil ---
        with tab.children[7]:
            clear_output()
            print(f"--- 9. Courbe ROC [{dataset_label}] ---")
            tracer_courbe_roc(current_fitted_pipelines_eval, X_eval, y_eval, model_names=top_model_names)
            print(f"\n--- 10. Courbe Precision-Recall [{dataset_label}] ---")
            tracer_courbe_precision_recall(current_fitted_pipelines_eval, X_eval, y_eval, model_names=top_model_names)
            print(f"\n--- 11. Courbe de calibration [{dataset_label}] ---")
            tracer_courbe_calibration(current_fitted_pipelines_eval, X_eval, y_eval, model_names=top_model_names)
            print(f"\n--- 12. Courbe d'apprentissage ---")
            meilleur_modele = top_model_names[0]
            # La courbe d'apprentissage refait ses propres CV internes : on lui
            # passe toujours le vrai pipeline (pas le proxy), quel que soit le mode.
            tracer_courbe_apprentissage(
                current_fitted_pipelines[meilleur_modele], X_train, y_train,
                scoring='roc_auc', nom_modele=meilleur_modele,
            )
            print(f"\n--- 13. Seuil optimal ---")
            for nom_modele in top_model_names:
                analyser_seuil_optimal(current_fitted_pipelines_eval[nom_modele], X_eval, y_eval, nom_modele=f"{nom_modele} ({dataset_label})")

    btn_eval.on_click(on_eval_clicked)

    return widgets.VBox([
        widgets.HBox([widgets.HTML("<b>Modèles à inclure manuellement (en plus du Top N) :</b>"),
                      btn_refresh_modeles]),
        zone_selecteur,
        widgets.HBox([metrique_dropdown, top_n_models_slider, dataset_dropdown]),
        btn_eval,
        tab
    ])