"""
etude_model.py
===============
Module regroupant les analyses post-entraînement du modèle retenu pour la
prédiction de l'attrition (a_quitte_l_entreprise) :

- analyser_residus()                         : diagnostic des résidus (4 graphiques + tests statistiques)
- distribution_probabilites_par_type() : distribution des probabilités prédites par modèle
- comparer_train_test()                      : graphique en barres horizontales Train (CV) vs Test
- importance_permutation()                 : classement des variables par importance (permutation importance)
- analyser_distribution_erreurs()        : distribution globale des erreurs (histogramme & KDE)
- afficher_grille_true_vs_pred()         : grille intelligente de graphiques True vs Predicted
- importance_permutation_avec_erreur()     : permutation importance avec écart-type et affichage tabulaire
- afficher_matrice_confusion()             : matrice de confusion sous forme de heatmap
- analyser_correlations_features()         : matrices Pearson, Spearman, paires corrélées & pairplot
- interface_etude_modeles_etape()          : interface interactive à 6 onglets avec choix Train/Test

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
from sklearn.metrics import confusion_matrix
from IPython.display import display, clear_output
import ipywidgets as widgets

# ----------------------------------------------------------------------
# 1. Analyse des résidus
# ----------------------------------------------------------------------
def analyser_residus(modele, X_test, y_test, nom_modele=None):
    if nom_modele:
        print(f"Modèle utilisé pour l'analyse des résidus : {nom_modele}")

    common_idx = X_test.index.intersection(y_test.index)
    X_test_aligned = X_test.loc[common_idx]
    y_test_aligned = y_test.loc[common_idx]

    # Gestion sécurisée pour les modèles sans predict_proba (ex: RidgeClassifier)
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
    """
    Affiche la distribution des probabilités prédites en séparant les 4 catégories (TN, FP, FN, TP)
    à la fois pour le jeu d'entraînement (Train) et le jeu de test (Test) pour chaque modèle.
    """
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
# 3. Comparaison Train (CV) vs Test (Filtré sur le Top N)
# ----------------------------------------------------------------------
def comparer_train_test(
    df_res,
    top_model_names=None,
    col_train='ROC_AUC_cv',
    col_test='ROC_AUC_test',
    titre="Comparaison des performances Train (CV) vs Test (Top N)",
    xlabel="Score",
    xlim=(0.4, 1.0),
):
    if top_model_names is not None:
        df_plot = df_res.loc[df_res.index.intersection(top_model_names)]
        df_plot = df_plot.reindex([m for m in top_model_names if m in df_plot.index])
    else:
        df_plot = df_res

    modeles = df_plot.index.tolist()
    y_pos = np.arange(len(modeles))
    hauteur = 0.35

    fig, ax = plt.subplots(figsize=(9, max(len(modeles) * 0.6, 4)))

    ax.barh(y_pos + hauteur / 2, df_plot[col_train], height=hauteur,
            color='skyblue', edgecolor='black', label='Train (CV Mean)')
    ax.barh(y_pos - hauteur / 2, df_plot[col_test], height=hauteur,
            color='green', edgecolor='black', label='Test')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(modeles)
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(titre)
    ax.legend(loc='lower right')

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
# 6. Distribution globale des erreurs
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
# 7. Importance des variables par permutation (jeu de test, avec écart-type)
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
# 8. Matrice de confusion
# ----------------------------------------------------------------------
def afficher_matrice_confusion(y_true, y_pred, nom_modele="Modèle"):
    """Affiche la matrice de confusion sous forme de heatmap seaborn."""
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
# 9. Grille intelligente de probabilités prédites vs Réalité (Classification)
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
        ax.text(0.5, threshold + 0.03, "prédit 1", color="darkorange", ha="center", fontsize=8)
        ax.text(0.5, threshold - 0.07, "prédit 0", color="steelblue", ha="center", fontsize=8)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 10. Analyse des corrélations (Pearson & Spearman) et Pairplot
# ----------------------------------------------------------------------
def analyser_correlations_features(X, seuil_pearson=0.85):
    """
    Calcule et affiche :
    1. La matrice de corrélation de Pearson pour détecter/éliminer les fortes corrélations linéaires.
    2. Une liste de paires de features fortement corrélées (linéairement).
    3. La matrice de corrélation de Spearman pour les relations non-linéaires.
    4. Un pairplot pour visualiser l'intensité des corrélations.
    """
    print("--- 1. Analyse de la corrélation de Pearson (Linéaire) ---")
    corr_pearson = X.select_dtypes(include=[np.number]).corr(method='pearson')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_pearson, annot=False, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Matrice de corrélation de Pearson")
    plt.tight_layout()
    plt.show()

    # Identification des paires fortement corrélées selon le seuil
    hautes_corrs = []
    columns = corr_pearson.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            val = corr_pearson.iloc[i, j]
            if abs(val) >= seuil_pearson:
                hautes_corrs.append((columns[i], columns[j], val))
                
    if hautes_corrs:
        print(f"⚠️ Paires avec une corrélation linéaire |r| >= {seuil_pearson} à envisager d'éliminer :")
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
    print("(Note : Si le nombre de variables est très élevé, le pairplot peut être restreint aux variables les plus importantes)")
    
    cols_a_tracer = X.select_dtypes(include=[np.number]).columns
    if len(cols_a_tracer) > 6:
        print("Information : Plus de 6 variables détectées, affichage du pairplot sur les 6 premières colonnes numériques par souci de lisibilité.")
        cols_a_tracer = cols_a_tracer[:6]

    sns.pairplot(X[cols_a_tracer], diag_kind='kde', corner=True)
    plt.suptitle("Pairplot des features (intensités des relations non-linéaires)", y=1.02)
    plt.show()

    return corr_pearson, corr_spearman


# ----------------------------------------------------------------------
# Widget UI pour l'étape d'analyse approfondie des modèles (6 Onglets)
# ----------------------------------------------------------------------
def interface_etude_modeles_etape():
    """Crée l'interface complète avec 6 onglets et un choix Train/Test pour l'analyse approfondie."""
    
    try:
        import sys
        main_ns = sys.modules['__main__'].__dict__
        max_models = len(main_ns.get("df_res_widget", [1, 2, 3, 4, 5]))
    except Exception:
        max_models = 5 

    metrique_dropdown = widgets.Dropdown(
        options=['Recall', 'F1', 'ROC_AUC', 'Precision', 'Accuracy'],
        value='Recall',
        description='Métrique :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )

    top_n_models_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=max_models,
        step=1,
        description='Top N :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    dataset_dropdown = widgets.Dropdown(
        options=[('Jeu de Test', 'test'), ('Jeu d\'Entraînement (Train)', 'train')],
        value='test',
        description='Données :',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='240px')
    )

    btn_eval = widgets.Button(
        description="🚀 Charger / Actualiser les analyses",
        button_style='primary',
        layout=widgets.Layout(width='300px', height='40px')
    )

    # Création de 6 onglets
    tab_contents = [widgets.Output(), widgets.Output(), widgets.Output(), widgets.Output(), widgets.Output(), widgets.Output()]
    tab = widgets.Tab()
    tab.children = tab_contents
    tab.set_title(0, "1. Probas & Erreurs")
    tab.set_title(1, "2. Résidus & CV")
    tab.set_title(2, "3. Importances")
    tab.set_title(3, "4. Matrices de confusion")
    tab.set_title(4, "5. Grille True vs Pred")
    tab.set_title(5, "6. Corrélations & Pairplot")

    def on_eval_clicked(b):
        import sys
        main_ns = sys.modules['__main__'].__dict__
        
        if "df_res_widget" not in main_ns or "fitted_pipelines_widget" not in main_ns:
            with tab.children[0]:
                clear_output()
                print("❌ Erreur : Veuillez d'abord exécuter l'entraînement des modèles via le panneau de contrôle !")
            return

        current_df_res = main_ns["df_res_widget"]
        current_fitted_pipelines = main_ns["fitted_pipelines_widget"]
        X_train, y_train = main_ns["X_train"], main_ns["y_train"]
        X_test, y_test = main_ns["X_test"], main_ns["y_test"]

        selected_metric = metrique_dropdown.value.lower()
        current_top_n = top_n_models_slider.value
        use_train = (dataset_dropdown.value == 'train')
        
        X_eval, y_eval = (X_train, y_train) if use_train else (X_test, y_test)
        dataset_label = "Entraînement (Train)" if use_train else "Test"

        colonnes_possibles = [c for c in current_df_res.columns if selected_metric in c.lower()]
        colonne_selection = colonnes_possibles[0] if colonnes_possibles else current_df_res.columns[0]
            
        top_models_df = current_df_res.sort_values(by=colonne_selection, ascending=False).head(current_top_n)
        top_model_names = top_models_df.index.tolist()

        # --- Onglet 0 : Probas & Erreurs ---
        with tab.children[0]:
            clear_output()
            print(f"--- 1. Distribution des probabilités & Erreurs [{dataset_label}] ---")
            distribution_probabilites_par_type(
                current_fitted_pipelines, current_df_res, X_train, y_train, X_test, y_test,
                threshold=0.5, nom_modele=top_model_names
            )
            for nom_modele in top_model_names:
                analyser_distribution_erreurs(current_fitted_pipelines[nom_modele], X_eval, y_eval, nom_modele=f"{nom_modele} ({dataset_label})")

        # --- Onglet 1 : Résidus & Comparaison Train/Test ---
        with tab.children[1]:
            clear_output()
            print(f"--- 2. Analyse des résidus [{dataset_label}] ---")
            for nom_modele in top_model_names:
                analyser_residus(current_fitted_pipelines[nom_modele], X_eval, y_eval, nom_modele=f"{nom_modele} ({dataset_label})")
            
            print(f"\n--- 3. Comparaison Train (CV) vs Test (Global) ---")
            comparer_train_test(current_df_res, top_model_names=top_model_names)

        # --- Onglet 2 : Importances ---
        with tab.children[2]:
            clear_output()
            print(f"--- 4. Importances par permutation ({dataset_label}) ---")
            importances_pivot = importance_permutation_tous_modeles(
                current_fitted_pipelines, X_eval, y_eval, models=top_model_names, scoring='f1'
            )
            plot_importance_heatmap(importances_pivot, top_n=10)
            plot_importance_grille(importances_pivot, top_n=10, ncols=2)

        # --- Onglet 3 : Matrices de confusion ---
        with tab.children[3]:
            clear_output()
            print(f"--- 6. Matrices de confusion [{dataset_label}] ---")
            for nom_modele in top_model_names:
                pipeline = current_fitted_pipelines[nom_modele]
                y_pred = pipeline.predict(X_eval)
                afficher_matrice_confusion(y_eval, y_pred, nom_modele=f"{nom_modele} ({dataset_label})")

        # --- Onglet 4 : Grille True vs Pred ---
        with tab.children[4]:
            clear_output()
            print(f"--- 7. Grille comparative Probabilités vs Réel [{dataset_label}] ---")
            afficher_grille_true_vs_pred(current_fitted_pipelines, top_model_names, X_eval, y_eval, ncols=2)

        # --- Onglet 5 : Corrélations de Pearson, Spearman & Pairplot ---
        with tab.children[5]:
            clear_output()
            print(f"--- 8. Analyse des corrélations & Pairplot [{dataset_label}] ---")
            analyser_correlations_features(X_eval, seuil_pearson=0.85)

    btn_eval.on_click(on_eval_clicked)

    return widgets.VBox([
        widgets.HBox([metrique_dropdown, top_n_models_slider, dataset_dropdown]),
        btn_eval,
        tab
    ])