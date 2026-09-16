"""
shap_analyse.py
===============
Module gérant l'interprétabilité des modèles via SHAP avec des boutons de type
On/Off indépendants (style pilules), un bouton de rafraîchissement, un choix
d'analyse Globale / Locale, et options globales.
"""

import sys
import os
import shap
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Configuration de l'environnement pour éviter les conflits de threads
os.environ["OMP_NUM_THREADS"] = "2"

# Libellés des modes d'analyse (utilisés comme constantes de comparaison)
MODE_GLOBALE = "🌍 Globale"
MODE_LOCALE = "🔍 Locale"
MODE_LES_DEUX = "🌍🔍 Globale + Locale"

# Libellés des sources de modèles (utilisés comme constantes de comparaison)
SOURCE_BASE = "📦 Modèles de base"
SOURCE_OPTIMISE = "⚙️ Modèles optimisés (hyperparamètres)"

# Nom de la variable, dans le namespace du notebook, associée à chaque source
VARIABLE_PAR_SOURCE = {
    SOURCE_BASE: "fitted_pipelines_widget",
    SOURCE_OPTIMISE: "best_pipelines",
}


def interface_analyse_shap():
    """Crée et retourne l'interface widget avec rafraîchissement dynamique,
    boutons On/Off, et choix d'analyse Globale / Locale."""

    output_shap = widgets.Output()
    model_toggles = {}

    # Conteneur interne pour la liste des modèles (mis à jour lors du rafraîchissement)
    models_box = widgets.VBox()

    # Case pour tout cocher/décocher d'un coup
    select_all_cb = widgets.Checkbox(
        value=True,
        description="Activer / Désactiver tout",
        style={'description_width': 'initial'}
    )

    # ========================================================
    # CHOIX DE LA SOURCE : MODELES DE BASE / MODELES OPTIMISES
    # ========================================================
    #
    # - Modèles de base : ceux entraînés avec les hyperparamètres
    #   par défaut (main_ns["fitted_pipelines_widget"], issu de
    #   l'étape d'entraînement / d'étude des modèles).
    # - Modèles optimisés : ceux issus du tuning d'hyperparamètres
    #   (main_ns["best_pipelines"], issu de l'interface de tuning).
    #
    # On peut basculer de l'un à l'autre directement ici, sans
    # relancer le notebook : la liste des modèles proposés et les
    # analyses SHAP se recalculent sur la source sélectionnée.

    source_selector = widgets.ToggleButtons(
        options=[SOURCE_BASE, SOURCE_OPTIMISE],
        value=SOURCE_BASE,
        description="Modèles à expliquer :",
        style={'description_width': 'initial'},
        button_style=''
    )

    def _get_pipelines_dict():
        """Retourne le dict {nom_modele: pipeline} correspondant à la
        source actuellement sélectionnée (base ou optimisée)."""

        main_ns = sys.modules['__main__'].__dict__
        nom_variable = VARIABLE_PAR_SOURCE[source_selector.value]
        return main_ns.get(nom_variable, {}) or {}

    # ========================================================
    # CHOIX DU TYPE D'ANALYSE : GLOBALE / LOCALE / LES DEUX
    # ========================================================
    #
    # - Globale : shap.summary_plot sur l'ensemble de X_test
    #   (importance et effet des variables sur toute la population).
    # - Locale : shap.plots.waterfall sur UNE observation précise de
    #   X_test (pourquoi le modèle a pris CETTE décision pour CE cas).

    analysis_type = widgets.ToggleButtons(
        options=[MODE_GLOBALE, MODE_LOCALE, MODE_LES_DEUX],
        value=MODE_GLOBALE,
        description="Type d'analyse :",
        style={'description_width': 'initial'},
        button_style=''
    )

    instance_selector = widgets.BoundedIntText(
        value=0,
        min=0,
        max=10**6,
        step=1,
        description="N° observation (X_test) :",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='260px')
    )

    instance_hint = widgets.HTML(
        "<i>Index de la ligne dans X_test (0 = première ligne).</i>"
    )

    local_options_box = widgets.HBox(
        [instance_selector, instance_hint],
        layout=widgets.Layout(margin='5px 0px', display='none')
    )

    def on_analysis_type_change(change):
        if change['new'] in (MODE_LOCALE, MODE_LES_DEUX):
            local_options_box.layout.display = 'flex'
        else:
            local_options_box.layout.display = 'none'

    analysis_type.observe(on_analysis_type_change, names='value')

    # Fonction pour charger ou recharger les modèles depuis l'environnement global
    def refresh_models(b=None):
        nonlocal model_toggles
        try:
            model_names = list(_get_pipelines_dict().keys())
        except Exception:
            model_names = []

        # Met a jour la borne max du selecteur d'observation si X_test est disponible
        try:
            main_ns = sys.modules['__main__'].__dict__
            if "X_test" in main_ns:
                nb_lignes = len(main_ns["X_test"])
                instance_selector.max = max(nb_lignes - 1, 0)
        except Exception:
            pass

        model_toggles = {}
        for name in model_names:
            btn = widgets.ToggleButton(
                value=select_all_cb.value,
                description=f"{name}",
                button_style='success' if select_all_cb.value else '',
                layout=widgets.Layout(margin='4px', height='32px')
            )

            def make_observer(b_toggle):
                def on_change(change):
                    if change['new']:
                        b_toggle.button_style = 'success'
                    else:
                        b_toggle.button_style = ''
                return on_change

            btn.observe(make_observer(btn), names='value')
            model_toggles[name] = btn

        toggles_box = widgets.HBox(
            list(model_toggles.values()),
            layout=widgets.Layout(flex_flow='row wrap', margin='5px 0px 10px 0px')
        )

        # Mise à jour du contenu du bloc des modèles
        models_box.children = [
            widgets.HTML("<b>Sélectionne les modèles à inclure (clique pour activer/désactiver) :</b>"),
            select_all_cb,
            toggles_box
        ]

        if b is not None:
            with output_shap:
                clear_output()
                nom_variable = VARIABLE_PAR_SOURCE[source_selector.value]
                if model_names:
                    print(f"🔄 Liste rafraîchie ({source_selector.value}) : {len(model_names)} modèle(s) trouvé(s).")
                else:
                    if source_selector.value == SOURCE_OPTIMISE:
                        print(
                            f"⚠️ Aucun modèle trouvé dans `{nom_variable}`. "
                            "Avez-vous lancé l'optimisation d'hyperparamètres ?"
                        )
                    else:
                        print(
                            f"⚠️ Aucun modèle trouvé dans `{nom_variable}`. "
                            "Avez-vous exécuté l'entraînement ?"
                        )

    # Rafraîchit la liste des modèles à chaque changement de source
    def on_source_change(change):
        refresh_models(b=1)

    source_selector.observe(on_source_change, names='value')

    # Logique de la case "Activer/Désactiver tout"
    def on_select_all_change(change):
        val = change['new']
        for btn in model_toggles.values():
            btn.value = val
    select_all_cb.observe(on_select_all_change, names='value')

    # Chargement initial des modèles
    refresh_models()

    # Bouton de rafraîchissement
    btn_refresh = widgets.Button(
        description="🔄 Rafraîchir les modèles",
        button_style='info',
        layout=widgets.Layout(width='220px', height='40px')
    )
    btn_refresh.on_click(refresh_models)

    # Boutons d'action principaux
    btn_all = widgets.Button(
        description="🔥 Lancer pour TOUS les modèles",
        button_style='danger',
        layout=widgets.Layout(width='260px', height='40px')
    )

    btn_selected = widgets.Button(
        description="🎯 Lancer pour la SÉLECTION",
        button_style='success',
        layout=widgets.Layout(width='260px', height='40px')
    )

    def _analyse_globale(shap_values, X_test_transformed, feature_names, model_name):
        """Trace le summary_plot (importance des variables sur toute la population)."""

        print("   🌍 Analyse globale (summary plot)")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=True)
        plt.close()

    def _analyse_locale(shap_values, pipeline, current_X_test, feature_names, idx):
        """Trace un waterfall plot (explication de la prédiction) pour UNE observation."""

        if idx < 0 or idx >= len(current_X_test):
            print(f"   ⚠️ Index d'observation {idx} hors limites (0 à {len(current_X_test) - 1}).")
            return

        print(f"   🔍 Analyse locale — observation n°{idx} (index réel dans X_test : {current_X_test.index[idx]})")

        try:
            proba = pipeline.predict_proba(current_X_test.iloc[[idx]])[0]
            print(f"      Probabilités prédites : {proba}")
        except Exception:
            pass

        try:
            shap_values_locaux = shap_values[idx]
            shap_values_locaux.feature_names = list(feature_names)
        except Exception as e:
            print(f"   ⚠️ Impossible de préparer l'explication locale : {e}")
            return

        shap.plots.waterfall(shap_values_locaux, show=True)
        plt.close()

    def run_shap_analysis(models_to_run):
        with output_shap:
            clear_output()

            try:
                main_ns = sys.modules['__main__'].__dict__
                current_fitted_pipelines = _get_pipelines_dict()

                if not current_fitted_pipelines or "X_test" not in main_ns:
                    nom_variable = VARIABLE_PAR_SOURCE[source_selector.value]
                    if source_selector.value == SOURCE_OPTIMISE:
                        print(
                            f"❌ Erreur : `{nom_variable}` est vide ou absent. "
                            "Veuillez d'abord lancer l'optimisation d'hyperparamètres !"
                        )
                    else:
                        print(
                            f"❌ Erreur : `{nom_variable}` est vide ou absent. "
                            "Veuillez d'abord exécuter l'entraînement des modèles !"
                        )
                    return

                current_X_test = main_ns["X_test"]
            except Exception as e:
                print(f"❌ Erreur lors de la récupération des variables : {e}")
                return

            if not models_to_run:
                print("⚠️ Aucun modèle sélectionné.")
                return

            print(f"📦 Source utilisée : {source_selector.value}\n")

            mode = analysis_type.value
            faire_globale = mode in (MODE_GLOBALE, MODE_LES_DEUX)
            faire_locale = mode in (MODE_LOCALE, MODE_LES_DEUX)
            idx_local = instance_selector.value

            print(f"🚀 Lancement de l'analyse SHAP ({mode}) pour : {list(models_to_run)}...\n")

            for model_name in models_to_run:
                if model_name not in current_fitted_pipelines:
                    print(f"⚠️ Modèle {model_name} introuvable dans les pipelines.")
                    continue

                print("=" * 60)
                print(f"📊 Modèle : {model_name}")
                print("=" * 60)

                try:
                    pipeline = current_fitted_pipelines[model_name]
                    preprocessor = pipeline.named_steps['preprocessor']
                    classifier = pipeline.named_steps['classifier']

                    X_test_transformed = preprocessor.transform(current_X_test)

                    try:
                        feature_names = preprocessor.get_feature_names_out()
                    except:
                        feature_names = current_X_test.columns

                    if any(tree_type in model_name for tree_type in ["XGBoost", "GradientBoosting", "RandomForest"]):
                        explainer = shap.TreeExplainer(classifier)
                        shap_values = explainer(X_test_transformed)

                        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                            shap_values = shap_values[..., 1]

                    else:
                        if any(lin in model_name for lin in ["LogisticRegression", "LogReg", "ElasticNet", "Lasso", "Ridge"]):
                            # Correction pour LinearExplainer : utilisation d'un masker Indépendant
                            masker = shap.maskers.Independent(X_test_transformed, max_samples=50)
                            explainer = shap.LinearExplainer(classifier, masker)
                        else:
                            background = shap.kmeans(X_test_transformed, 10)
                            explainer = shap.KernelExplainer(classifier.predict_proba, background)

                        shap_values = explainer(X_test_transformed)

                        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                            shap_values = shap_values[..., 1]

                    # Affichage du/des graphique(s) SHAP selon le mode choisi
                    if faire_globale:
                        _analyse_globale(shap_values, X_test_transformed, feature_names, model_name)

                    if faire_locale:
                        _analyse_locale(shap_values, pipeline, current_X_test, feature_names, idx_local)

                except Exception as e:
                    print(f"   ⚠️ Impossible de générer SHAP pour {model_name} (Erreur : {e})")

            print("\n✅ Analyse SHAP terminée !")

    def on_click_all(b):
        try:
            all_models = list(_get_pipelines_dict().keys())
        except:
            all_models = []
        run_shap_analysis(all_models)

    def on_click_selected(b):
        selected_models = [name for name, btn in model_toggles.items() if btn.value]
        run_shap_analysis(selected_models)

    btn_all.on_click(on_click_all)
    btn_selected.on_click(on_click_selected)

    return widgets.VBox([
        source_selector,
        widgets.HTML("<hr style='margin: 10px 0px;'>"),
        btn_refresh,
        widgets.HTML("<hr style='margin: 10px 0px;'>"),
        models_box,
        widgets.HTML("<hr style='margin: 10px 0px;'>"),
        analysis_type,
        local_options_box,
        widgets.HBox([btn_all, btn_selected], layout=widgets.Layout(grid_gap='10px', margin='10px 0px')),
        output_shap
    ])