"""
shap_analyse.py
===============
Module gérant l'interprétabilité des modèles via SHAP avec des boutons de type
On/Off indépendants (style pilules), un bouton de rafraîchissement, un choix
d'analyse Globale / Locale, et options globales.
"""

import sys
import os
import traceback
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
# Volontairement courts : ToggleButtons tronque le texte si le libellé est
# trop long pour la largeur de bouton par défaut. Le détail complet est
# donné juste en dessous, dans source_selector_hint.
SOURCE_BASE = "📦 Base"
SOURCE_AVEC_RECHERCHE = "🔍 Optimisé (avec recherche)"
SOURCE_SANS_RECHERCHE = "⚙️ Optimisé (sans recherche)"

# Description humaine de ce que chaque source va chercher, pour les messages d'erreur
DESCRIPTION_PAR_SOURCE = {
    SOURCE_BASE: "`fitted_pipelines_widget` (étape d'entraînement de base)",
    SOURCE_AVEC_RECHERCHE: "les résultats du mode « 🔍 Avec recherche d'hyperparamètres » de l'interface de tuning",
    SOURCE_SANS_RECHERCHE: "les résultats du mode « 📦 Modèle de base (sans recherche) » de l'interface de tuning",
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
    # CHOIX DE LA SOURCE : MODELES DE BASE / OPTIMISES (AVEC OU SANS RECHERCHE)
    # ========================================================
    #
    # - Modèles de base : ceux entraînés avec les hyperparamètres par
    #   défaut, hors de toute interface de tuning
    #   (main_ns["fitted_pipelines_widget"], issu de l'étape
    #   d'entraînement / d'étude des modèles).
    # - Optimisés — avec recherche : ceux issus du mode
    #   "🔍 Avec recherche d'hyperparamètres" de l'interface de tuning
    #   (GridSearchCV a exploré une grille de valeurs).
    # - Optimisés — sans recherche : ceux issus du mode
    #   "📦 Modèle de base (sans recherche)" de l'interface de tuning
    #   (un seul fit, hyperparamètres par défaut, mais repassés par le
    #   même pipeline / pré-traitement que le tuning).
    #
    # Ces deux derniers sont lus dans main_ns["resultats_par_mode"],
    # rempli par hyperparameters_interface.py, ce qui permet de garder
    # les deux résultats disponibles en même temps et de basculer de
    # l'un à l'autre ici sans rien relancer.

    source_selector = widgets.ToggleButtons(
        options=[SOURCE_BASE, SOURCE_AVEC_RECHERCHE, SOURCE_SANS_RECHERCHE],
        value=SOURCE_BASE,
        description="Modèles à expliquer :",
        style={'description_width': 'initial', 'button_width': 'auto'},
        button_style='',
        layout=widgets.Layout(width='auto')
    )

    source_selector_hint = widgets.HTML(
        "<i>📦 Base = <code>fitted_pipelines_widget</code> (entraînement initial, "
        "sans passer par le tuning) &nbsp;•&nbsp; "
        "🔍 Optimisé (avec recherche) = mode « Avec recherche d'hyperparamètres » "
        "de l'interface de tuning &nbsp;•&nbsp; "
        "⚙️ Optimisé (sans recherche) = mode « Modèle de base (sans recherche) » "
        "de l'interface de tuning.</i>"
    )

    def _get_pipelines_dict():
        """Retourne le dict {nom_modele: pipeline} correspondant à la
        source actuellement sélectionnée (base, avec ou sans recherche
        d'hyperparamètres)."""

        main_ns = sys.modules['__main__'].__dict__

        if source_selector.value == SOURCE_BASE:
            return main_ns.get("fitted_pipelines_widget", {}) or {}

        cle_mode = (
            "avec_recherche"
            if source_selector.value == SOURCE_AVEC_RECHERCHE
            else "sans_recherche"
        )

        resultats_par_mode = main_ns.get("resultats_par_mode", {}) or {}

        if cle_mode in resultats_par_mode:
            return resultats_par_mode[cle_mode].get("best_pipelines", {}) or {}

        # Repli de compatibilité : si l'interface de tuning utilisée ne
        # gère pas encore resultats_par_mode (ancienne version), on ne
        # peut retomber sur `best_pipelines` que si l'on est certain que
        # c'est bien le mode demandé qui a été exécuté en dernier.
        if main_ns.get("mode_optimisation_actif") == cle_mode:
            return main_ns.get("best_pipelines", {}) or {}

        return {}

    def _diagnostic_source_vide():
        """Message expliquant pourquoi la source sélectionnée est vide,
        pour distinguer « module de tuning pas à jour » de « mode pas
        encore lancé »."""

        main_ns = sys.modules['__main__'].__dict__

        if source_selector.value == SOURCE_BASE:
            return (
                f"Aucun modèle trouvé dans {DESCRIPTION_PAR_SOURCE[SOURCE_BASE]}. "
                "Avez-vous exécuté l'entraînement ?"
            )

        if "resultats_par_mode" not in main_ns:
            return (
                "`resultats_par_mode` est introuvable dans le notebook : la "
                "version de `hyperparameters_interface.py` chargée ici ne "
                "gère pas encore les modes séparés. Copiez la dernière "
                "version du module, redémarrez le kernel (ou ré-exécutez "
                "l'import), puis relancez la cellule qui affiche "
                "`interface_tuning(...)`."
            )

        return (
            f"Aucun modèle trouvé pour {DESCRIPTION_PAR_SOURCE[source_selector.value]}. "
            "Ce mode n'a pas encore été lancé — allez dans l'interface de "
            "tuning, sélectionnez ce mode, puis cliquez sur "
            "« 🚀 Optimisation automatique » (ou « ⚙️ Optimisation manuelle »)."
        )

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

    fiches_multi = widgets.SelectMultiple(
        options=[(f"n° {i}", i) for i in range(20)],
        value=(0,),
        description="Fiches :",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width="240px", height="160px"),
    )
    instance_hint = widgets.HTML(
        "<i>Ctrl+clic pour plusieurs fiches. n° 0 = 1re ligne de X_test.</i>"
    )
    output_local = widgets.Output()
    _cache_shap = {"items": []}

    local_options_box = widgets.VBox(
        [
            widgets.HTML("<span style='color:#555'>Salariés à expliquer (choix multiple) :</span>"),
            fiches_multi,
            instance_hint,
        ],
        layout=widgets.Layout(margin="5px 0px", display="none"),
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
                deja = tuple(i for i in (fiches_multi.value or ()) if i < nb_lignes)
                fiches_multi.options = [(f"n° {i}", i) for i in range(nb_lignes)]
                fiches_multi.value = deja if deja else ((0,) if nb_lignes else ())
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
                if model_names:
                    print(f"🔄 Liste rafraîchie ({source_selector.value}) : {len(model_names)} modèle(s) trouvé(s).")
                else:
                    print(f"⚠️ {_diagnostic_source_vide()}")

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
            traceback.print_exc()
            return

        # Pour un modèle mono-sortie (régression / score linéaire), base_values peut être
        # un vecteur de longueur 1 au lieu d'un scalaire : shap.plots.waterfall exige un
        # scalaire, d'où un échec silencieux "capté" plus haut. On normalise ici.
        try:
            base = shap_values_locaux.base_values
            if hasattr(base, "__len__") and len(base) == 1:
                shap_values_locaux.base_values = base[0]
        except Exception:
            pass

        shap.plots.waterfall(shap_values_locaux, show=True)
        plt.close()

    def _indices_a_afficher():
        choisis = [int(i) for i in (fiches_multi.value or ())]
        return choisis if choisis else [0]

    def _afficher_locaux_cache(indices=None):
        items = _cache_shap.get("items") or []
        if not items:
            with output_local:
                clear_output()
                print("Lance d'abord l'analyse (Locale ou Globale + Locale).")
            return
        if indices is None:
            indices = _indices_a_afficher()
        n = items[0]["n"]
        indices = [i for i in indices if 0 <= i < n]
        if not indices:
            indices = [0]
        with output_local:
            clear_output(wait=True)
            print(f"Fiches {indices} — change la sélection dans la liste pour d'autres salariés.")
            for idx in indices:
                for item in items:
                    try:
                        _analyse_locale(
                            item["shap_values"],
                            item["pipeline"],
                            item["X"],
                            item["feature_names"],
                            idx,
                        )
                    except Exception as e:
                        print(f"   ⚠️ {item['name']} fiche {idx} : {e}")

    def _on_multi_change(change):
        if change.get("name") == "value" and _cache_shap.get("items"):
            _afficher_locaux_cache(list(change["new"] or ()))

    fiches_multi.observe(_on_multi_change, names="value")

    def run_shap_analysis(models_to_run):
        with output_shap:
            clear_output()

            try:
                main_ns = sys.modules['__main__'].__dict__
                current_fitted_pipelines = _get_pipelines_dict()

                if not current_fitted_pipelines:
                    print(f"❌ Erreur : {_diagnostic_source_vide()}")
                    return

                if "X_test" not in main_ns:
                    print(
                        "❌ Erreur : `X_test` est introuvable dans le notebook. "
                        "Veuillez d'abord exécuter l'étape qui le crée."
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

            print(f"🚀 Lancement de l'analyse SHAP ({mode}) pour : {list(models_to_run)}...\n")
            _cache_shap["items"] = []

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

                    n_samples_bg = X_test_transformed.shape[0]

                    if any(tree_type in model_name for tree_type in ["XGBoost", "GradientBoosting", "RandomForest"]):
                        explainer = shap.TreeExplainer(classifier)
                        shap_values = explainer(X_test_transformed)

                        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                            shap_values = shap_values[..., 1]

                    else:
                        if any(lin in model_name for lin in ["LogisticRegression", "LogReg", "ElasticNet", "Lasso", "Ridge"]):
                            # Masker Indépendant : on utilise tout l'échantillon disponible pour
                            # éviter l'avertissement "Background dataset has N samples but
                            # max_samples=50" (le sous-échantillonnage à 50 n'était qu'une
                            # valeur arbitraire, pas une nécessité).
                            masker = shap.maskers.Independent(X_test_transformed, max_samples=n_samples_bg)
                            explainer = shap.LinearExplainer(classifier, masker)
                            shap_values = explainer(X_test_transformed)
                        else:
                            # Modèles "boîte noire" (SVM, KNN, ...) : on choisit la fonction à
                            # expliquer selon ce que le modèle expose réellement. Un régresseur
                            # (ex : SVR_Linear) n'a pas de predict_proba -> AttributeError
                            # immédiate sinon, ce qui faisait échouer TOUTE l'analyse (globale
                            # ET locale) pour ce modèle.
                            background = shap.kmeans(X_test_transformed, min(10, n_samples_bg))

                            if hasattr(classifier, "predict_proba"):
                                fonction_cible = classifier.predict_proba
                            elif hasattr(classifier, "decision_function"):
                                fonction_cible = classifier.decision_function
                            else:
                                fonction_cible = classifier.predict

                            explainer = shap.KernelExplainer(fonction_cible, background)
                            shap_values = explainer(X_test_transformed)

                        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                            shap_values = shap_values[..., 1]

                except Exception as e:
                    print(f"   ⚠️ Impossible de calculer les valeurs SHAP pour {model_name} (Erreur : {e})")
                    traceback.print_exc()
                    continue

                # Globale et locale sont maintenant dans des try/except séparés : un échec sur
                # l'une des deux n'empêche plus d'afficher (ni de diagnostiquer) l'autre.
                if faire_globale:
                    try:
                        _analyse_globale(shap_values, X_test_transformed, feature_names, model_name)
                    except Exception as e:
                        print(f"   ⚠️ Échec de l'analyse globale pour {model_name} (Erreur : {e})")
                        traceback.print_exc()
                if faire_locale:
                    _cache_shap["items"].append(
                        {
                            "name": model_name,
                            "shap_values": shap_values,
                            "pipeline": pipeline,
                            "X": current_X_test,
                            "feature_names": feature_names,
                            "n": len(current_X_test),
                        }
                    )

            if faire_locale and _cache_shap["items"]:
                n_fiches = _cache_shap["items"][0]["n"]
                print(
                    f"\n🔍 Waterfalls : choisis une ou plusieurs fiches dans la liste "
                    f"({n_fiches} salariés)."
                )
                _afficher_locaux_cache()

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
        source_selector_hint,
        widgets.HTML("<hr style='margin: 10px 0px;'>"),
        btn_refresh,
        widgets.HTML("<hr style='margin: 10px 0px;'>"),
        models_box,
        widgets.HTML("<hr style='margin: 10px 0px;'>"),
        analysis_type,
        local_options_box,
        widgets.HBox([btn_all, btn_selected], layout=widgets.Layout(grid_gap='10px', margin='10px 0px')),
        output_shap,
        widgets.HTML("<b>Fiches une par une (local)</b>"),
        output_local,
    ])