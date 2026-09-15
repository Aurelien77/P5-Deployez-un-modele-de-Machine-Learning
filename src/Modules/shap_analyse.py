"""
shap_analyse.py
===============
Module gérant l'interprétabilité des modèles via SHAP avec des boutons de type 
On/Off indépendants (style pilules), un bouton de rafraîchissement et options globales.
"""

import sys
import os
import shap
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Configuration de l'environnement pour éviter les conflits de threads
os.environ["OMP_NUM_THREADS"] = "2"

def interface_analyse_shap():
    """Crée et retourne l'interface widget avec rafraîchissement dynamique et boutons On/Off."""
    
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

    # Fonction pour charger ou recharger les modèles depuis l'environnement global
    def refresh_models(b=None):
        nonlocal model_toggles
        try:
            main_ns = sys.modules['__main__'].__dict__
            model_names = list(main_ns["fitted_pipelines_widget"].keys()) if "fitted_pipelines_widget" in main_ns else []
        except Exception:
            model_names = []

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
                    print(f"🔄 Liste des modèles rafraîchie avec succès ({len(model_names)} trouvés).")
                else:
                    print("⚠️ Aucun modèle trouvé dans `fitted_pipelines_widget`. Avez-vous exécuté l'entraînement ?")

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

    def run_shap_analysis(models_to_run):
        with output_shap:
            clear_output()
            
            try:
                main_ns = sys.modules['__main__'].__dict__
                if "fitted_pipelines_widget" not in main_ns or "X_test" not in main_ns:
                    print("❌ Erreur : Veuillez d'abord exécuter l'entraînement des modèles !")
                    return
                
                current_fitted_pipelines = main_ns["fitted_pipelines_widget"]
                current_X_test = main_ns["X_test"]
            except Exception as e:
                print(f"❌ Erreur lors de la récupération des variables : {e}")
                return

            if not models_to_run:
                print("⚠️ Aucun modèle sélectionné.")
                return

            print(f"🚀 Lancement de l'analyse SHAP pour : {list(models_to_run)}...\n")

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
                
                    # Affichage du graphique SHAP
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=True)
                    plt.close()
                    
                except Exception as e:
                    print(f"   ⚠️ Impossible de générer SHAP pour {model_name} (Erreur : {e})")

            print("\n✅ Analyse SHAP terminée !")

    def on_click_all(b):
        try:
            main_ns = sys.modules['__main__'].__dict__
            all_models = list(main_ns["fitted_pipelines_widget"].keys())
        except:
            all_models = []
        run_shap_analysis(all_models)

    def on_click_selected(b):
        selected_models = [name for name, btn in model_toggles.items() if btn.value]
        run_shap_analysis(selected_models)

    btn_all.on_click(on_click_all)
    btn_selected.on_click(on_click_selected)

    return widgets.VBox([
        btn_refresh,
        widgets.HTML("<hr style='margin: 10px 0px;'>"),
        models_box,
        widgets.HBox([btn_all, btn_selected], layout=widgets.Layout(grid_gap='10px', margin='10px 0px')),
        output_shap
    ])