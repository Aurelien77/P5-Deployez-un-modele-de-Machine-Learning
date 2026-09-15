"""
Module du panneau de sélection interactive des variables (ON/OFF),
groupées par type (linéaires, non-linéaires, qualitatives, booléennes,
nouvelles features).
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import ipywidgets as widgets

try:
    from IPython.display import display, clear_output
except Exception:  # pragma: no cover
    def display(obj: Any) -> None:
        print(obj)

    def clear_output(wait: bool = False) -> None:
        pass


GROUP_TITLES = {
    "lineaires": "Quantitatives linéaires",
    "non_lineaires": "Quantitatives non-linéaires",
    "qualitatives": "Qualitatives / catégorielles",
    "booleennes": "Booléennes / binaires",
    "nouvelles_features": "Nouvelles features",
}

FEATURES_METIER_PAR_DEFAUT = [
    "ratio_anciennete_carriere",
    "inertie_poste",
    "stagnation_promotion",
    "revenu_par_annee_experience",
    "revenu_par_niveau",
    "satisfaction_globale",
    "satisfaction_globale_moyenne",
    "satisfaction_min",
    "delta_performance",
    "montant_augmentation_precedente",
    "hs_et_salaire_bas",
    "jeune_faible_anciennete",
    "trajet_long",
    "job_hopper",
]


def _charger_classifier_et_detecter():
    candidats = (
        "exploration.exploration_y",
        "exploration_y",
        "exploration.features",
        "features.exploration_y",
    )
    erreurs = []
    for nom in candidats:
        try:
            module = importlib.import_module(nom)
        except Exception as exc:
            erreurs.append(f"{nom} : {exc}")
            continue
        fn = getattr(module, "classifier_et_detecter", None)
        if callable(fn):
            return fn
        erreurs.append(f"{nom} : pas de classifier_et_detecter")
    raise ImportError(
        "classifier_et_detecter introuvable (attendu dans exploration/exploration_y.py).\n- "
        + "\n- ".join(erreurs)
    )


def _preparer_classification(
    df_clean: pd.DataFrame,
    target_col: str,
    classification: Mapping[str, list[str]] | None,
    features_metier: Sequence[str],
) -> dict[str, list[str]]:
    """Récupère (ou calcule) la classification et isole les features métier à part."""
    if classification is None:
        classifier_et_detecter = _charger_classifier_et_detecter()
        classification = classifier_et_detecter(df_clean, target_col)
    classification = {k: list(v) for k, v in classification.items()}
    features_existantes = [f for f in features_metier if f in df_clean.columns]
    for k in classification:
        classification[k] = [v for v in classification[k] if v not in features_existantes]
    classification["nouvelles_features"] = features_existantes
    return classification


def creer_panneau_selection(
    df_clean: pd.DataFrame,
    target_col: str = "a_quitte_l_entreprise",
    classification: Mapping[str, list[str]] | None = None,
    features_metier: Sequence[str] = FEATURES_METIER_PAR_DEFAUT,
) -> dict[str, list[tuple[str, widgets.ToggleButton]]]:
    """Construit et affiche le panneau de sélection ON/OFF des variables."""
    classification = _preparer_classification(df_clean, target_col, classification, features_metier)
    var_buttons: dict[str, list[tuple[str, widgets.ToggleButton]]] = {}
    ui_blocks: list[widgets.VBox] = []

    for group_key, title in GROUP_TITLES.items():
        if group_key not in classification or not classification[group_key]:
            continue

        grp_chk = widgets.Checkbox(
            value=True,
            description=f"Activer tout : {title} ({len(classification[group_key])})",
            style={"description_width": "initial"},
        )
        var_buttons[group_key] = []
        btn_list: list[widgets.ToggleButton] = []

        for var in classification[group_key]:
            t_btn = widgets.ToggleButton(
                value=True,
                description=f"ON : {var}",
                button_style="success",
                layout=widgets.Layout(width="auto", margin="2px"),
            )

            def make_toggle_observer(btn, variable_name):
                def on_change(change):
                    if change["new"]:
                        btn.description = f"ON : {variable_name}"
                        btn.button_style = "success"
                    else:
                        btn.description = f"OFF : {variable_name}"
                        btn.button_style = ""

                return on_change

            t_btn.observe(make_toggle_observer(t_btn, var), names="value")
            var_buttons[group_key].append((var, t_btn))
            btn_list.append(t_btn)

        vars_box = widgets.HBox(
            btn_list, layout=widgets.Layout(flex_flow="wrap", margin="5px 0 10px 20px")
        )

        def make_group_observer(btn_list_inner):
            def on_change(change):
                if change["name"] == "value":
                    for btn in btn_list_inner:
                        btn.value = change["new"]

            return on_change

        grp_chk.observe(make_group_observer(btn_list), names="value")
        border_style = "solid 2px #3498db" if group_key == "nouvelles_features" else "solid 1px #ddd"
        ui_blocks.append(
            widgets.VBox(
                [grp_chk, vars_box],
                layout=widgets.Layout(border=border_style, padding="10px", margin="5px 0"),
            )
        )

    display(
        widgets.VBox(
            [
                widgets.HTML(
                    "<h3>Panneau de contrôle ON / OFF</h3>"
                    "<p>Choisis les variables, puis lance l’évaluation dans la cellule suivante.</p>"
                ),
                *ui_blocks,
            ]
        )
    )
    return var_buttons


def obtenir_variables_selectionnees(
    var_buttons: Mapping[str, list[tuple[str, widgets.ToggleButton]]],
    par_groupe: bool = False,
):
    """Lit l'état actuel des ToggleButton et retourne les variables cochées ON."""
    if par_groupe:
        return {
            groupe: [var for var, btn in paires if btn.value]
            for groupe, paires in var_buttons.items()
        }
    selection: list[str] = []
    for paires in var_buttons.values():
        selection.extend(var for var, btn in paires if btn.value)
    return selection


def bouton_features_selection(
    get_df_callback,
    on_demarrer_callback=None,
    target_col: str = "a_quitte_l_entreprise",
    classification: Mapping[str, list[str]] | None = None,
    features_metier: Sequence[str] = FEATURES_METIER_PAR_DEFAUT,
    on_panel_ready_callback: Callable[
        [Mapping[str, list[tuple[str, widgets.ToggleButton]]]], None
    ]
    | None = None,
) -> widgets.VBox:
    """Crée un bouton 'Afficher le panneau' et gère le retour de var_buttons."""
    bouton_afficher = widgets.Button(
        description="Afficher le panneau",
        button_style="info",
        icon="sliders",
        layout=widgets.Layout(width="280px", height="40px"),
    )
    sortie = widgets.Output()

    def au_clic_afficher(_):
        with sortie:
            clear_output(wait=True)
            df_entree = get_df_callback()
            var_buttons = creer_panneau_selection(
                df_entree,
                target_col=target_col,
                classification=classification,
                features_metier=features_metier,
            )
            if on_panel_ready_callback:
                on_panel_ready_callback(var_buttons)

            bouton_demarrer = widgets.Button(
                description="Démarrer avec cette sélection",
                button_style="success",
                icon="play",
                layout=widgets.Layout(width="280px", height="40px"),
            )
            sortie_demarrage = widgets.Output()

            def au_clic_demarrer(_):
                with sortie_demarrage:
                    clear_output(wait=True)
                    variables_choisies = obtenir_variables_selectionnees(var_buttons)
                    print(f"=== {len(variables_choisies)} VARIABLES SÉLECTIONNÉES ===")
                    print(variables_choisies)
                    if on_demarrer_callback:
                        on_demarrer_callback(variables_choisies)

            bouton_demarrer.on_click(au_clic_demarrer)
            display(widgets.VBox([bouton_demarrer, sortie_demarrage]))

    bouton_afficher.on_click(au_clic_afficher)
    return widgets.VBox([bouton_afficher, sortie])
