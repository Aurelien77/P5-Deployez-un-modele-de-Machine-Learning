"""
Module de séparation Train / Test, à partir des variables activées
dans le panneau de sélection (features_selection.creer_panneau_selection).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import pandas as pd
import ipywidgets as widgets
from sklearn.model_selection import train_test_split

try:
    from IPython.display import display, clear_output
except Exception:  # pragma: no cover
    def display(obj: Any) -> None:
        print(obj)

    def clear_output(wait: bool = False) -> None:
        pass


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def cible_numerique(s: pd.Series) -> pd.Series:
    """Convertit la cible en 0/1 si elle est encore textuelle."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return s.astype(str).str.lower().str.strip().isin(["oui", "yes", "true", "1", "1.0"]).astype(int)


def features_actives(
    df_clean: pd.DataFrame,
    var_buttons: Mapping[str, list[tuple[str, widgets.ToggleButton]]],
    target_col: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """
    Lit l'état des ToggleButton du panneau de sélection (features_selection)
    et retourne (lin, non_lin, qual, bools, ratios, selected).
    """
    def prises(groupe: str) -> list[str]:
        return [var for var, btn in var_buttons.get(groupe, []) if btn.value]

    lin = prises("lineaires")
    non_lin = prises("non_lineaires")
    qual = prises("qualitatives")
    bools = prises("booleennes")
    ratios = prises("ratios")
    nouv = prises("nouvelles_features")

    nouv_bin, nouv_num = [], []
    for c in nouv:
        if c in df_clean.columns and df_clean[c].nunique(dropna=True) <= 2:
            nouv_bin.append(c)
        else:
            nouv_num.append(c)

    lin = list(dict.fromkeys(lin + nouv_num))
    bools = list(dict.fromkeys(bools + nouv_bin))

    # Fusion directe de toutes les features actives
    selected = list(dict.fromkeys(lin + non_lin + qual + bools + ratios + nouv))
    selected = [c for c in selected if c in df_clean.columns and c != target_col]

    return lin, non_lin, qual, bools, ratios, selected


# ---------------------------------------------------------------------------
# CONSTRUCTION DU PANNEAU (réglages + bouton de split)
# ---------------------------------------------------------------------------

def _construire_panneau(
    get_df_callback: Callable[[], pd.DataFrame],
    get_var_buttons_callback: Callable[[], Mapping[str, list[tuple[str, widgets.ToggleButton]]]],
    on_split_callback: Callable[[dict], None] | None,
    target_col: str,
) -> widgets.VBox:
    slider_test = widgets.FloatSlider(
        value=0.20, min=0.10, max=0.40, step=0.05,
        description="Taille test :",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
    )
    chk_stratify = widgets.Checkbox(
        value=True,
        description="Stratifier sur la cible (recommandé)",
        style={"description_width": "initial"},
    )
    slider_seed = widgets.IntSlider(
        value=42, min=0, max=100, step=1,
        description="Graine (random_state) :",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
    )
    chk_detail = widgets.Checkbox(
        value=True,
        description="Afficher le détail des effectifs",
        style={"description_width": "initial"},
    )
    btn_split = widgets.Button(
        description="→→→→ 2 → Lancer le split train / test  ←←←←",
        button_style="success",
        icon="random",
        layout=widgets.Layout(width="300px", height="40px"),
    )
    out_split = widgets.Output()

    def on_split(_):
        with out_split:
            clear_output()

            df_clean = get_df_callback()
            var_buttons = get_var_buttons_callback()

            if not var_buttons:
                print("Panneau de sélection introuvable : lance d'abord features_selection.")
                return

            lin_actives, non_lin_actives, qual_actives, bool_actives, ratios_actives, features_selectionnees = (
                features_actives(df_clean, var_buttons, target_col)
            )

            if not features_selectionnees:
                print("Aucune variable sélectionnée dans le panneau ON/OFF.")
                return

            X = df_clean[features_selectionnees].copy()
            y = cible_numerique(df_clean[target_col])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=slider_test.value,
                random_state=slider_seed.value,
                stratify=y if chk_stratify.value else None,
            )

            print(f"Test : {slider_test.value:.0%} | Train : {1 - slider_test.value:.0%}")
            print(f"Stratify : {'oui' if chk_stratify.value else 'non'} | seed : {slider_seed.value}")
            print(f"Variables : {len(features_selectionnees)}")
            print(f"Train {X_train.shape} | Test {X_test.shape}")
            print(f"Départs train : {y_train.mean():.1%} | test : {y_test.mean():.1%}")

            if chk_detail.value:
                print("\nEffectifs cible")
                print("Train :", y_train.value_counts().to_dict())
                print("Test  :", y_test.value_counts().to_dict())

            print("\nÉtape 1 OK → cellule préprocesseur.")

            if on_split_callback:
                on_split_callback({
                    "X": X, "y": y,
                    "X_train": X_train, "X_test": X_test,
                    "y_train": y_train, "y_test": y_test,
                    "features_selectionnees": features_selectionnees,
                    "lin_actives": lin_actives,
                    "non_lin_actives": non_lin_actives,
                    "qual_actives": qual_actives,
                    "bool_actives": bool_actives,
                    "ratios_actives": ratios_actives,
                })

    btn_split.on_click(on_split)

    return widgets.VBox([
        widgets.HTML("<b>Réglages du split</b>"),
        slider_test, chk_stratify, slider_seed, chk_detail,
        btn_split, out_split,
    ])


# ---------------------------------------------------------------------------
# BOUTON UNIQUE — démarre le module
# ---------------------------------------------------------------------------

def bouton_train_test(
    get_df_callback: Callable[[], pd.DataFrame],
    get_var_buttons_callback: Callable[[], Mapping[str, list[tuple[str, widgets.ToggleButton]]]],
    on_split_callback: Callable[[dict], None] | None = None,
    target_col: str = "a_quitte_l_entreprise",
) -> widgets.VBox:
    """
    Bouton unique "1. Déployer la configuration" : au clic, affiche le panneau de réglages
    (taille test, stratify, seed, détail) et son bouton de split.
    """
    bouton_demarrer = widgets.Button(
        description="1 → Déployer",
        button_style="info",
        icon="play",
        layout=widgets.Layout(width="280px", height="40px"),
    )
    sortie = widgets.Output()

    def au_clic(_):
        with sortie:
            clear_output(wait=True)
            display(_construire_panneau(get_df_callback, get_var_buttons_callback, on_split_callback, target_col))

    bouton_demarrer.on_click(au_clic)

    return widgets.VBox([bouton_demarrer, sortie])