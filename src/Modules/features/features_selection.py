"""
Module du panneau de sélection interactive des variables (ON/OFF),
groupées par type (linéaires, non-linéaires, qualitatives, booléennes,
nouvelles features).

Les unités (€, %, ans, 0/1, texte, niveau) viennent de new_features.py
(source unique). Le panneau n'en redéfinit pas une deuxième copie.
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

FEATURES_METIER_PAR_DEFAUT: list[str] | None = None  # ne plus coder en dur : voir _charger_features_metier()

_SUFFIXE_VALUE = "_value"
_CANDIDATS_NEW_FEATURES = (
    "new_features",
    "exploration.new_features",
    "features.new_features",
)


# ---------------------------------------------------------------------------
# Chargement dynamique (même contrat qu'avant)
# ---------------------------------------------------------------------------

def _importer_module_avec_attr(candidats: Sequence[str], attr: str, erreur_intro: str):
    """Essaie une liste de chemins d'import et retourne l'attribut demandé."""
    erreurs: list[str] = []
    for nom in candidats:
        try:
            module = importlib.import_module(nom)
        except Exception as exc:
            erreurs.append(f"{nom} : {exc}")
            continue
        objet = getattr(module, attr, None)
        if objet is not None:
            return objet
        erreurs.append(f"{nom} : pas de {attr}")
    raise ImportError(erreur_intro + "\n- " + "\n- ".join(erreurs))


def _charger_module_new_features():
    erreurs: list[str] = []
    for nom in _CANDIDATS_NEW_FEATURES:
        try:
            return importlib.import_module(nom)
        except Exception as exc:
            erreurs.append(f"{nom} : {exc}")
    raise ImportError(
        "new_features introuvable.\n- " + "\n- ".join(erreurs)
    )


def _charger_explications_features() -> dict[str, str]:
    """Récupère EXPLICATIONS_FEATURES depuis new_features.py (plusieurs chemins)."""
    explications = _importer_module_avec_attr(
        _CANDIDATS_NEW_FEATURES,
        "EXPLICATIONS_FEATURES",
        "EXPLICATIONS_FEATURES introuvable (attendu dans new_features.py).",
    )
    if not isinstance(explications, dict):
        raise ImportError("EXPLICATIONS_FEATURES n'est pas un dict.")
    return dict(explications)


def _charger_features_metier() -> list[str]:
    """Récupère la liste à jour des features métier depuis EXPLICATIONS_FEATURES.

    Lit directement le dictionnaire de new_features.py plutôt qu'une liste codée
    en dur ici : dès qu'une feature est ajoutée / retirée / renommée là-bas,
    ce panneau la reflète automatiquement, sans double maintenance ni risque
    d'oubli (noms obsolètes, features manquantes...).
    """
    return list(_charger_explications_features().keys())


def _charger_inferer_unite():
    """Fonction d'unité définie dans new_features.py — pas de second dictionnaire ici."""
    fn = _importer_module_avec_attr(
        _CANDIDATS_NEW_FEATURES,
        "inferer_unite",
        "inferer_unite introuvable (attendu dans new_features.py).",
    )
    if not callable(fn):
        raise ImportError("inferer_unite n'est pas appelable.")
    return fn


def _charger_classifier_et_detecter():
    candidats = (
        "exploration.exploration_y",
        "exploration_y",
        "exploration.features",
        "features.exploration_y",
    )
    fn = _importer_module_avec_attr(
        candidats,
        "classifier_et_detecter",
        "classifier_et_detecter introuvable (attendu dans exploration/exploration_y.py).",
    )
    if not callable(fn):
        raise ImportError("classifier_et_detecter n'est pas appelable.")
    return fn


# ---------------------------------------------------------------------------
# Noms de colonnes : base vs suffixe _value
# ---------------------------------------------------------------------------

def _nom_sans_value(nom: str) -> str:
    if nom.endswith(_SUFFIXE_VALUE):
        return nom[: -len(_SUFFIXE_VALUE)]
    return nom


def _variantes_nom(nom: str) -> list[str]:
    """Nom canonique + variante *_value (ordre : tel quel, puis l'autre)."""
    base = _nom_sans_value(nom)
    variantes = [nom]
    if nom.endswith(_SUFFIXE_VALUE):
        if base not in variantes:
            variantes.append(base)
    else:
        variantes.append(f"{nom}{_SUFFIXE_VALUE}")
    if base not in variantes:
        variantes.append(base)
    return variantes


def _colonne_reelle(df: pd.DataFrame, nom: str) -> str | None:
    """Retourne le nom réellement présent dans df (gère le suffixe _value)."""
    for candidat in _variantes_nom(nom):
        if candidat in df.columns:
            return candidat
    return None


def _resoudre_features_dans_df(df: pd.DataFrame, features: Sequence[str]) -> list[str]:
    """Mappe une liste de noms métier vers les colonnes réellement présentes."""
    resolues: list[str] = []
    vus: set[str] = set()
    for nom in features:
        reel = _colonne_reelle(df, nom)
        if reel is not None and reel not in vus:
            resolues.append(reel)
            vus.add(reel)
    return resolues


# ---------------------------------------------------------------------------
# Unités et libellés d'affichage (délègue à new_features)
# ---------------------------------------------------------------------------

def _unite_colonne(nom: str, unites_df: Mapping[str, str] | None = None) -> str:
    if unites_df:
        if nom in unites_df:
            return unites_df[nom]
        base = _nom_sans_value(nom)
        if base in unites_df:
            return unites_df[base]
    try:
        inferer_unite = _charger_inferer_unite()
        return inferer_unite(nom) or ""
    except Exception:
        return ""


def _libelle_bouton(allume: bool, nom: str, unites_df: Mapping[str, str] | None = None) -> str:
    prefixe = "ON" if allume else "OFF"
    unite = _unite_colonne(nom, unites_df)
    suffixe = f"  [{unite}]" if unite else ""
    return f"{prefixe} : {nom}{suffixe}"


def _tooltip_variable(
    nom: str,
    explications: Mapping[str, str] | None,
    unites_df: Mapping[str, str] | None = None,
) -> str:
    base = _nom_sans_value(nom)
    expl = ""
    if explications:
        expl = explications.get(nom) or explications.get(base) or ""
    unite = _unite_colonne(nom, unites_df)
    parties = [nom]
    if nom.endswith(_SUFFIXE_VALUE):
        parties.append("colonne numérique (_value) extraite du texte source")
    if unite:
        parties.append(f"unité : {unite}")
    if expl:
        parties.append(expl)
    return " — ".join(parties)


def _unites_depuis_df(df: pd.DataFrame) -> dict[str, str]:
    try:
        module = _charger_module_new_features()
        fn = getattr(module, "unites_du_dataframe", None)
        if callable(fn):
            return dict(fn(df))
    except Exception:
        pass
    stockees = dict(df.attrs.get("unites_colonnes", {}) or {})
    try:
        inferer_unite = _charger_inferer_unite()
    except Exception:
        return stockees
    for col in df.columns:
        if col not in stockees:
            unite = inferer_unite(col)
            if unite:
                stockees[col] = unite
    return stockees


# ---------------------------------------------------------------------------
# Classification des groupes
# ---------------------------------------------------------------------------

def _preparer_classification(
    df_clean: pd.DataFrame,
    target_col: str,
    classification: Mapping[str, list[str]] | None,
    features_metier: Sequence[str] | None,
) -> dict[str, list[str]]:
    """Récupère (ou calcule) la classification et isole les features métier à part.

    Les features métier sont résolues vers le nom réel dans df_clean
    (avec ou sans suffixe _value) pour que le bouton correspond toujours
    à une colonne existante.
    """
    if features_metier is None:
        features_metier = _charger_features_metier()
    if classification is None:
        classifier_et_detecter = _charger_classifier_et_detecter()
        classification = classifier_et_detecter(df_clean, target_col)
    classification = {k: list(v) for k, v in classification.items()}

    features_existantes = _resoudre_features_dans_df(df_clean, features_metier)
    aliases_metier = set()
    for f in features_existantes:
        aliases_metier.update(_variantes_nom(f))
        aliases_metier.add(_nom_sans_value(f))

    for k in classification:
        classification[k] = [v for v in classification[k] if v not in aliases_metier]
    classification["nouvelles_features"] = features_existantes
    return classification


# ---------------------------------------------------------------------------
# Panneau ON / OFF
# ---------------------------------------------------------------------------

def creer_panneau_selection(
    df_clean: pd.DataFrame,
    target_col: str = "a_quitte_l_entreprise",
    classification: Mapping[str, list[str]] | None = None,
    features_metier: Sequence[str] | None = None,
) -> dict[str, list[tuple[str, widgets.ToggleButton]]]:
    """Construit et affiche le panneau de sélection ON/OFF des variables.

    Chaque bouton affiche le nom de colonne + un suffixe d'unité
    (ex. « ON : performance  [niveau] », « ON : departement  [texte] »).
    Le tooltip reprend l'explication métier quand elle existe.
    """
    classification = _preparer_classification(df_clean, target_col, classification, features_metier)
    try:
        explications = _charger_explications_features()
    except Exception:
        explications = {}
    unites_df = _unites_depuis_df(df_clean)

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
                description=_libelle_bouton(True, var, unites_df),
                button_style="success",
                tooltip=_tooltip_variable(var, explications, unites_df),
                layout=widgets.Layout(width="auto", margin="2px"),
            )

            def make_toggle_observer(btn, variable_name):
                def on_change(change):
                    if change["new"]:
                        btn.description = _libelle_bouton(True, variable_name, unites_df)
                        btn.button_style = "success"
                    else:
                        btn.description = _libelle_bouton(False, variable_name, unites_df)
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
                    "<p style='color:#555;font-size:90%'>"
                    "Suffixe = unité : "
                    "<code>€</code> montant, "
                    "<code>%</code> ratio / taux, "
                    "<code>ans</code> durée, "
                    "<code>0/1</code> binaire, "
                    "<code>texte</code> catégorie, "
                    "<code>niveau</code> échelle (note, fréquence, participation, perf). "
                    "Survole un bouton pour l’explication métier."
                    "</p>"
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
    """Lit l'état actuel des ToggleButton et retourne les variables cochées ON.

    Les noms renvoyés sont ceux réellement présents dans le DataFrame
    (donc avec le suffixe _value s'il est sur la colonne utilisée).
    """
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
    features_metier: Sequence[str] | None = None,
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
