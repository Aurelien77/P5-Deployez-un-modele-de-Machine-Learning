"""
Module de création de features métier RH.

Les unités (€, %, ans, 0/1, texte, niveau) sont définies ici une seule fois :
elles sont attachées au DataFrame à la création, affichées dans les logs,
et relues par le panneau de sélection.
"""
from __future__ import annotations

import re

import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output

EXPLICATIONS_FEATURES = {
    "montant_augmentation_precedente": "Montant en € de la dernière augmentation : impact réel sur la paie, pas seulement le %.",
    "inertie_poste": "Années poste / années entreprise → stagnation dans le rôle.",
    "revenu_par_niveau": "Salaire / niveau hiérarchique → écart au niveau de poste.",
    "satisfaction_globale": "Moyenne des 4 satisfactions → climat social.",
    "satisfaction_min": "Minimum des 4 satisfactions → un axe très bas suffit.",
    "satisfaction_max": "Maximum des 4 satisfactions → le meilleur axe, même si le reste va mal.",
    "performance": "Note actuelle − note précédente → hausse ou baisse d'éval.",
    "hs_et_salaire_bas": "HS et salaire < médiane → surcharge + bas salaire.",
    "jeune_faible_anciennete": "Âge ≤ 30 et ancienneté ≤ 2 ans → départ précoce.",
    "serial-worker": "≥ 5 employeurs précédents → habitude de partir.",
    "poste_x_niveau": "Croisement poste × niveau hiérarchique : le taux de départ n'est pas le même pour un commercial N1 et un commercial N4.",
    "formations_par_an": "Nombre de formations / années dans l'entreprise → rythme de formation, pas le stock accumulé avec l'ancienneté.",
    "part_sans_promotion": "Années depuis la dernière promotion / années dans l'entreprise → proche de 1 = pas promu depuis l'arrivée.",
}

# Vocabulaire unique d'unités (affichage + métadonnées, les colonnes ne sont pas renommées).
#   €      montants
#   %      ratios et taux
#   ans    durées / âge
#   0/1    binaires (genre, flags)
#   texte  catégories libres (département, poste…)
#   niveau notes, échelles, fréquences, participations, perf
UNITES_PAR_NOM: dict[str, str] = {
    # features métier
    "montant_augmentation_precedente": "€",
    "inertie_poste": "%",
    "revenu_par_niveau": "€",
    "satisfaction_globale": "niveau",
    "satisfaction_min": "niveau",
    "satisfaction_max": "niveau",
    "performance": "niveau",
    "hs_et_salaire_bas": "0/1",
    "jeune_faible_anciennete": "0/1",
    "serial-worker": "0/1",
    "poste_x_niveau": "texte",
    "formations_par_an": "%",
    "part_sans_promotion": "%",
    # salaire / taux
    "revenu_mensuel": "€",
    "augementation_salaire_precedente": "%",
    "augmentation_salaire_precedente": "%",
    # satisfactions & évaluations
    "satisfaction_employee_environnement": "niveau",
    "satisfaction_employee_nature_travail": "niveau",
    "satisfaction_employee_equipe": "niveau",
    "satisfaction_employee_equilibre_pro_perso": "niveau",
    "note_evaluation_actuelle": "niveau",
    "note_evaluation_precedente": "niveau",
    "niveau_hierarchique_poste": "niveau",
    "niveau_education": "niveau",
    "education": "niveau",
    # participations / formations / expériences (échelles)
    "nombre_participation_formation": "niveau",
    "nombre_participation_formations": "niveau",
    "nombre_formations": "niveau",
    "nombre_formations_suivies": "niveau",
    "nb_formations": "niveau",
    "nombre_experiences_precedentes": "niveau",
    "nombre_entreprises_precedentes": "niveau",
    # fréquence de déplacement = échelle ordinale
    "frequence_deplacement": "niveau",
    "frequence_de_deplacement": "niveau",
    "frequence_des_deplacements": "niveau",
    "deplacement_professionnel": "niveau",
    "distance_domicile_travail": "niveau",
    # durées
    "age": "ans",
    "annee_experience_totale": "ans",
    "annees_experience_totale": "ans",
    "annees_dans_le_poste_actuel": "ans",
    "annees_dans_l_entreprise": "ans",
    "annees_depuis_la_derniere_promotion": "ans",
    "annees_sous_responsable_actuel": "ans",
    "annees_avec_le_responsable_actuel": "ans",
    "annees_avec_responsable_actuel": "ans",
    "annees_avec_le_manager_actuel": "ans",
    "annees_avec_manager_actuel": "ans",
    # binaires
    "genre": "0/1",
    "sexe": "0/1",
    "heure_supplementaires": "0/1",
    "a_quitte_l_entreprise": "0/1",
    # texte
    "departement": "texte",
    "department": "texte",
    "poste": "texte",
    "intitule_poste": "texte",
    "domaine_etude": "texte",
    "filiere_etudes": "texte",
    "statut_marital": "texte",
    "situation_familiale": "texte",
}

_SUFFIXE_VALUE = "_value"
_ATTR_UNITES = "unites_colonnes"


def _nom_sans_value(nom: str) -> str:
    if nom.endswith(_SUFFIXE_VALUE):
        return nom[: -len(_SUFFIXE_VALUE)]
    return nom


def inferer_unite(nom: str) -> str:
    """Retourne l'unité d'affichage d'une colonne : €, %, ans, 0/1, texte, niveau."""
    base = _nom_sans_value(nom)
    if base in UNITES_PAR_NOM:
        return UNITES_PAR_NOM[base]
    if nom in UNITES_PAR_NOM:
        return UNITES_PAR_NOM[nom]

    n = base.lower()

    if n == "performance" or n.startswith("performance"):
        return "niveau"
    if re.search(r"(satisfaction|note_evaluation|evaluation|niveau_hierarchique|niveau_education)", n):
        return "niveau"
    if re.search(r"(frequence.*deplac|deplacement_professionnel|distance_domicile)", n):
        return "niveau"
    if re.search(r"(participation|formation|nb_formation)", n):
        return "niveau"
    if re.search(r"(nombre_experience|nombre_entreprise)", n):
        return "niveau"

    if re.search(r"(revenu|salaire|montant|prime|cout|coût|remuneration|rémunération)", n):
        if re.search(r"(pct|pourcent|taux|augmentation)", n):
            return "%"
        return "€"
    if re.search(r"(pct|pourcent|taux_augmentation|augmentation_salaire|inertie)", n):
        return "%"

    if re.search(
        r"(^age$|anciennete|ancienneté|^annee_|^annees_|^années_|experience_totale|expérience_totale|sous_responsable|avec_le_responsable|avec_manager)",
        n,
    ):
        return "ans"

    if re.search(
        r"(genre|^sexe$|heure_supplementaire|heures_sup|hs_et_|serial.?worker|jeune_faible|a_quitte)",
        n,
    ):
        return "0/1"

    if re.search(
        r"(departement|department|poste|intitule|domaine_etude|filiere|statut_marital|situation_familiale)",
        n,
    ):
        return "texte"

    return ""


def libelle_avec_unite(nom: str) -> str:
    """Nom + suffixe d'unité, ex. « performance  [niveau] »."""
    unite = inferer_unite(nom)
    return f"{nom}  [{unite}]" if unite else nom


def _enregistrer_unites(df: pd.DataFrame, colonnes: list[str] | None = None) -> None:
    """Attache le dictionnaire d'unités sur df.attrs (lu ensuite par le panneau)."""
    cibles = list(colonnes) if colonnes is not None else list(df.columns)
    deja = dict(df.attrs.get(_ATTR_UNITES, {}) or {})
    for col in cibles:
        unite = inferer_unite(col)
        if unite:
            deja[col] = unite
    df.attrs[_ATTR_UNITES] = deja


def unites_du_dataframe(df: pd.DataFrame) -> dict[str, str]:
    """Unités déjà stockées + inférence pour les colonnes encore absentes du dict."""
    stockees = dict(df.attrs.get(_ATTR_UNITES, {}) or {})
    for col in df.columns:
        if col not in stockees:
            unite = inferer_unite(col)
            if unite:
                stockees[col] = unite
    return stockees


def creer_features(df_clean: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Ajoute les features métier RH à df_clean (sans écraser celles déjà présentes)."""
    df_clean = df_clean.copy()
    nouvelles_features: list[str] = []
    features_deja_presentes: list[str] = []

    def ajouter_feature(nom: str, serie: pd.Series) -> bool:
        if nom in df_clean.columns:
            features_deja_presentes.append(nom)
            return False
        df_clean[nom] = serie
        nouvelles_features.append(nom)
        return True

    col_aug = "augementation_salaire_precedente"
    if col_aug not in df_clean.columns and "augmentation_salaire_precedente" in df_clean.columns:
        col_aug = "augmentation_salaire_precedente"

    cols_satis = [
        "satisfaction_employee_environnement",
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso",
    ]
    cols_satis_presents = [c for c in cols_satis if c in df_clean.columns]
    col_hs = "heure_supplementaires" if "heure_supplementaires" in df_clean.columns else None

    if col_aug in df_clean.columns and "revenu_mensuel" in df_clean.columns:
        if "montant_augmentation_precedente" not in df_clean.columns:
            taux_pct = pd.to_numeric(
                df_clean[col_aug].astype(str).str.replace("%", "", regex=False).str.strip(),
                errors="coerce",
            ) / 100.0
            ancien_salaire = df_clean["revenu_mensuel"] / (1.0 + taux_pct)
            df_clean["montant_augmentation_precedente"] = (
                df_clean["revenu_mensuel"] - ancien_salaire
            ).round().astype("Int64")
            df_clean[col_aug] = taux_pct * 100.0
            nouvelles_features.append("montant_augmentation_precedente")

    if {"annees_dans_le_poste_actuel", "annees_dans_l_entreprise"} <= set(df_clean.columns):
        ajouter_feature(
            "inertie_poste",
            (df_clean["annees_dans_le_poste_actuel"] / (df_clean["annees_dans_l_entreprise"] + 1e-5)).round(3),
        )
    if {"revenu_mensuel", "niveau_hierarchique_poste"} <= set(df_clean.columns):
        ajouter_feature(
            "revenu_par_niveau",
            (df_clean["revenu_mensuel"] / (df_clean["niveau_hierarchique_poste"] + 1e-5)).round(2),
        )
    if cols_satis_presents:
        ajouter_feature("satisfaction_globale", df_clean[cols_satis_presents].mean(axis=1).round(2))
        ajouter_feature("satisfaction_min", df_clean[cols_satis_presents].min(axis=1))
        ajouter_feature("satisfaction_max", df_clean[cols_satis_presents].max(axis=1))
    if {"note_evaluation_actuelle", "note_evaluation_precedente"} <= set(df_clean.columns):
        ajouter_feature(
            "performance",
            df_clean["note_evaluation_actuelle"] - df_clean["note_evaluation_precedente"],
        )
    if col_hs is not None and "revenu_mensuel" in df_clean.columns:
        hs_txt = df_clean[col_hs].astype(str).str.lower().str.strip()
        hs_oui = hs_txt.isin(["1", "1.0", "oui", "yes", "true", "o", "y"]) | (
            pd.to_numeric(df_clean[col_hs], errors="coerce") == 1
        )
        salaire_bas = df_clean["revenu_mensuel"] < df_clean["revenu_mensuel"].median()
        ajouter_feature("hs_et_salaire_bas", (hs_oui & salaire_bas).astype(int))
    elif verbose:
        print("⚠️ Heures supplémentaires introuvable : 'hs_et_salaire_bas' non créée.")

    if {"age", "annees_dans_l_entreprise"} <= set(df_clean.columns):
        ajouter_feature(
            "jeune_faible_anciennete",
            ((df_clean["age"] <= 30) & (df_clean["annees_dans_l_entreprise"] <= 2)).astype(int),
        )
    if "nombre_experiences_precedentes" in df_clean.columns:
        ajouter_feature("serial-worker", (df_clean["nombre_experiences_precedentes"] >= 5).astype(int))

    col_poste = next(
        (c for c in ("poste", "intitule_poste", "emploi") if c in df_clean.columns),
        None,
    )
    col_niv = next(
        (
            c
            for c in ("niveau_hierarchique_poste", "niveau_hierarchique", "grade")
            if c in df_clean.columns
        ),
        None,
    )
    if col_poste and col_niv:
        poste_txt = df_clean[col_poste].astype(str).str.strip()
        niveau_num = pd.to_numeric(df_clean[col_niv], errors="coerce")
        niveau_txt = niveau_num.map(lambda v: f"N{int(v)}" if pd.notna(v) else "N?")
        ajouter_feature(
            "poste_x_niveau",
            (poste_txt + " | " + niveau_txt).where(poste_txt.ne("nan") & niveau_num.notna(), pd.NA),
        )

    col_form = next(
        (
            c
            for c in (
                "nb_formations_suivies",
                "nombre_formations_suivies",
                "nombre_formations",
                "nombre_participation_formation",
                "nombre_participation_formations",
            )
            if c in df_clean.columns
        ),
        None,
    )
    if col_form and "annees_dans_l_entreprise" in df_clean.columns:
        ajouter_feature(
            "formations_par_an",
            (
                pd.to_numeric(df_clean[col_form], errors="coerce")
                / (pd.to_numeric(df_clean["annees_dans_l_entreprise"], errors="coerce") + 1.0)
            ).round(3),
        )

    if {
        "annees_depuis_la_derniere_promotion",
        "annees_dans_l_entreprise",
    } <= set(df_clean.columns):
        ajouter_feature(
            "part_sans_promotion",
            (
                pd.to_numeric(df_clean["annees_depuis_la_derniere_promotion"], errors="coerce")
                / (pd.to_numeric(df_clean["annees_dans_l_entreprise"], errors="coerce") + 1e-5)
            ).round(3),
        )

    if "id_employee_clean" in df_clean.columns:
        df_clean = df_clean.drop(columns=["id_employee_clean"])

    _enregistrer_unites(df_clean)

    if verbose:
        print("=== FEATURES CRÉÉES À CETTE ÉTAPE ===")
        if nouvelles_features:
            for feat in nouvelles_features:
                print(f"✓ {libelle_avec_unite(feat)}")
                print(f"    → {EXPLICATIONS_FEATURES.get(feat, 'Feature métier ajoutée pour l’analyse RH.')}")
        else:
            print("Aucune nouvelle feature : elles existaient déjà.")
        if features_deja_presentes:
            print("\n=== DÉJÀ PRÉSENTES (non recréées) ===")
            for feat in features_deja_presentes:
                print(f"• {libelle_avec_unite(feat)} — {EXPLICATIONS_FEATURES.get(feat, '')}")
        print(f"\n• Forme de df_clean : {df_clean.shape[0]} lignes, {df_clean.shape[1]} colonnes")
        print("• genre / Oui-Non : encore en texte → encodeur ensuite")
        verifier_dataframe(
            df_clean,
            nom="df_clean après features",
            colonnes_cles=nouvelles_features or features_deja_presentes,
        )
    return df_clean


def verifier_dataframe(
    df: pd.DataFrame,
    nom: str = "df_clean",
    colonnes_cles: list[str] | None = None,
) -> None:
    """Contrôles de fin : dimensions, types, describe, NA, features métier."""
    print("\n" + "=" * 50)
    print(f"        VÉRIFICATIONS : {nom}")
    print("=" * 50)
    print(f"• Lignes : {df.shape[0]}  |  Colonnes : {df.shape[1]}")
    print("\n--- df.info() ---")
    df.info()

    print("\n--- describe(include='all') ---")
    try:
        from IPython.display import display

        with pd.option_context("display.max_columns", None, "display.width", 220):
            display(df.describe(include="all"))
    except Exception:
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(df.describe(include="all").to_string())

    manquantes = df.isna().sum()
    manquantes = manquantes[manquantes > 0].sort_values(ascending=False)
    print("\n--- Valeurs manquantes ---")
    if manquantes.empty:
        print("Aucune.")
    else:
        print(manquantes.to_string())

    attendues = list(EXPLICATIONS_FEATURES.keys())
    presentes = [c for c in attendues if c in df.columns]
    absentes = [c for c in attendues if c not in df.columns]
    print("\n--- Features métier ---")
    print("Présentes :", [libelle_avec_unite(c) for c in presentes] or "aucune")
    if absentes:
        print("Absentes  :", absentes)

    unites = unites_du_dataframe(df)
    if unites:
        print("\n--- Unités des colonnes ---")
        for col in df.columns:
            if col in unites:
                print(f"  {col}  [{unites[col]}]")

    if colonnes_cles:
        extra = [c for c in colonnes_cles if c in df.columns]
        if extra:
            print("\n--- Aperçu des features ajoutées ---")
            try:
                from IPython.display import display

                display(df[extra].describe(include="all"))
            except Exception:
                print(df[extra].describe(include="all").to_string())
    print("=" * 50 + "\n")


def preparer_X_y(df, target_col="a_quitte_l_entreprise", id_col="id_employee"):
    """Sépare le DataFrame en X (features) et y (target)."""
    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' est absente du DataFrame.")

    y = df[target_col]
    exclure = [target_col, id_col]
    X = df.drop(columns=[c for c in exclure if c in df.columns], errors="ignore")
    if hasattr(df, "attrs"):
        X.attrs[_ATTR_UNITES] = unites_du_dataframe(df)

    print("=" * 50)
    print("        PRÉPARATION DES DONNÉES (X et y)")
    print("=" * 50)
    print(f"🔹 DataFrame X créé : {X.shape[0]} lignes | {X.shape[1]} colonnes (features)")
    print(f"🔹 Series y créée     : {len(y)} lignes (Cible : '{target_col}')")
    print("=" * 50)
    return X, y


def bouton_creer_features(get_df_callback, set_df_callback) -> widgets.VBox:
    """Widget notebook : création des features métier."""
    bouton = widgets.Button(
        description="Créer les Features Métier",
        button_style="primary",
        icon="magic",
        layout=widgets.Layout(width="280px", height="40px"),
    )
    sortie = widgets.Output()

    def au_clic(_):
        with sortie:
            clear_output(wait=True)
            df_entree = get_df_callback()
            res = creer_features(df_entree)
            if set_df_callback:
                set_df_callback(res)

    bouton.on_click(au_clic)
    return widgets.VBox([bouton, sortie])
