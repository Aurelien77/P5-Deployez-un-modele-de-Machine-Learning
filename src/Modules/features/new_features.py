"""
Module de création de features métier RH.
"""
from __future__ import annotations

import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output

EXPLICATIONS_FEATURES = {
    "montant_augmentation_precedente": "Montant en € de la dernière augmentation : impact réel sur la paie, pas seulement le %.",
    "ratio_anciennete_carriere": "Années entreprise / expérience totale → loyauté vs mobilité.",
    "inertie_poste": "Années poste / années entreprise → stagnation dans le rôle.",
    "stagnation_promotion": "Années sans promo / ancienneté → carrière bloquée.",
    "revenu_par_annee_experience": "Salaire / (expérience + 1) → sous-paiement relatif.",
    "revenu_par_niveau": "Salaire / niveau hiérarchique → écart au niveau de poste.",
    "satisfaction_globale": "Moyenne des 4 satisfactions → climat social.",
    "satisfaction_min": "Minimum des 4 satisfactions → un axe très bas suffit.",
    "delta_performance": "Note actuelle − note précédente → hausse ou baisse d'éval.",
    "hs_et_salaire_bas": "HS et salaire < médiane → surcharge + bas salaire.",
    "jeune_faible_anciennete": "Âge ≤ 30 et ancienneté ≤ 2 ans → départ précoce.",
    "trajet_long": "Distance domicile-travail ≥ 15 → pénibilité du trajet.",
    "job_hopper": "≥ 5 employeurs précédents → habitude de partir.",
}


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

    if {"annees_dans_l_entreprise", "annee_experience_totale"} <= set(df_clean.columns):
        ajouter_feature(
            "ratio_anciennete_carriere",
            (df_clean["annees_dans_l_entreprise"] / (df_clean["annee_experience_totale"] + 1e-5)).round(3),
        )
    if {"annees_dans_le_poste_actuel", "annees_dans_l_entreprise"} <= set(df_clean.columns):
        ajouter_feature(
            "inertie_poste",
            (df_clean["annees_dans_le_poste_actuel"] / (df_clean["annees_dans_l_entreprise"] + 1e-5)).round(3),
        )
    if {"annees_depuis_la_derniere_promotion", "annees_dans_l_entreprise"} <= set(df_clean.columns):
        ajouter_feature(
            "stagnation_promotion",
            (
                df_clean["annees_depuis_la_derniere_promotion"]
                / (df_clean["annees_dans_l_entreprise"] + 1e-5)
            ).round(3),
        )
    if {"revenu_mensuel", "annee_experience_totale"} <= set(df_clean.columns):
        ajouter_feature(
            "revenu_par_annee_experience",
            (df_clean["revenu_mensuel"] / (df_clean["annee_experience_totale"] + 1)).round(2),
        )
    if {"revenu_mensuel", "niveau_hierarchique_poste"} <= set(df_clean.columns):
        ajouter_feature(
            "revenu_par_niveau",
            (df_clean["revenu_mensuel"] / (df_clean["niveau_hierarchique_poste"] + 1e-5)).round(2),
        )
    if cols_satis_presents:
        ajouter_feature("satisfaction_globale", df_clean[cols_satis_presents].mean(axis=1).round(2))
        ajouter_feature("satisfaction_min", df_clean[cols_satis_presents].min(axis=1))
    if {"note_evaluation_actuelle", "note_evaluation_precedente"} <= set(df_clean.columns):
        ajouter_feature(
            "delta_performance",
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
    if "distance_domicile_travail" in df_clean.columns:
        ajouter_feature("trajet_long", (df_clean["distance_domicile_travail"] >= 15).astype(int))
    if "nombre_experiences_precedentes" in df_clean.columns:
        ajouter_feature("job_hopper", (df_clean["nombre_experiences_precedentes"] >= 5).astype(int))

    if "id_employee_clean" in df_clean.columns:
        df_clean = df_clean.drop(columns=["id_employee_clean"])

    if verbose:
        print("=== FEATURES CRÉÉES À CETTE ÉTAPE ===")
        if nouvelles_features:
            for feat in nouvelles_features:
                print(f"✓ {feat}")
                print(f"    → {EXPLICATIONS_FEATURES.get(feat, 'Feature métier ajoutée pour l’analyse RH.')}")
        else:
            print("Aucune nouvelle feature : elles existaient déjà.")
        if features_deja_presentes:
            print("\n=== DÉJÀ PRÉSENTES (non recréées) ===")
            for feat in features_deja_presentes:
                print(f"• {feat} — {EXPLICATIONS_FEATURES.get(feat, '')}")
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
    print("Présentes :", presentes or "aucune")
    if absentes:
        print("Absentes  :", absentes)
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
