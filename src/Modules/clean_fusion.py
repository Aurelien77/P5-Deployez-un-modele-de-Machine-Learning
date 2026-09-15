"""
Module de nettoyage et de fusion des 3 sources (SIRH, Sondage, Évaluation).
"""
from __future__ import annotations

import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output

# Colonnes sans aucune variance (constantes sur tout le fichier) :
# - nombre_employee_sous_responsabilite : toujours 1
# - nombre_heures_travailless           : toujours 80
# Inutiles pour KDE, corrélation ou modèle — à retirer dès le nettoyage.
COLONNES_CONSTANTES = (
    "nombre_heures_travailless",
    "nombre_employee_sous_responsabilite",
)


def _retirer_colonnes_constantes(*dataframes: pd.DataFrame) -> list[pd.DataFrame]:
    """Retire les colonnes constantes connues si elles existent."""
    out = []
    for df in dataframes:
        presentes = [c for c in COLONNES_CONSTANTES if c in df.columns]
        if presentes:
            df = df.drop(columns=presentes)
        out.append(df)
    return out


def identifier_colonnes_ids(
    df_sirh: pd.DataFrame,
    df_sondage: pd.DataFrame,
    df_eval: pd.DataFrame,
) -> dict[str, object]:
    """Identification des colonnes IDs et des intersections entre sources."""
    print("## Identification des colonnes IDs")
    print(
        "On extrait les identifiants de chaque source, on mesure les IDs "
        "communs (ceux qu'on pourra fusionner) et les colonnes partagées "
        "(hors clés de jointure).\n"
    )

    id_sirh = set(df_sirh["id_employee"].astype(str).str.strip())
    id_sondage = set(df_sondage["code_sondage"].astype(str).str.strip())
    id_eval = set(
        df_eval["eval_number"]
        .astype(str)
        .str.replace("E_", "", regex=False)
        .str.strip()
    )

    commun_sirh_sondage = id_sirh.intersection(id_sondage)
    commun_sirh_eval = id_sirh.intersection(id_eval)
    commun_tous = id_sirh.intersection(id_sondage).intersection(id_eval)

    print(f"Nombre d'IDs uniques dans df_sirh : {len(id_sirh)}")
    print(f"Nombre d'IDs uniques dans df_sondage : {len(id_sondage)}")
    print(f"Nombre d'IDs uniques dans df_eval : {len(id_eval)}")
    print("-" * 50)
    print(f"IDs communs entre SIRH et Sondage : {len(commun_sirh_sondage)}")
    print(f"IDs communs entre SIRH et Eval : {len(commun_sirh_eval)}")
    print(f"IDs communs aux 3 DataFrames : {len(commun_tous)}")

    n_ref = max(len(id_sirh), 1)
    print(
        f"Taux de recouvrement vs SIRH — "
        f"Sondage : {len(commun_sirh_sondage) / n_ref:.1%} | "
        f"Éval : {len(commun_sirh_eval) / n_ref:.1%} | "
        f"les 3 : {len(commun_tous) / n_ref:.1%}\n"
    )

    cles = {"id_employee", "code_sondage", "eval_number", "id_employees"}
    colonnes_sirh = set(df_sirh.columns)
    colonnes_sondage = set(df_sondage.columns)
    colonnes_eval = set(df_eval.columns)

    print(
        "Communes entre sirh et sondage :",
        sorted((colonnes_sirh & colonnes_sondage) - cles) or "{}",
    )
    print(
        "Communes entre sirh et eval :",
        sorted((colonnes_sirh & colonnes_eval) - cles) or "{}",
    )
    print(
        "Communes entre sondage et eval :",
        sorted((colonnes_sondage & colonnes_eval) - cles) or "{}",
    )
    print()

    return {
        "id_sirh": id_sirh,
        "id_sondage": id_sondage,
        "id_eval": id_eval,
        "commun_sirh_sondage": commun_sirh_sondage,
        "commun_sirh_eval": commun_sirh_eval,
        "commun_tous": commun_tous,
    }


def nettoyer_et_fusionner(
    df_sirh: pd.DataFrame,
    df_sondage: pd.DataFrame,
    df_eval: pd.DataFrame,
) -> pd.DataFrame:
    """Nettoie les 3 DataFrames, effectue la fusion directe via df_sirh et retourne df_clean."""

    print("=== 0. IDENTIFICATION DES COLONNES IDs ===")
    identifier_colonnes_ids(df_sirh, df_sondage, df_eval)

    print("=== 1. DIMENSIONS AVANT NETTOYAGE ===")
    print(f"• df_sirh    : {df_sirh.shape[0]} lignes | {df_sirh.shape[1]} colonnes")
    print(f"• df_sondage : {df_sondage.shape[0]} lignes | {df_sondage.shape[1]} colonnes")
    print(f"• df_eval    : {df_eval.shape[0]} lignes | {df_eval.shape[1]} colonnes\n")
    cols_avant = {
        "df_sirh": set(df_sirh.columns),
        "df_sondage": set(df_sondage.columns),
        "df_eval": set(df_eval.columns),
    }

    df_sirh = df_sirh.copy()
    df_sondage = df_sondage.copy()
    df_eval = df_eval.copy()

    df_sirh["id_employee"] = df_sirh["id_employee"].astype(str).str.strip()

    if "code_sondage" in df_sondage.columns:
        df_sondage["id_employee"] = df_sondage["code_sondage"].astype(str).str.strip()

    if "eval_number" in df_eval.columns:
        df_eval["id_employee"] = (
            df_eval["eval_number"]
            .astype(str)
            .str.replace("E_", "", regex=False)
            .str.strip()
        )

    id_sirh = set(df_sirh["id_employee"])
    id_sondage = set(df_sondage["id_employee"])
    id_eval = set(df_eval["id_employee"])

    print("=== DÉTAIL DES IDENTIFIANTS AVANT FUSION ===")
    print(f"• df_sirh    : {len(id_sirh)} IDs uniques")
    print(f"• df_sondage : {len(id_sondage)} IDs uniques")
    print(f"• df_eval    : {len(id_eval)} IDs uniques\n")

    commun_tous = id_sirh.intersection(id_sondage).intersection(id_eval)
    exclus_sirh = id_sirh - commun_tous
    exclus_sondage = id_sondage - commun_tous
    exclus_eval = id_eval - commun_tous

    print("=== ANALYSE DES EXCLUSIONS (IDs absents d'au moins une source) ===")
    print(f"• IDs exclus de df_sirh    : {len(exclus_sirh)}")
    print(f"• IDs exclus de df_sondage : {len(exclus_sondage)}")
    print(f"• IDs exclus de df_eval    : {len(exclus_eval)}")
    print(f"• IDs communs aux 3 sources : {len(commun_tous)}\n")

    print("=== COLONNES CONSTANTES RETIRÉES (variance nulle, hors clés) ===")
    print("• nombre_employee_sous_responsabilite (toujours 1)")
    print("• nombre_heures_travailless (toujours 80)\n")
    df_sirh, df_sondage, df_eval = _retirer_colonnes_constantes(df_sirh, df_sondage, df_eval)

    print("=== 2. DIMENSIONS APRÈS NETTOYAGE (AVANT FUSION) ===")
    print(f"• df_sirh    : {df_sirh.shape[0]} lignes | {df_sirh.shape[1]} colonnes")
    print(f"• df_sondage : {df_sondage.shape[0]} lignes | {df_sondage.shape[1]} colonnes")
    print(f"• df_eval    : {df_eval.shape[0]} lignes | {df_eval.shape[1]} colonnes")
    for nom, df in (("df_sirh", df_sirh), ("df_sondage", df_sondage), ("df_eval", df_eval)):
        avant = cols_avant[nom]
        apres = set(df.columns)
        ajoutees = sorted(apres - avant)
        retirees = sorted(avant - apres)
        print(f"  → {nom}  ajoutées : {ajoutees or 'aucune'}  |  retirées : {retirees or 'aucune'}")
    print()

    df_clean = pd.merge(
        df_sirh,
        df_sondage,
        on="id_employee",
        how="inner",
        suffixes=("", "_sondage"),
    )
    df_clean = pd.merge(
        df_clean,
        df_eval,
        on="id_employee",
        how="inner",
        suffixes=("", "_eval"),
    )

    cols_id_a_supprimer = ["code_sondage", "eval_number", "id_employees"]
    df_clean = df_clean.drop(columns=[c for c in cols_id_a_supprimer if c in df_clean.columns])

    cols_redondantes = ["note_evaluation_precedente_num", "note_evaluation_actuelle_num"]
    df_clean = df_clean.drop(columns=[c for c in cols_redondantes if c in df_clean.columns])

    df_clean = _retirer_colonnes_constantes(df_clean)[0]

    cols_constantes_apres = list(COLONNES_CONSTANTES)
    if "ayant_enfants" in df_clean.columns:
        valeurs_uniques_enfants = df_clean["ayant_enfants"].dropna().unique()
        if len(valeurs_uniques_enfants) <= 1:
            df_clean = df_clean.drop(columns=["ayant_enfants"])
            cols_constantes_apres.append("ayant_enfants")

    if "id_employee" in df_clean.columns:
        df_clean = df_clean.drop(columns=["id_employee"])

    print("=== 3. DIMENSIONS FINALES (APRÈS FUSION & NETTOYAGE FINAL) ===")
    print(f"• Lignes finales : {df_clean.shape[0]}")
    print(f"• Colonnes finales : {df_clean.shape[1]}")
    print(f"• Colonnes redondantes retirées : {cols_redondantes}")
    print(f"• Colonnes constantes retirées  : {cols_constantes_apres}")
    print("• id_employee retiré du df final (identifiant technique)\n")

    # --- Test réintégré : affichage des types de toutes les colonnes du df final ---
    print("=== TYPES DE TOUTES LES COLONNES DANS DF_CLEAN ===")
    print(df_clean.dtypes)
    print()

    nom_fichier = "df_clean.csv"
    df_clean.to_csv(nom_fichier, index=False)
    print(f"• Fichier sauvegardé localement : {nom_fichier}")

    # Copies de travail seulement (les df du notebook restent tant qu'on ne les del pas là-bas)
    del df_sirh, df_sondage, df_eval
    print("• Copies locales SIRH / Sondage / Éval effacées → il reste df_clean\n")
    return df_clean


def bouton_nettoyer_fusionner(get_dfs_callback, set_df_clean_callback) -> widgets.VBox:
    """Crée un bloc interactif (bouton + zone de texte) pour le nettoyage et la fusion."""
    bouton = widgets.Button(
        description="Lancer Nettoyage & Fusion",
        button_style="success",
        icon="check",
        layout=widgets.Layout(width="280px", height="40px"),
    )
    sortie = widgets.Output()

    def au_clic(_):
        with sortie:
            clear_output(wait=True)
            d_sirh, d_sondage, d_eval = get_dfs_callback()
            res = nettoyer_et_fusionner(d_sirh, d_sondage, d_eval)
            if set_df_clean_callback:
                set_df_clean_callback(res)

    bouton.on_click(au_clic)
    return widgets.VBox([bouton, sortie])