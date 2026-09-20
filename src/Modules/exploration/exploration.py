"""
Module d'exploration et de diagnostic des DataFrames RH.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import ipywidgets as widgets
except ImportError:  # notebook optionnel
    widgets = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Colonnes techniques (identifiants) à ne jamais analyser statistiquement
# ---------------------------------------------------------------------------
# Liste de secours / noms connus. La détection automatique (unicité +
# régularité des écarts) remplace désormais l'exclusion manuelle comme
# mécanisme principal.
COLONNES_ID_PAR_DEFAUT: tuple[str, ...] = (
    "id_employee",
    "id_employe",
    "code_sondage",
    "id",
    "eval_number",
)

CANDIDATS_CIBLE = (
    "a_quitte_l_entreprise",
    "depart",
    "départ",
    "a_quitte",
    "attrition",
    "quitte_entreprise",
)

_LABELS_METHODE_CORRELATION = {
    "pearson": "r de Pearson",
    "spearman": "ρ de Spearman",
    "kendall": "τ de Kendall",
}

_ETIQUETTES_STATUT = {
    0: "Reste",
    1: "Part",
    "0": "Reste",
    "1": "Part",
    0.0: "Reste",
    1.0: "Part",
    "Non": "Reste",
    "Oui": "Part",
    "non": "Reste",
    "oui": "Part",
    False: "Reste",
    True: "Part",
}


def _filtrer_colonnes_id(colonnes: Iterable[str], exclure: Iterable[str]) -> list[str]:
    """Retire les colonnes d'identifiant (comparaison insensible à la casse)."""
    exclure_lower = {c.lower().strip() for c in exclure}
    return [c for c in colonnes if c.lower().strip() not in exclure_lower]


# ---------------------------------------------------------------------------
# Détection automatique des colonnes séquentielles (identifiant / compteur)
# ---------------------------------------------------------------------------
def est_colonne_sequentielle(
    serie: pd.Series,
    seuil_unicite: float = 0.95,
    seuil_regularite: float = 0.90,
    tolerance_ecart: float = 0.05,
    min_valeurs: int = 5,
) -> bool:
    """Indique si UNE variable, prise isolément, se comporte comme un identifiant
    ou un compteur plutôt que comme une vraie variable statistique.

    Ce n'est PAS un test de corrélation (linéaire / non-linéaire) entre deux
    colonnes. On examine uniquement l'évolution interne de la série :

    1. Taux d'unicité proche de 100 % (peu ou pas de valeurs répétées).
    2. Une fois les valeurs triées, les écarts entre valeurs consécutives
       sont quasi constants (ex. +1 à chaque fois, comme un compteur).

    Les deux critères doivent être satisfaits simultanément.
    """
    serie_num = _vers_numerique(serie).dropna()
    n = int(len(serie_num))
    if n < min_valeurs:
        return False

    n_uniques = int(serie_num.nunique(dropna=True))
    if n_uniques < min_valeurs:
        return False

    taux_unicite = n_uniques / n
    if taux_unicite < seuil_unicite:
        return False

    valeurs = np.sort(serie_num.unique().astype(float))
    diffs = np.diff(valeurs)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return False

    # Un compteur a un pas strictement positif. Un pas nul = constantes.
    pas = float(np.median(diffs))
    if pas <= 0:
        return False

    # Écart relatif au pas médian : un vrai compteur (1,2,3… ou 10,20,30…)
    # a presque tous ses diffs égaux au pas.
    ecarts_relatifs = np.abs(diffs - pas) / pas
    taux_regularite = float(np.mean(ecarts_relatifs <= tolerance_ecart))
    return taux_regularite >= seuil_regularite


def detecter_colonnes_sequentielles(
    df: pd.DataFrame,
    colonnes: Iterable[str] | None = None,
    seuil_unicite: float = 0.95,
    seuil_regularite: float = 0.90,
    tolerance_ecart: float = 0.05,
    min_valeurs: int = 5,
    verbose: bool = False,
) -> list[str]:
    """Détecte automatiquement les colonnes de type identifiant / compteur.

    Une colonne est retenue si, individuellement :
    - presque toutes ses valeurs sont uniques ;
    - ses valeurs triées progressent à intervalles réguliers.
    """
    cibles = list(colonnes) if colonnes is not None else list(df.columns)
    detectees: list[str] = []

    for col in cibles:
        if col not in df.columns:
            continue
        if not est_colonne_sequentielle(
            df[col],
            seuil_unicite=seuil_unicite,
            seuil_regularite=seuil_regularite,
            tolerance_ecart=tolerance_ecart,
            min_valeurs=min_valeurs,
        ):
            continue
        detectees.append(col)
        if verbose:
            serie_num = _vers_numerique(df[col]).dropna()
            n = len(serie_num)
            n_uniques = int(serie_num.nunique())
            valeurs = np.sort(serie_num.unique().astype(float))
            diffs = np.diff(valeurs)
            pas = float(np.median(diffs)) if len(diffs) else float("nan")
            print(
                f"  • {col}: unicité={n_uniques}/{n} ({n_uniques / n:.1%}), "
                f"pas médian={pas:g} → exclue (identifiant / compteur)"
            )
    return detectees


def _colonnes_a_exclure(
    df: pd.DataFrame,
    exclure: Iterable[str] | None = None,
    verbose: bool = False,
) -> list[str]:
    """Union de la liste nominale et de la détection automatique."""
    nominaux = list(exclure) if exclure is not None else list(COLONNES_ID_PAR_DEFAUT)
    auto = detecter_colonnes_sequentielles(df, verbose=verbose)

    vus: set[str] = set()
    fusion: list[str] = []
    for col in list(nominaux) + auto:
        cle = col.lower().strip()
        if cle in vus:
            continue
        vus.add(cle)
        fusion.append(col)
    return fusion


# ---------------------------------------------------------------------------
# Utilitaires d'affichage et de conversion
# ---------------------------------------------------------------------------
def _afficher(obj: Any) -> None:
    """Affiche un objet sans tronquer les cellules de texte."""
    if isinstance(obj, pd.DataFrame):
        options = (
            "display.max_columns",
            None,
            "display.max_rows",
            None,
            "display.max_colwidth",
            None,
            "display.width",
            220,
            "display.expand_frame_repr",
            True,
        )
        try:
            from IPython.display import display

            with pd.option_context(*options):
                try:
                    styler = (
                        obj.style.set_properties(
                            **{
                                "text-align": "left",
                                "white-space": "pre-wrap",
                                "max-width": "520px",
                            }
                        ).set_table_styles(
                            [
                                {"selector": "th", "props": [("text-align", "left")]},
                                {
                                    "selector": "td",
                                    "props": [
                                        ("white-space", "pre-wrap"),
                                        ("word-wrap", "break-word"),
                                    ],
                                },
                            ]
                        )
                    )
                    display(styler)
                except Exception:
                    display(obj)
            return
        except Exception:
            with pd.option_context(*options):
                print(obj.to_string())
            return
    try:
        from IPython.display import display

        display(obj)
    except Exception:
        print(obj)


def _vers_numerique(serie: pd.Series) -> pd.Series:
    """Convertit une série vers un type numérique, y compris les pourcentages texte."""
    if serie.dtype == "object" or str(serie.dtype).startswith("str"):
        return pd.to_numeric(
            serie.astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )
    return pd.to_numeric(serie, errors="coerce")


def _serie_kde_ok(serie: pd.Series, min_obs: int = 5) -> bool:
    """True seulement si un KDE a un sens (assez d'observations et une vraie variance)."""
    s = pd.Series(serie).dropna()
    if len(s) < min_obs:
        return False
    if int(s.nunique()) < 2:
        return False
    std = float(s.std(ddof=0))
    return bool(np.isfinite(std) and std > 0)


def _label_statut(val: Any) -> str:
    if val in _ETIQUETTES_STATUT:
        return _ETIQUETTES_STATUT[val]
    texte = str(val).strip()
    return _ETIQUETTES_STATUT.get(texte, texte)


def _poser_legende(ax, title: str | None = None, **kwargs) -> None:
    """Ajoute une légende seulement s'il existe au moins un artiste labellisé."""
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title=title, **kwargs)


def _correlation_sure(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    """Corrélation sans warning numpy si une série est constante ou vide."""
    d = pd.DataFrame({"x": _vers_numerique(x), "y": _vers_numerique(y)}).dropna()
    if len(d) < 3 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return float("nan")
    if float(d["x"].std()) == 0.0 or float(d["y"].std()) == 0.0:
        return float("nan")
    val = d["x"].corr(d["y"], method=method)
    return float(val) if val is not None and np.isfinite(val) else float("nan")


def bilan_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    manquantes = (df.isna().mean() * 100).round(2)
    return pd.DataFrame(
        {
            "Type": df.dtypes.astype(str),
            "Valeurs Pleines (%)": 100 - manquantes,
            "Valeurs Vides (%)": manquantes,
            "Valeurs Uniques": df.nunique(),
        }
    )


def _radiographie_remplissage(resultat: pd.DataFrame, nom: str) -> None:
    """Bande compacte : une case par colonne, verte = pleine, rouge = trouée.

    Volontairement petit (une seule ligne de hauteur) — sert de coup d'œil
    rapide avant de lire le tableau détaillé de `tester_remplissage`.
    """
    n = len(resultat)
    if n == 0:
        return

    taux = resultat["Taux de remplissage (%)"].to_numpy().reshape(1, -1)
    largeur = max(4, min(14, 0.35 * n + 1.5))
    fig, ax = plt.subplots(figsize=(largeur, 1.3))
    im = ax.imshow(taux, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(resultat["Colonne"], rotation=90, fontsize=6)
    ax.set_yticks([])
    for i, val in enumerate(taux[0]):
        couleur_texte = "black" if 30 < val < 90 else "white"
        ax.text(i, 0, f"{val:.0f}", ha="center", va="center", fontsize=6, color=couleur_texte)

    ax.set_title(f"Radiographie du remplissage — {nom}", fontsize=9, pad=6)
    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.01, label="% plein")
    fig.tight_layout()
    plt.show()


def tester_remplissage(df: pd.DataFrame, nom: str = "dataframe") -> pd.DataFrame:
    """Teste le remplissage de chaque colonne, en distinguant deux types de manquants :

    - NaN / None (valeurs réellement manquantes au sens pandas) ;
    - chaînes vides ou blanches ("", "   ") pour les colonnes texte — souvent
      invisibles à `isna()` mais tout aussi vides fonctionnellement.

    Affiche un résumé global puis retourne un tableau détaillé trié (colonnes
    les moins remplies en premier).
    """
    n_lignes = len(df)

    n_nan_par_col = df.isna().sum()

    n_vides_texte = pd.Series(0, index=df.columns, dtype=int)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            masque_vide = df[col].notna() & (df[col].astype(str).str.strip() == "")
            n_vides_texte[col] = int(masque_vide.sum())

    n_manquant_total = n_nan_par_col + n_vides_texte

    def _taux(n_manque: pd.Series) -> pd.Series:
        return (100 - (n_manque / n_lignes * 100)).round(2) if n_lignes else n_manque * 0

    resultat = pd.DataFrame(
        {
            "Colonne": df.columns,
            "Nb NaN": n_nan_par_col.values,
            "Nb vides ('' texte)": n_vides_texte.values,
            "Nb manquant (total)": n_manquant_total.values,
            "Taux de remplissage (%)": _taux(n_manquant_total).values,
            "Taux de manquant (%)": (100 - _taux(n_manquant_total)).values,
        }
    ).sort_values("Taux de remplissage (%)").reset_index(drop=True)

    total_cellules = df.shape[0] * df.shape[1]
    total_nan = int(n_nan_par_col.sum())
    total_vides_texte = int(n_vides_texte.sum())
    total_manquant = total_nan + total_vides_texte
    taux_global = round(100 - (total_manquant / total_cellules * 100), 2) if total_cellules else 0.0

    print(f"\n=== TEST DE REMPLISSAGE : {nom} ===")
    print(f"• Lignes : {n_lignes} | Colonnes : {df.shape[1]}")
    print(f"• NaN au total : {total_nan} / {total_cellules}")
    print(f"• Chaînes vides ('') au total : {total_vides_texte} / {total_cellules}")
    print(f"• Manquant total (NaN + vides) : {total_manquant} / {total_cellules} ({100 - taux_global:.2f}%)")
    print(f"• Taux de remplissage global : {taux_global}%")

    colonnes_incompletes = resultat[resultat["Nb manquant (total)"] > 0]
    if colonnes_incompletes.empty:
        print("• Aucune colonne incomplète : toutes les colonnes sont remplies à 100%.\n")
    else:
        print(f"• {len(colonnes_incompletes)} colonne(s) avec au moins une valeur manquante :\n")
        _afficher(colonnes_incompletes)

    _radiographie_remplissage(resultat, nom)

    return resultat


def dataframe_numerique(df: pd.DataFrame) -> pd.DataFrame:
    data = {}
    exclus = {c.lower().strip() for c in _colonnes_a_exclure(df)}
    for col in df.columns:
        if col.lower().strip() in exclus:
            continue
        serie_num = _vers_numerique(df[col])
        if serie_num.notna().mean() >= 0.5 and serie_num.nunique(dropna=True) > 1:
            data[col] = serie_num
    return pd.DataFrame(data, index=df.index)


def preparer_donnees_correlation(
    df: pd.DataFrame,
    exclure: Iterable[str] = COLONNES_ID_PAR_DEFAUT,
    max_modalites: int = 10,
) -> pd.DataFrame:
    """Construit un DataFrame entièrement numérique afin que TOUTES les colonnes
    (y compris qualitatives) puissent entrer dans l'analyse de corrélation :

    - colonnes déjà numériques (ou convertibles)          -> gardées telles quelles ;
    - colonnes qualitatives à 2 modalités (ex : genre)      -> encodées en 0/1
      (équivalent à une corrélation "point-bisériale" avec Pearson) ;
    - colonnes qualitatives à plusieurs modalités           -> indicatrices one-hot ;
    - colonnes avec trop de modalités (> max_modalites)     -> ignorées ;
    - colonnes séquentielles (identifiant / compteur)       -> ignorées.
    """
    exclus = {c.lower().strip() for c in _colonnes_a_exclure(df, exclure)}
    morceaux: list[pd.Series] = []

    for col in df.columns:
        if col.lower().strip() in exclus:
            continue
        if est_colonne_sequentielle(df[col]):
            continue

        serie_num = _vers_numerique(df[col])
        if serie_num.notna().mean() >= 0.5 and serie_num.nunique(dropna=True) > 1:
            morceaux.append(serie_num.rename(col))
            continue

        serie_qual = df[col]
        n_modalites = serie_qual.nunique(dropna=True)
        if n_modalites < 2:
            continue
        if n_modalites == 2:
            codes = pd.factorize(serie_qual, sort=True)[0].astype(float)
            codes[codes == -1] = np.nan
            morceaux.append(pd.Series(codes, index=df.index, name=col))
        elif n_modalites <= max_modalites:
            dummies = pd.get_dummies(serie_qual, prefix=col, dtype=float)
            for c in dummies.columns:
                morceaux.append(dummies[c])

    if not morceaux:
        return pd.DataFrame(index=df.index)
    return pd.concat(morceaux, axis=1)


def _encoder_cible(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """0 = reste, 1 = départ."""
    out = df.copy()
    if pd.api.types.is_numeric_dtype(out[target_col]):
        return out
    out[target_col] = (
        out[target_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin(["oui", "yes", "true", "1", "1.0"])
        .astype(int)
    )
    return out


def _quantitatives_hors_id(df: pd.DataFrame, cible: str | None) -> list[str]:
    exclus = set(_colonnes_a_exclure(df))
    if cible:
        exclus.add(cible)
    cols: list[str] = []
    for col in df.columns:
        if col in exclus or est_colonne_sequentielle(df[col]):
            continue
        serie = _vers_numerique(df[col])
        if serie.notna().mean() < 0.5:
            continue
        n_uniques = int(serie.nunique(dropna=True))
        if n_uniques <= 2:
            continue  # binaire : pas de forme à tester
        cols.append(col)
    return cols


def _dessiner_heatmap_correlation(
    ax,
    corr: pd.DataFrame,
    methode: str,
    nom: str,
    titre_extra: str = "",
) -> None:
    n = corr.shape[0]
    if n == 0:
        ax.axis("off")
        ax.set_title(f"{methode.capitalize()} — {nom}\nAucune colonne")
        return
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=n <= 12,
        annot_kws={"size": 8},
        fmt=".2f",
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": _LABELS_METHODE_CORRELATION[methode]},
    )
    bas, haut = ax.get_ylim()
    ax.set_ylim(bas + 0.5, haut - 0.5)
    titre = f"{methode.capitalize()} — {nom}"
    if titre_extra:
        titre += f"\n{titre_extra}"
    ax.set_title(titre, fontsize=11, pad=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")


def classement_correlations(
    corr_pearson: pd.DataFrame,
    corr_spearman: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classement des paires : une ligne = une paire, colonnes r et ρ."""
    colonnes = ["Variable 1", "Variable 2", "Pearson r", "Spearman ρ", "|r|", "|ρ − r|"]
    if corr_pearson is None or corr_pearson.empty or corr_pearson.shape[0] < 2:
        return pd.DataFrame(columns=colonnes)

    communs = list(corr_pearson.index)
    if corr_spearman is not None and not corr_spearman.empty:
        communs = [c for c in communs if c in corr_spearman.index]
    if len(communs) < 2:
        return pd.DataFrame(columns=colonnes)

    cp = corr_pearson.loc[communs, communs]
    cs = (
        corr_spearman.loc[communs, communs]
        if corr_spearman is not None and not corr_spearman.empty
        else None
    )
    arr_p = cp.to_numpy(dtype=float)
    arr_s = cs.to_numpy(dtype=float) if cs is not None else None

    lignes = []
    for i in range(len(communs)):
        for j in range(i + 1, len(communs)):
            r = arr_p[i, j]
            if not np.isfinite(r):
                continue
            rho = arr_s[i, j] if arr_s is not None else float("nan")
            lignes.append(
                {
                    "Variable 1": communs[i],
                    "Variable 2": communs[j],
                    "Pearson r": float(r),
                    "Spearman ρ": float(rho) if np.isfinite(rho) else float("nan"),
                }
            )
    if not lignes:
        return pd.DataFrame(columns=colonnes)

    paires = pd.DataFrame(lignes)
    paires["|r|"] = paires["Pearson r"].abs()
    paires["|ρ − r|"] = (paires["Spearman ρ"] - paires["Pearson r"]).abs()
    return paires.sort_values("|r|", ascending=False).reset_index(drop=True)



def _afficher_classement_et_barres(
    classement: pd.DataFrame,
    nom: str,
    max_paires: int = 15,
) -> None:
    if classement.empty:
        return

    top_paires = (
        classement[["Variable 1", "Variable 2", "Pearson r", "Spearman ρ"]]
        .head(max_paires)
        .reset_index(drop=True)
    )
    top_paires.insert(0, "rang", np.arange(1, len(top_paires) + 1))
    print(f"\n--- TOP {len(top_paires)} paires les plus corrélées : {nom} ---")
    _afficher(top_paires.round(3))

    pieces = []
    for a, b, r, rho in classement[["Variable 1", "Variable 2", "Pearson r", "Spearman ρ"]].itertuples(
        index=False
    ):
        pieces.append((a, b, r, rho))
        pieces.append((b, a, r, rho))
    par_feat = pd.DataFrame(pieces, columns=["feature", "partenaire", "Pearson r", "Spearman ρ"])
    par_feat["|r|"] = par_feat["Pearson r"].abs()
    top_feat = (
        par_feat.sort_values("|r|", ascending=False)
        .drop_duplicates("feature")
        .drop(columns="|r|")
        .reset_index(drop=True)
    )
    top_feat.insert(0, "rang", np.arange(1, len(top_feat) + 1))
    print(f"\n--- TOP features (lien le plus fort avec une autre) : {nom} ---")
    _afficher(top_feat.round(3))

    barre = top_paires.iloc[::-1]
    etiquettes = barre["Variable 1"] + "  ×  " + barre["Variable 2"]
    y = np.arange(len(barre))
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.38 * len(barre))))
    ax.barh(y + 0.18, barre["Pearson r"], height=0.35, color="#4C72B0", label="Pearson r")
    ax.barh(y - 0.18, barre["Spearman ρ"], height=0.35, color="#DD8452", label="Spearman ρ")
    ax.set_yticks(y)
    ax.set_yticklabels(etiquettes, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"TOP paires corrélées — {nom}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plt.show()


def matrice_correlation(
    df: pd.DataFrame,
    nom: str = "dataframe",
    exclure: Iterable[str] = COLONNES_ID_PAR_DEFAUT,
    max_paires_barchart: int = 20,
    max_modalites: int = 10,
    seuil_non_lineaire: float = 0.08,
    seuil_signal: float = 0.20,
) -> dict[str, pd.DataFrame] | None:
    """Corrélations entre features quantitatives (Pearson + Spearman).

    La cible ``a_quitte_l_entreprise`` est volontairement ignorée ici :
    son étude est dans ``analyser_correlations_vs_cible`` (bouton Attrition).
    """
    del exclure, max_modalites, seuil_non_lineaire, seuil_signal

    auto = detecter_colonnes_sequentielles(df)
    if auto:
        print(f"\n--- Identifiants / compteurs exclus ({nom}) ---")
        for col in auto:
            print(f"  • {col}")

    cible = _trouver_colonne(df, CANDIDATS_CIBLE)
    quantitatives = _quantitatives_hors_id(df, cible)
    if len(quantitatives) < 2:
        return None

    df_q = df[quantitatives].apply(_vers_numerique)
    df_q = df_q.loc[:, df_q.nunique(dropna=True) > 1]
    if df_q.shape[1] < 2:
        return None

    corr_p = df_q.corr(method="pearson")
    corr_s = df_q.corr(method="spearman")

    cote = max(6, min(12, 0.55 * df_q.shape[1] + 3))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(cote * 2, cote))
    _dessiner_heatmap_correlation(ax1, corr_p, "pearson", nom)
    _dessiner_heatmap_correlation(ax2, corr_s, "spearman", nom)
    fig.suptitle(f"Matrices de corrélation — {nom}", fontsize=13)
    fig.tight_layout()
    plt.show()
   

    classement = classement_correlations(corr_p, corr_s)
    _afficher_classement_et_barres(classement, nom, max_paires_barchart)
 
    
    # Ajout pour afficher la matrice de nuages pour ce DataFrame
    top_vars = df_q.columns[:6].tolist()
    plot_matrice_nuages(df, top_vars, cible=cible, nom=nom)

    return {"pearson": corr_p, "spearman": corr_s, "classement": classement}


def _cramer_v(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan")
    n = float(tab.to_numpy().sum())
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import chi2_contingency

        chi2 = float(chi2_contingency(tab, correction=False)[0])
    except Exception:
        attendu = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / n
        chi2 = float(((tab.to_numpy() - attendu) ** 2 / np.clip(attendu, 1e-12, None)).sum())
    k = min(tab.shape) - 1
    if k <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / (n * k)))


def _plot_kde_vs_cible(df: pd.DataFrame, colonnes: list[str], cible: str, ncols: int = 2) -> None:
    """KDE Reste vs Part, plusieurs variables côte à côte."""
    y = df[cible]
    a_tracer: list[tuple[str, list[tuple[Any, pd.Series]]]] = []
    ignorees: list[str] = []

    for col in colonnes:
        if col not in df.columns:
            continue
        x = _vers_numerique(df[col])
        if int(x.nunique(dropna=True)) < 2:
            ignorees.append(col)
            continue
        groupes = []
        for val in sorted(pd.unique(y.dropna()), key=str):
            donnees = x[(y == val) & x.notna()]
            groupes.append((val, donnees))
        tracables = [(val, d) for val, d in groupes if _serie_kde_ok(d)]
        if not tracables:
            ignorees.append(col)
            continue
        a_tracer.append((col, tracables))

    if not a_tracer:
        if ignorees:
            print("Aucun KDE traçable : " + ", ".join(ignorees))
        return

    n = len(a_tracer)
    cols = max(1, min(ncols, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, (col, tracables) in enumerate(a_tracer):
        ax = axes[i]
        for val, donnees in tracables:
            sns.kdeplot(
                donnees,
                ax=ax,
                fill=True,
                alpha=0.35,
                linewidth=2,
                label=_label_statut(val),
                warn_singular=False,
            )
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Densité", fontsize=8)
        ax.tick_params(labelsize=7)
        _poser_legende(ax, title="Statut", fontsize=7)
        ax.grid(alpha=0.25)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"KDE Reste vs Part ({cible})", fontsize=13)
    fig.tight_layout()
    plt.show()

    if ignorees:
        print(
            "KDE ignoré (variable constante ou sans variance exploitable) : "
            + ", ".join(ignorees)
        )




def _plot_taux_depart_qualitatif(df: pd.DataFrame, colonnes: list[str], cible: str, ncols: int = 3) -> None:
    """Taux de départ par modalité (genre, heures sup, poste…)."""
    utiles = [c for c in colonnes if c in df.columns]
    if not utiles:
        return

    nrows = int(np.ceil(len(utiles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    y = pd.to_numeric(df[cible], errors="coerce")

    for i, col in enumerate(utiles):
        ax = axes[i]
        tmp = pd.DataFrame({"mod": df[col].astype(str), "y": y}).dropna()
        if tmp.empty:
            ax.axis("off")
            continue
        taux = tmp.groupby("mod")["y"].mean().sort_values(ascending=False)
        if len(taux) > 12:
            taux = taux.head(12)
        ax.barh(taux.index.astype(str)[::-1], taux.values[::-1], color="#DD8452")
        ax.set_xlim(0, 1)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Taux de départ", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(utiles), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Taux de départ par modalité ({cible})", fontsize=12)
    fig.tight_layout()
    plt.show()


def _matrices_pearson_spearman(
    df: pd.DataFrame,
    colonnes: Sequence[str],
    cible: str | None = None,
    nom: str = "Attrition",
) -> dict[str, pd.DataFrame] | None:
    """Deux heatmaps côte à côte : Pearson | Spearman, cible incluse si fournie."""
    cols = [c for c in colonnes if c in df.columns]
    if cible and cible in df.columns and cible not in cols:
        cols = cols + [cible]
    if len(cols) < 2:
        return None

    df_q = df[cols].apply(_vers_numerique)
    df_q = df_q.loc[:, df_q.nunique(dropna=True) > 1]
    if cible and cible in df_q.columns:
        autres = [c for c in df_q.columns if c != cible]
        df_q = df_q[autres + [cible]]
    if df_q.shape[1] < 2:
        return None

    corr_p = df_q.corr(method="pearson")
    corr_s = df_q.corr(method="spearman")

    print(f"\n--- Matrices Pearson et Spearman côte à côte ({nom}) ---")
    cote = max(7, min(14, 0.48 * df_q.shape[1] + 3.5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(cote * 2, cote))
    extra = f"cible : {cible}" if cible else ""
    _dessiner_heatmap_correlation(ax1, corr_p, "pearson", nom, titre_extra=extra)
    _dessiner_heatmap_correlation(ax2, corr_s, "spearman", nom, titre_extra=extra)
    fig.suptitle(
        f"Corrélations Pearson (gauche) et Spearman (droite) — {nom}",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()
    return {"pearson": corr_p, "spearman": corr_s}


_LIBELLES_AXES_NUAGES = {
    "nombre_participation_pee": "Participations\nPEE",
    "nb_formations_suivies": "Formations\nsuivies",
    "distance_domicile_travail": "Distance\ndomicile–travail",
    "niveau_education": "Niveau\nd'éducation",
    "annees_depuis_la_derniere_promotion": "Ans depuis\nla promotion",
    "annees_sous_responsable_actuel": "Ans sous\nle responsable",
    "annees_dans_l_entreprise": "Ans dans\nl'entreprise",
    "annees_dans_le_poste_actuel": "Ans dans\nle poste",
    "annee_experience_totale": "Expérience\ntotale (ans)",
    "annees_experience_totale": "Expérience\ntotale (ans)",
    "revenu_mensuel": "Revenu\nmensuel",
    "nombre_experiences_precedentes": "Expériences\nprécédentes",
    "heure_supplementaires": "Heures\nsupplémentaires",
    "niveau_hierarchique_poste": "Niveau\nhiérarchique",
    "satisfaction_employee_environnement": "Satisfaction\nenvironnement",
    "satisfaction_employee_nature_travail": "Satisfaction\ntravail",
    "satisfaction_employee_equilibre_pro_perso": "Équilibre\npro / perso",
}


def _libelle_axe(nom: str, largeur: int = 18) -> str:
    """Libellé court et lisible pour les axes du pairplot."""
    import textwrap

    if nom in _LIBELLES_AXES_NUAGES:
        return _LIBELLES_AXES_NUAGES[nom]
    nom = nom.replace("_", " ").strip()
    return "\n".join(textwrap.wrap(nom, width=largeur)) or nom


def plot_matrice_nuages(
    df: pd.DataFrame,
    colonnes: Sequence[str],
    cible: str | None = None,
    max_vars: int = 6,
    nom: str = "corrélations",
) -> None:
    """Matrice de nuages (pairplot) : diagonale = distribution, hors diagonale = nuage."""
    cols_brutes = [c for c in colonnes if c in df.columns]
    if cible and cible in cols_brutes:
        cols_brutes = [c for c in cols_brutes if c != cible]
    if len(cols_brutes) < 2:
        return

    data = df[cols_brutes + ([cible] if cible and cible in df.columns else [])].copy()
    for c in cols_brutes:
        data[c] = _vers_numerique(data[c])
    cols = cols_brutes[:max_vars]
    data = data[cols + ([cible] if cible and cible in data.columns else [])].dropna()
    if len(data) < 20 or data[cols].shape[1] < 2:
        print("(Matrice de nuages : pas assez de données.)")
        return

    hue = None
    if cible and cible in data.columns:
        data["Départ"] = data[cible].map(_label_statut)
        hue = "Départ"

    print(f"\n--- Matrice de nuages ({nom}) : {', '.join(cols)} ---")
    print("Diagonale = répartition de chaque variable. Hors diagonale = lien deux à deux.")
    if hue:
        print("Couleur orange = Part.  Couleur bleue-gris = Reste.")

    alias = {c: _libelle_axe(c) for c in cols}
    data_plot = data.rename(columns=alias)
    vars_plot = [alias[c] for c in cols]
    if hue:
        data_plot[hue] = data[hue]

    grid = sns.pairplot(
        data_plot,
        vars=vars_plot,
        hue=hue,
        hue_order=["Reste", "Part"] if hue else None,
        corner=False,
        diag_kind="hist",
        plot_kws={"alpha": 0.35, "s": 18, "edgecolor": "none"},
        diag_kws={"alpha": 0.7},
        height=2.7,
        palette={"Reste": "#7BA3C9", "Part": "#E67E22"} if hue else None,
    )
    leg = getattr(grid, "_legend", None) or getattr(grid, "legend", None)
    if leg is not None:
        leg.set_title("Départ")
        try:
            leg.set_bbox_to_anchor((1.02, 1))
        except Exception:
            pass
        leg.set_frame_on(True)
    for ax in grid.axes[-1, :]:
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=7)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
    for ax in grid.axes[:, 0]:
        plt.setp(ax.get_yticklabels(), fontsize=7)
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)
    for ax in grid.axes.ravel():
        if ax is not None:
            ax.tick_params(labelsize=7)
    grid.fig.suptitle(f"Nuages de points — {nom}", y=1.02, fontsize=13)
    grid.fig.tight_layout()
    grid.fig.subplots_adjust(top=0.93, bottom=0.10, left=0.10, right=0.88)
    plt.show()


def _noms_features_metier() -> list[str]:
    """Liste des features créées dans new_features.EXPLICATIONS_FEATURES."""
    import importlib

    for nom in ("new_features", "exploration.new_features", "features.new_features"):
        try:
            module = importlib.import_module(nom)
            expl = getattr(module, "EXPLICATIONS_FEATURES", None)
            if isinstance(expl, dict):
                return list(expl.keys())
        except Exception:
            continue
    try:
        from .new_features import EXPLICATIONS_FEATURES as expl
        if isinstance(expl, dict):
            return list(expl.keys())
    except Exception:
        pass
    return []


def _colonnes_features_metier_dans_df(df: pd.DataFrame) -> list[str]:
    """Noms métier réellement présents dans df (gère éventuellement le suffixe _value)."""
    presentes: list[str] = []
    vus: set[str] = set()
    for nom in _noms_features_metier():
        for candidat in (nom, f"{nom}_value"):
            if candidat in df.columns and candidat not in vus:
                presentes.append(candidat)
                vus.add(candidat)
                break
    return presentes


def _filtrer_classification(
    classif: Mapping[str, list[str]],
    colonnes: Sequence[str] | None,
) -> dict[str, list[str]]:
    """Restreint chaque groupe de classification à une liste de colonnes."""
    if colonnes is None:
        return {k: list(v) for k, v in classif.items()}
    autorisees = set(colonnes)
    return {k: [c for c in v if c in autorisees] for k, v in classif.items()}


def analyser_correlations_vs_cible(
    df: pd.DataFrame,
    target_col: str | None = None,
    nom: str = "Attrition",
    colonnes: Sequence[str] | None = None,
) -> dict | None:
    """Étude des quantitatives CONTRE le départ (bouton Attrition globale).

    Test de forme : logit(cible) ~ x + x²  (exploration_y / features).

    `colonnes` : si fourni, n'analyse que ces variables. Absent = analyse
    complète (comportement historique, utilisé ailleurs dans le projet).
    """
    cible = target_col or _trouver_colonne(df, CANDIDATS_CIBLE)
    if not cible or cible not in df.columns:
        print("❌ Colonne d'attrition introuvable : pas d'étude vs y.")
        return None

    from .exploration_y import (
        classifier_et_detecter,
        afficher_classification,
        detecter_non_linearite,
        plot_derivees,
        tester_hypothese_attrition,
    )

    df_lin = _encoder_cible(df, cible)
    auto = detecter_colonnes_sequentielles(df_lin)
    exclus = set(auto) | set(_colonnes_a_exclure(df_lin))

    classif = classifier_et_detecter(df_lin, cible)
    classif["lineaires"] = [c for c in classif["lineaires"] if c not in exclus]
    classif["non_lineaires"] = [c for c in classif["non_lineaires"] if c not in exclus]
    classif = _filtrer_classification(classif, colonnes)
    if colonnes is not None:
        print(f"• Analyse restreinte à {len(colonnes)} colonne(s) : {list(colonnes)}")
        if not any(classif.get(k) for k in ("lineaires", "non_lineaires", "qualitatives", "booleennes")):
            print("❌ Aucune des colonnes demandées n'est analysable (absente ou exclue).")
            return None

    print(f"\n--- Forme de chaque quantitative vs '{cible}' (logit ~ x + x²) ---")
    afficher_classification(classif)

    _nl, pvals, models = detecter_non_linearite(
        df_lin, cible, classif["non_lineaires"] + classif["lineaires"]
    )
    if pvals:
        print("\n--- p-value du terme x² (plus petit = plus courbe vs le départ) ---")
        for col, p in sorted(pvals.items(), key=lambda kv: kv[1]):
            print(f"  • {col}: p = {p:.4g}")

    if classif["non_lineaires"]:
        print("\n--- Dérivée du risque de départ (variables courbes) ---")
        plot_derivees(classif["non_lineaires"], models, ncols=2)

    cols_p = [c for c in classif["lineaires"] if c in df_lin.columns]
    cols_s = [c for c in classif["non_lineaires"] if c in df_lin.columns]

    _matrices_pearson_spearman(
        df_lin,
        cols_p + cols_s,
        cible=cible,
        nom=nom,
    )

    print(f"\n--- Lien de chaque quantitative avec '{cible}' ---")
    lignes = []
    for col in cols_p:
        r = _correlation_sure(df_lin[col], df_lin[cible], method="pearson")
        lignes.append({"colonne": col, "forme": "linéaire", "coef": r, "methode": "Pearson r"})
    for col in cols_s:
        r = _correlation_sure(df_lin[col], df_lin[cible], method="spearman")
        lignes.append({"colonne": col, "forme": "non-linéaire", "coef": r, "methode": "Spearman ρ"})

    liens = pd.DataFrame(lignes)
    if not liens.empty:
        liens = (
            liens.dropna(subset=["coef"])
            .sort_values("coef", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True)
        )

    if liens.empty:
        print("Aucun coefficient calculable.")
    else:
        _afficher(liens.round(3))
        figb, axb = plt.subplots(figsize=(9, max(3, 0.35 * len(liens))))
        couleurs = ["#4C72B0" if f == "linéaire" else "#DD8452" for f in liens["forme"].iloc[::-1]]
        axb.barh(liens["colonne"].iloc[::-1], liens["coef"].iloc[::-1], color=couleurs)
        axb.axvline(0, color="black", linewidth=0.8)
        axb.set_xlabel("Corrélation avec le départ")
        axb.set_title(f"Quantitatives × {cible}  (bleu = droite, orange = courbe)")
        figb.tight_layout()
        plt.show()
        top_nuages = liens["colonne"].head(6).tolist()
        plot_matrice_nuages(df_lin, top_nuages, cible=cible, nom=nom)

    cats = [
        c
        for c in classif["booleennes"] + classif["qualitatives"]
        if c not in exclus and c in df_lin.columns and c != cible
    ]
    lignes_cat = []
    for col in cats:
        n_mod = df_lin[col].nunique(dropna=True)
        if n_mod <= 1:
            continue
        if n_mod == 2:
            codes = pd.Series(
                pd.factorize(df_lin[col], sort=True)[0], index=df_lin.index
            ).replace(-1, np.nan)
            r = _correlation_sure(codes, df_lin[cible], method="pearson")
            lignes_cat.append(
                {"colonne": col, "forme": "binaire / 2 modalités", "coef": r, "methode": "point-bisériel"}
            )
        else:
            v = _cramer_v(df_lin[col], df_lin[cible])
            lignes_cat.append(
                {"colonne": col, "forme": "qualitative", "coef": v, "methode": "Cramér V"}
            )

    liens_cat = pd.DataFrame(lignes_cat)
    if not liens_cat.empty:
        liens_cat = (
            liens_cat.dropna(subset=["coef"])
            .sort_values("coef", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True)
        )
        print(f"\n--- Lien des qualitatives / binaires avec '{cible}' ---")
        print("(point-bisériel signé pour 2 modalités, Cramér V entre 0 et 1 pour le reste)")
        _afficher(liens_cat.round(3))
        figc, axc = plt.subplots(figsize=(9, max(3, 0.35 * len(liens_cat))))
        axc.barh(liens_cat["colonne"].iloc[::-1], liens_cat["coef"].iloc[::-1], color="#55A868")
        axc.axvline(0, color="black", linewidth=0.8)
        axc.set_xlabel("Association avec le départ")
        axc.set_title(f"Qualitatives / binaires × {cible}")
        figc.tight_layout()
        plt.show()

    print(f"\n--- KDE : quantitatives selon '{cible}' (Reste vs Part) ---")
    _plot_kde_vs_cible(df_lin, cols_p + cols_s, cible)
    print(f"\n--- Taux de départ par modalité (qualitatives / binaires) ---")
    _plot_taux_depart_qualitatif(df_lin, cats, cible)
    tests = tester_hypothese_attrition(df_lin, cible)

    return {
        "lineaires": cols_p,
        "non_lineaires": cols_s,
        "qualitatives": cats,
        "cible": cible,
        "liens": liens,
        "liens_cat": liens_cat if not liens_cat.empty else pd.DataFrame(),
        "tests_hypothese": tests,
        "nom": nom,
    }


def analyser_dataframe(
    df: pd.DataFrame,
    nom: str = "dataframe",
    n_head: int = 5,
) -> None:
    print(f"\n{'=' * 40}\nANALYSE GLOBALE : {nom}\n{'=' * 40}")
    print(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")
    _afficher(df.head(n_head))
    _afficher(df.describe(include="all"))
    matrice_correlation(df, nom=nom)


def afficher_correlations_triees(corr: pd.DataFrame) -> pd.DataFrame:
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    df_pairs = corr.where(mask).stack().reset_index()
    df_pairs.columns = ["Variable 1", "Variable 2", "Coefficient (r)"]
    df_pairs = df_pairs.dropna(subset=["Coefficient (r)"]).reset_index(drop=True)
    df_pairs["abs_r"] = df_pairs["Coefficient (r)"].abs()
    df_pairs = df_pairs.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)
    return df_pairs


# ---------------------------------------------------------------------------
# Fonctions d'exploration globale
# ---------------------------------------------------------------------------
def explorer_sirh(df: pd.DataFrame) -> None:
    tester_remplissage(df, nom="SIRH")
    analyser_dataframe(df, nom="SIRH")
    analyser_quantitatif(df, titre_prefixe="SIRH - QUANTITATIF")
    analyser_qualitatif(df, titre_prefixe="SIRH - QUALITATIF")


def explorer_sondage(df: pd.DataFrame) -> None:
    tester_remplissage(df, nom="Sondage")
    analyser_dataframe(df, nom="Sondage")
    analyser_quantitatif(df, titre_prefixe="Sondage - QUANTITATIF")
    analyser_qualitatif(df, titre_prefixe="Sondage - QUALITATIF")


def explorer_eval(df: pd.DataFrame) -> None:
    tester_remplissage(df, nom="Évaluation")
    analyser_dataframe(df, nom="Évaluation")
    analyser_quantitatif(df, titre_prefixe="Évaluation - QUANTITATIF")
    analyser_qualitatif(df, titre_prefixe="Évaluation - QUALITATIF")


def explorer(
    df_sirh: pd.DataFrame | None = None,
    df_sondage: pd.DataFrame | None = None,
    df_eval: pd.DataFrame | None = None,
) -> None:
    if df_sirh is not None:
        explorer_sirh(df_sirh)
    if df_sondage is not None:
        explorer_sondage(df_sondage)
    if df_eval is not None:
        explorer_eval(df_eval)


def classer_colonnes_sondage(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "numerique": list(dataframe_numerique(df).columns),
        "qualitatif": [c for c in df.columns if c not in dataframe_numerique(df).columns],
    }


def bouton_explorer():
    if widgets is None:
        raise ImportError("ipywidgets est requis pour bouton_explorer().")
    return widgets.Button(
        description="Lancer l'exploration",
        button_style="primary",
        icon="play",
        layout=widgets.Layout(width="280px", height="40px"),
    )


def _est_quantitative_continue(serie: pd.Series) -> bool:
    """Vrai seulement pour une quantitative à beaucoup de valeurs (pas un Likert 1-4)."""
    x = _vers_numerique(serie).dropna()
    if len(x) < 10 or x.nunique() <= 8:
        return False
    return True


def analyser_quantitatif(
    df: pd.DataFrame,
    colonnes: Sequence[str] | None = None,
    seuils: Mapping[str, tuple[float, float]] | None = None,
    titre_prefixe: str = "QUANTITATIF",
    ylabel: str = "Valeurs",
    max_par_figure: int = 6,
    exclure: Iterable[str] = COLONNES_ID_PAR_DEFAUT,
) -> None:
    del seuils, ylabel, max_par_figure

    if colonnes is None:
        colonnes = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) or _vers_numerique(df[c]).notna().mean() >= 0.5
        ]

    colonnes_existantes = [c for c in colonnes if c in df.columns]
    exclus_effectifs = _colonnes_a_exclure(df, exclure)
    colonnes_existantes = _filtrer_colonnes_id(colonnes_existantes, exclus_effectifs)
    colonnes_existantes = [c for c in colonnes_existantes if not est_colonne_sequentielle(df[c])]
    if not colonnes_existantes:
        return

    donnees = {c: _vers_numerique(df[c]).dropna() for c in colonnes_existantes}
    donnees = {c: s for c, s in donnees.items() if len(s) > 0}
    if not donnees:
        return

    print(f"\n--- Boxplots regroupés : {titre_prefixe} ---")
    z = {}
    for c, s in donnees.items():
        std = float(s.std())
        z[c] = (s - float(s.mean())) / std if std else s * 0

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(z) + 4), 6))
    ax.boxplot(
        [z[c] for c in z],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="steelblue"),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    ax.set_xticks(range(1, len(z) + 1))
    ax.set_xticklabels(list(z.keys()))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("z-score")
    ax.set_title(f"Boxplots — {titre_prefixe}")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()

    continues = [c for c in donnees if _est_quantitative_continue(df[c])]
    if not continues:
        print("(Pas d'IQR : aucune quantitative continue, seulement des échelles 1-4 / ordinales.)")
        return

    print(f"\n--- IQR / outliers (quantitatives continues seulement) : {titre_prefixe} ---")
    n = len(continues)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, col in enumerate(continues):
        ax = axes[i // ncols][i % ncols]
        s = donnees[col]
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        b_inf, b_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        hors = (s < b_inf) | (s > b_sup)
        x_n = np.random.normal(0, 0.08, size=int((~hors).sum()))
        x_o = np.random.normal(0, 0.08, size=int(hors.sum()))
        ax.scatter(x_n, s[~hors], s=14, alpha=0.45, color="steelblue", label="Normal")
        if hors.any():
            ax.scatter(x_o, s[hors], s=18, alpha=0.8, color="crimson", label=f"Outlier ({int(hors.sum())})")
        ax.axhline(b_sup, color="crimson", linestyle="--", linewidth=1, label=f"sup {b_sup:.1f}")
        ax.axhline(b_inf, color="crimson", linestyle="--", linewidth=1, label=f"inf {b_inf:.1f}")
        ax.set_title(f"{col}\nIQR = {iqr:.1f}", fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for j in range(len(continues), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig2.suptitle(f"IQR & outliers — quantitatives continues ({titre_prefixe})", fontsize=13)
    fig2.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Analyse qualitative
# ---------------------------------------------------------------------------
def _plot_qualitatives_empilees(df: pd.DataFrame, colonnes: list[str], nom: str = "QUALITATIF") -> None:
    """Un seul graphique : composition (%) de chaque qualitative."""
    utiles = []
    parts = []
    for col in colonnes:
        if col not in df.columns:
            continue
        vc = df[col].astype(str).replace({"": "(vide)", "nan": "(vide)"}).value_counts(dropna=False)
        if vc.empty or vc.size < 2:
            continue
        pct = (vc / vc.sum() * 100).sort_values(ascending=False)
        utiles.append(col)
        parts.append(pct)
    if not utiles:
        return

    print(f"\n--- Composition des qualitatives : {nom} ---")
    for col, pct in zip(utiles, parts):
        detail = ", ".join(f"{mod} {val:.1f}%" for mod, val in pct.items())
        print(f"  • {col} ({len(pct)} modalités) : {detail}")

    simples = [(c, p) for c, p in zip(utiles, parts) if len(p) <= 4]
    detaillees = [(c, p) for c, p in zip(utiles, parts) if len(p) > 4]
    n_det = max(len(detaillees), 1)
    haut_haut = max(2.8, 0.7 * max(len(simples), 1) + 1.2)
    haut_bas = max(3.5, 0.38 * max((len(p) for _, p in detaillees), default=4) + 1.2)

    fig = plt.figure(figsize=(16, haut_haut + (haut_bas if detaillees else 0)))
    if simples and detaillees:
        gs = fig.add_gridspec(2, n_det, height_ratios=[haut_haut, haut_bas], hspace=0.45, wspace=0.35)
        ax_stack = fig.add_subplot(gs[0, :])
        axes_det = [fig.add_subplot(gs[1, i]) for i in range(n_det)]
    elif simples:
        ax_stack = fig.add_subplot(1, 1, 1)
        axes_det = []
    else:
        ax_stack = None
        axes_det = [fig.add_subplot(1, n_det, i + 1) for i in range(n_det)]

    if ax_stack is not None and simples:
        palette = sns.color_palette("Set2", 8)
        for i, (col, pct) in enumerate(simples):
            gauche = 0.0
            for j, (mod, val) in enumerate(pct.items()):
                ax_stack.barh(i, val, left=gauche, color=palette[j % len(palette)], height=0.7, edgecolor="white")
                if val >= 3:
                    ax_stack.text(
                        gauche + val / 2,
                        i,
                        f"{mod}\n{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
                gauche += float(val)
        ax_stack.set_yticks(range(len(simples)))
        ax_stack.set_yticklabels([c for c, _ in simples])
        ax_stack.set_xlim(0, 100)
        ax_stack.set_xlabel("Part des salariés (%)")
        ax_stack.set_title("Composition (variables à peu de modalités)")
        ax_stack.grid(axis="x", alpha=0.25)

    for ax2, (col, pct) in zip(axes_det, detaillees):
        ordre = pct.sort_values(ascending=True)
        couleurs = sns.color_palette("tab20", len(ordre))
        ax2.barh(ordre.index.astype(str), ordre.values, color=couleurs)
        for y, val in enumerate(ordre.values):
            ax2.text(val + 0.3, y, f"{val:.1f}%", va="center", fontsize=8)
        ax2.set_xlabel("%")
        ax2.set_xlim(0, float(ordre.max()) + 8)
        ax2.set_title(col)
        ax2.grid(axis="x", alpha=0.25)

    fig.suptitle(f"Qualitatives — {nom}", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.08, hspace=0.45, wspace=0.35)
    plt.show()


def analyser_qualitatif(
    df: pd.DataFrame,
    colonnes: Sequence[str] | None = None,
    titre_prefixe: str = "QUALITATIF",
    titre_graphique: str = "Répartition",
    max_par_figure: int = 6,
    exclure: Iterable[str] = COLONNES_ID_PAR_DEFAUT,
) -> None:
    del titre_graphique, max_par_figure
    if colonnes is None:
        colonnes = [
            c
            for c in df.columns
            if df[c].dtype == "object" or str(df[c].dtype).startswith("str") or df[c].nunique() <= 15
        ]
    colonnes_existantes = [c for c in colonnes if c in df.columns]
    exclus_effectifs = _colonnes_a_exclure(df, exclure)
    colonnes_existantes = _filtrer_colonnes_id(colonnes_existantes, exclus_effectifs)
    colonnes_existantes = [c for c in colonnes_existantes if not est_colonne_sequentielle(df[c])]
    if not colonnes_existantes:
        return
    _plot_qualitatives_empilees(df, colonnes_existantes, nom=titre_prefixe)


def _analyser_qualitatif_lot(
    df: pd.DataFrame,
    colonnes_existantes: list[str],
    titre_prefixe: str,
    titre_graphique: str,
) -> None:
    del titre_prefixe
    fig, axes = plt.subplots(
        len(colonnes_existantes),
        2,
        figsize=(14, 4 * len(colonnes_existantes)),
        squeeze=False,
    )
    for i, col in enumerate(colonnes_existantes):
        serie = df[col]
        nb_nulls = int(serie.isna().sum())
        nb_vides = int((serie.astype(str).str.strip() == "").sum()) if serie.dtype == "object" else 0
        effectifs = serie.value_counts(dropna=False)
        top_valeurs = effectifs.head(2).to_dict()
        sns.countplot(
            data=df,
            x=col,
            ax=axes[i, 0],
            palette="Set2",
            hue=col,
            legend=False,
            order=effectifs.index,
        )
        axes[i, 0].set_title(f"{titre_graphique} : {col}", fontsize=14)
        axes[i, 0].tick_params(axis="x", rotation=45)
        axes[i, 0].grid(axis="y", alpha=0.3)
        axes[i, 1].axis("off")
        top_str = ", ".join([f"'{k}' ({v})" for k, v in top_valeurs.items()])
        stats_text = (
            f"--- Analyse Fréquentielle ---\n\n"
            f"• Valeurs nulles (NaN) : {nb_nulls}\n"
            f"• Valeurs vides ('') : {nb_vides}\n"
            f"• Catégories : {len(effectifs)}\n"
            f"• Majoritaires :\n  -> {top_str}"
        )
        axes[i, 1].text(
            0.05,
            0.5,
            stats_text,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="whitesmoke", edgecolor="gray", alpha=0.9),
        )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Analyses spécifiques
# ---------------------------------------------------------------------------
def _trouver_colonne(df: pd.DataFrame, candidats: Sequence[str]) -> str | None:
    lookup = {c.lower().strip(): c for c in df.columns}
    for nom in candidats:
        if nom.lower() in lookup:
            return lookup[nom.lower()]
    return None


def _ajouter_depart(df_eval: pd.DataFrame, df_sirh: pd.DataFrame | None) -> tuple[pd.DataFrame, str | None]:
    candidats_depart = ("depart", "départ", "a_quitte", "attrition", "quitte_entreprise")
    col_depart = _trouver_colonne(df_eval, candidats_depart)
    if col_depart or df_sirh is None:
        return df_eval, col_depart

    col_depart = _trouver_colonne(df_sirh, candidats_depart)
    if not col_depart:
        return df_eval, None

    cles = ("id_employee", "id_employe", "identifiant", "id")
    cle_eval = _trouver_colonne(df_eval, cles)
    cle_sirh = _trouver_colonne(df_sirh, cles)
    if not cle_eval or not cle_sirh:
        return df_eval, None

    extra = df_sirh[[cle_sirh, col_depart]].drop_duplicates(subset=[cle_sirh])
    return df_eval.merge(extra, left_on=cle_eval, right_on=cle_sirh, how="left"), col_depart


def comparer_notes_evaluation(df_eval: pd.DataFrame, df_sirh: pd.DataFrame | None = None) -> None:
    col_prec = "note_evaluation_precedente"
    col_actu = "note_evaluation_actuelle"
    if col_prec not in df_eval.columns or col_actu not in df_eval.columns:
        print("Colonnes de notes d'évaluation introuvables.")
        return

    df = df_eval.copy()
    df, _col_depart = _ajouter_depart(df, df_sirh)
    df[col_prec + "_num"] = _vers_numerique(df[col_prec])
    df[col_actu + "_num"] = _vers_numerique(df[col_actu])

    plt.figure(figsize=(12, 6))
    series = (
        (col_prec + "_num", "Note précédente", "skyblue"),
        (col_actu + "_num", "Note actuelle", "orange"),
    )
    for col_note, label, color in series:
        s = df[col_note].dropna()
        if _serie_kde_ok(s):
            sns.kdeplot(
                s,
                label=label,
                fill=True,
                color=color,
                alpha=0.4,
                linewidth=2,
                warn_singular=False,
            )
        elif len(s) > 0:
            plt.axvline(
                float(s.median()),
                linestyle="--",
                color=color,
                linewidth=2,
                label=f"{label} (constante, n={len(s)})",
            )

    plt.title("Comparaison des courbes de répartition (KDE)", fontsize=14)
    plt.xlabel("Valeur de la note")
    plt.ylabel("Densité")
    handles, _labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyser_attrition_croisee(
    df: pd.DataFrame,
    colonnes_ref: Sequence[str] | None = None,
    col_cible: str | None = None,
) -> None:
    """
    Affiche une seule fois la répartition globale de l'attrition,
    puis génère une grille de graphiques pour croiser plusieurs variables.
    """
    if col_cible is None:
        col_cible = _trouver_colonne(df, CANDIDATS_CIBLE)
    if not col_cible or col_cible not in df.columns:
        print("❌ Colonne d'attrition (départ) introuvable dans ce DataFrame.")
        return

    if colonnes_ref is None:
        candidats_defaut = [
            "revenu_mensuel",
            "age",
            "annees_dans_l_entreprise",
            "annees_dans_le_poste_actuel",
            "distance_domicile_travail",
            "annees_depuis_la_derniere_promotion",
            "heure_supplementaires",
            "satisfaction_employee_environnement",
            "satisfaction_employee_equilibre_pro_perso",
            "note_evaluation_actuelle",
        ]
        colonnes_ref = [c for c in candidats_defaut if c in df.columns]

    print(f"\n{'=' * 60}")
    print(f"ANALYSE MULTI-CRITÈRES DE L'ATTRITION (Cible : '{col_cible}')")
    print(f"{'=' * 60}")

    effectifs_bruts = df[col_cible].value_counts(dropna=False)
    pourcentages = df[col_cible].value_counts(normalize=True, dropna=False) * 100
    res_synthese = pd.DataFrame({"Effectif": effectifs_bruts, "Pourcentage (%)": pourcentages.round(2)})
    _afficher(res_synthese)


def analyser_attrition_globale(
    df_sirh: pd.DataFrame,
    df_sondage: pd.DataFrame,
    df_eval: pd.DataFrame,
) -> None:
    """Fusionne temporairement les dataframes, lance l'analyse d'attrition, puis nettoie la mémoire."""
    print("🚀 Fusion temporaire des sources (SIRH + Sondage + Éval)...")
    df_merged = df_sirh.copy()

    if df_sondage is not None:
        if "id_employee" in df_merged.columns and "id_employee" in df_sondage.columns:
            df_merged = df_merged.merge(
                df_sondage, on="id_employee", how="inner", suffixes=("", "_sondage")
            )
        elif "id_employee" in df_merged.columns and "code_sondage" in df_sondage.columns:
            df_merged = df_merged.merge(
                df_sondage,
                left_on="id_employee",
                right_on="code_sondage",
                how="inner",
                suffixes=("", "_sondage"),
            )
        else:
            df_merged = pd.concat(
                [df_merged, df_sondage.loc[:, ~df_sondage.columns.isin(df_merged.columns)]],
                axis=1,
            )

    if df_eval is not None:
        if "id_employee" in df_merged.columns and "id_employee" in df_eval.columns:
            df_merged = df_merged.merge(df_eval, on="id_employee", how="inner", suffixes=("", "_eval"))
        else:
            df_merged = pd.concat(
                [df_merged, df_eval.loc[:, ~df_eval.columns.isin(df_merged.columns)]],
                axis=1,
            )

    analyser_correlations_vs_cible(df_merged, nom="Attrition globale")
    analyser_attrition_croisee(df_merged)

    del df_merged
    print("🧹 Mémoire nettoyée : le DataFrame fusionné a été supprimé.")


# Variables sources à recroiser avec les features métier (bloc carrière + signaux hors bloc).
_ANCIENNES_POUR_CROISEMENT: tuple[str, ...] = (
    "age",
    "revenu_mensuel",
    "annee_experience_totale",
    "annees_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "annees_sous_responsable_actuel",
    "niveau_hierarchique_poste",
    "annees_depuis_la_derniere_promotion",
    "nb_formations_suivies",
    "nombre_participation_pee",
    "note_evaluation_precedente",
    "satisfaction_employee_nature_travail",
    "satisfaction_employee_environnement",
    "heure_supplementaires",
    "distance_domicile_travail",
)


def _verdict_conservation(
    lien_y: float,
    forme: str,
    max_abs_r: float,
    jumeau: str | None,
) -> tuple[str, str]:
    """Règle unique : lié à Y + peu lié aux sources → garder."""
    ly = abs(lien_y) if lien_y is not None and np.isfinite(lien_y) else 0.0
    mr = max_abs_r if max_abs_r is not None and np.isfinite(max_abs_r) else 0.0
    nom_jumeau = jumeau or "une variable source"

    if forme == "qualitative" and ly >= 0.15:
        return "GARDER", "Cramér V utile : ce n'est pas un doublon numérique"
    if ly < 0.08:
        if mr >= 0.70:
            return "RETIRER", f"lien avec Y plat et copie de {nom_jumeau}"
        return "RETIRER", "presque aucun lien avec le départ"
    if mr >= 0.70:
        return "RETIRER", f"redondant avec {nom_jumeau} (ρ={mr:.2f})"
    if mr >= 0.55:
        return "OPTIONNEL", f"proche de {nom_jumeau} (ρ={mr:.2f}) : n'en garder qu'une"
    return "GARDER", "lié au départ et peu redondant avec les sources"


def tableau_decision_features(
    tableau_y: pd.DataFrame,
    extraire: pd.DataFrame | None,
    cible: str,
) -> pd.DataFrame:
    """Une ligne par feature métier : lien Y, doublon éventuel, décision."""
    lignes = []
    for _, row in tableau_y.iterrows():
        feat = row["feature"]
        lien_y = float(row["lien_avec_Y"]) if pd.notna(row["lien_avec_Y"]) else float("nan")
        max_abs_r, jumeau = float("nan"), None
        if extraire is not None and feat in extraire.index:
            serie = extraire.loc[feat].drop(labels=[cible], errors="ignore").abs()
            if not serie.empty and serie.notna().any():
                jumeau = str(serie.idxmax())
                max_abs_r = float(serie.max())
        decision, motif = _verdict_conservation(lien_y, row["forme"], max_abs_r, jumeau)
        lignes.append(
            {
                "feature": feat,
                "lien_avec_Y": lien_y,
                "mesure": row["methode_Y"],
                "plus_proche": jumeau,
                "ρ_plus_proche": max_abs_r,
                "décision": decision,
                "pourquoi": motif,
            }
        )
    out = pd.DataFrame(lignes)
    ordre = {"GARDER": 0, "OPTIONNEL": 1, "RETIRER": 2}
    if not out.empty:
        out["_ord"] = out["décision"].map(ordre)
        out["_abs"] = out["lien_avec_Y"].abs()
        out = out.sort_values(["_ord", "_abs"], ascending=[True, False])
        out = out.drop(columns=["_ord", "_abs"]).reset_index(drop=True)
    return out


def comparer_nouvelles_et_anciennes(
    df: pd.DataFrame,
    target_col: str,
    nom: str = "nouvelles × anciennes",
) -> pd.DataFrame | None:
    """Petite section : lien des features métier avec Y et avec les variables sources."""
    nouvelles = _colonnes_features_metier_dans_df(df)
    anciennes = [c for c in _ANCIENNES_POUR_CROISEMENT if c in df.columns and c not in nouvelles]
    if not nouvelles:
        print("⚠️ Pas de features métier dans le DataFrame : section nouvelles × anciennes ignorée.")
        return None

    print("\n" + "=" * 60)
    print("NOUVELLES FEATURES × ANCIENNES VARIABLES")
    print("=" * 60)
    print(f"• Nouvelles ({len(nouvelles)}) : {nouvelles}")
    print(f"• Anciennes retenues ({len(anciennes)}) : {anciennes}")

    y = _vers_numerique(df[target_col])
    lignes = []
    for col in nouvelles:
        serie = df[col]
        n_mod = int(serie.nunique(dropna=True))
        est_texte = (
            serie.dtype == "object"
            or str(serie.dtype).startswith("str")
            or pd.api.types.is_string_dtype(serie)
        )
        if est_texte or (n_mod > 2 and n_mod <= 25 and not pd.api.types.is_numeric_dtype(serie)):
            coef = _cramer_v(serie, df[target_col])
            methode, forme = "Cramér V", "qualitative"
        elif n_mod <= 2:
            codes = pd.Series(pd.factorize(serie, sort=True)[0], index=df.index).replace(-1, np.nan)
            coef = _correlation_sure(codes, y, method="pearson")
            methode, forme = "point-bisériel", "binaire"
        else:
            coef = _correlation_sure(_vers_numerique(serie), y, method="spearman")
            methode, forme = "Spearman ρ", "numérique"
        lignes.append({"feature": col, "lien_avec_Y": coef, "methode_Y": methode, "forme": forme})

    tableau_y = pd.DataFrame(lignes)
    if not tableau_y.empty:
        tableau_y = tableau_y.sort_values("lien_avec_Y", key=lambda s: s.abs(), ascending=False)
        print("\n--- Nouvelles features × cible ---")
        _afficher(tableau_y.round(3).reset_index(drop=True))

        fig, ax = plt.subplots(figsize=(9, max(3.2, 0.38 * len(tableau_y))))
        couleurs = [
            "#55A868" if f == "qualitative" else ("#4C72B0" if f == "binaire" else "#DD8452")
            for f in tableau_y["forme"].iloc[::-1]
        ]
        ax.barh(tableau_y["feature"].iloc[::-1], tableau_y["lien_avec_Y"].iloc[::-1], color=couleurs)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lien avec le départ")
        ax.set_title("Nouvelles features × Y")
        fig.tight_layout()
        plt.show()

    extraire = None
    # Heatmap nouvelles (lignes) × (Y + anciennes numériques) (colonnes)
    cols_num_anciennes = []
    for c in anciennes:
        s = _vers_numerique(df[c])
        if s.notna().mean() >= 0.5 and int(s.nunique(dropna=True)) >= 3:
            cols_num_anciennes.append(c)

    nouvelles_num = []
    for c in nouvelles:
        s = _vers_numerique(df[c])
        if s.notna().mean() >= 0.5 and int(s.nunique(dropna=True)) >= 3:
            nouvelles_num.append(c)

    if nouvelles_num and (cols_num_anciennes or target_col in df.columns):
        bloc = nouvelles_num + cols_num_anciennes + [target_col]
        bloc = list(dict.fromkeys(bloc))
        data = pd.DataFrame({c: _vers_numerique(df[c]) for c in bloc})
        corr = data.corr(method="spearman")
        extraire = corr.loc[nouvelles_num, [c for c in ([target_col] + cols_num_anciennes) if c in corr.columns]]

        print("\n--- Spearman : nouvelles (lignes) × anciennes + Y (colonnes) ---")
        print("Case foncée = la feature métier redit la même chose que la variable source.")
        print("Case pâle + lien avec Y = signal complémentaire (à garder).")
        _afficher(extraire.round(2))

        fig2, ax2 = plt.subplots(figsize=(max(8, 0.55 * extraire.shape[1] + 4), max(3.5, 0.42 * extraire.shape[0] + 1.8)))
        sns.heatmap(
            extraire,
            ax=ax2,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            linewidths=0.4,
            linecolor="white",
        )
        ax2.set_title(f"Nouvelles × anciennes — {nom}")
        plt.setp(ax2.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
        fig2.tight_layout()
        plt.show()

    if not tableau_y.empty:
        decision = tableau_decision_features(tableau_y, extraire, target_col)
        print("\n--- Décision simple : garder ou retirer ---")
        print("GARDER     = lié à Y et pas une copie d'une variable déjà là")
        print("OPTIONNEL  = un peu utile mais proche d'une source : n'en garder qu'une")
        print("RETIRER    = lien plat avec Y, ou doublon (ρ ≥ 0,70 avec une ancienne)")
        _afficher(decision.round(2))
        print("\nÀ cocher ON  :", decision.loc[decision["décision"] == "GARDER", "feature"].tolist())
        print("À discuter    :", decision.loc[decision["décision"] == "OPTIONNEL", "feature"].tolist())
        print("À décocher    :", decision.loc[decision["décision"] == "RETIRER", "feature"].tolist())

    return tableau_y if not tableau_y.empty else None


def analyser_features(
    df: pd.DataFrame,
    uniquement_nouvelles: bool = False,
    colonnes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Analyse d'attrition sur df_clean (cible encodée si besoin).

    Par défaut : analyse complète (anciennes + nouvelles), puis une section
    dédiée « nouvelles × anciennes ».
    uniquement_nouvelles=True : saute l'analyse globale et ne garde que
    les features métier + le croisement.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print("❌ df_clean est vide ou introuvable. Lance d'abord le nettoyage / fusion.")
        return df

    print(f"DataFrame reçu : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    cible = _trouver_colonne(df, CANDIDATS_CIBLE)
    if not cible:
        print("❌ Colonne d'attrition introuvable.")
        return df

    df_travail = _encoder_cible(df, cible)
    print(f"Cible utilisée : '{cible}'  (0 = reste, 1 = part)")

    if colonnes is None and uniquement_nouvelles:
        colonnes = _colonnes_features_metier_dans_df(df_travail)
        if colonnes:
            print("• Périmètre : nouvelles features métier uniquement")
        else:
            print("⚠️ Aucune feature métier trouvée dans le DataFrame → analyse complète.")
            colonnes = None

    nom = "nouvelles features" if colonnes is not None else "df_clean"
    analyser_correlations_vs_cible(
        df_travail, target_col=cible, nom=nom, colonnes=colonnes
    )
    comparer_nouvelles_et_anciennes(df_travail, target_col=cible, nom=nom)
    return df_travail


def bouton_analyse_features(
    get_current_df: Callable[[], pd.DataFrame],
    set_current_df: Callable[[pd.DataFrame], None] | None = None,
    uniquement_nouvelles: bool = False,
):
    """Widget : analyse complète, plus la section nouvelles × anciennes.

    uniquement_nouvelles=True : uniquement les features métier + le croisement.
    Les autres appels à exploration.py ne passent pas par ce bouton.
    """
    if widgets is None:
        raise ImportError("ipywidgets est requis pour bouton_analyse_features().")

    try:
        from IPython.display import clear_output
    except ImportError:
        clear_output = lambda wait=False: None  # noqa: E731

    bouton = widgets.Button(
        description="Analyser les features",
        button_style="primary",
        icon="bar-chart",
        layout=widgets.Layout(width="280px", height="40px"),
    )
    sortie = widgets.Output()

    def au_clic(_):
        with sortie:
            clear_output(wait=True)
            df = get_current_df()
            res = analyser_features(df, uniquement_nouvelles=uniquement_nouvelles)
            if set_current_df is not None and res is not None:
                set_current_df(res)

    bouton.on_click(au_clic)
    return widgets.VBox([bouton, sortie])