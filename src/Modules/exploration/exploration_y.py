"""
exploration_y.py
================
Module regroupant les fonctions d'analyse et de classification des variables
pour l'étude de l'attrition (a_quitte_l_entreprise).

Fonctions principales :
- classifier_variables()      : classe les colonnes en booléennes / qualitatives / numériques
- detecter_non_linearite()    : teste la non-linéarité des variables numériques (via terme quadratique)
- classifier_et_detecter()    : pipeline complet (booléennes / qualitatives / linéaires / non-linéaires)
- afficher_classification()   : affichage texte structuré du résultat
- plot_derivees()             : trace les dérivées du risque pour les variables non-linéaires
- tester_hypothese_attrition() : taux de départ par tranches et à niveau de poste comparable
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError
except Exception:  # versions statsmodels un peu anciennes
    ConvergenceWarning = UserWarning  # type: ignore[misc, assignment]
    PerfectSeparationError = Exception  # type: ignore[misc, assignment]


def classifier_variables(df, target_col, id_col="id_employee"):
    """
    Identifie les variables booléennes, qualitatives et numériques d'un DataFrame.

    Retourne un dict avec les clés :
        'booleennes', 'qualitatives', 'numeriques'
    """
    exclure = [target_col, id_col]

    cols_booleennes = []
    for col in df.columns:
        if col in exclure:
            continue
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2 and (
            pd.api.types.is_bool_dtype(df[col])
            or set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False})
        ):
            cols_booleennes.append(col)

    cols_qualitatives = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cols_qualitatives = [c for c in cols_qualitatives if c not in cols_booleennes and c not in exclure]

    cols_numeriques = df.select_dtypes(include=["number"]).columns.drop(exclure, errors="ignore").tolist()
    cols_numeriques = [
        c
        for c in cols_numeriques
        if c not in cols_booleennes and int(pd.Series(df[c]).nunique(dropna=True)) >= 3
    ]

    return {
        "booleennes": cols_booleennes,
        "qualitatives": cols_qualitatives,
        "numeriques": cols_numeriques,
    }


def _preparer_xy(df, col, target_col):
    """Extrait x, y numériques, sans NA, avec assez de variation."""
    temp = df[[col, target_col]].dropna()
    if temp.empty:
        return None

    x = pd.to_numeric(temp[col], errors="coerce")
    y = pd.to_numeric(temp[target_col], errors="coerce")
    masque = x.notna() & y.notna()
    x, y = x[masque], y[masque]

    if len(x) < 30:
        return None
    if int(x.nunique()) < 3:
        return None
    if int(y.nunique()) < 2:
        return None
    if float(x.std(ddof=0)) == 0:
        return None
    # Trop peu d'événements (départs ou restants) → logit instable
    if int(y.value_counts().min()) < 8:
        return None
    return x.astype(float), y.astype(float)


def _standardiser(x: pd.Series):
    mu = float(x.mean())
    sigma = float(x.std(ddof=0))
    if not np.isfinite(sigma) or sigma == 0:
        return None
    z = (x - mu) / sigma
    return z, mu, sigma


def _ajuster_logit_quadratique(z: pd.Series, y: pd.Series):
    """
    Ajuste logit(y) ~ z + z² après standardisation.

    Retourne le modèle seulement s'il a convergé. Sinon None.
    On standardise pour éviter que x et x² (salaire, ancienneté…) fassent
    exploser le Newton-Raphson.
    """
    X = sm.add_constant(np.column_stack((z.to_numpy(), np.square(z.to_numpy()))))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")
        warnings.filterwarnings("ignore", message="overflow encountered")

        model = None
        try:
            model = sm.Logit(y.to_numpy(), X).fit(disp=0, maxiter=200, method="newton")
        except (PerfectSeparationError, np.linalg.LinAlgError, Exception):
            model = None

        converged = bool(getattr(model, "mle_retvals", {}).get("converged", False)) if model is not None else False
        if model is not None and converged:
            return model

        # Repli : pénalisation légère, plus stable en cas de quasi-séparation
        try:
            model_reg = sm.Logit(y.to_numpy(), X).fit_regularized(
                disp=0,
                alpha=1e-3,
                L1_wt=0.0,  # ridge
            )
            # fit_regularized n'a pas toujours mle_retvals['converged']
            if model_reg is not None and np.all(np.isfinite(model_reg.params)):
                return model_reg
        except Exception:
            return None

    return None


def detecter_non_linearite(df, target_col, colonnes_numeriques, p_threshold=0.05):
    """
    Pour chaque colonne numérique, ajuste un modèle logit avec terme quadratique
    (z, z²) après standardisation et considère la variable comme non-linéaire
    si le coefficient du terme quadratique est significatif (p < p_threshold)
    ET si le modèle a convergé.

    Retourne :
        colonnes_non_lineaires : liste des colonnes non-linéaires
        results_non_lineaires  : dict {col: p_value du terme quadratique}
        models_data            : dict {col: (modele, x_brut, mu, sigma)}
    """
    colonnes_non_lineaires = []
    results_non_lineaires = {}
    models_data = {}

    for col in colonnes_numeriques:
        prepare = _preparer_xy(df, col, target_col)
        if prepare is None:
            continue
        x, y = prepare
        scaled = _standardiser(x)
        if scaled is None:
            continue
        z, mu, sigma = scaled

        model_quad = _ajuster_logit_quadratique(z, y)
        if model_quad is None:
            continue

        try:
            p_val_sq = float(model_quad.pvalues[2])
        except Exception:
            continue
        if not np.isfinite(p_val_sq):
            continue

        if p_val_sq < p_threshold:
            colonnes_non_lineaires.append(col)
            results_non_lineaires[col] = p_val_sq
            models_data[col] = (model_quad, x, mu, sigma)

    return colonnes_non_lineaires, results_non_lineaires, models_data


def classifier_et_detecter(df, target_col, id_col="id_employee", p_threshold=0.05):
    """
    Pipeline complet :
        - quantitatives linéaires
        - quantitatives non-linéaires
        - qualitatives / catégorielles
        - booléennes / binaires
    """
    base = classifier_variables(df, target_col, id_col)
    colonnes_numeriques = base["numeriques"]
    colonnes_non_lineaires, _, models_data = detecter_non_linearite(
        df, target_col, colonnes_numeriques, p_threshold
    )
    colonnes_lineaires = [c for c in colonnes_numeriques if c not in colonnes_non_lineaires]
    return {
        "lineaires": colonnes_lineaires,
        "non_lineaires": colonnes_non_lineaires,
        "qualitatives": base["qualitatives"],
        "booleennes": base["booleennes"],
        "models_data": models_data,
    }


def afficher_classification(classification):
    """Affiche proprement le résultat de classifier_et_detecter()."""
    print("=" * 50)
    print("     CLASSIFICATION GLOBALE DES VARIABLES")
    print("=" * 50 + "\n")

    print(f"--- 1. QUANTITATIVES : LINÉAIRES ({len(classification['lineaires'])}) ---")
    for col in classification["lineaires"]:
        print(f"  • {col}")

    print(f"\n--- 2. QUANTITATIVES : NON-LINÉAIRES ({len(classification['non_lineaires'])}) ---")
    for col in classification["non_lineaires"]:
        print(f"  • {col}")

    print(f"\n--- 3. QUALITATIVES / CATÉGORIELLES ({len(classification['qualitatives'])}) ---")
    for col in classification["qualitatives"]:
        print(f"  • {col}")

    print(f"\n--- 4. BOOLÉENNES / BINAIRES ({len(classification['booleennes'])}) ---")
    for col in classification["booleennes"]:
        print(f"  • {col}")

    print("\n" + "=" * 50)


def _coefs_logit(model) -> list[float] | None:
    """params est une Series après .fit(), un ndarray après .fit_regularized()."""
    params = getattr(model, "params", None)
    if params is None:
        return None
    if hasattr(params, "iloc"):
        valeurs = [float(params.iloc[i]) for i in range(len(params))]
    else:
        valeurs = [float(v) for v in np.asarray(params).ravel()]
    if len(valeurs) < 3 or not np.all(np.isfinite(valeurs[:3])):
        return None
    return valeurs[:3]


def plot_derivees(colonnes_non_lineaires, models_data, ncols=2, taille_case=(5, 3.5)):
    """
    Trace la dérivée du risque (vitesse de variation de la probabilité de départ)
    pour chaque variable non-linéaire, dans une grille de 'ncols' graphiques par ligne.

    Compatible avec l'ancien format models_data {col: (modele, x)}
    et le nouveau {col: (modele, x, mu, sigma)}.
    """
    n_features = len(colonnes_non_lineaires)
    if n_features == 0:
        print("Aucune variable non-linéaire détectée.")
        return

    nrows = int(np.ceil(n_features / ncols))
    figsize = (taille_case[0] * ncols, taille_case[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(colonnes_non_lineaires):
        pack = models_data.get(col)
        if pack is None:
            axes[i].axis("off")
            continue

        if len(pack) == 4:
            model_quad, x, mu, sigma = pack
        else:
            model_quad, x = pack
            mu = float(pd.Series(x).mean())
            sigma = float(pd.Series(x).std(ddof=0)) or 1.0

        coefs = _coefs_logit(model_quad)
        if coefs is None:
            axes[i].axis("off")
            continue
        beta_0, beta_1, beta_2 = coefs

        x_range = np.linspace(float(np.min(x)), float(np.max(x)), 300)
        z_range = (x_range - mu) / sigma
        logit_val = beta_0 + beta_1 * z_range + beta_2 * np.square(z_range)
        # clip pour éviter overflow de exp
        logit_val = np.clip(logit_val, -30, 30)
        p_val = 1.0 / (1.0 + np.exp(-logit_val))
        l_prime_z = beta_1 + 2.0 * beta_2 * z_range
        # dP/dx = dP/dz * dz/dx
        derivative_p = p_val * (1.0 - p_val) * l_prime_z / sigma

        ax = axes[i]
        ax.plot(x_range, derivative_p, color="purple", linewidth=2, label="Dérivée")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="Seuil neutre")
        ax.set_title(f"{col}", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Taux de variation", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Tests d'hypothèse : tranches + croisement à niveau comparable
# ---------------------------------------------------------------------------
def _trouver_col_y(df, candidats):
    lookup = {c.lower().strip(): c for c in df.columns}
    for nom in candidats:
        if nom.lower() in lookup:
            return lookup[nom.lower()]
    return None


def _cible_01(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")
    return (
        serie.astype(str)
        .str.lower()
        .str.strip()
        .isin(["oui", "yes", "true", "1", "1.0"])
        .astype(float)
    )


def _couper_tranches(serie: pd.Series, bornes, libelles) -> pd.Series:
    x = pd.to_numeric(serie, errors="coerce")
    return pd.cut(x, bins=bornes, labels=libelles, right=False, include_lowest=True)


def plot_taux_depart_tranches(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Taux de départ par tranches d'ancienneté, d'âge et de salaire."""
    y = _cible_01(df[target_col])
    plans = [
        (
            ("annees_dans_l_entreprise", "annees_dans_le_poste_actuel"),
            [0, 1, 3, 5, 10, np.inf],
            ["0-1 an", "1-3 ans", "3-5 ans", "5-10 ans", "10 ans +"],
            "Ancienneté dans l'entreprise",
        ),
        (
            ("age",),
            [0, 25, 30, 35, 45, np.inf],
            ["<25", "25-29", "30-34", "35-44", "45+"],
            "Âge",
        ),
        (
            ("revenu_mensuel", "salaire_mensuel"),
            None,
            None,
            "Revenu mensuel (quartiles)",
        ),
    ]

    morceaux = []
    axes_plans = []
    for candidats, bornes, libelles, titre in plans:
        col = _trouver_col_y(df, candidats)
        if not col:
            continue
        if bornes is None:
            x = pd.to_numeric(df[col], errors="coerce")
            try:
                tranche = pd.qcut(x, 4, duplicates="drop")
            except Exception:
                continue
        else:
            tranche = _couper_tranches(df[col], bornes, libelles)
        tmp = pd.DataFrame({"tranche": tranche, "y": y}).dropna()
        if tmp.empty:
            continue
        stats = (
            tmp.groupby("tranche", observed=True)["y"]
            .agg(taux="mean", n="size")
            .reset_index()
        )
        stats.insert(0, "variable", titre)
        morceaux.append(stats)
        axes_plans.append((titre, stats))

    if not axes_plans:
        print("Aucune variable de tranche trouvée (ancienneté / âge / revenu).")
        return pd.DataFrame()

    print("\n--- Taux de départ par tranches (ancienneté, âge, salaire) ---")
    fig, axes = plt.subplots(1, len(axes_plans), figsize=(5.2 * len(axes_plans), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (titre, stats) in zip(axes, axes_plans):
        etiquettes = [str(v) for v in stats["tranche"]]
        ax.bar(etiquettes, stats["taux"], color="#DD8452", edgecolor="white")
        for i, row in enumerate(stats.itertuples(index=False)):
            ax.text(
                i,
                row.taux + 0.01,
                f"{row.taux:.0%}\nn={int(row.n)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0, max(0.35, float(stats["taux"].max()) + 0.08))
        ax.set_ylabel("Taux de départ")
        ax.set_title(titre, fontsize=11)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Taux de départ par tranches ({target_col})", fontsize=13)
    fig.tight_layout()
    plt.show()
    out = pd.concat(morceaux, ignore_index=True)
    return out


def plot_taux_depart_strate(df: pd.DataFrame, target_col: str) -> dict[str, pd.DataFrame]:
    """Taux de départ à niveau de poste comparable (heures sup et poste)."""
    y = _cible_01(df[target_col])
    col_niv = _trouver_col_y(df, ("niveau_hierarchique_poste", "niveau_hierarchique", "grade"))
    col_hs = _trouver_col_y(df, ("heure_supplementaires", "heures_supplementaires", "heures_sup"))
    col_poste = _trouver_col_y(df, ("poste", "intitule_poste", "emploi"))

    resultats: dict[str, pd.DataFrame] = {}
    if not col_niv:
        print("Pas de niveau hiérarchique : croisement à grade comparable impossible.")
        return resultats

    print("\n--- Taux de départ à niveau de poste comparable ---")
    n_plots = int(col_hs is not None) + int(col_poste is not None)
    if n_plots == 0:
        print("Ni heures sup ni poste trouvés pour le croisement.")
        return resultats

    fig, axes = plt.subplots(1, n_plots, figsize=(7.5 * n_plots, 4.6))
    axes = np.atleast_1d(axes)
    i_ax = 0

    if col_hs is not None:
        tmp = pd.DataFrame(
            {
                "niveau": pd.to_numeric(df[col_niv], errors="coerce"),
                "hs": df[col_hs].astype(str),
                "y": y,
            }
        ).dropna()
        stats = (
            tmp.groupby(["niveau", "hs"], observed=True)["y"]
            .agg(taux="mean", n="size")
            .reset_index()
        )
        resultats["heures_sup_x_niveau"] = stats
        ax = axes[i_ax]
        i_ax += 1
        modalites = list(stats["hs"].unique())
        niveaux = sorted(stats["niveau"].unique())
        largeur = 0.8 / max(len(modalites), 1)
        for k, mod in enumerate(modalites):
            sous = stats[stats["hs"] == mod].set_index("niveau").reindex(niveaux)
            xpos = np.arange(len(niveaux)) + (k - (len(modalites) - 1) / 2) * largeur
            ax.bar(xpos, sous["taux"].fillna(0), width=largeur * 0.95, label=str(mod))
        ax.set_xticks(np.arange(len(niveaux)))
        ax.set_xticklabels([str(int(v) if float(v).is_integer() else v) for v in niveaux])
        ax.set_ylabel("Taux de départ")
        ax.set_xlabel(col_niv)
        ax.set_title("Heures sup × niveau de poste")
        ax.legend(title=col_hs, fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    if col_poste is not None:
        tmp = pd.DataFrame(
            {
                "niveau": pd.to_numeric(df[col_niv], errors="coerce"),
                "poste": df[col_poste].astype(str),
                "y": y,
            }
        ).dropna()
        stats = (
            tmp.groupby(["niveau", "poste"], observed=True)["y"]
            .agg(taux="mean", n="size")
            .reset_index()
        )
        resultats["poste_x_niveau"] = stats
        # Trop de postes : on garde ceux avec le plus d'effectifs
        top_postes = tmp["poste"].value_counts().head(8).index
        pivot = (
            stats[stats["poste"].isin(top_postes)]
            .pivot(index="poste", columns="niveau", values="taux")
        )
        ax = axes[i_ax]
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Oranges", vmin=0, vmax=max(0.4, float(np.nanmax(pivot.to_numpy()))))
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(list(pivot.index), fontsize=8)
        ax.set_xlabel(col_niv)
        ax.set_title("Poste × niveau de poste (taux de départ)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Départ à grade comparable ({target_col})", fontsize=13)
    fig.tight_layout()
    plt.show()
    return resultats


def tester_hypothese_attrition(df: pd.DataFrame, target_col: str) -> dict:
    """Enchaîne tranches + croisements pour tester l'hypothèse 'profil junior'."""
    print(
        "\n--- Test d'hypothèse ---\n"
        "Question : le départ des bas salaires / heures sup / certains postes\n"
        "reste-t-il visible à ancienneté, âge ou niveau hiérarchique comparable ?"
    )
    tranches = plot_taux_depart_tranches(df, target_col)
    strates = plot_taux_depart_strate(df, target_col)
    plot_nuages_profil_junior(df, target_col)
    return {"tranches": tranches, "strates": strates}


def plot_nuages_profil_junior(df: pd.DataFrame, target_col: str, max_vars: int = 5) -> None:
    """Pairplot du bloc carrière (âge, salaire, ancienneté, grade) coloré par le départ."""
    candidats = [
        "age",
        "revenu_mensuel",
        "annee_experience_totale",
        "annees_dans_l_entreprise",
        "niveau_hierarchique_poste",
        "annees_dans_le_poste_actuel",
    ]
    cols = [c for c in candidats if c in df.columns][:max_vars]
    if len(cols) < 2 or target_col not in df.columns:
        return

    data = df[cols + [target_col]].copy()
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()
    if len(data) < 20:
        return

    y = data[target_col]
    if pd.api.types.is_numeric_dtype(y):
        data["_statut"] = np.where(y.astype(float) >= 0.5, "Part", "Reste")
    else:
        data["_statut"] = (
            y.astype(str).str.lower().str.strip()
            .isin(["oui", "yes", "true", "1", "1.0"])
            .map({True: "Part", False: "Reste"})
        )

    print("\n--- Nuages du profil junior (variables les plus liées entre elles) ---")
    print("Une droite de points = lien fort (âge / salaire / ancienneté).")
    print("Deux nuages qui se séparent en couleur = le départ n'est pas au même endroit.")

    # Palette contrastée : bleu vif et saturé pour "Part", orange clair et discret pour "Reste".
    # "Reste" est presque toujours majoritaire, donc on le rend volontairement pâle
    # pour qu'il ne noie pas visuellement le bleu.
    palette = {"Reste": "#F5C29B", "Part": "#0B5FFF"}

    # hue_order définit aussi l'ORDRE DE DESSIN : "Reste" est tracé en premier (en dessous),
    # "Part" est tracé en dernier (par-dessus) → le bleu n'est jamais recouvert.
    hue_order = ["Reste", "Part"]

    grid = sns.PairGrid(
        data,
        vars=cols,
        hue="_statut",
        hue_order=hue_order,
        palette=palette,
        corner=False,
        height=2.15,
    )

    def _scatter_par_groupe(x, y, color, label, **kwargs):
        """Style différent selon le groupe : Part = plus gros, plus opaque, dessiné au-dessus."""
        if label == "Part":
            plt.scatter(x, y, color=palette["Part"], alpha=0.75, s=32,
                        edgecolor="white", linewidth=0.3, zorder=3, label=label)
        else:
            plt.scatter(x, y, color=palette["Reste"], alpha=0.25, s=14,
                        edgecolor="none", zorder=1, label=label)

    grid.map_offdiag(_scatter_par_groupe)
    grid.map_diag(sns.histplot, multiple="layer", alpha=0.5, element="step")
    grid.add_legend(title="_statut")

    grid.fig.suptitle("Profil junior — nuages deux à deux", y=1.02, fontsize=13)
    plt.show()