import os
import time
import pandas as pd
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, Integer, Float, String, text
from sqlalchemy.exc import OperationalError

# --- CONFIGURATION DE LA BASE DE DONNÉES POSTGRESQL ---
# (mêmes variables d'environnement que dans app.py, pour rester cohérent)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rh_predictions_db")

DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Dossier contenant X.csv et y.csv à l'intérieur du conteneur
DATA_DIR = os.getenv("DATA_DIR", "/app/Data")
X_CSV = os.path.join(DATA_DIR, "X.csv")
Y_CSV = os.path.join(DATA_DIR, "y.csv")

TABLE_X = "employes_features"
TABLE_Y = "employes_cible"


def attendre_connexion(engine, tentatives: int = 10, delai: int = 3):
    """Attend que PostgreSQL soit prêt à accepter des connexions."""
    for i in range(tentatives):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Connexion PostgreSQL OK.")
            return
        except OperationalError as e:
            print(f"DB pas encore prête (essai {i + 1}/{tentatives}) : {e}")
            time.sleep(delai)
    raise RuntimeError("Impossible de se connecter à PostgreSQL après plusieurs tentatives.")


def type_sqlalchemy_depuis_pandas(series: pd.Series):
    """Déduit un type de colonne SQLAlchemy à partir du dtype pandas."""
    if pd.api.types.is_integer_dtype(series):
        return Integer
    elif pd.api.types.is_float_dtype(series):
        return Float
    else:
        return String


def creer_table_si_absente(engine, metadata: MetaData, nom_table: str, chemin_csv: str) -> bool:
    """
    Vérifie via SQLAlchemy (inspect) si la table existe déjà.
    - Si oui : ne fait rien (pas de requête de création).
    - Si non : lit les colonnes/types du CSV et crée la table via l'ORM.
    Retourne True si la table vient d'être créée, False si elle existait déjà.
    """
    inspecteur = inspect(engine)
    if inspecteur.has_table(nom_table):
        print(f"Table '{nom_table}' déjà existante : aucune création nécessaire.")
        return False

    print(f"Table '{nom_table}' absente : lecture de {chemin_csv} pour déduire le schéma...")
    df = pd.read_csv(chemin_csv)

    colonnes = [Column("id", Integer, primary_key=True, autoincrement=True)]
    for nom_colonne in df.columns:
        colonnes.append(Column(nom_colonne, type_sqlalchemy_depuis_pandas(df[nom_colonne])))

    table = Table(nom_table, metadata, *colonnes)
    metadata.create_all(engine, tables=[table])
    print(f"Table '{nom_table}' créée avec {len(df.columns)} colonnes (+ id).")
    return True


def charger_donnees_si_vide(engine, nom_table: str, chemin_csv: str):
    """Charge le CSV dans la table uniquement si celle-ci est vide (évite les doublons au redémarrage)."""
    with engine.connect() as conn:
        nb_lignes = conn.execute(text(f'SELECT COUNT(*) FROM "{nom_table}"')).scalar()

    if nb_lignes and nb_lignes > 0:
        print(f"Table '{nom_table}' contient déjà {nb_lignes} lignes : pas de rechargement.")
        return

    print(f"Chargement des données de {chemin_csv} dans '{nom_table}'...")
    df = pd.read_csv(chemin_csv)
    with engine.begin() as connexion:
        df.to_sql(nom_table, connexion, if_exists="append", index=False)
    print(f"{len(df)} lignes insérées dans '{nom_table}'.")


def main():
    engine = create_engine(DATABASE_URL)
    attendre_connexion(engine)

    metadata = MetaData()

    creer_table_si_absente(engine, metadata, TABLE_X, X_CSV)
    creer_table_si_absente(engine, metadata, TABLE_Y, Y_CSV)

    charger_donnees_si_vide(engine, TABLE_X, X_CSV)
    charger_donnees_si_vide(engine, TABLE_Y, Y_CSV)

    print("Initialisation des tables 'employes_features' et 'employes_cible' terminée.")


if __name__ == "__main__":
    main()