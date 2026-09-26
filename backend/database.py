import os
import time

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rh_predictions_db")

DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def attendre_et_creer_tables(tentatives: int = 10, delai: int = 3):
    """Attend que Postgres soit prêt, puis vérifie/crée les tables."""
    for i in range(tentatives):
        try:
            Base.metadata.create_all(bind=engine)
            print("Connexion DB OK, tables vérifiées/créées.")
            return
        except OperationalError as e:
            print(f"DB pas encore prête (essai {i + 1}/{tentatives}) : {e}")
            time.sleep(delai)
    raise RuntimeError("Impossible de se connecter à la base après plusieurs tentatives.")


def assurer_colonne_details():
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE resultats ADD COLUMN IF NOT EXISTS details TEXT"))
            conn.commit()
    except Exception as exc:
        print(f"Note migration colonne details : {exc}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def initialiser_base():
    attendre_et_creer_tables()
    assurer_colonne_details()
