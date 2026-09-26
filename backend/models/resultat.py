from sqlalchemy import Column, Integer, String, Float

from database import Base


class ResultatDB(Base):
    __tablename__ = "resultats"

    id = Column(Integer, primary_key=True, index=True)
    prenom = Column(String, default="John")
    nom = Column(String, default="Doe")
    modele_utilise = Column(String)
    probabilite_de_quitter = Column(Float)
    prediction = Column(Integer)
    libelle_prediction = Column(String)
    seuil_applique = Column(Float)
    details = Column(String)
