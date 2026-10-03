from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text

from database import Base


class ResultatDB(Base):
    __tablename__ = "resultats"

    id = Column(Integer, primary_key=True, index=True)
    prenom = Column(String, index=True)
    nom = Column(String, index=True)
    modele_utilise = Column(String)
    probabilite_de_quitter = Column(Float)
    prediction = Column(Integer)
    libelle_prediction = Column(String)
    seuil_applique = Column(Float)
    employe_id = Column(Integer, nullable=True)
    details = Column(Text, nullable=True)
    id_user = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
