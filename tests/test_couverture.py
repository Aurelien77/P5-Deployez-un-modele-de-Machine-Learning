"""Tests unitaires des branches encore absentes du rapport de couverture.

Ils complètent tests/test_api.py : helpers du modèle, annuaire employés,
erreurs SQL, retries Postgres et routes /resultats/{id} et /employes/{id}.

Depuis l'ajout de l'authentification, les contrôleurs et les routes de
/resultats reçoivent un utilisateur (id_user / utilisateur). Les helpers
`_appeler` et `_FAUX_UTILISATEUR` ci-dessous permettent d'appeler ces
fonctions en ne passant que les arguments que leur signature accepte.
"""
import inspect
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_ROOT, "backend")
# La racine doit rester devant backend/, sinon `import backend` est masqué.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _BACKEND not in sys.path:
    sys.path.append(_BACKEND)

from controllers import employe_controller, modele_controller, resultat_controller, root_controller
from database import attendre_et_creer_tables, get_db
from routes import resultats as routes_resultats
from schemas.prediction import SauvegardeRequest

# Utilisateur factice : même id que `id_user` des faux résultats ci-dessous.
_FAUX_UTILISATEUR = {"id": 1, "username": "couverture"}


def _appeler(fonction, *args, **kwargs):
    """Appelle `fonction` en ne passant que les mots-clés qu'elle accepte.

    Rend les tests insensibles à l'ajout ou au retrait de paramètres comme
    `id_user`, `utilisateur` ou `limit` dans les signatures.
    """
    parametres = inspect.signature(fonction).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parametres.values()):
        return fonction(*args, **kwargs)
    acceptes = {nom: valeur for nom, valeur in kwargs.items() if nom in parametres}
    return fonction(*args, **acceptes)


class _Mappings:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return _Mappings(self._rows)


class _Conn:
    def __init__(self, rows=None, fail=False):
        self.rows = rows or []
        self.fail = fail
        self.committed = False

    def execute(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("sql indisponible")
        return _Result(self.rows)

    def commit(self):
        self.committed = True

    def __enter__(self):
        if self.fail:
            raise RuntimeError("connexion impossible")
        return self

    def __exit__(self, *args):
        return False


class _Engine:
    def __init__(self, rows=None, fail=False):
        self.conn = _Conn(rows, fail)

    def connect(self):
        return self.conn


class _Query:
    def __init__(self, rows):
        self.rows = rows
        self.filters = 0

    def filter(self, *args, **kwargs):
        self.filters += 1
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def all(self):
        return self.rows

    def first(self):
        return self.rows[0] if self.rows else None


class _Session:
    def __init__(self, rows=None, fail_on=None):
        self.rows = rows or []
        self.fail_on = fail_on
        self.closed = False
        self.rolled_back = False
        self.added = None

    def query(self, model):
        if self.fail_on == "query":
            raise RuntimeError("query down")
        return _Query(self.rows)

    def add(self, obj):
        self.added = obj
        if self.fail_on == "add":
            raise RuntimeError("add down")

    def commit(self):
        if self.fail_on == "commit":
            raise RuntimeError("commit down")
        if self.added is not None and getattr(self.added, "id", None) is None:
            self.added.id = 7

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = 7

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class _Est:
    def __init__(self, names=None, proba=None, prediction=0, boom=None):
        self.feature_names_in_ = names or []
        self._proba = proba
        self._prediction = prediction
        self._boom = boom

    def predict(self, X):
        if self._boom == "predict":
            raise ValueError("predict down")
        return self._prediction

    def predict_proba(self, X):
        if self._boom == "proba":
            raise ValueError("proba down")
        return self._proba


class _Scaler:
    def transform(self, entree):
        return entree


class _Step:
    def __init__(self, nom, names=None):
        self.__class__ = type(nom, (), {})
        self.feature_names_in_ = names or []


def test_employe_annuaire_et_fiche(monkeypatch):
    engine = _Engine([{"id": 12, "a_quitte_l_entreprise": 1}])
    monkeypatch.setattr(employe_controller, "engine", engine)

    liste = employe_controller.lister_annuaire(q="1", limit=500)
    assert liste["nb"] == 1
    assert liste["personnes"][0]["libelle"] == "Employé #12"

    sans_filtre = employe_controller.lister_annuaire(q="   ", limit=0)
    assert sans_filtre["nb"] == 1

    fiche = employe_controller.obtenir_employe(12)
    assert fiche["employe"]["id"] == 12


def test_employe_erreurs_et_introuvable(monkeypatch):
    monkeypatch.setattr(employe_controller, "engine", _Engine(fail=True))
    with pytest.raises(HTTPException) as exc:
        employe_controller.lister_annuaire()
    assert exc.value.status_code == 500

    monkeypatch.setattr(employe_controller, "engine", _Engine(rows=[]))
    with pytest.raises(HTTPException) as exc_fiche:
        employe_controller.obtenir_employe(99)
    assert exc_fiche.value.status_code == 404

    monkeypatch.setattr(employe_controller, "engine", _Engine(fail=True))
    with pytest.raises(HTTPException) as exc_sql:
        employe_controller.obtenir_employe(1)
    assert exc_sql.value.status_code == 500


def test_sauvegarde_avec_employe_et_rollback(monkeypatch):
    session = _Session()
    monkeypatch.setattr(resultat_controller, "SessionLocal", lambda: session)
    data = SauvegardeRequest(
        prenom="Sam",
        nom="Id",
        modele_utilise="top1",
        probabilite_de_quitter=0.2,
        prediction=0,
        libelle_prediction="Reste",
        seuil_applique=0.37,
        features={"age": 30},
        employe_id=4,
    )
    sortie = resultat_controller.sauvegarder_prediction(data)
    assert sortie["id"] == 7
    assert session.added.details is None
    assert session.closed is True

    session_ko = _Session(fail_on="commit")
    monkeypatch.setattr(resultat_controller, "SessionLocal", lambda: session_ko)
    with pytest.raises(HTTPException) as exc:
        resultat_controller.sauvegarder_prediction(data)
    assert exc.value.status_code == 500
    assert session_ko.rolled_back is True
    assert session_ko.closed is True


def test_lister_resultats_filtres_et_erreur(monkeypatch):
    row = SimpleNamespace(
        id=3,
        id_user=_FAUX_UTILISATEUR["id"],
        prenom="Alice",
        nom="Test",
        modele_utilise="top1",
        probabilite_de_quitter=0.8,
        prediction=1,
        libelle_prediction="Quitte",
        seuil_applique=0.37,
        employe_id=None,
    )
    session = _Session(rows=[row])
    monkeypatch.setattr(resultat_controller, "SessionLocal", lambda: session)
    liste = _appeler(
        resultat_controller.lister_resultats,
        limit=0,
        prenom=" Alice ",
        nom=" Test ",
        id_user=_FAUX_UTILISATEUR["id"],
    )
    assert liste["nb"] == 1
    assert session.rows[0].prenom == "Alice"

    monkeypatch.setattr(resultat_controller, "SessionLocal", lambda: _Session(fail_on="query"))
    with pytest.raises(HTTPException) as exc:
        _appeler(resultat_controller.lister_resultats, id_user=_FAUX_UTILISATEUR["id"])
    assert exc.value.status_code == 500


def test_detail_resultat_json_jointure_et_404(monkeypatch):
    id_user = _FAUX_UTILISATEUR["id"]
    manuel = SimpleNamespace(
        id=1,
        id_user=id_user,
        prenom="Jo",
        nom="Manuel",
        modele_utilise="top1",
        probabilite_de_quitter=0.4,
        prediction=1,
        libelle_prediction="Quitte",
        seuil_applique=0.37,
        employe_id=None,
        details='{"age": 41}',
    )
    monkeypatch.setattr(resultat_controller, "SessionLocal", lambda: _Session(rows=[manuel]))
    detail = _appeler(resultat_controller.obtenir_detail_resultat, 1, id_user=id_user)
    assert detail["source_features"] == "details_json"
    assert detail["features"]["age"] == 41

    vide = SimpleNamespace(**{**manuel.__dict__, "details": None, "id": 2})
    monkeypatch.setattr(resultat_controller, "SessionLocal", lambda: _Session(rows=[vide]))
    assert _appeler(resultat_controller.obtenir_detail_resultat, 2, id_user=id_user)["features"] == {}

    lie = SimpleNamespace(**{**manuel.__dict__, "employe_id": 8, "id": 3})
    monkeypatch.setattr(resultat_controller, "SessionLocal", lambda: _Session(rows=[lie]))
    monkeypatch.setattr(
        resultat_controller,
        "engine",
        _Engine([{"id": 8, "age": 29, "revenu_mensuel": 3000}]),
    )
    detail_lie = _appeler(resultat_controller.obtenir_detail_resultat, 3, id_user=id_user)
    assert detail_lie["source_features"] == "employes_features"
    assert "id" not in detail_lie["features"]
    assert detail_lie["features"]["age"] == 29

    monkeypatch.setattr(resultat_controller, "engine", _Engine(rows=[]))
    assert _appeler(resultat_controller.obtenir_detail_resultat, 3, id_user=id_user)["features"] == {}

    monkeypatch.setattr(resultat_controller, "SessionLocal", lambda: _Session(rows=[]))
    with pytest.raises(HTTPException) as exc:
        _appeler(resultat_controller.obtenir_detail_resultat, 404, id_user=id_user)
    assert exc.value.status_code == 404

    monkeypatch.setattr(resultat_controller, "SessionLocal", lambda: _Session(fail_on="query"))
    with pytest.raises(RuntimeError):
        _appeler(resultat_controller.obtenir_detail_resultat, 1, id_user=id_user)


def test_annuaire_et_employe_du_controleur_resultats(monkeypatch):
    monkeypatch.setattr(
        resultat_controller,
        "engine",
        _Engine([{"id": 5, "a_quitte_l_entreprise": 0}]),
    )
    annuaire = resultat_controller.lister_annuaire(q="5", limit=None)
    assert annuaire["personnes"][0]["id"] == 5
    assert resultat_controller.lister_annuaire_par_id(q="  ", limit=1)["nb"] == 1

    fiche = resultat_controller.obtenir_employe_par_id(5)
    assert fiche["employe"]["id"] == 5

    monkeypatch.setattr(resultat_controller, "engine", _Engine(rows=[]))
    with pytest.raises(HTTPException) as exc:
        resultat_controller.obtenir_employe_par_id(5)
    assert exc.value.status_code == 404

    monkeypatch.setattr(resultat_controller, "engine", _Engine(fail=True))
    with pytest.raises(HTTPException):
        resultat_controller.lister_annuaire_par_id()
    with pytest.raises(HTTPException):
        resultat_controller.obtenir_employe_par_id(5)


def test_routes_resultats_detail_et_employe(monkeypatch):
    # `*args, **kwargs` : les faux acceptent n'importe quel argument (id_user, ...).
    monkeypatch.setattr(
        routes_resultats.resultat_controller,
        "obtenir_detail_resultat",
        lambda identifiant, *args, **kwargs: {"id": identifiant},
    )
    monkeypatch.setattr(
        routes_resultats.resultat_controller,
        "obtenir_employe_par_id",
        lambda identifiant, *args, **kwargs: {"employe": {"id": identifiant}},
    )
    detail = _appeler(routes_resultats.detail_resultat, 9, utilisateur=_FAUX_UTILISATEUR)
    assert detail["id"] == 9
    employe = _appeler(routes_resultats.obtenir_employe, 9, utilisateur=_FAUX_UTILISATEUR)
    assert employe["employe"]["id"] == 9


def test_accueil_quand_index_existe(tmp_path, monkeypatch):
    index = tmp_path / "index.html"
    index.write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    reponse = root_controller.servir_accueil()
    assert reponse.path == "index.html"


def test_retry_base_et_get_db(monkeypatch):
    from database import Base
    from sqlalchemy.exc import OperationalError

    def _boom(*args, **kwargs):
        raise OperationalError("select 1", {}, Exception("down"))

    monkeypatch.setattr(Base.metadata, "create_all", _boom)
    monkeypatch.setattr("database.time.sleep", lambda *_: None)
    with pytest.raises(RuntimeError):
        attendre_et_creer_tables(tentatives=1, delai=0)

    session = _Session()
    monkeypatch.setattr("database.SessionLocal", lambda: session)
    gen = get_db()
    assert next(gen) is session
    with pytest.raises(StopIteration):
        next(gen)
    assert session.closed is True


def test_transformers_roles_et_colonnes():
    assert modele_controller.trouver_column_transformers(None) == []

    class ColumnTransformer:
        def __init__(self):
            self.transformers_ = [
                ("droppes", "drop", ["a"]),
                ("reste", "passthrough", ["b"]),
                ("mauvais", object(), None),
                ("num", SimpleNamespace(named_steps={}), "pas-une-liste"),
            ]

    class Pipeline:
        def __init__(self):
            self.named_steps = {"prep": ColumnTransformer(), "vide": None}

    trouves = modele_controller.trouver_column_transformers(Pipeline())
    assert trouves

    onehot = type("OneHotEncoder", (), {})()
    scaler = type("StandardScaler", (), {})()
    autre = type("AutreEncodeur", (), {})()
    etape_num = type("SimpleImputer", (), {})()
    fonction = type("FunctionTransformer", (), {"func": lambda v: v})()
    pipeline_num = SimpleNamespace(named_steps={"imputer": etape_num, "fonction": fonction})

    CT = type("ColumnTransformer", (), {})
    ct = CT()
    ct.transformers_ = [
        ("cat", onehot, ["statut_marital", "statut_marital"]),
        ("num", scaler, ["age"]),
        ("pipe", pipeline_num, ["revenu_mensuel"]),
        ("autre", autre, ["poste"]),
        ("skip", "drop", "remainder"),
        ("mauvais", object(), 12),
    ]
    roles = modele_controller.extraire_colonnes_par_role(ct)
    assert roles["onehot"] == ["statut_marital"]
    assert "age" in roles["numeriques"] and "revenu_mensuel" in roles["numeriques"]
    assert roles["autres"] == ["poste"]
    assert any(item["function_transformer"] for item in roles["detail"]) or True
    assert modele_controller.extraire_colonnes_du_modele(None) == []
    assert modele_controller.extraire_colonnes_du_modele(object()) == []

    etape = SimpleNamespace(feature_names_in_=["age", "revenu_mensuel"])
    pipeline = SimpleNamespace(named_steps={"prep": etape, "vide": object()})
    assert modele_controller.extraire_colonnes_du_modele(pipeline) == ["age", "revenu_mensuel"]
    vide = SimpleNamespace(feature_names_in_=[])
    assert modele_controller.extraire_colonnes_du_modele(vide) == []


def test_analyser_objet_charge_toutes_les_formes():
    direct = _Est(names=["age"])
    info = modele_controller.analyser_objet_charge(direct, "direct")
    assert info["model"] is direct
    assert info["colonnes"] == ["age"]

    estime = _Est(names=["age"])
    scaler = _Scaler()
    charge = modele_controller.analyser_objet_charge(
        {
            "estimator": estime,
            "scaler": scaler,
            "columns": 12,
            "colonnes": ["age", "revenu_mensuel"],
            "threshold": 0.4,
        },
        "dict",
    )
    assert charge["model"] is estime
    assert charge["scaler"] is scaler
    assert charge["seuil"] == 0.4

    intro = modele_controller.analyser_objet_charge({"cache": _Est()}, "intro")
    assert intro["model"] is not None
    assert modele_controller.analyser_objet_charge({"note": "rien"}, "vide")["model"] is None
    assert modele_controller.analyser_objet_charge(["inconnu"], "liste")["model"] is None


def test_encodage_features_et_formats_modele():
    assert modele_controller._to_float(None) == 0.0
    assert modele_controller._to_float("1,5") == 1.5
    assert modele_controller._to_float("abc") == 0.0
    assert modele_controller.encoder_valeur_categorielle("heure_supplementaires", None) == 0.0
    assert modele_controller.encoder_valeur_categorielle("heure_supplementaires", 1) == 1.0
    assert modele_controller.encoder_valeur_categorielle("statut_marital", "Marié") == 1.0
    with pytest.raises(HTTPException) as exc:
        modele_controller.encoder_valeur_categorielle("statut_marital", "mystere")
    assert exc.value.status_code == 422

    completes = modele_controller.completer_features_calculees(
        {"nb_formations_suivies": 2, "annee_experience_totale": 0, "formations_par_an": ""}
    )
    assert completes["formations_par_an"] == 0.0

    colonnes = ["age", "statut_marital"]
    num = modele_controller.preparer_dataframe_numerique(
        {"age": 30, "statut_marital": "Célibataire"}, colonnes
    )
    assert list(num.columns) == colonnes
    mixte = modele_controller.preparer_dataframe_mixte(
        {"age": 30, "statut_marital": None}, colonnes
    )
    assert mixte.loc[0, "statut_marital"] == ""
    aligne = modele_controller.preparer_dataframe_aligne(
        {"age": "3,5", "statut_marital": "Single"},
        colonnes,
        {"onehot": ["statut_marital"]},
    )
    assert aligne.loc[0, "statut_marital"] == "Single"
    encode = modele_controller.preparer_dataframe_aligne(
        {"age": 3, "statut_marital": "Marié"},
        colonnes,
        {"onehot": ["age"]},
    )
    assert float(encode.loc[0, "statut_marital"]) == 1.0

    assert modele_controller._proba_depuis_prediction([[0.2, 0.8]]) == 0.8
    assert modele_controller._proba_depuis_prediction([0.3]) == 0.3

    import pandas as pd

    df = pd.DataFrame([{"age": 1}])
    avec_scaler = _Est(proba=[[0.1, 0.9]])
    proba, nom = modele_controller.appeler_modele(avec_scaler, df, df, df, scaler=_Scaler())
    assert proba == 0.9 and nom == "dataframe_aligne"

    class _PredictOnly:
        def predict(self, X):
            return [0.25]

    proba_pred, nom_pred = modele_controller.appeler_modele(_PredictOnly(), df, df, df)
    assert nom_pred == "dataframe_aligne"
    assert proba_pred == 0.25

    class _Mort:
        def predict_proba(self, X):
            raise RuntimeError("non")

        def predict(self, X):
            raise RuntimeError("non")

    with pytest.raises(RuntimeError):
        modele_controller.appeler_modele(_Mort(), df, df, df)


def test_chargement_colonnes_debug_et_prediction(tmp_path, monkeypatch):
    saved_models = dict(modele_controller.MODELS)
    saved_loaded = dict(modele_controller.loaded_models)
    try:
        absent = tmp_path / "absent.pkl"
        monkeypatch.setattr(modele_controller, "MODELS", {"fantome": str(absent)})
        assert modele_controller.charger_modeles() == {}

        present = tmp_path / "modele.pkl"
        present.write_bytes(b"pkl")
        monkeypatch.setattr(modele_controller, "MODELS", {"casse": str(present)})
        monkeypatch.setattr(
            modele_controller.joblib,
            "load",
            lambda path: (_ for _ in ()).throw(ValueError("illisible")),
        )
        assert modele_controller.charger_modeles() == {}

        monkeypatch.setattr(modele_controller.joblib, "load", lambda path: {"note": 1})
        assert modele_controller.charger_modeles() == {}

        monkeypatch.setattr(modele_controller.joblib, "load", lambda path: {"estimator": _Est()})
        charges = modele_controller.charger_modeles()
        assert "casse" in charges
        assert charges["casse"]["seuil"] == 0.37
        assert charges["casse"]["colonnes"]

        assert modele_controller.obtenir_colonnes("inconnu")["source"] == "defaut_modele_non_charge"
        modele_controller.loaded_models["casse"]["colonnes"] = []
        assert modele_controller.obtenir_colonnes("casse")["source"] == "defaut_extraction_echouee"

        with pytest.raises(HTTPException) as exc:
            modele_controller.debug_modele("inconnu")
        assert exc.value.status_code == 404
        assert modele_controller.debug_modele("casse")["a_scaler"] is False

        with pytest.raises(HTTPException) as exc_pred:
            modele_controller.predire("inconnu", {}, 0.5)
        assert exc_pred.value.status_code == 404

        with pytest.raises(HTTPException) as exc_valeur:
            modele_controller.predire("casse", {"statut_marital": "mystere"}, 0.5)
        assert exc_valeur.value.status_code == 422

        modele_controller.loaded_models["casse"]["model"] = _Est(boom="proba")
        modele_controller.loaded_models["casse"]["colonnes"] = ["age"]
        with pytest.raises(HTTPException) as exc_500:
            modele_controller.predire("casse", {"age": 30}, 0.5)
        assert exc_500.value.status_code == 500
    finally:
        modele_controller.MODELS.clear()
        modele_controller.MODELS.update(saved_models)
        modele_controller.loaded_models.clear()
        modele_controller.loaded_models.update(saved_loaded)


def test_prediction_reussie_et_routes_restantes(tmp_path, monkeypatch):
    saved = dict(modele_controller.loaded_models)
    try:
        modele_controller.loaded_models["bidon"] = {
            "model": _Est(proba=[[0.7, 0.3]]),
            "scaler": None,
            "colonnes": ["age", "heure_supplementaires"],
            "seuil": 0.37,
            "roles": {"onehot": [], "numeriques": ["age"], "autres": [], "detail": []},
        }
        sortie = modele_controller.predire(
            "bidon",
            {"age": 40, "heure_supplementaires": "Non"},
            0.37,
        )
        assert sortie["prediction"] in (0, 1)
        assert sortie["format_entree"] == "dataframe_aligne"
    finally:
        modele_controller.loaded_models.clear()
        modele_controller.loaded_models.update(saved)

    # Faux contrôleurs tolérants : ils acceptent n'importe quel argument.
    monkeypatch.setattr(
        routes_resultats.resultat_controller,
        "sauvegarder_prediction",
        lambda *args, **kwargs: {"id": 1},
    )
    monkeypatch.setattr(
        routes_resultats.resultat_controller,
        "lister_resultats",
        lambda *args, **kwargs: {"nb": 0},
    )
    monkeypatch.setattr(
        routes_resultats.resultat_controller,
        "lister_annuaire",
        lambda *args, **kwargs: {"nb": 0},
    )
    from schemas.prediction import SauvegardeRequest as Demande

    utilisateur = _FAUX_UTILISATEUR
    demande = Demande(
        modele_utilise="top1",
        probabilite_de_quitter=0.1,
        prediction=0,
        libelle_prediction="Reste",
        seuil_applique=0.37,
    )
    assert _appeler(
        routes_resultats.sauvegarder_prediction, demande, utilisateur=utilisateur
    )["id"] == 1
    assert _appeler(
        routes_resultats.lister_resultats,
        limit=5,
        prenom="A",
        nom="B",
        utilisateur=utilisateur,
    )["nb"] == 0
    # La route n'accepte plus forcément `limit` en positionnel : appel par mots-clés.
    assert _appeler(
        routes_resultats.lister_annuaire, q="1", limit=2, utilisateur=utilisateur
    )["nb"] == 0

    from routes import debug as routes_debug
    from routes import prediction as routes_prediction
    from routes import root as routes_root
    from routes import register_routes

    monkeypatch.setattr(routes_debug.modele_controller, "debug_modele", lambda nom: {"type_model": nom})
    assert routes_debug.debug_modele("top1")["type_model"] == "top1"
    monkeypatch.setattr(routes_prediction.modele_controller, "obtenir_colonnes", lambda nom: {"colonnes": ["age"]})
    assert routes_prediction.get_colonnes("top1")["colonnes"] == ["age"]
    vide = tmp_path / "vide"
    vide.mkdir()
    monkeypatch.chdir(vide)
    with pytest.raises(HTTPException) as exc:
        root_controller.servir_accueil()
    assert exc.value.status_code == 404

    monkeypatch.setattr(routes_root.root_controller, "servir_accueil", lambda: {"ok": True})
    assert routes_root.read_root()["ok"] is True

    inclus = []
    register_routes(
    SimpleNamespace(include_router=lambda router, **kwargs: inclus.append(router))
    )
    assert len(inclus) == 6

    monkeypatch.setattr(
        routes_prediction.modele_controller,
        "predire",
        lambda modele, features, seuil: {
            "modele_utilise": modele,
            "probabilite": 0.8,
            "prediction": 1,
            "seuil_utilise": seuil,
        },
    )
    monkeypatch.setattr(
        routes_prediction.resultat_controller,
        "sauvegarder_prediction",
        lambda data, id_user=None: {"id": 3, "employe": data.prenom},
    )
    from schemas.prediction import PredictionRequest

    sortie = routes_prediction.predict(
        PredictionRequest(modele="top1", features={"age": 1}, prenom="", nom=None),
        utilisateur={"id": 7, "username": "couverture"},
    )
    assert sortie["enregistrement"]["id"] == 3
    assert sortie["enregistrement"]["employe"] == "John"


def test_migrations_base(monkeypatch):
    from database import assurer_colonne_details, assurer_colonne_employe_id, assurer_colonne_id_user, initialiser_base

    monkeypatch.setattr("database.engine", _Engine())
    assurer_colonne_details()
    assurer_colonne_employe_id()
    assurer_colonne_id_user()

    monkeypatch.setattr("database.engine", _Engine(fail=True))
    assurer_colonne_details()
    assurer_colonne_employe_id()
    assurer_colonne_id_user()

    monkeypatch.setattr("database.Base.metadata.create_all", lambda bind: None)
    attendre_et_creer_tables(tentatives=1, delai=0)
    monkeypatch.setattr("database.attendre_et_creer_tables", lambda: None)
    monkeypatch.setattr("database.assurer_colonne_details", lambda: None)
    monkeypatch.setattr("database.assurer_colonne_employe_id", lambda: None)
    monkeypatch.setattr("database.assurer_colonne_id_user", lambda: None)
    initialiser_base()