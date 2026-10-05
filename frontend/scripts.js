

    async function chargerListeQuittes() {
    const select = document.getElementById("select-quittes");
    try {
        const response = await fetch(`${API_BASE}/annuaire`);
        const data = await response.json();
        const quittes = (data.personnes || []).filter(p => p.a_quitte_l_entreprise === "Oui");

        quittes.forEach(p => {
            const option = document.createElement("option");
            option.value = p.id;
            option.innerText = p.libelle;
            select.appendChild(option);
        });

        if (!quittes.length) {
            const option = document.createElement("option");
            option.disabled = true;
            option.innerText = "Aucun employé ayant quitté trouvé";
            select.appendChild(option);
        }
    } catch (e) {
        console.error("Erreur chargement liste 'quittés'", e);
    }
}

document.getElementById("select-quittes").addEventListener("change", function () {
    if (this.value) {
        document.getElementById("recherche-annuaire").value = this.value;
        chargerFicheParId(this.value);
    }
});
    const API_BASE = "/api";
    const TOKEN_KEY = "rh_access_token";
    const USER_KEY = "rh_username";

    function tokenCourant() {
        return localStorage.getItem(TOKEN_KEY) || "";
    }

    function entetesAuth(extra) {
        const headers = Object.assign({ "Content-Type": "application/json" }, extra || {});
        const token = tokenCourant();
        if (token) headers.Authorization = "Bearer " + token;
        return headers;
    }


    function viderDernierePrediction() {
        const texte = document.getElementById("resultat-texte");
        const msg = document.getElementById("msg-sauvegarde");
        const bloc = document.getElementById("resultat");
        if (texte) texte.innerHTML = "";
        if (msg) msg.innerText = "";
        if (bloc) bloc.style.display = "none";
    }

    function majAuthUi() {
        const connecte = Boolean(tokenCourant());
        document.getElementById("compte-menu").style.display = connecte ? "inline-block" : "none";
        document.getElementById("btn-auth-submit").style.display = connecte ? "none" : "inline-block";
        if (!connecte) document.getElementById("compte-dropdown").classList.remove("ouvert");
        document.getElementById("btn-compte").textContent = localStorage.getItem(USER_KEY) || "Compte";
        document.getElementById("champs-auth").style.display = connecte ? "none" : "grid";
        document.getElementById("mode-switch").style.display = connecte ? "none" : "flex";
        document.getElementById("app-connectee").style.display = connecte ? "block" : "none";
        const etat = document.getElementById("auth-etat");
        etat.textContent = connecte
            ? "Connecté en tant que " + (localStorage.getItem(USER_KEY) || "utilisateur") + ". Jeton valable 12 h."
            : (modeInscription
                ? "Choisissez un nom et un mot de passe, puis créez le compte."
                : "Connectez-vous pour accéder à l'interface.");
        document.getElementById("auth-titre").textContent = connecte
            ? "Compte"
            : (modeInscription ? "Créer un compte" : "Connexion");
    }

    let modeInscription = false;

    function messageApi(data, status) {
        if (data && typeof data.detail === "string") return data.detail;
        if (status === 404) return "Service d'authentification introuvable. Reconstruisez l'image API.";
        return "Échec de l'authentification.";
    }

    function basculerMode(inscription) {
        modeInscription = inscription;
        document.getElementById("mode-switch").classList.toggle("inscription", inscription);
        document.getElementById("label-login").classList.toggle("actif", !inscription);
        document.getElementById("label-register").classList.toggle("actif", inscription);
        document.getElementById("btn-auth-submit").textContent = inscription ? "Créer le compte" : "Se connecter";
        document.getElementById("auth-password").autocomplete = inscription ? "new-password" : "current-password";
        majAuthUi();
    }

    async function authentifier() {
        const username = document.getElementById("auth-username").value.trim();
        const password = document.getElementById("auth-password").value;
        const msg = document.getElementById("auth-message");
        const chemin = modeInscription ? "/auth/register" : "/auth/login";
        msg.style.color = "#7f8c8d";
        msg.textContent = "En cours...";
        try {
            const response = await fetch(`${API_BASE}${chemin}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password })
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                msg.style.color = "#c0392b";
                msg.textContent = messageApi(data, response.status);
                return;
            }
            if (modeInscription) {
                document.getElementById("auth-password").value = "";
                basculerMode(false);
                msg.style.color = "#27ae60";
                msg.textContent = "Compte créé. Vous pouvez vous connecter.";
                return;
            }
            localStorage.setItem(TOKEN_KEY, data.access_token);
            localStorage.setItem(USER_KEY, data.username);
            if (data.id) localStorage.setItem("rh_user_id", String(data.id));
            viderDernierePrediction();
            msg.style.color = "#27ae60";
            msg.textContent = "Session ouverte pour " + data.username + " (12 h).";
            majAuthUi();
            chargerResultats();
        } catch (error) {
            msg.style.color = "#c0392b";
            msg.textContent = "Impossible de joindre l'API.";
        }
    }

    document.getElementById("btn-slider").addEventListener("click", () => basculerMode(!modeInscription));
    document.getElementById("label-login").addEventListener("click", () => basculerMode(false));
    document.getElementById("label-register").addEventListener("click", () => basculerMode(true));
    document.getElementById("btn-auth-submit").addEventListener("click", authentifier);
    document.getElementById("btn-compte").addEventListener("click", (event) => {
        event.stopPropagation();
        document.getElementById("compte-dropdown").classList.toggle("ouvert");
    });
    document.addEventListener("click", () => {
        document.getElementById("compte-dropdown").classList.remove("ouvert");
    });
    document.getElementById("btn-ouvrir-mdp").addEventListener("click", (event) => {
        event.stopPropagation();
        const form = document.getElementById("form-mdp");
        form.style.display = form.style.display === "none" ? "block" : "none";
    });
    document.getElementById("form-mdp").addEventListener("click", (event) => event.stopPropagation());
    document.getElementById("form-mdp").addEventListener("submit", async (event) => {
        event.preventDefault();
        const msg = document.getElementById("auth-message");
        const response = await fetch(`${API_BASE}/auth/mot-de-passe`, {
            method: "POST",
            headers: entetesAuth(),
            body: JSON.stringify({
                ancien_password: document.getElementById("ancien-mdp").value,
                nouveau_password: document.getElementById("nouveau-mdp").value
            })
        });
        const data = await response.json().catch(() => ({}));
        msg.style.color = response.ok ? "#27ae60" : "#c0392b";
        msg.textContent = data.message || data.detail || "Modification impossible.";
        if (response.ok) {
            document.getElementById("ancien-mdp").value = "";
            document.getElementById("nouveau-mdp").value = "";
            document.getElementById("form-mdp").style.display = "none";
            document.getElementById("compte-dropdown").classList.remove("ouvert");
        }
    });
    document.getElementById("btn-logout").addEventListener("click", () => {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        localStorage.removeItem("rh_user_id");
        document.getElementById("auth-message").textContent = "Déconnecté.";
        majAuthUi();
        document.getElementById("liste-resultats").innerText = "Connectez-vous pour voir vos prédictions.";
        viderDernierePrediction();
    });
    document.getElementById("btn-supprimer-compte").addEventListener("click", async () => {
        if (!confirm("Supprimer le compte et toutes ses prédictions ?")) return;
        const response = await fetch(`${API_BASE}/auth/compte`, {
            method: "DELETE",
            headers: entetesAuth()
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            document.getElementById("auth-message").textContent = data.detail || "Suppression impossible.";
            return;
        }
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        localStorage.removeItem("rh_user_id");
        document.getElementById("auth-message").style.color = "#27ae60";
        document.getElementById("auth-message").textContent = data.message || "Compte supprimé.";
        majAuthUi();
        document.getElementById("liste-resultats").innerText = "Compte supprimé.";
        viderDernierePrediction();
    });
    majAuthUi();

    const COLONNES_MODELE = [
        "age",
        "revenu_mensuel",
        "annees_dans_l_entreprise",
        "distance_domicile_travail",
        "annes_sous_responsable_actuel",
        "satisfaction_employee_environnement",
        "niveau_hierarchique_poste",
        "satisfaction_min",
        "poste_x_niveau",
        "formations_par_an",
        "statut_marital",
        "frequence_deplacement",
        "heure_supplementaires",
        "hs_et_salaire_bas",
        "jeune_faible_anciennete"
    ];

    const optionsCategoriques = {
        "statut_marital": ["Célibataire", "Divorcé(e)", "Marié(e)"],
        "frequence_deplacement": ["Aucun", "Frequent", "Occasionnel"],
        "heure_supplementaires": ["Non", "Oui"]
    };

    const valeursParDefaut = {
        "age": 36,
        "revenu_mensuel": 4919,
        "annees_dans_l_entreprise": 5,
        "distance_domicile_travail": 7,
        "annes_sous_responsable_actuel": 3,
        "satisfaction_employee_environnement": 3,
        "niveau_hierarchique_poste": 2,
        "satisfaction_min": 2,
        "poste_x_niveau": 6,
        "formations_par_an": 0.3,
        "hs_et_salaire_bas": 0,
        "jeune_faible_anciennete": 0
    };

    const colonnesCalculees = ["poste_x_niveau", "formations_par_an"];

    let colonnesActuelles = COLONNES_MODELE.slice();
    let dernierePredictionData = null;

    function toNumber(id, fallback = 0) {
        const el = document.getElementById(id);
        const n = parseFloat(el && el.value);
        return Number.isFinite(n) ? n : fallback;
    }

    function recalculerFeaturesDerivees() {
        const exp = toNumber("annee_experience_totale", 0);
        const nbForm = toNumber("nb_formations_suivies", 0);
        const formations = exp ? nbForm / exp : 0;

        const nivPoste = toNumber("niveau_hierarchique_poste", 0);
        const nivEduc = toNumber("niveau_education", 0);
        const posteXNiveau = nivPoste * nivEduc;

        const champForm = document.getElementById("formations_par_an");
        const champPoste = document.getElementById("poste_x_niveau");
        if (champForm) champForm.value = Number(formations.toFixed(4));
        if (champPoste) champPoste.value = Number(posteXNiveau.toFixed(4));
    }

    function creerChamp(col) {
        const div = document.createElement("div");
        div.className = "form-group";

        const label = document.createElement("label");
        label.htmlFor = col;
        label.innerHTML = col;
        if (colonnesCalculees.includes(col)) {
            label.innerHTML += ' <span class="hint">(calculé auto)</span>';
        }
        div.appendChild(label);

        if (optionsCategoriques[col]) {
            const select = document.createElement("select");
            select.id = col;
            select.name = col;
            optionsCategoriques[col].forEach(opt => {
                const option = document.createElement("option");
                option.value = opt;
                option.innerText = opt;
                select.appendChild(option);
            });
            div.appendChild(select);
        } else {
            const input = document.createElement("input");
            input.id = col;
            input.name = col;
            input.type = "number";
            input.step = "any";
            input.value = valeursParDefaut[col] !== undefined ? valeursParDefaut[col] : 0;
            if (colonnesCalculees.includes(col)) {
                input.readOnly = true;
            }
            if (col === "niveau_hierarchique_poste") {
                input.addEventListener("input", recalculerFeaturesDerivees);
            }
            div.appendChild(input);
        }
        return div;
    }

    async function chargerColonnes() {
        const modele = "top1";
        try {
            const response = await fetch(`${API_BASE}/colonnes?modele=${modele}`);
            const data = await response.json();
            colonnesActuelles = (data.colonnes && data.colonnes.length) ? data.colonnes : COLONNES_MODELE.slice();
        } catch (e) {
            console.error("Erreur lors du chargement des colonnes", e);
            colonnesActuelles = COLONNES_MODELE.slice();
        }

        const container = document.getElementById("fields-container");
        container.innerHTML = "";
        colonnesActuelles.forEach(col => container.appendChild(creerChamp(col)));
        recalculerFeaturesDerivees();
    }

    ["nb_formations_suivies", "annee_experience_totale", "niveau_education"].forEach(id => {
        document.getElementById(id).addEventListener("input", recalculerFeaturesDerivees);
    });


    function ajouterResultatLocal(r) {
        const zone = document.getElementById("liste-resultats");
        let table = zone.querySelector("table");
        if (!table) {
            zone.innerHTML = `
                <table>
                    <thead>
                        <tr><th>ID</th><th>Employé</th><th>Prédiction</th><th>Risque</th></tr>
                    </thead>
                    <tbody></tbody>
                </table>`;
            table = zone.querySelector("table");
        }
        const body = table.querySelector("tbody");
        const ligne = document.createElement("tr");
        ligne.innerHTML = `
            <td>${r.id}</td>
            <td>${r.prenom} ${r.nom}</td>
            <td>${r.libelle_prediction}</td>
            <td>${(r.probabilite_de_quitter * 100).toFixed(2)}%</td>`;
        body.prepend(ligne);
    }

async function chargerResultats() {
    const zone = document.getElementById("liste-resultats");
    try {
        if (!tokenCourant()) {
            zone.innerText = "Connectez-vous pour voir vos prédictions.";
            return;
        }
        const response = await fetch(`${API_BASE}/resultats`, {
            headers: entetesAuth(),
            signal: AbortSignal.timeout(8000)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(JSON.stringify(data.detail || data));
        const lignesSource = data.resultats || [];
        if (!lignesSource.length) {
            zone.innerHTML = "<p class='hint'>Aucune prédiction enregistrée pour ce compte.</p>";
            return;
        }
        const lignes = lignesSource.map(r => `
            <tr>
                <td>${r.id}</td>
                <td>${r.prenom} ${r.nom}</td>
                <td>${r.libelle_prediction}</td>
                <td>${(r.probabilite_de_quitter * 100).toFixed(2)}%</td>
            </tr>
        `).join("");
        zone.innerHTML = `
            <table>
                <thead>
                    <tr>
                        <th>ID</th><th>Employé</th><th>Prédiction</th>
                        <th>Risque</th>
                    </tr>
                </thead>
                <tbody>${lignes}</tbody>
            </table>
        `;
    } catch (e) {
        zone.innerHTML = "<p class='hint'>Impossible de lire les prédictions : " + e.message + "</p>";
    }
}

    
    function demarrer() {
        chargerColonnes();
        chargerResultats();
        chargerListeQuittes();
    }
    if (document.readyState === "loading") {
        window.addEventListener("DOMContentLoaded", demarrer);
    } else {
        demarrer();
    }

    let employeIdCourant = null;


    function appliquerValeur(id, valeur) {
        const el = document.getElementById(id);
        if (!el || valeur === undefined || valeur === null) return;
        el.value = valeur;
    }

async function chargerFicheParId(identifiant) {
    const idNum = parseInt(identifiant, 10);
    const alerte = document.getElementById("alerte-employe-quitte");

    if (!idNum) {
        employeIdCourant = null;
        alerte.style.display = "none";
        return;
    }
    employeIdCourant = idNum;
    try {
        const response = await fetch(`${API_BASE}/employes/${idNum}`);
        const data = await response.json();
        if (!response.ok) throw new Error(JSON.stringify(data.detail || data));
        const fiche = data.employe || {};
        Object.keys(fiche).forEach((cle) => appliquerValeur(cle, fiche[cle]));

        if (fiche.a_quitte_l_entreprise === "Oui") {
            alerte.style.display = "inline-block";
        } else {
            alerte.style.display = "none";
        }

        if (typeof recalculerFeaturesDerivees === "function") {
            recalculerFeaturesDerivees();
        }
    } catch (error) {
        alerte.style.display = "none";
        alert("Aucun employé pour l'id " + identifiant + ".");
        console.error(error);
    }
}
document.getElementById("recherche-annuaire").addEventListener("input", function () {
    clearTimeout(window._annuaireTimer);
    const terme = this.value.trim();
    if (!terme) {
        employeIdCourant = null;
        document.getElementById("alerte-employe-quitte").style.display = "none";  // ← AJOUT
    }
    window._annuaireTimer = setTimeout(() => {
        if (/^\d+$/.test(terme)) chargerFicheParId(terme);
    }, 250);
});

    document.getElementById("recherche-annuaire").addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
            e.preventDefault();
            chargerFicheParId(this.value.trim());
        }
    });

    document.getElementById("btn-charger-id").addEventListener("click", function () {
        chargerFicheParId(document.getElementById("recherche-annuaire").value.trim());
    });


    document.getElementById("prediction-form").addEventListener("submit", async function(e) {
        e.preventDefault();
        const idSaisi = document.getElementById("recherche-annuaire").value.trim();
        if (!idSaisi) {
            alert("Renseignez un ID employé avant de lancer la prédiction.");
            document.getElementById("recherche-annuaire").focus();
            return;
        }
        employeIdCourant = Number(idSaisi);
        if (!Number.isInteger(employeIdCourant)) {
            alert("L'ID employé doit être un nombre.");
            return;
        }
        if (!colonnesActuelles.length) {
            alert("Les colonnes du modèle ne sont pas chargées. L'API ne répond pas.");
            return;
        }
        recalculerFeaturesDerivees();

        const modeleChoisi = "top1";
        const seuilSaisi = 0.37;

        const features = {};
        colonnesActuelles.forEach(col => {
            const element = document.getElementById(col);
            if (!element) {
                features[col] = valeursParDefaut[col] !== undefined ? valeursParDefaut[col] : 0;
                return;
            }
            let val = element.value;
            if (!optionsCategoriques[col]) {
                val = val.includes(".") ? parseFloat(val) : parseInt(val, 10);
                if (Number.isNaN(val)) val = parseFloat(element.value) || 0;
            }
            features[col] = val;
        });

        // Sources utiles si l'API doit recalculer côté serveur
        features.nb_formations_suivies = toNumber("nb_formations_suivies");
        features.annee_experience_totale = toNumber("annee_experience_totale");
        features.niveau_education = toNumber("niveau_education");
        features.genre = document.getElementById("genre").value;
        features.departement = document.getElementById("departement").value;
        features.poste = document.getElementById("poste").value;
        features.domaine_etude = document.getElementById("domaine_etude").value;

        const prenomSaisi = "Employe";
        const nomSaisi = employeIdCourant ? String(employeIdCourant) : "N/A";

        const payload = {
            modele: modeleChoisi,
            features: features,
            seuil: seuilSaisi,
            prenom: prenomSaisi,
            nom: nomSaisi,
            employe_id: employeIdCourant
        };

        try {
            if (!tokenCourant()) {
                alert("Connectez-vous pour lancer une prédiction.");
                return;
            }
            const response = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                headers: entetesAuth(),
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            const resultatDiv = document.getElementById("resultat");
            const resultatTexte = document.getElementById("resultat-texte");

            if (response.ok) {
                resultatDiv.style.display = "block";

                const probaDepart = data.probabilite * 100;
                const probaRester = 100 - probaDepart;
                const libellePrediction = data.prediction === 1 ? "⚠️ Quitte (1)" : "✅ Reste (0)";
                const libelleTexteBrut = data.prediction === 1 ? "Quitte" : "Reste";

                resultatTexte.innerHTML = `
                    <strong>Prédiction :</strong> ${libellePrediction} <br>
                    <strong>Risque de départ :</strong> ${probaDepart.toFixed(2)}% <br>
                    <strong>Probabilité de rester :</strong> ${probaRester.toFixed(2)}%
                `;

                dernierePredictionData = {
                    prenom: prenomSaisi,
                    nom: nomSaisi,
                    modele_utilise: data.modele_utilise,
                    probabilite_de_quitter: data.probabilite,
                    prediction: data.prediction,
                    libelle_prediction: libelleTexteBrut,
                    seuil_applique: data.seuil_utilise,
                    features: features
                };

                const msgSpan = document.getElementById("msg-sauvegarde");
                if (data.enregistrement && data.enregistrement.id) {
                    msgSpan.style.color = "#27ae60";
                    msgSpan.innerText = "✔ Enregistré automatiquement #" + data.enregistrement.id + " pour " + prenomSaisi + " " + nomSaisi;
                    ajouterResultatLocal({
                        id: data.enregistrement.id,
                        prenom: prenomSaisi,
                        nom: nomSaisi,
                        libelle_prediction: libelleTexteBrut,
                        probabilite_de_quitter: data.probabilite,
                        id_user: data.enregistrement.id_user
                    });
    
                } else {
                    msgSpan.innerText = "";
                }
            } else {
                alert("Erreur de l'API : " + JSON.stringify(data.detail));
            }
        } catch (error) {
            alert("Impossible de joindre l'API.");
            console.error(error);
        }
    });

