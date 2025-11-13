import os, io, glob, json, time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import unicodedata

from google.cloud import bigquery
from google.oauth2 import service_account
import asyncio

@st.cache_data(ttl=3600)
def get_mycapa_data(username: str, password: str, url: str) -> pd.DataFrame:
    """
    Déclenche la fonction asynchrone pour se connecter à MyCapa et télécharger le fichier.

    Attention: Le code Streamlit est synchrone, on utilise asyncio.run pour exécuter la tâche.
    Ceci peut ralentir l'application si l'exécution prend du temps.
    """

    # Créer un dossier temporaire pour le téléchargement
    DOWNLOAD_DIR = "temp_mycapa_downloads"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    try:
        # Exécute la fonction asynchrone
        df = asyncio.run(_run_playwright_download(username, password, url, DOWNLOAD_DIR))
        return df
    except Exception as e:
        st.error(f"🚨 Échec de l'agent MyCapa : {e}")
        return pd.DataFrame()


async def _run_playwright_download(username: str, password: str, url: str, download_path: str) -> pd.DataFrame:
    """
    Fonction asynchrone pour l'automatisation de la navigation et du téléchargement.
    """
    # ⚠️ REMPLACEZ CES SÉLECTEURS PAR LES VRAIS SÉLECTEURS HTML DE MYCAPA ⚠️
    SELECTORS = {
        "login_input": 'input[type="email"]', # Exemple générique
        "password_input": 'input[type="password"]', # Exemple générique
        "login_button": 'button[type="submit"]', # Exemple générique
        "capacity_page_link": 'a[href*="/capacity/exports"]', # Lien vers la page d'export
        "export_button": 'button:has-text("Exporter la Capacité")', # Bouton d'export
        "download_link": '.download-container a' # Exemple de lien final
    }

    async with async_playwright() as p:
        # Lance le navigateur en mode sans tête (headless=True) pour la vitesse
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        st.toast("🌐 Connexion à MyCapa en cours...")
        await page.goto(url)

        # 1. Connexion
        await page.fill(SELECTORS["login_input"], username)
        await page.fill(SELECTORS["password_input"], password)
        await page.click(SELECTORS["login_button"])

        # Attendre la navigation après la connexion (par exemple, vers le dashboard)
        await page.wait_for_url("**/dashboard**", timeout=30000)

        # 2. Navigation vers la page de Capacité/Export
        st.toast("🔗 Navigation vers la page d'export...")
        await page.click(SELECTORS["capacity_page_link"])
        await page.wait_for_selector(SELECTORS["export_button"])

        # 3. Déclenchement du Téléchargement
        # Ceci est la partie la plus critique et dépend de la réponse du site.
        async with page.expect_download(timeout=60000) as download_info:
            await page.click(SELECTORS["export_button"])

        download = await download_info.value

        # Enregistrement du fichier
        final_path = os.path.join(download_path, download.suggested_filename)
        await download.save_as(final_path)

        await browser.close()
        st.toast(f"✅ Fichier téléchargé et enregistré : {final_path}")

        # 4. Lecture et Nettoyage des données
        df_capa = pd.read_excel(final_path) # Assumer un format Excel

        # Nettoyage et standardisation (comme dans votre logique manuelle)
        df_capa = df_capa.rename(columns={
            "Nom_du_Projet_dans_MyCapa": "Projet",
            "Capacite_Engagee_Colonne": "Capacité Engagée"
        })
        df_capa["Projet"] = df_capa["Projet"].astype(str).str.strip().str.upper()

        return df_capa[['Projet', 'Capacité Engagée']].groupby('Projet').sum().reset_index()

# Notez que le code ci-dessus est générique. Vous devez l'intégrer à votre application.
def load_data_from_bigquery(project_id: str, query: str) -> pd.DataFrame:
    """
    Se connecte à BigQuery et exécute la requête, en utilisant les secrets Streamlit (.streamlit/secrets.toml).
    """

    # --- ÉTAPE 1 : AUTHENTIFICATION ---
    try:
        # Tente d'utiliser st.secrets pour l'authentification (doit correspondre à [gcp_service_account])
        credentials_info = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
    except KeyError:
        st.error("🚨 BigQuery : La clé de service [gcp_service_account] n'a pas été trouvée dans .streamlit/secrets.toml. Veuillez configurer le fichier.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"🚨 BigQuery : Erreur lors du chargement des secrets : {e}")
        return pd.DataFrame()


    # --- ÉTAPE 2 : CONNEXION & REQUÊTE ---
    client = bigquery.Client(project=project_id, credentials=credentials)

    try:
        # Exécute la requête
        query_job = client.query(query)
        df = query_job.result().to_dataframe()
        return df
    except Exception as e:
        st.error(f"🚨 Erreur BigQuery. Vérifiez votre requête SQL et les permissions: {e}")
        return pd.DataFrame()


st.set_page_config(page_title="AI Planning Hub (Effectif Réel)", layout="wide", initial_sidebar_state="expanded")

try:
    # Largeur réduite à 600 pixels
    st.image(IMAGE_BANNER_PATH, width=300)
except Exception:
    st.warning("⚠️ Image bannière non trouvée ou chemin inaccessible. Assurez-vous que le chemin G:\\... est correct et accessible localement.")

# --- FIN BANNIÈRE ---


DATA_DIR = "data"; OUT_DIR = "out"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
STATE_FILE = os.path.join(OUT_DIR, "validation_state.json")

# Style CSS V2 (Thème bleu marine/blanc)
st.markdown("""
<style>
/* Arrière-plan plus sobre */
.stApp {
    background-color: #ffffff; /* Fond blanc propre */
}
/* Entêtes */
h1, h2, h3, h4 {
    color: #004c8c; /* Bleu foncé/Marine (Plus institutionnel) */
    font-weight: 700;
}
/* Cartes d'information */
div[data-testid="stMetric"] > div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    color: #007bff; /* Bleu vif conservé */
}
div[data-testid="stMetric"] > div[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #495057; /* Gris sombre */
}
/* Conteneurs principaux/cartes */
.block-container {
    padding-top: 12px !important;
}
.card {
    background: #f8f9fa; /* Gris très clair pour les cartes */
    padding: .75rem 1rem;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); /* Ombre légère */
    border: 1px solid #e9ecef; /* Bordure très fine */
}
/* Messages d'information/notes */
.note {
    background:#e6f3ff; /* Bleu très clair */
    color:#004c8c; /* Texte bleu marine */
    padding:.6rem .8rem;
    border-left:4px solid #004c8c; /* Barre bleu marine */
    border-radius:.5rem;
    margin-bottom:.5rem;
    font-size: 0.9rem;
}
.stTabs [data-testid="stTab"] {
    color: #004c8c; /* Bleu marine pour les titres d'onglets */
    font-weight: 700;
}
/* Réduire la marge sous la bannière */
div.css-1r6dm7m, div.css-1r6dm7m > img {
    margin-bottom: -15px !important; /* Ajustement fin pour l'espace sous l'image */
}
</style>
""", unsafe_allow_html=True)

def read_any(path_or_file, sheet_name=None, dtype: Optional[Dict[str, type]] = None):
    """Lit CSV/XLSX depuis chemin ou uploader Streamlit, avec support de sheet_name et dtype."""
    if hasattr(path_or_file, "name"):
        n = path_or_file.name.lower()
        if n.endswith(".csv"):
            return pd.read_csv(path_or_file, dtype=dtype)
        if n.endswith((".xlsx",".xls")):
            # Utilise dtype pour la lecture Excel
            return pd.read_excel(path_or_file, sheet_name=sheet_name, engine="openpyxl", dtype=dtype)
    else:
        p = str(path_or_file).lower()
        if p.endswith(".csv"):
            return pd.read_csv(path_or_file, dtype=dtype)
        if p.endswith((".xlsx",".xls")):
            # Utilise dtype pour la lecture Excel
            return pd.read_excel(path_or_file, sheet_name=sheet_name, engine="openpyxl", dtype=dtype)
    raise ValueError("Format non supporté (CSV/XLSX)")

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export en CSV bytes avec BOM pour les accents."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def guess_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Match tolérant pour deviner le nom de colonne (ID, Projet, OU...)."""
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
        s = s.lower()
        for ch in [" ", "-", "_", "/", "\\", ":"]: s = s.replace(ch, "")
        return s
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        nc = _norm(cand)
        if nc in norm_map: return norm_map[nc]
    for key, real in norm_map.items():
        for cand in candidates:
            if _norm(cand) in key: return real
    return None

def initial_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage initial: supprime lignes NaN, normalise les noms de colonnes."""
    df = df.dropna(how='all')
    new_cols = {}
    for col in df.columns:
        clean_col = str(col).strip()
        clean_col = clean_col.replace('\n', ' ').replace('\r', '')
        new_cols[col] = clean_col
    df = df.rename(columns=new_cols)
    return df.copy()

# Initialisation de l'état de session
if "df_eff" not in st.session_state: st.session_state["df_eff"] = pd.DataFrame()
if "df_eff_filtered" not in st.session_state: st.session_state["df_eff_filtered"] = pd.DataFrame()
if "df_cong" not in st.session_state: st.session_state["df_cong"] = pd.DataFrame()
if "df_mapping" not in st.session_state: st.session_state["df_mapping"] = pd.DataFrame()
if "s1_total" not in st.session_state: st.session_state["s1_total"] = 0.0
if "s2_total" not in st.session_state: st.session_state["s2_total"] = 0.0
if "kpis_data" not in st.session_state: st.session_state["kpis_data"] = pd.DataFrame()

# ---------------------------------------------------------
# SIDEBAR — chemins (Placeholder) ET INFO UTILISATEUR
# ---------------------------------------------------------
st.sidebar.title("Sources")
default_g = r"G:\Drive partagés\DIRECTION WFM\WFM\Commun\Outil de travail\Modulation\Extractions"
g_dir = st.sidebar.text_input("Dossier G: (Effectif & MyCongé)", value=default_g)
st.sidebar.caption("Chemin d'accès principal pour les extractions.")

st.sidebar.markdown("---") # Séparateur

# --- AJOUT DE L'IDENTIFIANT WINDOWS ---
try:
    windows_id = os.environ.get('USERNAME')
    if windows_id:
        st.sidebar.info(f"👤 Utilisateur : **{windows_id}**")
    else:
        st.sidebar.warning("Nom d'utilisateur Windows non trouvé.")
except Exception as e:
    st.sidebar.error(f"Erreur d'accès à l'ID Windows : {e}")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
t0, tmap, t1, t2, t3, t4, t5, t6, tcapa = st.tabs([
    "① Effectifs & Congés (Filtrage) 👥",
    "② Mapping Planificateur ↔ Projet 🔗",
    "③ Inputs Center 📥",
    "④ Analyse 4 semaines 📊",
    "⑤ Scénarios S1/S2 🧮",
    "⑥ Validation ✅",
    "⑦ Répartition 📤",
    "⑧ Effectifs par OU/Projet 📋",
    "⑨ Capacité Engagée vs Réelle ⚖️"
])


with t0:
    st.header("📂 Étape 1 : Calcul de l'Effectif Réel Planifiable")

    st.markdown("""
        <div class='note'>
            **Séquence :** Nettoyage ➡️ Garder les agents **NON** en congés ➡️ Effectif Réel.
        </div>
    """, unsafe_allow_html=True)

    # 1. Lecture EFFECTIF (Feuille: The teams|Les équipes)
    EFF_SHEET = "The teams|Les équipes"
    eff_upload = st.file_uploader(f"Uploader Effectif Total (Feuille: **{EFF_SHEET}**)", type=["xlsx"], key="eff_upl_1")
    df_eff = pd.DataFrame()
    if eff_upload:
        try:
            # Deviner la colonne ID avant le chargement pour forcer le type string
            df_eff_temp = pd.read_excel(eff_upload, sheet_name=EFF_SHEET, nrows=1, engine="openpyxl")
            id_col_name = guess_col(df_eff_temp, ["Employee ID / Matricule", "Employee ID", "Matricule", "Employee ID(D)"])

            # --- CORRECTION ERREUR M50246 ---
            dtype_map = {id_col_name: str} if id_col_name else None

            # Lecture finale avec type string forcé sur la colonne ID
            df_eff = read_any(eff_upload, sheet_name=EFF_SHEET, dtype=dtype_map)
            df_eff = initial_data_cleaning(df_eff) # Nettoyage
            st.success(f"Effectif Total chargé et nettoyé: **{len(df_eff)}** lignes. (**Matricule ID forcé en texte**)")
        except Exception as e:
            st.error(f"Erreur lecture Effectif (Feuille: '{EFF_SHEET}'): {e}")

    # 2. Lecture CONGÉS (Feuille: Détail par agent - Agent Detail)
    CONG_SHEET = "Détail par agent - Agent Detail"
    cong_upload = st.file_uploader(f"Uploader Congés (Feuille: **{CONG_SHEET}**)", type=["xlsx"], key="cong_upl_1")
    df_cong = pd.DataFrame()
    if cong_upload:
        try:
            # Deviner la colonne ID avant le chargement pour forcer le type string
            df_cong_temp = pd.read_excel(cong_upload, sheet_name=CONG_SHEET, nrows=1, engine="openpyxl")
            id_col_name_cong = guess_col(df_cong_temp, ["Matricule/ID", "Matricule", "Employee ID", "Matricule"])

            # --- CORRECTION ERREUR M50246 ---
            dtype_map_cong = {id_col_name_cong: str} if id_col_name_cong else None

            # Lecture finale avec type string forcé sur la colonne ID
            df_cong = read_any(cong_upload, sheet_name=CONG_SHEET, dtype=dtype_map_cong)
            df_cong = initial_data_cleaning(df_cong) # Nettoyage
            st.success(f"Détail Congés chargé et nettoyé: **{len(df_cong)}** lignes. (**Matricule ID forcé en texte**)")
        except Exception as e:
            st.error(f"Erreur lecture Congés (Feuille: '{CONG_SHEET}'): {e}")

    st.session_state["df_eff"] = df_eff.copy()
    df_eff_filtered = pd.DataFrame()

    # --- Étape de Filtrage (Logique pour exclure les IDs en congés) ---
    if not df_eff.empty and not df_cong.empty:
        st.subheader("Résultats du Filtrage : Effectif Réel")

        # Deviner les colonnes ID
        id_eff_col = guess_col(df_eff, ["Employee ID / Matricule", "Employee ID", "Matricule", "Employee ID(D)"])
        id_cong_col = guess_col(df_cong, ["Matricule/ID", "Matricule", "Employee ID", "Matricule"])

        if not id_eff_col or not id_cong_col:
            st.error("🚨 Impossible de trouver la colonne **Matricule/ID** dans l'un des fichiers.")
        else:
            st.info(f"IDs utilisés : Effectif **'{id_eff_col}'** | Congés **'{id_cong_col}'**")

            # Agents à exclure
            absent_ids = df_cong[id_cong_col].astype(str).str.strip().unique()
            eff_ids = df_eff[id_eff_col].astype(str).str.strip()

            # Filtrage: seuls les IDs NON trouvés dans la liste des congés sont conservés
            mask_present = ~eff_ids.isin(absent_ids)
            df_eff_filtered = df_eff[mask_present].copy()

            # Affichage des résultats
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Effectif Total", len(df_eff))
            with c2: st.metric("Agents Absents Filtrés", len(df_eff) - len(df_eff_filtered))
            with c3: st.metric("**Effectif Réel Planifiable**", len(df_eff_filtered))

            st.session_state["df_eff_filtered"] = df_eff_filtered.copy()

            with st.expander("Aperçu de l'Effectif Réel Filtré"):
                st.dataframe(df_eff_filtered.head(50), use_container_width=True)

        # === Résumé : nombre d'agents par Projet & OU/UO (sur l'effectif réel filtré) ===
        ou_eff_col = guess_col(df_eff_filtered, ["OU/UO", "UO/OU", "OU", "UO"])
        proj_eff_col = guess_col(df_eff_filtered, ["Project/Projet", "Projects/Projets", "Project", "Projet"])
        id_eff_col2 = guess_col(
            df_eff_filtered,
            ["Employee ID / Matricule", "Employee ID", "Matricule", "Employee ID(D)"]
        )
        if ou_eff_col and proj_eff_col and id_eff_col2:
            df_tmp = df_eff_filtered.copy()
            # normaliser les IDs pour éviter les doublons (espaces, etc.)
            df_tmp[id_eff_col2] = df_tmp[id_eff_col2].astype(str).str.strip()
            nb_par_ou = (
                df_tmp
                .groupby([proj_eff_col, ou_eff_col])[id_eff_col2]
                .nunique()
                .reset_index(name="Nb Agents (ID uniques)")
                .sort_values("Nb Agents (ID uniques)", ascending=False)
            )
            with st.expander("Résumé : Nb d'agents par Projet & OU/UO (Effectif Réel)"):
                st.dataframe(nb_par_ou, use_container_width=True)
        else:
            st.info("Colonnes Projet / OU/UO / ID non trouvées pour le résumé par UO.")

    # Fallback
    if st.session_state["df_eff_filtered"].empty and not df_eff.empty:
        st.info("L'Effectif Total non filtré sera utilisé pour le mapping faute de données de congés.")
        st.session_state["df_eff_filtered"] = df_eff.copy()

# ---------------------------------------------------------
# ② MAPPING Planificateur → Projet (SÉQUENCE 2 : CALCUL)
# ---------------------------------------------------------
with tmap:
    st.header("🔗 Étape 2 : Croisement et Comptage par OU/UO")
    # UTILISE L'EFFECTIF FILTRÉ DE L'ÉTAPE 1
    df_eff_used  = st.session_state.get("df_eff_filtered",  pd.DataFrame())
    if df_eff_used.empty:
        st.error("🚨 Effectif Réel non disponible. Veuillez le charger dans l'onglet ①.")
        st.stop()
    # --- Chargement Mapping (inchangé) ---
    map_default = r"G:\Drive partagés\DIRECTION WFM\WFM\Prév&Stats\upload_extractions\Mapping\mapping_planif\fichier_mapping_planif_V2.xlsx"
    map_path = st.text_input("Chemin mapping (Planificateur ↔ Projet/File)", value=map_default, key="map_path_input")
    mdf = st.session_state.get("df_mapping", pd.DataFrame())
    # Bouton de chargement (simple)
    if st.button("Charger/Rafraîchir Mapping", key="refresh_map_btn"):
        try:
            mxls = pd.ExcelFile(map_path)
            m_sheet = st.selectbox("Feuille du mapping", options=mxls.sheet_names, index=0, key="map_sheet_select_reload")
            mdf_raw = pd.read_excel(map_path, sheet_name=m_sheet, engine="openpyxl")
            planif_col = guess_col(mdf_raw, ["Planificateur","Planner","Planif"]) or mdf_raw.columns[0]
            map_proj   = guess_col(mdf_raw, ["Projet","Project"]) or (mdf_raw.columns[1] if len(mdf_raw.columns)>1 else mdf_raw.columns[0])
            map_file   = guess_col(mdf_raw, ["File","Files","Fichier"])
            rename_map = {planif_col:"Planificateur", map_proj:"Projet"}
            keep = ["Planificateur","Projet"]
            if map_file: rename_map[map_file] = "File"; keep.append("File")
            mdf = mdf_raw.rename(columns=rename_map)[keep].copy()
            for c in keep: mdf[c] = mdf[c].astype(str).str.strip()
            st.session_state["df_mapping"] = mdf
            st.session_state["last_map_path"] = map_path
        except Exception as e:
            st.error(f"Erreur lecture Mapping : {e}")
    if mdf.empty: st.warning("Veuillez charger le fichier de mapping ci-dessus."); st.stop()
    # --- Sélection Planificateur ---
    st.markdown("---")
    planifs = sorted(mdf["Planificateur"].dropna().unique().tolist())
    selected_plan = st.selectbox("Sélectionner un Planificateur", options=planifs, key="planif_select_map")
    plan_map = mdf[mdf["Planificateur"] == selected_plan]
    # 1. Normalisation des critères du Mapping (Projet/File) en majuscules
    projects_set = {p.strip().upper() for p in plan_map["Projet"].dropna().astype(str)}
    files_set = {f.strip().upper() for f in plan_map["File"].dropna().astype(str)} if "File" in plan_map.columns else set()
    col_info1, col_info2 = st.columns(2)
    with col_info1: st.info(f"**Projets ({len(projects_set)})** : {', '.join(sorted(projects_set)[:5])}... (pour **{selected_plan}**)")
    if files_set:
        with col_info2: st.info(f"**Files (OU/UO - {len(files_set)})** : {', '.join(sorted(files_set)[:5])}... (pour **{selected_plan}**)")
    # Détection des colonnes Effectif
    proj_col = guess_col(df_eff_used, ["Project/Projet","Projects/Projets","Project","Projet"]) or df_eff_used.columns[0]
    ou_col   = guess_col(df_eff_used, ["OU/UO","UO/OU","OU","UO"]) or (df_eff_used.columns[1] if len(df_eff_used.columns)>1 else df_eff_used.columns[0])
    team_col = guess_col(df_eff_used, ["Team/Equipe","Equipe","Team"]) or (df_eff_used.columns[2] if len(df_eff_used.columns)>2 else df_eff_used.columns[0])
    st.markdown("---")
    # --- NOUVELLE OPTION DE FILTRAGE ---
    filter_mode = st.radio(
        "Mode de Filtrage (si le croisement échoue à 0)",
        ("Projet ET File (OU/UO)", "Projet SEULEMENT (pour debug OU si Mapping File est faux)"),
        index=0,
        key="filter_mode_select"
    )
    # --- Croisement avec Normalisation (Rend le matching insensible à la casse) ---
    # Colonnes Effectif nettoyées
    proj_stripped = df_eff_used[proj_col].fillna("").astype(str).str.strip()
    ou_stripped   = df_eff_used[ou_col].fillna("").astype(str).str.strip()
    # Colonnes Effectif en majuscules (pour le matching)
    proj_upper = proj_stripped.str.upper()
    ou_upper   = ou_stripped.str.upper()
    # Application des masques (matching insensible à la casse)
    mask_proj = proj_upper.isin(projects_set)
    mask_file = ou_upper.isin(files_set)
    # DÉCISION DU MASQUE FINAL BASÉE SUR L'OPTION CHOISIE
    if files_set and filter_mode == "Projet ET File (OU/UO)":
        mask_both = mask_proj & mask_file
        st.warning("⚠️ Attention: Si 'Lignes correspondant' est à 0, vous êtes probablement confronté à l'incohérence de nom (ex: MA- vs WA-) et devriez changer le mode de filtrage.")
    else:
        # Utilise le match Projet seulement (CONTOURNEMENT DU PROBLÈME MA-/WA-)
        mask_both = mask_proj
        st.info("ℹ️ Filtrage effectué uniquement par **Projet**. Le comptage par OU/UO sera affiché pour tous les agents de ce Projet.")
    filtered_both = df_eff_used.loc[mask_both].copy()
    # Affichage des métriques de matching
    st.subheader("Résultats du Croisement sur Effectif Réel")
    c_proj, c_file, c_final = st.columns(3)
    c_proj.metric("Lignes match Projet", int(mask_proj.sum()))
    c_file.metric("Lignes match File (OU/UO)", int(mask_file.sum()))
    c_final.metric("**Lignes correspondant (Final)**", len(filtered_both))
    if filtered_both.empty:
        st.warning("Aucun agent de l'effectif réel ne correspond à ce Planificateur après filtrage.")
        st.stop()
    # --- Étape Finale : Comptage TOTAL par Projet et OU/UO (VOTRE REQUÊTE) ---
    # Assigner les colonnes d'origine au DataFrame filtré (pour un affichage propre)
    filtered_both[ou_col] = ou_stripped[filtered_both.index]
    filtered_both[proj_col] = proj_stripped[filtered_both.index]
    # Comptage par Projet ET OU/UO
    total_par_ou = (
        filtered_both
        .groupby([proj_col, ou_col]).size().reset_index(name="Total Agents")
        .sort_values("Total Agents", ascending=False)
    )
    st.subheader("Total Agents Réels Planifiables par Projet et Unité Opérationnelle (OU/UO) 📊")
    st.dataframe(total_par_ou, use_container_width=True)
    # Export des totaux par OU/UO
    st.download_button(
        "💾 Export Total Agents par Projet et OU/UO (CSV)",
        data=to_csv_bytes(total_par_ou),
        file_name=f"total_agents_par_ou_projet_{selected_plan}.csv",
        mime="text/csv"
    )
    # Détail par Team/Equipe (facultatif)
    if team_col in filtered_both.columns:
        grp = (
            filtered_both
            .groupby([proj_col, ou_col, team_col]).size().reset_index(name="Nombre")
            .sort_values([ou_col, "Nombre"], ascending=[True, False])
        )
        with st.expander("Détail du Comptage par Team/Équipe"):
            st.dataframe(grp, use_container_width=True)

# ---------------------------------------------------------
# ⑧ EFFECTIFS PAR OU/PROJET - VUE GLOBALE
# ---------------------------------------------------------
with t6:
    st.header("📋 Effectifs par OU/UO et Projet - Vue Globale")
    st.markdown("""
    <div class='note'>
        Cette vue affiche pour <strong>chaque UO/OU le nombre d'agents affectés</strong> et le <strong>projet auquel ils appartiennent</strong>,
        sur la base de l'effectif réel planifiable (après exclusion des congés).
    </div>
    """, unsafe_allow_html=True)
    # Utiliser l'effectif filtré de la session
    df_eff_global = st.session_state.get("df_eff_filtered", pd.DataFrame())
    if df_eff_global.empty:
        st.warning("⚠️ Veuillez d'abord charger et filtrer les effectifs dans l'onglet '① Effectifs & Congés'")
        st.stop()
    # Détection des colonnes
    proj_col_global = guess_col(df_eff_global, ["Project/Projet","Projects/Projets","Project","Projet"])
    ou_col_global = guess_col(df_eff_global, ["OU/UO","UO/OU","OU","UO"])
    if not proj_col_global or not ou_col_global:
        st.error("🚨 Impossible de détecter les colonnes 'Projet' et 'OU/UO' dans le fichier d'effectif")
        st.info("Colonnes disponibles : " + ", ".join(df_eff_global.columns.tolist()))
        st.stop()
    st.success(f"✅ Colonnes détectées : **{proj_col_global}** (Projet) et **{ou_col_global}** (OU/UO)")
    # Calcul des effectifs par OU et Projet
    effectifs_par_ou_projet = (
        df_eff_global
        .groupby([ou_col_global, proj_col_global])
        .size()
        .reset_index(name="Nombre d'Agents")
        .sort_values([ou_col_global, "Nombre d'Agents"], ascending=[True, False])
    )
    # Statistiques globales
    total_agents = len(df_eff_global)
    total_ous = effectifs_par_ou_projet[ou_col_global].nunique()
    total_projets = effectifs_par_ou_projet[proj_col_global].nunique()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Agents Planifiables", total_agents)
    with col2:
        st.metric("Nombre d'OU/UO", total_ous)
    with col3:
        st.metric("Nombre de Projets", total_projets)
    st.markdown("---")
    # Affichage du tableau principal
    st.subheader("📊 Répartition des Effectifs par OU/UO et Projet")
    st.dataframe(effectifs_par_ou_projet, use_container_width=True)
    # Vue agrégée par OU seulement
    st.subheader("🧮 Total Agents par OU/UO")
    effectifs_par_ou = (
        effectifs_par_ou_projet
        .groupby(ou_col_global)["Nombre d'Agents"]
        .sum()
        .reset_index()
        .sort_values("Nombre d'Agents", ascending=False)
    )
    st.dataframe(effectifs_par_ou, use_container_width=True)
    # Vue agrégée par Projet seulement
    st.subheader("🏢 Total Agents par Projet")
    effectifs_par_projet = (
        effectifs_par_ou_projet
        .groupby(proj_col_global)["Nombre d'Agents"]
        .sum()
        .reset_index()
        .sort_values("Nombre d'Agents", ascending=False)
    )
    st.dataframe(effectifs_par_projet, use_container_width=True)
    # Export des données
    st.markdown("---")
    st.subheader("💾 Export des Données")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    with col_exp1:
        st.download_button(
            "Télécharger Effectifs par OU/Projet (CSV)",
            data=to_csv_bytes(effectifs_par_ou_projet),
            file_name="effectifs_par_ou_et_projet.csv",
            mime="text/csv"
        )
    with col_exp2:
        st.download_button(
            "Télécharger Total par OU (CSV)",
            data=to_csv_bytes(effectifs_par_ou),
            file_name="total_agents_par_ou.csv",
            mime="text/csv"
        )
    with col_exp3:
        st.download_button(
            "Télécharger Total par Projet (CSV)",
            data=to_csv_bytes(effectifs_par_projet),
            file_name="total_agents_par_projet.csv",
            mime="text/csv"
        )
    # Visualisations
    st.markdown("---")
    st.subheader("📈 Visualisations")
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.bar_chart(effectifs_par_ou.set_index(ou_col_global)["Nombre d'Agents"].head(10))
        st.caption("Top 10 des OU/UO par effectif")
    with viz_col2:
        st.bar_chart(effectifs_par_projet.set_index(proj_col_global)["Nombre d'Agents"].head(10))
        st.caption("Top 10 des projets par effectif")

# =========================================================
# ⑨ COMPARAISON CAPACITÉ (ENGAGÉE VS RÉELLE)
# =========================================================
with tcapa:
    st.header("⚖️ Comparaison Capacité : Engagée vs Réelle (API BQ)")

    col_intro1, col_intro2 = st.columns([0.7, 0.3])
    with col_intro1:
        st.markdown("""
            <div class='note'>
                La Capacité **Engagée** est chargée manuellement (MyCapa - Upload).
                La Capacité **Réelle** est chargée en direct depuis **BigQuery** (API BQ).
                **⚠️ N'oubliez pas d'ajuster** le nom de la table et des colonnes dans la requête SQL ci-dessous.
            </div>
        """, unsafe_allow_html=True)
    with col_intro2:
        # --- LIEN VERS MYCAPA (Raccourci) ---
        st.link_button("🔗 Accès MyCapa.Intelcia", url="https://mycapa.intelcia.com/#/clients", help="Cliquez pour accéder directement à l'outil MyCapa.")

    # --- Configuration BigQuery (VOTRE PROJECT_ID) ---
    BQ_PROJECT_ID = "dda-dpl-wfm-prd-hf"

    # --- REQUÊTE CAPACITÉ RÉELLE (À ADAPTER) ---
    BQ_CAPA_QUERY = f"""
        SELECT
            CAST(T1.Project_Name AS STRING) AS Projet,
            SUM(T1.Metric_Capacite) AS Capacité_Réelle
        FROM
            -- ⚠️ MODIFIEZ CECI : `dda-dpl-wfm-prd-hf.votre_dataset_gold.votre_table_kpis`
            `{BQ_PROJECT_ID}.votre_dataset_gold.votre_table_kpis` AS T1
        WHERE
            T1.Date_Reference = CURRENT_DATE() -- Adaptez la logique de date si nécessaire
        GROUP BY 1
    """

    st.markdown("### ⚙️ Requête SQL BigQuery (Capacité Réelle)")
    st.code(BQ_CAPA_QUERY, language="sql")

    c0_1, c0_2 = st.columns([3, 1])
    with c0_2:
        st.caption(f"Projet BQ : **{BQ_PROJECT_ID}**")
        if st.button("🔌 Exécuter Requête BigQuery", help="Déclenche la requête BQ et rafraîchit les données", key="refresh_capa_bq"):
            st.session_state["refresh_bq_capa"] = time.time() # Déclenche le refresh

    # ----------------------------------------------------
    # 1. Capacité Engagée (MyCapa - UPLOAD)
    # ----------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1️⃣ Capacité Engagée (MyCapa - Upload) 🤝")
        capa_engagee_upload = st.file_uploader("Uploader Capacité Engagée (CSV/XLSX)", type=["xlsx", "csv"], key="capa_eng_upl")
        df_eng = pd.DataFrame()

        PROJ_COLS = ["Projet", "Project", "Project/Projet"]
        CAPA_COLS = ["Effectif", "Capacité", "Volume", "Nb Agents"]

        if capa_engagee_upload:
            try:
                df_eng = read_any(capa_engagee_upload).copy()
                df_eng = initial_data_cleaning(df_eng)
                proj_col_eng = guess_col(df_eng, PROJ_COLS)
                capa_col_eng = guess_col(df_eng, CAPA_COLS)

                if proj_col_eng and capa_col_eng:
                    df_eng = df_eng.rename(columns={proj_col_eng: "Projet", capa_col_eng: "Capacité Engagée"})
                    df_eng["Projet"] = df_eng["Projet"].astype(str).str.strip().str.upper()
                    df_eng["Capacité Engagée"] = pd.to_numeric(df_eng["Capacité Engagée"], errors='coerce').fillna(0) # Nettoyage numérique
                    df_eng = df_eng[["Projet", "Capacité Engagée"]].groupby("Projet").sum().reset_index()
                    st.success(f"Capacité Engagée chargée pour {len(df_eng)} projets.")
                else:
                    st.error("Colonnes 'Projet' ou 'Capacité' non trouvées dans l'upload MyCapa.")
                    df_eng = pd.DataFrame()
            except Exception as e:
                st.error(f"Erreur lecture Capacité Engagée (MyCapa) : {e}")


    # ----------------------------------------------------
    # 2. Capacité Réelle (BigQuery - API)
    # ----------------------------------------------------
    with c2:
        st.subheader("2️⃣ Capacité Réelle (BigQuery - API) 🎯")
        df_reel = pd.DataFrame()

        if "refresh_bq_capa" not in st.session_state:
            st.session_state["refresh_bq_capa"] = 0

        # Lancement de la requête BigQuery
        # La fonction load_data_from_bigquery est mise en cache (ttl=3600), le bouton rafraîchit la session state
        df_reel_raw = load_data_from_bigquery(BQ_PROJECT_ID, BQ_CAPA_QUERY)

        if not df_reel_raw.empty:
            df_reel = df_reel_raw.copy()

            # Assurez-vous que les colonnes Projet et Capacité Réelle sont standardisées
            if "Projet" in df_reel.columns and "Capacité_Réelle" in df_reel.columns:
                df_reel["Projet"] = df_reel["Projet"].astype(str).str.strip().str.upper()
                df_reel["Capacité Réelle"] = pd.to_numeric(df_reel["Capacité Réelle"], errors='coerce').fillna(0) # Nettoyage numérique
                df_reel = df_reel.rename(columns={"Capacité_Réelle": "Capacité Réelle"})
                st.success(f"Capacité Réelle chargée pour {len(df_reel)} projets depuis BigQuery.")
            else:
                st.error("Colonnes 'Projet' ou 'Capacité_Réelle' non trouvées dans le résultat BigQuery. Vérifiez votre requête SQL.")
                df_reel = pd.DataFrame()
        elif "refresh_bq_capa" in st.session_state and st.session_state["refresh_bq_capa"] != 0:
             st.warning("La requête BigQuery n'a retourné aucune donnée. Vérifiez les permissions et la requête.")
        else:
             st.info("Cliquez sur 'Exécuter Requête BigQuery' pour charger les données réelles.")


    # 3. Comparaison et Affichage
    if not df_eng.empty and not df_reel.empty:
        st.markdown("---")
        st.subheader("Résultats de la Comparaison (Delta) 📈")

        # Fusion des deux DataFrames sur la colonne 'Projet'
        df_compare = pd.merge(
            df_eng,
            df_reel,
            on="Projet",
            how="outer"
        ).fillna(0)

        # Les colonnes sont déjà numériques grâce au nettoyage effectué précédemment
        df_compare["Delta (Réel - Engagé)"] = df_compare["Capacité Réelle"] - df_compare["Capacité Engagée"]

        # Gestion de la division par zéro
        with np.errstate(divide='ignore', invalid='ignore'):
            df_compare["% Delta (Réel / Engagé)"] = np.where(
                df_compare["Capacité Engagée"] != 0,
                (df_compare["Capacité Réelle"] / df_compare["Capacité Engagée"]) * 100,
                np.nan
            )

        # Formatage pour l'affichage
        df_compare = df_compare.sort_values("Delta (Réel - Engagé)", ascending=True)

        # Métriques globales
        total_eng = df_compare["Capacité Engagée"].sum()
        total_reel = df_compare["Capacité Réelle"].sum()
        total_delta = total_reel - total_eng

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Engagé", f"{total_eng:,.0f}")
        col_m2.metric("Total Réel", f"{total_reel:,.0f}")
        col_m3.metric(
            "Delta Global (Réel - Engagé)",
            f"{total_delta:,.0f}",
            delta=f"{total_delta:,.0f} agents"
        )

        st.markdown("---")

        # Affichage du tableau de comparaison
        st.subheader("Détail par Projet")
        display_df = df_compare.style.format({
            "Capacité Engagée": "{:,.0f}",
            "Capacité Réelle": "{:,.0f}",
            "Delta (Réel - Engagé)": "{:,.0f}",
            "% Delta (Réel / Engagé)": lambda x: f"{x:,.1f}%" if pd.notna(x) else "N/A"
        })

        st.dataframe(display_df, use_container_width=True)

        st.download_button(
            "💾 Télécharger la Comparaison (CSV)",
            data=to_csv_bytes(df_compare),
            file_name="comparaison_capacite_delta.csv",
            mime="text/csv"
        )

# =========================================================
# ⑥ VALIDATION (KPI BIGQUERY)
# =========================================================
with t4:
    st.header("✅ Étape 6 : Validation des KPIs (BigQuery)")
    st.markdown("""
    <div class='note'>
        Utilisez cette section pour charger les KPIs de validation (Ex: SLA, Taux de conformité)
        directement depuis votre entrepôt de données BigQuery.
    </div>
    """, unsafe_allow_html=True)

    # --- Configuration BigQuery (Utilise le même PROJECT_ID) ---
    BQ_PROJECT_ID = "dda-dpl-wfm-prd-hf"

    # --- REQUÊTE KPI DE VALIDATION (À ADAPTER) ---
    BQ_KPI_QUERY = f"""
        SELECT
            CAST(T2.Metric_Name AS STRING) AS KPI,
            CAST(T2.Value AS FLOAT64) AS Valeur,
            CAST(T2.Reference_Date AS DATE) AS Date
        FROM
            -- ⚠️ MODIFIEZ CECI : `dda-dpl-wfm-prd-hf.votre_dataset_gold.votre_table_validation_kpis`
            `{BQ_PROJECT_ID}.votre_dataset_gold.votre_table_validation_kpis` AS T2
        WHERE
            T2.Reference_Date = CURRENT_DATE() - 1 -- Exemple: KPIs de la veille
            AND T2.Metric_Name IN ('SLA', 'Conformité Planning', 'Adhérence')
    """

    st.markdown("### ⚙️ Requête SQL BigQuery pour KPIs de Validation")
    st.code(BQ_KPI_QUERY, language="sql")

    c_bq_btn, c_bq_info = st.columns([1, 3])
    with c_bq_btn:
        if st.button("🔌 Charger les KPIs de Validation", key="load_kpis_btn"):
            with st.spinner('Chargement des KPIs depuis BigQuery...'):
                df_kpis = load_data_from_bigquery(BQ_PROJECT_ID, BQ_KPI_QUERY)
                st.session_state["kpis_data"] = df_kpis
                st.session_state["kpis_load_time"] = datetime.now().strftime("%H:%M:%S")

    with c_bq_info:
        if "kpis_load_time" in st.session_state:
            st.success(f"✅ KPIs chargés. Dernier rafraîchissement : {st.session_state['kpis_load_time']}")

    # --- Affichage des KPIs ---
    df_kpis = st.session_state.get("kpis_data", pd.DataFrame())

    if not df_kpis.empty:
        st.markdown("---")
        st.subheader("Résultats des KPIs de Validation 🎯")

        # Affichage des métriques si le format est simple (KPI, Valeur, Date)
        if all(col in df_kpis.columns for col in ["KPI", "Valeur"]):
            # On prend un maximum de 4 colonnes pour la présentation
            unique_kpis = df_kpis["KPI"].unique()
            num_cols = min(len(unique_kpis), 4)
            kpi_cols = st.columns(num_cols)

            for i, kpi_name in enumerate(unique_kpis):
                row = df_kpis[df_kpis["KPI"] == kpi_name].iloc[0]
                kpi_value = row["Valeur"]

                # Formattage conditionnel : pourcentages
                if "SLA" in kpi_name or "Adhérence" in kpi_name or "Conformité" in kpi_name:
                    display_value = f"{kpi_value:.1f}%"
                else:
                    display_value = f"{kpi_value:,.2f}"

                delta_val = None # La colonne Target/Précédent n'est pas dans la requête actuelle

                with kpi_cols[i % num_cols]: # Assure la distribution des métriques
                    st.metric(kpi_name, display_value, delta=delta_val)

            st.markdown("---")
            st.info("Tableau détaillé des KPIs :")
            st.dataframe(df_kpis, use_container_width=True)

        else:
            st.error("Le format des colonnes KPI n'est pas standard (attendu: 'KPI', 'Valeur', 'Date'). Affichage brut :")
            st.dataframe(df_kpis, use_container_width=True)

    elif "kpis_load_time" in st.session_state and st.session_state["kpis_load_time"] != 0:
        st.warning("⚠️ La requête BigQuery a retourné un DataFrame vide. Vérifiez la requête et la disponibilité des données.")


# =========================================================
# PLACEHOLDERS POUR LES AUTRES ONGLETS
# =========================================================
with t1:
    st.header("📥 Inputs Center")
    st.info("Cette fonctionnalité sera implémentée ultérieurement")
with t2:
    st.header("📊 Analyse 4 semaines")
    st.info("Cette fonctionnalité sera implémentée ultérieurement")
with t3:
    st.header("🧮 Scénarios S1/S2")
    st.info("Cette fonctionnalité sera implémentée ultérieurement")
with t5:
    st.header("📤 Répartition")
    st.info("Cette fonctionnalité sera implémentée ultérieurement")
