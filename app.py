"""
Haloscan SEO Diff Analyzer
Version corrigée pour le format exact du fichier Baptiste
Avec intégration des données de leads par URL
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json

st.set_page_config(
    page_title="Haloscan SEO Diff Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Charge le CSV avec le bon séparateur (virgule)"""
    
    # Toujours utiliser la virgule comme séparateur
    try:
        df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=',', encoding='latin-1')
    
    # Nettoyage des noms de colonnes
    df.columns = (df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace(';', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace('é', 'e', regex=False)
        .str.replace('è', 'e', regex=False)
    )
    
    # Mapping vers noms standards
    mapping = {
        'mot-cle_(mc)': 'mot_cle',
        'plus_vieille_pos': 'ancienne_pos',
    }
    df = df.rename(columns=mapping)
    
    # Créer colonne 'volume' à partir de 'volumeh' si elle n'existe pas
    if 'volume' not in df.columns and 'volumeh' in df.columns:
        df['volume'] = df['volumeh']
    
    # Conversion numérique
    for col in ['derniere_pos', 'ancienne_pos', 'meilleure_pos', 'diff_pos', 'volume', 'volumeh', 'trafic', 'cpc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calcul du score de priorité
    vol = df['volume'].fillna(0) if 'volume' in df.columns else 0
    diff = df['diff_pos'].fillna(0).abs() if 'diff_pos' in df.columns else 0
    df['priority_score'] = vol * diff
    
    return df


def normalize_url(url):
    """Normalise une URL pour la comparaison"""
    if pd.isna(url):
        return ""
    url = str(url).lower().strip()
    # Retirer le protocole
    url = url.replace('https://', '').replace('http://', '')
    # Retirer www.
    url = url.replace('www.', '')
    # Retirer les doubles slashes (problème fréquent)
    while '//' in url:
        url = url.replace('//', '/')
    # Retirer le slash final
    url = url.rstrip('/')
    # Retirer le slash initial si présent
    url = url.lstrip('/')
    return url

# =============================================================================
# INTERFACE
# =============================================================================

st.title("📊 Haloscan SEO Diff Analyzer")

with st.sidebar:
    st.header("📁 Import des données")
    
    st.subheader("📊 Fichiers Haloscan")
    uploaded_file_p1 = st.file_uploader("1️⃣ CSV Haloscan Période 1", type=['csv'], key="haloscan_p1")
    uploaded_file_p2 = st.file_uploader("2️⃣ CSV Haloscan Période 2", type=['csv'], key="haloscan_p2")
    
    # Labels des périodes (personnalisables)
    if uploaded_file_p1 and uploaded_file_p2:
        st.caption("📅 Nommez vos périodes :")
        col1, col2 = st.columns(2)
        with col1:
            label_debut_p1 = st.text_input("Début P1", value="Jan 2025", key="label_debut_p1")
            label_fin_p1 = st.text_input("Fin P1 / Début P2", value="Sept 2025", key="label_fin_p1")
        with col2:
            label_fin_p2 = st.text_input("Fin P2", value="Fév 2026", key="label_fin_p2")
    else:
        label_debut_p1 = "Début P1"
        label_fin_p1 = "Fin P1"
        label_fin_p2 = "Fin P2"
    
    st.subheader("💰 Données business")
    uploaded_leads = st.file_uploader("3️⃣ Excel Leads par URL (optionnel)", type=['xlsx', 'xls'], 
                                       help="Fichier avec colonnes: url, puis une colonne par mois (YYYY_MM)")
    
    st.subheader("🔍 Search Console")
    uploaded_gsc = st.file_uploader("4️⃣ ZIP Search Console (optionnel)", type=['zip'],
                                     help="Export ZIP de Google Search Console (Performance)")
    
    st.subheader("🤖 Analyse IA")
    anthropic_api_key = st.text_input("Clé API Anthropic", type="password", 
                                       help="Pour générer l'analyse IA du rapport")

# Variables globales pour les leads
leads_df = None
has_leads = False
month_cols = []
periode_avant = []
periode_apres = []

# Variables pour le mode multi-périodes
has_dual_haloscan = False
df_p1 = None
df_p2 = None

# Variables pour Search Console
gsc_queries_df = None
gsc_pages_df = None
has_gsc = False

if uploaded_leads:
    # Lire la feuille "Leads totaux par urls" (pas la première feuille qui contient les visites)
    try:
        xlsx = pd.ExcelFile(uploaded_leads)
        
        # Afficher les feuilles disponibles
        st.sidebar.caption(f"Feuilles : {xlsx.sheet_names}")
        
        # Chercher la feuille des leads par son nom exact ou contenant "lead"
        leads_sheet = None
        for sheet in xlsx.sheet_names:
            if 'lead' in sheet.lower():
                leads_sheet = sheet
                break
        
        if leads_sheet:
            leads_df_raw = pd.read_excel(xlsx, sheet_name=leads_sheet)
            st.sidebar.success(f"📊 Feuille chargée : {leads_sheet} ({len(leads_df_raw)} lignes)")
        else:
            # IMPORTANT: La feuille des leads est généralement la 2ème (index 1)
            # La 1ère feuille (index 0) contient les visites
            if len(xlsx.sheet_names) > 1:
                leads_df_raw = pd.read_excel(xlsx, sheet_name=1)
                st.sidebar.success(f"📊 Feuille chargée : {xlsx.sheet_names[1]} ({len(leads_df_raw)} lignes)")
            else:
                leads_df_raw = pd.read_excel(xlsx, sheet_name=0)
                st.sidebar.warning(f"⚠️ Une seule feuille : {xlsx.sheet_names[0]}")
        
        # VÉRIFICATION : Les leads doivent avoir des valeurs faibles (< 1000 en général)
        # Si la moyenne est > 500, c'est probablement les visites
        month_cols_check = [c for c in leads_df_raw.columns if '2025' in str(c) or '2024' in str(c)]
        if month_cols_check:
            mean_val = leads_df_raw[month_cols_check].mean().mean()
            if mean_val > 500:
                st.sidebar.error(f"⚠️ ATTENTION : Moyenne = {mean_val:.0f} — Ce sont probablement les VISITES, pas les leads !")
                st.sidebar.info("Vérifiez que la feuille 'Leads totaux par urls' est bien dans le fichier")
            else:
                st.sidebar.info(f"✅ Moyenne = {mean_val:.1f} — Données leads OK")
        
        # Debug : afficher un aperçu pour confirmer
        with st.sidebar.expander("🔍 Vérification données leads", expanded=False):
            st.write(f"Feuilles disponibles : {xlsx.sheet_names}")
            st.write(f"Lignes : {len(leads_df_raw)}")
            # Trouver une colonne de mois pour montrer un exemple
            sample_cols = [c for c in leads_df_raw.columns if '2025' in str(c)][:2]
            if sample_cols and 'url' in leads_df_raw.columns:
                st.write(f"Exemple (premières lignes) :")
                st.dataframe(leads_df_raw[['url'] + sample_cols].head(3))
                
    except Exception as e:
        leads_df_raw = pd.read_excel(uploaded_leads)
        st.sidebar.warning(f"Lecture par défaut (erreur: {e})")
    
    # Identifier les colonnes de mois
    month_cols = [col for col in leads_df_raw.columns if col != 'url' and '_' in str(col)]
    month_cols_sorted = sorted(month_cols)
    
    has_leads = True
    
    # Fonction pour convertir un label ("Jan 2025", "Sept 2025") en format YYYY_MM
    def label_to_month_format(label):
        """Convertit 'Jan 2025' ou 'Janvier 2025' en '2025_01'"""
        mois_map = {
            'jan': '01', 'fev': '02', 'fév': '02', 'mar': '03', 'avr': '04', 'apr': '04',
            'mai': '05', 'may': '05', 'jun': '06', 'jui': '07', 'jul': '07',
            'aou': '08', 'aoû': '08', 'aug': '08', 'sep': '09', 'oct': '10', 
            'nov': '11', 'dec': '12', 'déc': '12'
        }
        label_lower = label.lower().strip()
        year = None
        month = None
        
        # Extraire l'année (4 chiffres)
        import re
        year_match = re.search(r'20\d{2}', label_lower)
        if year_match:
            year = year_match.group()
        
        # Extraire le mois
        for mois_key, mois_val in mois_map.items():
            if mois_key in label_lower:
                month = mois_val
                break
        
        if year and month:
            return f"{year}_{month}"
        return None
    
    # Détecter automatiquement les périodes si labels Haloscan sont définis
    auto_detected = False
    if uploaded_file_p1 and uploaded_file_p2:
        debut_p1_month = label_to_month_format(label_debut_p1)
        fin_p1_month = label_to_month_format(label_fin_p1)
        fin_p2_month = label_to_month_format(label_fin_p2)
        
        if debut_p1_month and fin_p1_month and fin_p2_month:
            # Période AVANT = du début P1 jusqu'à (fin P1 - 1 mois)
            # Période APRÈS = de fin P1 jusqu'à fin P2
            default_avant = [m for m in month_cols_sorted if debut_p1_month <= m < fin_p1_month]
            default_apres = [m for m in month_cols_sorted if fin_p1_month <= m <= fin_p2_month]
            
            if default_avant and default_apres:
                auto_detected = True
                st.sidebar.success(f"🎯 Périodes auto-détectées depuis labels Haloscan")
    
    # Si pas de détection auto, utiliser les valeurs par défaut
    if not auto_detected:
        default_avant = [c for c in month_cols_sorted if c.startswith('2025_09')]
        if not default_avant:
            default_avant = month_cols_sorted[-6:-3] if len(month_cols_sorted) >= 6 else month_cols_sorted[:3]
        
        default_apres = [c for c in month_cols_sorted if c.startswith('2025_11') or c.startswith('2026')]
        if not default_apres:
            default_apres = month_cols_sorted[-3:] if len(month_cols_sorted) >= 3 else month_cols_sorted[-1:]
    
    with st.sidebar:
        st.subheader("📅 Périodes leads à comparer")
        if auto_detected:
            st.caption(f"Basé sur vos labels : {label_debut_p1} → {label_fin_p1} → {label_fin_p2}")
        else:
            st.caption("Sélectionnez les mois correspondant à votre export Haloscan")
        
        # Période AVANT (ancienne position)
        st.markdown("**Période AVANT** (début analyse)")
        periode_avant = st.multiselect(
            "Mois période avant",
            options=month_cols_sorted,
            default=default_avant,
            key="avant"
        )
        
        # Période APRÈS (position actuelle)
        st.markdown("**Période APRÈS** (fin analyse)")
        periode_apres = st.multiselect(
            "Mois période après", 
            options=month_cols_sorted,
            default=default_apres,
            key="apres"
        )
    
    # Calculer les métriques leads sur les bonnes périodes
    leads_df = leads_df_raw.copy()
    
    # S'assurer que les colonnes de mois sont numériques
    for col in month_cols:
        if col in leads_df.columns:
            leads_df[col] = pd.to_numeric(leads_df[col], errors='coerce').fillna(0)
    
    # Créer les noms de colonnes dynamiques basés sur la sélection
    periode_avant_label = '+'.join(periode_avant) if periode_avant else 'N/A'
    periode_apres_label = '+'.join(periode_apres) if periode_apres else 'N/A'
    
    # Calculer les totaux sur TOUS les mois entre le début de période AVANT et la fin de période APRÈS
    if periode_avant and periode_apres:
        # Trouver le mois min (début période) et max (fin période)
        all_selected = periode_avant + periode_apres
        mois_min = min(all_selected)
        mois_max = max(all_selected)
        
        # Filtrer les colonnes de mois qui sont dans cette plage
        periode_complete = [m for m in month_cols_sorted if mois_min <= m <= mois_max]
        
        if periode_complete:
            leads_df['leads_total'] = leads_df[periode_complete].sum(axis=1)
            st.sidebar.caption(f"📊 Leads total : {mois_min} → {mois_max} ({len(periode_complete)} mois)")
        else:
            leads_df['leads_total'] = 0
    elif month_cols:
        leads_df['leads_total'] = leads_df[month_cols].sum(axis=1)
    else:
        leads_df['leads_total'] = 0
        
    leads_df['leads_avant'] = leads_df[periode_avant].sum(axis=1) if periode_avant else 0
    leads_df['leads_apres'] = leads_df[periode_apres].sum(axis=1) if periode_apres else 0
    leads_df['leads_evolution'] = leads_df['leads_apres'] - leads_df['leads_avant']
    leads_df['leads_evolution_pct'] = ((leads_df['leads_apres'] - leads_df['leads_avant']) / leads_df['leads_avant'].replace(0, 1) * 100).round(1)
    
    leads_df['url_normalized'] = leads_df['url'].apply(normalize_url)
    
    st.sidebar.success(f"✅ {len(leads_df):,} URLs avec données leads")
    if periode_avant and periode_apres:
        st.sidebar.info(f"Comparaison : {periode_avant_label} → {periode_apres_label}")

# Charger Search Console si uploadé
if uploaded_gsc:
    import zipfile
    import io
    
    try:
        with zipfile.ZipFile(uploaded_gsc, 'r') as z:
            # Chercher les fichiers Requêtes et Pages
            files_in_zip = z.namelist()
            
            queries_file = None
            pages_file = None
            
            for f in files_in_zip:
                if 'Requêtes' in f or 'Queries' in f or 'requetes' in f.lower():
                    queries_file = f
                elif 'Pages' in f or 'pages' in f.lower():
                    pages_file = f
            
            # Charger Requêtes
            if queries_file:
                with z.open(queries_file) as qf:
                    gsc_queries_df = pd.read_csv(qf)
                    # Normaliser les noms de colonnes
                    gsc_queries_df.columns = gsc_queries_df.columns.str.strip()
                    # Renommer la première colonne en 'query'
                    first_col = gsc_queries_df.columns[0]
                    gsc_queries_df = gsc_queries_df.rename(columns={first_col: 'query'})
                    # Convertir CTR en float
                    if 'CTR' in gsc_queries_df.columns:
                        gsc_queries_df['CTR'] = gsc_queries_df['CTR'].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
                    # Normaliser les requêtes pour le matching
                    gsc_queries_df['query_normalized'] = gsc_queries_df['query'].str.lower().str.strip()
            
            # Charger Pages
            if pages_file:
                with z.open(pages_file) as pf:
                    gsc_pages_df = pd.read_csv(pf)
                    # Normaliser les noms de colonnes
                    gsc_pages_df.columns = gsc_pages_df.columns.str.strip()
                    # Renommer la première colonne en 'url'
                    first_col = gsc_pages_df.columns[0]
                    gsc_pages_df = gsc_pages_df.rename(columns={first_col: 'url'})
                    # Convertir CTR en float
                    if 'CTR' in gsc_pages_df.columns:
                        gsc_pages_df['CTR'] = gsc_pages_df['CTR'].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
                    # Normaliser les URLs pour le matching
                    gsc_pages_df['url_normalized'] = gsc_pages_df['url'].apply(normalize_url)
            
            if gsc_queries_df is not None or gsc_pages_df is not None:
                has_gsc = True
                gsc_info = []
                if gsc_queries_df is not None:
                    gsc_info.append(f"{len(gsc_queries_df):,} requêtes")
                if gsc_pages_df is not None:
                    gsc_info.append(f"{len(gsc_pages_df):,} pages")
                st.sidebar.success(f"🔍 GSC : {' | '.join(gsc_info)}")
            else:
                st.sidebar.warning("⚠️ Fichiers Requêtes/Pages non trouvés dans le ZIP")
                
    except Exception as e:
        st.sidebar.error(f"❌ Erreur lecture ZIP GSC: {e}")

# Déterminer le mode de fonctionnement
uploaded_file = None
if uploaded_file_p1 and uploaded_file_p2:
    # Mode double période
    has_dual_haloscan = True
    st.sidebar.success("📊 Mode double période activé")
elif uploaded_file_p1:
    # Mode simple avec P1
    uploaded_file = uploaded_file_p1
elif uploaded_file_p2:
    # Mode simple avec P2
    uploaded_file = uploaded_file_p2

# Charger et fusionner les données si mode double période
if has_dual_haloscan:
    df_p1 = load_data(uploaded_file_p1)
    df_p2 = load_data(uploaded_file_p2)
    
    # Renommer les colonnes de position pour P1
    df_p1 = df_p1.rename(columns={
        'ancienne_pos': 'pos_debut_p1',
        'derniere_pos': 'pos_fin_p1',
        'diff_pos': 'diff_p1'
    })
    
    # Renommer les colonnes de position pour P2
    df_p2 = df_p2.rename(columns={
        'ancienne_pos': 'pos_debut_p2',
        'derniere_pos': 'pos_fin_p2',
        'diff_pos': 'diff_p2'
    })
    
    # Fusionner sur mot_cle + url
    df = df_p1.merge(
        df_p2[['mot_cle', 'url', 'pos_debut_p2', 'pos_fin_p2', 'diff_p2']],
        on=['mot_cle', 'url'],
        how='outer',
        suffixes=('', '_p2')
    )
    
    # Calculer les colonnes consolidées
    # Position de départ = pos_debut_p1 (ou pos_debut_p2 si pas de P1)
    df['ancienne_pos'] = df['pos_debut_p1'].fillna(df['pos_debut_p2'])
    # Position finale = pos_fin_p2 (ou pos_fin_p1 si pas de P2)
    df['derniere_pos'] = df['pos_fin_p2'].fillna(df['pos_fin_p1'])
    # Diff totale : positif = gain (ancienne - dernière, car passer de 96 à 1 = +95)
    df['diff_pos'] = df['ancienne_pos'] - df['derniere_pos']
    
    # Calculer la tendance multi-période
    def calc_tendance_multi(row):
        d1 = row.get('diff_p1', 0) or 0
        d2 = row.get('diff_p2', 0) or 0
        
        if pd.isna(d1): d1 = 0
        if pd.isna(d2): d2 = 0
        
        if d1 < -5 and d2 < -5:
            return "📉📉 Chute continue"
        elif d1 > 5 and d2 < -5:
            return "📈📉 Rebond puis rechute"
        elif d1 < -5 and d2 > 5:
            return "📉📈 Récupération"
        elif d1 > 5 and d2 > 5:
            return "📈📈 Hausse continue"
        elif abs(d1) <= 5 and abs(d2) <= 5:
            return "➡️ Stable"
        elif d1 < 0 or d2 < 0:
            return "📉 Baisse"
        else:
            return "📈 Hausse"
    
    df['tendance_multi'] = df.apply(calc_tendance_multi, axis=1)
    
    # Recalculer le volume si nécessaire
    if 'volume' not in df.columns and 'volumeh' in df.columns:
        df['volume'] = df['volumeh']
    
    # Recalculer priority_score
    if 'volume' in df.columns:
        df['priority_score'] = df['volume'].fillna(0) * df['diff_pos'].abs().fillna(0)
    else:
        df['priority_score'] = df['diff_pos'].abs().fillna(0)
    
    st.sidebar.info(f"🔗 {len(df):,} KW fusionnés (P1: {len(df_p1):,} | P2: {len(df_p2):,})")

elif uploaded_file:
    df = load_data(uploaded_file)
    has_dual_haloscan = False

# Suite du traitement si on a des données
if (has_dual_haloscan or uploaded_file) and 'df' in dir():
    
    # Croiser avec les données leads si disponibles
    if has_leads and 'url' in df.columns:
        df['url_normalized'] = df['url'].apply(normalize_url)
        df = df.merge(
            leads_df[['url_normalized', 'leads_total', 'leads_avant', 'leads_apres', 'leads_evolution', 'leads_evolution_pct']], 
            on='url_normalized', 
            how='left'
        )
        
        # Stocker les labels de période pour l'affichage
        df.attrs['periode_avant_label'] = periode_avant_label
        df.attrs['periode_apres_label'] = periode_apres_label
        
        # Créer indicateur visuel de tendance leads
        def tendance_leads(row):
            evol = row.get('leads_evolution', 0) or 0
            pct = row.get('leads_evolution_pct', 0) or 0
            if evol < -10 or pct < -20:
                return "🔻🔻 CHUTE"
            elif evol < 0:
                return "🔻 Baisse"
            elif evol == 0:
                return "➡️ Stable"
            elif evol > 10 or pct > 20:
                return "🔺🔺 BOOM"
            else:
                return "🔺 Hausse"
        
        df['tendance_leads'] = df.apply(tendance_leads, axis=1)
        
        # Score de priorité enrichi : 
        # - priority_score = volume recherche × |diff_pos|
        # - On booste si l'URL génère des leads (leads_total)
        # - On booste ENCORE PLUS si les leads sont en baisse (leads_evolution < 0)
        base_score = df['priority_score']
        leads_boost = (1 + df['leads_total'].fillna(0) / 100)  # Plus de leads = plus important
        
        # Malus si les leads baissent (évolution négative)
        leads_trend = df['leads_evolution'].fillna(0)
        trend_multiplier = 1 + (leads_trend.clip(upper=0).abs() / 100)  # Perte de leads = urgence
        
        df['priority_score_business'] = base_score * leads_boost * trend_multiplier
        
        # Flag pour identifier les URLs en double peine (perte SEO + perte leads)
        df['double_peine'] = (df['diff_pos'] < 0) & (df['leads_evolution'] < 0)
        
        # Créer indicateur visuel de tendance SEO (positions)
        def tendance_seo(diff):
            if pd.isna(diff):
                return "➡️ N/A"
            diff = int(diff)
            if diff <= -20:
                return "🔻🔻 CHUTE"
            elif diff < 0:
                return "🔻 Baisse"
            elif diff == 0:
                return "➡️ Stable"
            elif diff >= 20:
                return "🔺🔺 BOOM"
            else:
                return "🔺 Hausse"
        
        df['tendance_seo'] = df['diff_pos'].apply(tendance_seo)
        
        st.success(f"✅ {len(df):,} mots-clés chargés — Données leads croisées !")
        
        # Stats de matching
        urls_avec_leads = df[df['leads_total'].notna() & (df['leads_total'] > 0)]['url'].nunique()
        urls_double_peine = df[df['double_peine'] == True]['url'].nunique()
        st.info(f"📊 {urls_avec_leads} URLs avec leads | ⚠️ {urls_double_peine} URLs en double peine (perte SEO + perte leads)")
        
        has_leads_merged = True
    else:
        df['leads_total'] = 0
        df['leads_avant'] = 0
        df['leads_apres'] = 0
        df['leads_evolution'] = 0
        df['tendance_leads'] = "➡️ N/A"
        
        # Créer indicateur visuel de tendance SEO (positions) même sans leads
        def tendance_seo(diff):
            if pd.isna(diff):
                return "➡️ N/A"
            diff = int(diff)
            if diff <= -20:
                return "🔻🔻 CHUTE"
            elif diff < 0:
                return "🔻 Baisse"
            elif diff == 0:
                return "➡️ Stable"
            elif diff >= 20:
                return "🔺🔺 BOOM"
            else:
                return "🔺 Hausse"
        
        df['tendance_seo'] = df['diff_pos'].apply(tendance_seo)
        
        if has_leads:
            st.warning("⚠️ Fichier leads chargé mais colonne 'url' manquante dans le CSV Haloscan")
        st.success(f"✅ {len(df):,} mots-clés chargés")
        has_leads_merged = False
    
    # Debug colonnes
    with st.sidebar:
        with st.expander("🔍 Colonnes", expanded=True):
            st.write(list(df.columns))
    
    # Vérification diff_pos
    if 'diff_pos' not in df.columns:
        st.error(f"❌ Colonne 'diff_pos' non trouvée. Colonnes: {list(df.columns)}")
        st.stop()
    
    # ==========================================================================
    # FILTRES
    # ==========================================================================
    
    with st.sidebar:
        st.header("🎛️ Filtres")
        
        variation = st.multiselect("Variation", ['Pertes', 'Gains', 'Stables'], default=['Pertes', 'Gains', 'Stables'])
        
        if 'volume' in df.columns:
            vmin, vmax = int(df['volume'].min() or 0), int(df['volume'].max() or 10000)
            vol_range = st.slider("Volume", vmin, vmax, (vmin, vmax))
        else:
            vol_range = None
        
        search_kw = st.text_input("🔎 Mot-clé")
        search_url = st.text_input("🔎 URL contient")
    
    # Appliquer filtres
    df_f = df.copy()
    
    # Filtre variation
    masks = []
    if 'Pertes' in variation:
        masks.append(df_f['diff_pos'] < 0)
    if 'Gains' in variation:
        masks.append(df_f['diff_pos'] > 0)
    if 'Stables' in variation:
        masks.append(df_f['diff_pos'] == 0)
    if masks:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        df_f = df_f[combined]
    
    # Filtre volume
    if vol_range and 'volume' in df_f.columns:
        df_f = df_f[(df_f['volume'] >= vol_range[0]) & (df_f['volume'] <= vol_range[1])]
    
    # Filtre recherche
    if search_kw and 'mot_cle' in df_f.columns:
        df_f = df_f[df_f['mot_cle'].astype(str).str.contains(search_kw, case=False, na=False)]
    if search_url and 'url' in df_f.columns:
        df_f = df_f[df_f['url'].astype(str).str.contains(search_url, case=False, na=False)]
    
    # ==========================================================================
    # KPIs
    # ==========================================================================
    
    total = len(df_f)
    pertes = len(df_f[df_f['diff_pos'] < 0])
    gains = len(df_f[df_f['diff_pos'] > 0])
    stables = len(df_f[df_f['diff_pos'] == 0])
    
    vol_perdu = int(df_f[df_f['diff_pos'] < 0]['volume'].fillna(0).sum()) if 'volume' in df_f.columns else 0
    vol_gagne = int(df_f[df_f['diff_pos'] > 0]['volume'].fillna(0).sum()) if 'volume' in df_f.columns else 0
    
    # ==========================================================================
    # ONGLETS
    # ==========================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Dashboard", "🔴 Pertes", "📁 Par URL", "🟢 Gains", "🔄 Cannibalisation", "🔍 Search Console", "📝 Rapport"])
    
    # TAB 1: DASHBOARD
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", f"{total:,}")
        c2.metric("🔴 Pertes", f"{pertes:,}")
        c3.metric("🟢 Gains", f"{gains:,}")
        c4.metric("⚪ Stables", f"{stables:,}")
        
        st.divider()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📉 Volume perdu", f"{vol_perdu:,}")
        c2.metric("📈 Volume gagné", f"{vol_gagne:,}")
        
        # Métriques leads si disponibles
        if has_leads_merged:
            # Leads sur les URLs en perte - NE PAS compter plusieurs fois la même URL
            df_pertes_dash = df_f[df_f['diff_pos'] < 0]
            df_urls_perte_unique = df_pertes_dash.drop_duplicates(subset=['url']) if 'url' in df_pertes_dash.columns else df_pertes_dash
            
            leads_urls_perte = df_urls_perte_unique['leads_total'].fillna(0).sum()
            c3.metric("⚠️ Leads sur URLs en perte", f"{int(leads_urls_perte):,}")
            
            leads_evol = df_urls_perte_unique['leads_evolution'].fillna(0).sum()
            delta_color = "inverse" if leads_evol < 0 else "normal"
            c4.metric("📊 Évol. leads (période)", f"{int(leads_evol):+,}", delta_color=delta_color)
        
        # Section MULTI-PÉRIODES si disponible
        if has_dual_haloscan and 'tendance_multi' in df_f.columns:
            st.divider()
            st.subheader(f"📈 Analyse multi-périodes ({label_debut_p1} → {label_fin_p1} → {label_fin_p2})")
            
            # Compter les tendances
            tendances = df_f['tendance_multi'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📉📉 Chute continue", f"{tendances.get('📉📉 Chute continue', 0):,}", help="Perte P1 ET perte P2")
            col2.metric("📈📉 Rebond puis rechute", f"{tendances.get('📈📉 Rebond puis rechute', 0):,}", help="Gain P1 puis perte P2")
            col3.metric("📉📈 Récupération", f"{tendances.get('📉📈 Récupération', 0):,}", help="Perte P1 puis gain P2")
            col4.metric("📈📈 Hausse continue", f"{tendances.get('📈📈 Hausse continue', 0):,}", help="Gain P1 ET gain P2")
            
            # Tableau des KW en chute continue (priorité max)
            df_chute_continue = df_f[df_f['tendance_multi'] == '📉📉 Chute continue'].copy()
            if len(df_chute_continue) > 0:
                st.error(f"🚨 **{len(df_chute_continue):,}** mots-clés en CHUTE CONTINUE — Problème structurel à traiter !")
                
                # Afficher les colonnes pertinentes
                cols_multi = ['mot_cle', 'url', 'pos_debut_p1', 'pos_fin_p1', 'diff_p1', 'pos_fin_p2', 'diff_p2', 'diff_pos', 'volume']
                cols_multi = [c for c in cols_multi if c in df_chute_continue.columns]
                
                # Renommer pour clarté avec labels dynamiques
                df_chute_display = df_chute_continue[cols_multi].head(50).copy()
                rename_map = {
                    'pos_debut_p1': f'Pos {label_debut_p1}',
                    'pos_fin_p1': f'Pos {label_fin_p1}',
                    'diff_p1': f'Δ P1',
                    'pos_fin_p2': f'Pos {label_fin_p2}',
                    'diff_p2': f'Δ P2',
                    'diff_pos': 'Δ TOTAL',
                    'volume': 'Volume'
                }
                df_chute_display = df_chute_display.rename(columns=rename_map)
                
                st.dataframe(df_chute_display.sort_values('Δ TOTAL', ascending=True), use_container_width=True, height=300)
        
        # Section DOUBLE PEINE (suite du code existant)
        if has_leads_merged:
            if 'double_peine' in df_f.columns:
                df_double_peine = df_f[df_f['double_peine'] == True]
                if len(df_double_peine) > 0:
                    st.divider()
                    st.subheader("🚨 ALERTE : URLs en DOUBLE PEINE (perte SEO + perte leads)")
                    st.error(f"**{df_double_peine['url'].nunique()}** URLs perdent à la fois des positions ET des leads !")
                    
                    # Récupérer les labels de période
                    p_avant = df.attrs.get('periode_avant_label', 'AVANT')
                    p_apres = df.attrs.get('periode_apres_label', 'APRÈS')
                    
                    # Tableau des URLs double peine
                    agg_dp = {'diff_pos': ['count', 'sum']}
                    if 'tendance_seo' in df_double_peine.columns:
                        agg_dp['tendance_seo'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "➡️ N/A"
                    if 'leads_avant' in df_double_peine.columns:
                        agg_dp['leads_avant'] = 'first'
                    if 'leads_apres' in df_double_peine.columns:
                        agg_dp['leads_apres'] = 'first'
                    if 'leads_evolution' in df_double_peine.columns:
                        agg_dp['leads_evolution'] = 'first'
                    if 'tendance_leads' in df_double_peine.columns:
                        agg_dp['tendance_leads'] = 'first'
                    
                    df_dp_urls = df_double_peine.groupby('url').agg(agg_dp).reset_index()
                    df_dp_urls.columns = ['URL', 'KW perdus', 'Diff total'] + \
                                        (['📊 SEO'] if 'tendance_seo' in df_double_peine.columns else []) + \
                                        ([f'Leads {p_avant}'] if 'leads_avant' in df_double_peine.columns else []) + \
                                        ([f'Leads {p_apres}'] if 'leads_apres' in df_double_peine.columns else []) + \
                                        (['Évol. Leads'] if 'leads_evolution' in df_double_peine.columns else []) + \
                                        (['📊 LEADS'] if 'tendance_leads' in df_double_peine.columns else [])
                    
                    # Trier par évolution leads (les plus grosses pertes en premier)
                    if 'Évol. Leads' in df_dp_urls.columns:
                        df_dp_urls = df_dp_urls.sort_values('Évol. Leads', ascending=True)
                    elif 'Diff total' in df_dp_urls.columns:
                        df_dp_urls = df_dp_urls.sort_values('Diff total', ascending=True)
                    
                    st.dataframe(df_dp_urls.head(20), use_container_width=True, hide_index=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=[pertes, gains, stables], names=['Pertes', 'Gains', 'Stables'],
                        color_discrete_sequence=['#EF4444', '#22C55E', '#6B7280'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df_f, x='diff_pos', nbins=50)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top URLs impactées avec leads
        try:
            df_pertes_temp = df_f[df_f['diff_pos'] < 0]
            if has_leads_merged and len(df_pertes_temp) > 0 and 'leads_total' in df_pertes_temp.columns:
                st.subheader("🎯 URLs critiques : Pertes SEO + Impact Business")
                
                # Construire l'agrégation dynamiquement selon les colonnes disponibles
                agg_dict_dash = {'diff_pos': ('diff_pos', 'count')}
                if 'volume' in df_pertes_temp.columns:
                    agg_dict_dash['volume_perdu'] = ('volume', 'sum')
                if 'leads_total' in df_pertes_temp.columns:
                    agg_dict_dash['leads_total'] = ('leads_total', 'first')
                if 'leads_evolution' in df_pertes_temp.columns:
                    agg_dict_dash['leads_evolution'] = ('leads_evolution', 'first')
                if 'tendance_leads' in df_pertes_temp.columns:
                    agg_dict_dash['tendance_leads'] = ('tendance_leads', 'first')
                
                df_perte_urls = df_pertes_temp.groupby('url').agg(**agg_dict_dash).reset_index()
                df_perte_urls = df_perte_urls.rename(columns={'diff_pos': 'kw_perdus'})
                
                # Trier par évolution leads (les plus grosses pertes en premier)
                if 'leads_evolution' in df_perte_urls.columns:
                    df_perte_urls = df_perte_urls.sort_values('leads_evolution', ascending=True).head(15)
                else:
                    df_perte_urls = df_perte_urls.sort_values('kw_perdus', ascending=False).head(15)
                
                st.dataframe(df_perte_urls, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible d'afficher les URLs critiques: {e}")
    
    # TAB 2: PERTES
    with tab2:
        st.header("🔴 Pertes critiques")
        df_pertes = df_f[df_f['diff_pos'] < 0].sort_values('diff_pos', ascending=True)
        st.info(f"**{len(df_pertes):,}** mots-clés en perte")
        
        cols = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'tendance_seo', 'volume'] if c in df_pertes.columns]
        st.dataframe(df_pertes[cols], use_container_width=True, height=600)
        
        csv = df_pertes[cols].to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("📥 Export CSV", csv, "pertes.csv")
    
    # TAB 3: PAR URL
    with tab3:
        st.header("📁 Analyse par URL")
        if 'url' in df_f.columns:
            try:
                # Construire l'agrégation dynamiquement
                agg_funcs = {
                    'diff_pos': ['count', lambda x: (x < 0).sum(), lambda x: (x > 0).sum(), 'sum'],
                }
                if 'volume' in df_f.columns:
                    agg_funcs['volume'] = 'sum'
                if has_leads_merged:
                    if 'leads_total' in df_f.columns:
                        agg_funcs['leads_total'] = 'first'
                    if 'leads_avant' in df_f.columns:
                        agg_funcs['leads_avant'] = 'first'
                    if 'leads_apres' in df_f.columns:
                        agg_funcs['leads_apres'] = 'first'
                    if 'leads_evolution' in df_f.columns:
                        agg_funcs['leads_evolution'] = 'first'
                    if 'tendance_leads' in df_f.columns:
                        agg_funcs['tendance_leads'] = 'first'
                
                url_stats = df_f.groupby('url').agg(agg_funcs).reset_index()
                
                # Aplatir les colonnes multi-index
                new_cols = ['url']
                for col in url_stats.columns[1:]:
                    if isinstance(col, tuple):
                        new_cols.append(f"{col[0]}_{col[1]}" if col[1] != 'sum' and col[1] != 'first' else col[0])
                    else:
                        new_cols.append(col)
                url_stats.columns = new_cols
                
                # Renommer les colonnes pour plus de clarté
                rename_dict = {
                    'diff_pos_count': 'total_kw',
                    'diff_pos_<lambda_0>': 'kw_perte', 
                    'diff_pos_<lambda_1>': 'kw_gain',
                    'diff_pos_sum': 'diff_total'
                }
                url_stats = url_stats.rename(columns=rename_dict)
                
                # Ajouter indicateur tendance SEO basé sur diff_total
                def tendance_seo_url(diff):
                    if pd.isna(diff):
                        return "➡️ N/A"
                    diff = int(diff)
                    if diff <= -50:
                        return "🔻🔻 CHUTE"
                    elif diff < 0:
                        return "🔻 Baisse"
                    elif diff == 0:
                        return "➡️ Stable"
                    elif diff >= 50:
                        return "🔺🔺 BOOM"
                    else:
                        return "🔺 Hausse"
                
                if 'diff_total' in url_stats.columns:
                    url_stats['📊 SEO'] = url_stats['diff_total'].apply(tendance_seo_url)
                
                # Ajouter tendance leads si dispo
                if 'tendance_leads' in url_stats.columns:
                    url_stats = url_stats.rename(columns={'tendance_leads': '📊 LEADS'})
                
                # Tri par évolution leads ou par nombre de KW en perte
                if 'leads_evolution' in url_stats.columns:
                    url_stats = url_stats.sort_values('leads_evolution', ascending=True)
                elif 'kw_perte' in url_stats.columns:
                    url_stats = url_stats.sort_values('kw_perte', ascending=False)
                else:
                    url_stats = url_stats.sort_values('total_kw', ascending=False)
                
                st.info(f"**{len(url_stats):,}** URLs analysées")
                st.dataframe(url_stats, use_container_width=True, height=500)
                
                # Export
                csv_urls = url_stats.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button("📥 Exporter TOUTES les URLs (CSV)", csv_urls, "analyse_urls_complete.csv")
                
            except Exception as e:
                st.error(f"Erreur lors de l'analyse par URL: {e}")
            
            st.divider()
            
            # Détail URL
            st.subheader("🔍 Détail d'une URL")
            url_list = df_f['url'].unique().tolist()[:100]
            url_sel = st.selectbox("Sélectionner une URL", url_list)
            if url_sel:
                df_url = df_f[df_f['url'] == url_sel]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total KW", len(df_url))
                c2.metric("En perte", len(df_url[df_url['diff_pos'] < 0]))
                c3.metric("En gain", len(df_url[df_url['diff_pos'] > 0]))
                if 'volume' in df_url.columns:
                    c4.metric("Volume total", f"{int(df_url['volume'].fillna(0).sum()):,}")
                
                # Afficher les leads si dispo
                if has_leads_merged and 'leads_total' in df_url.columns:
                    # Récupérer les labels de période
                    p_avant = df.attrs.get('periode_avant_label', 'AVANT')
                    p_apres = df.attrs.get('periode_apres_label', 'APRÈS')
                    
                    c1, c2, c3, c4 = st.columns(4)
                    leads_t = df_url['leads_total'].iloc[0] if len(df_url) > 0 else 0
                    leads_av = df_url['leads_avant'].iloc[0] if len(df_url) > 0 and 'leads_avant' in df_url.columns else 0
                    leads_ap = df_url['leads_apres'].iloc[0] if len(df_url) > 0 and 'leads_apres' in df_url.columns else 0
                    leads_e = df_url['leads_evolution'].iloc[0] if len(df_url) > 0 and 'leads_evolution' in df_url.columns else 0
                    c1.metric("📊 Leads total", f"{int(leads_t or 0):,}")
                    c2.metric(f"📊 Leads {p_avant}", f"{int(leads_av or 0):,}")
                    c3.metric(f"📊 Leads {p_apres}", f"{int(leads_ap or 0):,}")
                    c4.metric("📈 Évolution", f"{int(leads_e or 0):+,}")
                
                cols = [c for c in ['mot_cle', 'diff_pos', 'volume', 'derniere_pos', 'ancienne_pos', 'meilleure_pos'] if c in df_url.columns]
                st.dataframe(df_url[cols].sort_values('diff_pos'), use_container_width=True)
                
                # Export détail URL
                csv_url_detail = df_url[cols].to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(f"📥 Exporter les KW de cette URL", csv_url_detail, f"detail_url.csv")
        else:
            st.warning("Colonne 'url' non trouvée")
    
    # TAB 4: GAINS
    with tab4:
        st.header("🟢 Gains")
        df_gains = df_f[df_f['diff_pos'] > 0].sort_values('diff_pos', ascending=False)
        st.success(f"**{len(df_gains):,}** mots-clés en gain")
        
        cols = [c for c in ['mot_cle', 'url', 'diff_pos', 'tendance_seo', 'volume', 'derniere_pos', 'ancienne_pos'] if c in df_gains.columns]
        st.dataframe(df_gains[cols], use_container_width=True, height=600)
        
        csv_gains = df_gains[cols].to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("📥 Exporter TOUS les gains (CSV)", csv_gains, "gains_complet.csv")
    
    # TAB 5: CANNIBALISATION
    with tab5:
        st.header("🔄 Détection de cannibalisation interne")
        st.info("**Objectif** : Identifier les KW où une URL perd des positions tandis qu'une autre URL du site en gagne. Avant de réoptimiser une page en perte, vérifiez qu'une autre page n'a pas pris le relais !")
        
        if 'mot_cle' in df.columns and 'url' in df.columns:
            with st.spinner("Analyse des cannibalisations en cours..."):
                # Travailler sur le df complet (pas filtré) pour détecter toutes les cannibalisations
                df_canni = df[['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume']].copy()
                
                # Pour chaque KW, trouver les URLs en perte et en gain
                df_pertes_canni = df_canni[df_canni['diff_pos'] < 0].copy()
                df_gains_canni = df_canni[df_canni['diff_pos'] > 0].copy()
                
                # Trouver les KW qui ont à la fois des pertes ET des gains (= cannibalisation potentielle)
                kw_en_perte = set(df_pertes_canni['mot_cle'].unique())
                kw_en_gain = set(df_gains_canni['mot_cle'].unique())
                kw_cannibalisation = kw_en_perte & kw_en_gain
                
                st.metric("🔄 KW avec cannibalisation potentielle", f"{len(kw_cannibalisation):,}")
                
                if len(kw_cannibalisation) > 0:
                    # Construire le tableau de cannibalisation
                    resultats_canni = []
                    
                    for kw in kw_cannibalisation:
                        # URLs en perte sur ce KW
                        urls_perte = df_pertes_canni[df_pertes_canni['mot_cle'] == kw].sort_values('diff_pos', ascending=True)
                        # URLs en gain sur ce KW
                        urls_gain = df_gains_canni[df_gains_canni['mot_cle'] == kw].sort_values('diff_pos', ascending=False)
                        
                        # Prendre la pire perte et le meilleur gain
                        if len(urls_perte) > 0 and len(urls_gain) > 0:
                            perte = urls_perte.iloc[0]
                            gain = urls_gain.iloc[0]
                            
                            # Volume du KW (prendre le max disponible)
                            vol = max(perte.get('volume', 0) or 0, gain.get('volume', 0) or 0)
                            
                            resultats_canni.append({
                                'mot_cle': kw,
                                'volume': vol,
                                'url_perte': perte['url'],
                                'ancienne_pos_perte': perte.get('ancienne_pos', 0),
                                'nouvelle_pos_perte': perte.get('derniere_pos', 0),
                                'diff_perte': perte.get('diff_pos', 0),
                                'url_gain': gain['url'],
                                'ancienne_pos_gain': gain.get('ancienne_pos', 0),
                                'nouvelle_pos_gain': gain.get('derniere_pos', 0),
                                'diff_gain': gain.get('diff_pos', 0),
                            })
                    
                    if resultats_canni:
                        df_resultats = pd.DataFrame(resultats_canni)
                        
                        # Trier par volume décroissant (les KW les plus importants d'abord)
                        df_resultats = df_resultats.sort_values('volume', ascending=False)
                        
                        # Filtres
                        col1, col2 = st.columns(2)
                        with col1:
                            vol_min_canni = st.number_input("Volume minimum", min_value=0, value=0, step=50, key="vol_canni")
                        with col2:
                            diff_min_canni = st.number_input("Perte minimum (positions)", min_value=0, value=0, step=1, key="diff_canni")
                        
                        # Appliquer filtres (fillna pour éviter que les NaN soient exclus)
                        df_resultats_f = df_resultats[
                            (df_resultats['volume'].fillna(0) >= vol_min_canni) & 
                            (df_resultats['diff_perte'].fillna(0).abs() >= diff_min_canni)
                        ]
                        
                        st.success(f"**{len(df_resultats_f):,}** cas de cannibalisation détectés (sur {len(df_resultats):,} total)")
                        
                        # Affichage du tableau
                        st.subheader("⚠️ KW à risque — Vérifier avant réoptimisation")
                        
                        # Formater pour l'affichage
                        df_display = df_resultats_f.copy()
                        df_display['📉 URL en perte'] = df_display['url_perte']
                        df_display['Était pos'] = df_display['ancienne_pos_perte'].apply(lambda x: int(x) if pd.notna(x) else 0)
                        df_display['→ Maintenant'] = df_display['nouvelle_pos_perte'].apply(lambda x: int(x) if pd.notna(x) else 0)
                        df_display['Diff'] = df_display['diff_perte'].apply(lambda x: int(x) if pd.notna(x) else 0)
                        df_display['📈 URL en hausse'] = df_display['url_gain']
                        df_display['Était pos '] = df_display['ancienne_pos_gain'].apply(lambda x: int(x) if pd.notna(x) else 0)
                        df_display['→ Maintenant '] = df_display['nouvelle_pos_gain'].apply(lambda x: int(x) if pd.notna(x) else 0)
                        df_display['Diff '] = df_display['diff_gain'].apply(lambda x: f"+{int(x)}" if pd.notna(x) else "+0")
                        df_display['Volume'] = df_display['volume'].apply(lambda x: int(x) if pd.notna(x) else 0)
                        
                        cols_display = ['mot_cle', 'Volume', '📉 URL en perte', 'Était pos', '→ Maintenant', 'Diff', '📈 URL en hausse', 'Était pos ', '→ Maintenant ', 'Diff ']
                        
                        st.dataframe(df_display[cols_display].head(100), use_container_width=True, height=500)
                        
                        # Alerte
                        st.warning("""
                        **⚠️ ATTENTION avant de réoptimiser une URL en perte :**
                        1. Vérifiez si l'URL en hausse répond mieux à l'intention de recherche
                        2. Si oui → renforcez l'URL en hausse plutôt que l'ancienne
                        3. Si non → vérifiez le maillage interne pour éviter la cannibalisation
                        4. Envisagez une redirection 301 si l'ancienne URL n'a plus de raison d'être
                        """)
                        
                        # Export
                        csv_canni = df_resultats_f.to_csv(index=False, sep=';').encode('utf-8')
                        st.download_button("📥 Exporter les cannibalisations (CSV)", csv_canni, "cannibalisations.csv")
                        
                else:
                    st.success("✅ Aucune cannibalisation détectée ! Chaque KW n'a qu'une seule URL qui bouge.")
        else:
            st.warning("Colonnes 'mot_cle' et 'url' nécessaires pour l'analyse de cannibalisation")
    
    # TAB 6: SEARCH CONSOLE
    with tab6:
        st.header("🔍 Données Search Console")
        
        if has_gsc:
            st.info("**Données réelles Google** : Clics, impressions, CTR et positions moyennes des 12 derniers mois")
            
            # Créer les sous-onglets GSC
            gsc_tab1, gsc_tab2, gsc_tab3 = st.tabs(["🚨 URLs en danger", "💡 Opportunités CTR", "📊 Vue globale"])
            
            # === ONGLET 1 : URLs EN DANGER ===
            with gsc_tab1:
                st.subheader("🚨 URLs en danger : Perte SEO + Trafic réel")
                st.caption("URLs qui perdent des positions Haloscan ET qui ont beaucoup de clics GSC → Perte de trafic réelle")
                
                if gsc_pages_df is not None and 'url' in df_f.columns:
                    # Agréger les données Haloscan par URL
                    df_haloscan_urls = df_f.groupby('url').agg({
                        'diff_pos': ['mean', 'sum', 'count'],
                        'volume': 'sum'
                    }).reset_index()
                    df_haloscan_urls.columns = ['url', 'diff_pos_mean', 'diff_pos_sum', 'nb_kw', 'volume_total']
                    df_haloscan_urls['url_normalized'] = df_haloscan_urls['url'].apply(normalize_url)
                    
                    # Fusionner avec GSC
                    df_danger = df_haloscan_urls.merge(
                        gsc_pages_df[['url_normalized', 'Clics', 'Impressions', 'CTR', 'Position']],
                        on='url_normalized',
                        how='inner'
                    )
                    
                    # URLs en danger = diff négative + beaucoup de clics
                    df_danger = df_danger[df_danger['diff_pos_mean'] < 0].copy()
                    df_danger['score_danger'] = df_danger['Clics'] * df_danger['diff_pos_mean'].abs()
                    df_danger = df_danger.sort_values('score_danger', ascending=False)
                    
                    # Métriques
                    col1, col2, col3 = st.columns(3)
                    col1.metric("URLs en danger", f"{len(df_danger):,}")
                    col2.metric("Clics totaux menacés", f"{int(df_danger['Clics'].sum()):,}")
                    col3.metric("Impressions menacées", f"{int(df_danger['Impressions'].sum()):,}")
                    
                    if len(df_danger) > 0:
                        # Afficher le tableau
                        df_danger_display = df_danger[['url', 'Clics', 'Impressions', 'CTR', 'Position', 'diff_pos_mean', 'nb_kw', 'volume_total']].copy()
                        df_danger_display = df_danger_display.rename(columns={
                            'Clics': '🖱️ Clics GSC',
                            'Impressions': '👁️ Impressions',
                            'CTR': '📊 CTR %',
                            'Position': '📍 Pos GSC',
                            'diff_pos_mean': '📉 Δ Haloscan',
                            'nb_kw': 'Nb KW',
                            'volume_total': 'Vol. total'
                        })
                        df_danger_display['📉 Δ Haloscan'] = df_danger_display['📉 Δ Haloscan'].round(1)
                        
                        st.dataframe(df_danger_display.head(50), use_container_width=True, height=400)
                        
                        st.error("""
                        **🚨 ACTION REQUISE** : Ces URLs perdent des positions ET génèrent du trafic réel.
                        → Prioriser leur réoptimisation pour éviter une perte de trafic
                        """)
                        
                        # Export
                        csv_danger = df_danger.to_csv(index=False, sep=';').encode('utf-8')
                        st.download_button("📥 Exporter les URLs en danger (CSV)", csv_danger, "urls_danger_gsc.csv")
                    else:
                        st.success("✅ Aucune URL en danger détectée !")
                else:
                    st.warning("Données Pages GSC ou URLs Haloscan non disponibles")
            
            # === ONGLET 2 : OPPORTUNITÉS CTR ===
            with gsc_tab2:
                st.subheader("💡 Opportunités CTR : Bien positionné mais peu cliqué")
                st.caption("URLs en Top 10 avec CTR < 5% → Title et meta description à optimiser")
                
                if gsc_pages_df is not None:
                    # Filtrer : Position < 10 et CTR < 5%
                    df_ctr_opps = gsc_pages_df[
                        (gsc_pages_df['Position'] <= 10) & 
                        (gsc_pages_df['CTR'] < 5) &
                        (gsc_pages_df['Impressions'] >= 100)  # Au moins 100 impressions pour être significatif
                    ].copy()
                    
                    # Calculer le potentiel de clics
                    # Si CTR passait à 5%, combien de clics en plus ?
                    df_ctr_opps['ctr_potentiel'] = 5.0
                    df_ctr_opps['clics_potentiels'] = (df_ctr_opps['Impressions'] * df_ctr_opps['ctr_potentiel'] / 100).astype(int)
                    df_ctr_opps['clics_supplementaires'] = df_ctr_opps['clics_potentiels'] - df_ctr_opps['Clics']
                    df_ctr_opps = df_ctr_opps.sort_values('clics_supplementaires', ascending=False)
                    
                    # Métriques
                    col1, col2, col3 = st.columns(3)
                    col1.metric("URLs à optimiser", f"{len(df_ctr_opps):,}")
                    col2.metric("Clics actuels", f"{int(df_ctr_opps['Clics'].sum()):,}")
                    col3.metric("Potentiel clics en +", f"+{int(df_ctr_opps['clics_supplementaires'].sum()):,}")
                    
                    if len(df_ctr_opps) > 0:
                        # Afficher le tableau
                        df_ctr_display = df_ctr_opps[['url', 'Position', 'CTR', 'Clics', 'Impressions', 'clics_supplementaires']].copy()
                        df_ctr_display = df_ctr_display.rename(columns={
                            'Position': '📍 Position',
                            'CTR': '📊 CTR actuel %',
                            'Clics': '🖱️ Clics',
                            'Impressions': '👁️ Impressions',
                            'clics_supplementaires': '🎯 Potentiel clics +'
                        })
                        df_ctr_display['📍 Position'] = df_ctr_display['📍 Position'].round(1)
                        
                        st.dataframe(df_ctr_display.head(50), use_container_width=True, height=400)
                        
                        st.warning("""
                        **💡 OPTIMISATION RECOMMANDÉE** :
                        - Revoir les **titles** pour les rendre plus attractifs
                        - Améliorer les **meta descriptions** avec des CTA
                        - Ajouter des **données structurées** pour enrichir les snippets
                        """)
                        
                        # Export
                        csv_ctr = df_ctr_opps.to_csv(index=False, sep=';').encode('utf-8')
                        st.download_button("📥 Exporter les opportunités CTR (CSV)", csv_ctr, "opportunites_ctr.csv")
                    else:
                        st.success("✅ Toutes les URLs en Top 10 ont un bon CTR !")
                else:
                    st.warning("Données Pages GSC non disponibles")
            
            # === ONGLET 3 : VUE GLOBALE ===
            with gsc_tab3:
                st.subheader("📊 Vue globale Search Console")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔍 Top Requêtes (par clics)**")
                    if gsc_queries_df is not None:
                        st.dataframe(
                            gsc_queries_df[['query', 'Clics', 'Impressions', 'CTR', 'Position']].head(20),
                            use_container_width=True,
                            height=400
                        )
                    else:
                        st.info("Données requêtes non disponibles")
                
                with col2:
                    st.markdown("**📄 Top Pages (par clics)**")
                    if gsc_pages_df is not None:
                        df_pages_display = gsc_pages_df[['url', 'Clics', 'Impressions', 'CTR', 'Position']].head(20).copy()
                        # Raccourcir les URLs pour l'affichage
                        df_pages_display['url'] = df_pages_display['url'].str.replace('https://www.ootravaux.fr', '...')
                        st.dataframe(df_pages_display, use_container_width=True, height=400)
                    else:
                        st.info("Données pages non disponibles")
                
                # Stats globales
                st.divider()
                st.markdown("**📈 Statistiques globales GSC**")
                col1, col2, col3, col4 = st.columns(4)
                
                if gsc_pages_df is not None:
                    col1.metric("Total Clics", f"{int(gsc_pages_df['Clics'].sum()):,}")
                    col2.metric("Total Impressions", f"{int(gsc_pages_df['Impressions'].sum()):,}")
                    col3.metric("CTR moyen", f"{gsc_pages_df['CTR'].mean():.2f}%")
                    col4.metric("Position moyenne", f"{gsc_pages_df['Position'].mean():.1f}")
        
        else:
            st.warning("👆 Uploadez un fichier ZIP Search Console pour voir les données de trafic réel")
            st.info("""
            **Comment obtenir l'export :**
            1. Allez sur [Google Search Console](https://search.google.com/search-console)
            2. Sélectionnez votre propriété
            3. Allez dans "Performances" > "Résultats de recherche"
            4. Cliquez sur "Exporter" > "Télécharger au format ZIP"
            """)
    
    # TAB 7: RAPPORT
    with tab7:
        st.header("📝 Rapport complet pour l'équipe édito")
        
        if st.button("🔄 Générer le rapport complet", type="primary"):
            
            # Calculs pour le rapport
            df_pertes_rapport = df_f[df_f['diff_pos'] < 0].sort_values('diff_pos', ascending=True)
            df_gains_rapport = df_f[df_f['diff_pos'] > 0].sort_values('diff_pos', ascending=False)
            
            # URLs les plus impactées
            urls_critiques = pd.DataFrame()  # Initialiser vide par défaut
            if 'url' in df_f.columns and len(df_pertes_rapport) > 0:
                # Construire l'agrégation dynamiquement selon les colonnes disponibles
                agg_url = {'diff_pos': 'count'}
                if 'volume' in df_pertes_rapport.columns:
                    agg_url['volume'] = 'sum'
                if has_leads_merged:
                    if 'leads_total' in df_pertes_rapport.columns:
                        agg_url['leads_total'] = 'first'
                    if 'leads_avant' in df_pertes_rapport.columns:
                        agg_url['leads_avant'] = 'first'
                    if 'leads_apres' in df_pertes_rapport.columns:
                        agg_url['leads_apres'] = 'first'
                    if 'leads_evolution' in df_pertes_rapport.columns:
                        agg_url['leads_evolution'] = 'first'
                    if 'tendance_leads' in df_pertes_rapport.columns:
                        agg_url['tendance_leads'] = 'first'
                
                try:
                    urls_critiques = df_pertes_rapport.groupby('url').agg(agg_url).reset_index()
                    
                    # Renommer les colonnes
                    rename_cols = {'diff_pos': 'nb_kw_perdus', 'volume': 'volume_impacte'}
                    urls_critiques = urls_critiques.rename(columns=rename_cols)
                    
                    # Trier par évolution leads ou par nb KW perdus
                    if 'leads_evolution' in urls_critiques.columns:
                        urls_critiques = urls_critiques.sort_values('leads_evolution', ascending=True)
                    else:
                        urls_critiques = urls_critiques.sort_values('nb_kw_perdus', ascending=False)
                except Exception as e:
                    st.warning(f"Erreur agrégation URLs: {e}")
                    urls_critiques = pd.DataFrame()
            
            # Calcul impact leads - ATTENTION : éviter de compter plusieurs fois la même URL
            if has_leads_merged:
                # Grouper par URL pour ne compter qu'une fois les leads de chaque URL
                urls_en_perte = df_pertes_rapport['url'].unique() if 'url' in df_pertes_rapport.columns else []
                df_urls_perte_unique = df_pertes_rapport.drop_duplicates(subset=['url'])
                
                total_leads_perte = int(df_urls_perte_unique['leads_total'].fillna(0).sum())
                total_leads_avant_perte = int(df_urls_perte_unique['leads_avant'].fillna(0).sum())
                total_leads_apres_perte = int(df_urls_perte_unique['leads_apres'].fillna(0).sum())
                leads_evolution_total = int(df_urls_perte_unique['leads_evolution'].fillna(0).sum())
            
            # Définir la période pour le titre du rapport
            if has_dual_haloscan:
                periode_rapport = f"{label_debut_p1} → {label_fin_p2}"
            else:
                periode_rapport = "Période analysée"
            
            report = f"""# 📊 RAPPORT D'ANALYSE SEO COMPLET
## Période : {periode_rapport}
## Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}

---

# 1. SYNTHÈSE GLOBALE

| Métrique | Valeur |
|----------|--------|
| **Total mots-clés analysés** | {total:,} |
| **Mots-clés en perte** | {pertes:,} ({pertes/total*100:.1f}%) |
| **Mots-clés en gain** | {gains:,} ({gains/total*100:.1f}%) |
| **Mots-clés stables** | {stables:,} ({stables/total*100:.1f}%) |
| **Volume de recherche perdu** | {vol_perdu:,} /mois |
| **Volume de recherche gagné** | {vol_gagne:,} /mois |
| **Bilan net volume** | {vol_gagne - vol_perdu:+,} /mois |
"""
            
            if has_leads_merged:
                periodes_info = f"Période AVANT: {', '.join(periode_avant) if periode_avant else 'N/A'} | Période APRÈS: {', '.join(periode_apres) if periode_apres else 'N/A'}"
                report += f"""
## 💰 IMPACT BUSINESS (Leads)

**{periodes_info}**

| Métrique | Valeur |
|----------|--------|
| **Leads historiques sur URLs en perte** | {total_leads_perte:,} |
| **Leads période AVANT** | {total_leads_avant_perte:,} |
| **Leads période APRÈS** | {total_leads_apres_perte:,} |
| **Évolution des leads** | {leads_evolution_total:+,} |

⚠️ **Ces URLs génèrent des leads et perdent en visibilité SEO = PRIORITÉ MAXIMALE**

"""

            # Section multi-périodes si disponible
            if has_dual_haloscan and 'tendance_multi' in df_f.columns:
                tendances = df_f['tendance_multi'].value_counts()
                chute_continue = tendances.get('📉📉 Chute continue', 0)
                rebond_rechute = tendances.get('📈📉 Rebond puis rechute', 0)
                recuperation = tendances.get('📉📈 Récupération', 0)
                hausse_continue = tendances.get('📈📈 Hausse continue', 0)
                
                report += f"""---

## 📈 ANALYSE MULTI-PÉRIODES ({label_debut_p1} → {label_fin_p1} → {label_fin_p2})

| Tendance | Nombre de KW | Signification |
|----------|--------------|---------------|
| 📉📉 **Chute continue** | {chute_continue:,} | Perte sur P1 ET P2 — **Problème structurel** |
| 📈📉 Rebond puis rechute | {rebond_rechute:,} | Gain sur P1 puis perte sur P2 |
| 📉📈 Récupération | {recuperation:,} | Perte sur P1 puis gain sur P2 |
| 📈📈 Hausse continue | {hausse_continue:,} | Gain sur P1 ET P2 |

"""
                # Ajouter les KW en chute continue (TOP 100)
                df_chute_continue = df_f[df_f['tendance_multi'] == '📉📉 Chute continue'].copy()
                if len(df_chute_continue) > 0:
                    report += f"""### 🚨 TOP 100 KW en CHUTE CONTINUE — Priorité maximale

| Mot-clé | URL | Pos {label_debut_p1} | Pos {label_fin_p1} | Δ P1 | Pos {label_fin_p2} | Δ P2 | Δ TOTAL | Volume |
|---------|-----|---------------------|--------------------|----- |--------------------|----- |---------|--------|
"""
                    # Trier par diff totale
                    df_chute_continue = df_chute_continue.sort_values('diff_pos', ascending=True)
                    
                    for _, row in df_chute_continue.head(100).iterrows():
                        mc = str(row.get('mot_cle', 'N/A'))[:40]
                        url = str(row.get('url', 'N/A'))
                        pos_debut = int(row.get('pos_debut_p1', 0)) if pd.notna(row.get('pos_debut_p1')) else 0
                        pos_mid = int(row.get('pos_fin_p1', 0)) if pd.notna(row.get('pos_fin_p1')) else 0
                        diff_p1 = int(row.get('diff_p1', 0)) if pd.notna(row.get('diff_p1')) else 0
                        pos_fin = int(row.get('pos_fin_p2', 0)) if pd.notna(row.get('pos_fin_p2')) else 0
                        diff_p2 = int(row.get('diff_p2', 0)) if pd.notna(row.get('diff_p2')) else 0
                        diff_tot = int(row.get('diff_pos', 0)) if pd.notna(row.get('diff_pos')) else 0
                        vol = int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0
                        report += f"| {mc} | {url} | {pos_debut} | {pos_mid} | {diff_p1} | {pos_fin} | {diff_p2} | {diff_tot} | {vol:,} |\n"
                    
                    if len(df_chute_continue) > 100:
                        report += f"\n_+ {len(df_chute_continue) - 100:,} autres KW en chute continue (non affichés)_\n"

            # Section Search Console si disponible
            if has_gsc and gsc_pages_df is not None and 'url' in df_f.columns:
                # Calculer les URLs en danger
                df_haloscan_urls_rpt = df_f.groupby('url').agg({
                    'diff_pos': 'mean',
                    'volume': 'sum'
                }).reset_index()
                df_haloscan_urls_rpt['url_normalized'] = df_haloscan_urls_rpt['url'].apply(normalize_url)
                
                df_danger_rpt = df_haloscan_urls_rpt.merge(
                    gsc_pages_df[['url_normalized', 'Clics', 'Impressions', 'CTR', 'Position']],
                    on='url_normalized',
                    how='inner'
                )
                df_danger_rpt = df_danger_rpt[df_danger_rpt['diff_pos'] < 0].copy()
                df_danger_rpt = df_danger_rpt.sort_values('Clics', ascending=False)
                
                # Opportunités CTR
                df_ctr_rpt = gsc_pages_df[
                    (gsc_pages_df['Position'] <= 10) & 
                    (gsc_pages_df['CTR'] < 5) &
                    (gsc_pages_df['Impressions'] >= 100)
                ].copy()
                df_ctr_rpt['clics_potentiels'] = (df_ctr_rpt['Impressions'] * 5 / 100 - df_ctr_rpt['Clics']).astype(int)
                df_ctr_rpt = df_ctr_rpt.sort_values('clics_potentiels', ascending=False)
                
                report += f"""---

## 🔍 DONNÉES SEARCH CONSOLE (12 derniers mois)

| Métrique | Valeur |
|----------|--------|
| **Total clics** | {int(gsc_pages_df['Clics'].sum()):,} |
| **Total impressions** | {int(gsc_pages_df['Impressions'].sum()):,} |
| **CTR moyen** | {gsc_pages_df['CTR'].mean():.2f}% |
| **URLs en danger (perte + clics)** | {len(df_danger_rpt):,} |
| **Opportunités CTR (Top 10, CTR < 5%)** | {len(df_ctr_rpt):,} |
| **Clics potentiels à gagner** | +{int(df_ctr_rpt['clics_potentiels'].sum()):,} |

"""
                if len(df_danger_rpt) > 0:
                    report += """### 🚨 TOP 20 URLs EN DANGER (Perte SEO + Trafic réel)

| URL | Clics GSC | Δ Haloscan | Position GSC | CTR |
|-----|-----------|------------|--------------|-----|
"""
                    for _, row in df_danger_rpt.head(20).iterrows():
                        url = str(row.get('url', 'N/A'))
                        clics = int(row.get('Clics', 0))
                        diff = round(row.get('diff_pos', 0), 1)
                        pos = round(row.get('Position', 0), 1)
                        ctr = round(row.get('CTR', 0), 2)
                        report += f"| {url} | {clics:,} | {diff} | {pos} | {ctr}% |\n"
                
                if len(df_ctr_rpt) > 0:
                    report += """

### 💡 TOP 20 OPPORTUNITÉS CTR (à optimiser)

| URL | Position | CTR actuel | Impressions | Potentiel clics + |
|-----|----------|------------|-------------|-------------------|
"""
                    for _, row in df_ctr_rpt.head(20).iterrows():
                        url = str(row.get('url', 'N/A'))
                        pos = round(row.get('Position', 0), 1)
                        ctr = round(row.get('CTR', 0), 2)
                        impr = int(row.get('Impressions', 0))
                        pot = int(row.get('clics_potentiels', 0))
                        report += f"| {url} | {pos} | {ctr}% | {impr:,} | +{pot:,} |\n"

            report += """---

# 2. DIAGNOSTIC

"""
            if gains == 0:
                report += f"""⚠️ **SITUATION CRITIQUE** : Le site n'a aucun gain de position.
- {pertes:,} mots-clés en perte
- Action recommandée : **Audit urgent des contenus**

"""
            elif pertes > gains:
                report += f"""⚠️ **SITUATION PRÉOCCUPANTE** : Le site perd plus de positions qu'il n'en gagne.
- Ratio pertes/gains : {pertes/gains:.1f}x plus de pertes
- Action recommandée : **Audit urgent des contenus impactés**

"""
            elif pertes == 0:
                report += f"""✅ **SITUATION EXCELLENTE** : Aucune perte de position !
- {gains:,} mots-clés en gain

"""
            else:
                report += f"""✅ **SITUATION POSITIVE** : Le site gagne plus de positions qu'il n'en perd.
- Ratio gains/pertes : {gains/pertes:.1f}x plus de gains

"""

            if len(urls_critiques) > 0:
                report += f"""---

# 3. TOUTES LES PAGES À TRAITER ({len(urls_critiques):,} URLs)

"""
                if has_leads_merged:
                    # Récupérer les labels de période
                    p_avant = df.attrs.get('periode_avant_label', 'AVANT')
                    p_apres = df.attrs.get('periode_apres_label', 'APRÈS')
                    
                    report += f"""**Triées par évolution des leads** — Les URLs avec les plus grosses pertes de leads en premier.

| Priorité | URL | KW perdus | Volume | Leads {p_avant} | Leads {p_apres} | 📊 TENDANCE |
|----------|-----|-----------|--------|-------------|-------------|-------------|
"""
                    for i, row in urls_critiques.iterrows():
                        leads_evol = row.get('leads_evolution', 0)
                        leads_evol = 0 if pd.isna(leads_evol) else leads_evol
                        tendance = row.get('tendance_leads', '➡️ N/A')
                        prio = "🔴 CRITIQUE" if leads_evol < -100 else \
                               "🟠 URGENT" if leads_evol < -20 else \
                               "🟡 MOYEN" if leads_evol < 0 else "⚪ STABLE/HAUSSE"
                        
                        # Sécuriser toutes les valeurs numériques
                        nb_kw = int(row.get('nb_kw_perdus', 0) or 0)
                        vol = row.get('volume_impacte', 0)
                        vol = 0 if pd.isna(vol) else int(vol)
                        l_avant = row.get('leads_avant', 0)
                        l_avant = 0 if pd.isna(l_avant) else int(l_avant)
                        l_apres = row.get('leads_apres', 0)
                        l_apres = 0 if pd.isna(l_apres) else int(l_apres)
                        
                        report += f"| {prio} | {row['url']} | {nb_kw} | {vol:,} | {l_avant:,} | {l_apres:,} | {tendance} |\n"
                else:
                    report += """**Triées par nombre de mots-clés perdus**

| Priorité | URL | KW perdus | Volume impacté |
|----------|-----|-----------|----------------|
"""
                    for i, row in urls_critiques.iterrows():
                        nb_kw = row['nb_kw_perdus']
                        prio = "🔴 URGENT" if nb_kw > 50 else "🟠 MOYEN" if nb_kw > 10 else "🟡 FAIBLE"
                        report += f"| {prio} | {row['url']} | {int(nb_kw)} | {int(row.get('volume_impacte', 0) or 0):,} |\n"
            else:
                report += """---

# 3. PAGES À TRAITER

_Aucune URL en perte détectée_

"""

            # Filtrer les KW qui ont vraiment morflé (grosses pertes uniquement)
            # Priorité : diff très négative + volume élevé
            df_pertes_critiques = df_pertes_rapport[df_pertes_rapport['diff_pos'] <= -5].copy()
            
            # Grouper par URL et garder le KW principal :
            # = celui avec le plus gros volume PARMI ceux où l'URL rankait bien avant (ancienne_pos ≤ 10)
            if 'volume' in df_pertes_critiques.columns and 'url' in df_pertes_critiques.columns:
                # Filtrer les KW où l'URL rankait vraiment bien (top 10)
                df_bien_ranke = df_pertes_critiques[df_pertes_critiques['ancienne_pos'] <= 10].copy()
                
                # Si pas de KW bien ranké pour une URL, on prend quand même le meilleur volume
                if len(df_bien_ranke) > 0:
                    idx_kw_principal = df_bien_ranke.groupby('url')['volume'].idxmax()
                    df_pertes_par_url = df_bien_ranke.loc[idx_kw_principal].copy()
                else:
                    idx_kw_principal = df_pertes_critiques.groupby('url')['volume'].idxmax()
                    df_pertes_par_url = df_pertes_critiques.loc[idx_kw_principal].copy()
                
                # Ajouter le nombre total de KW perdus par URL (tous les KW, pas que les bien rankés)
                kw_count = df_pertes_critiques.groupby('url').size().rename('nb_kw_perdus')
                df_pertes_par_url = df_pertes_par_url.merge(kw_count, on='url', how='left')
                
                # Trier par diff_pos (les pires en premier)
                df_pertes_par_url = df_pertes_par_url.sort_values('diff_pos', ascending=True)
            else:
                df_pertes_par_url = df_pertes_critiques.sort_values('diff_pos', ascending=True)
            
            # Limiter à 500 URLs max
            max_kw_rapport = 500
            df_pertes_limited = df_pertes_par_url.head(max_kw_rapport)
            
            report += f"""

---

# 4. PERTES CRITIQUES — TOP {len(df_pertes_limited):,} URLs (pertes ≥ 5 positions)

**⚠️ Priorité maximale — KW principal = plus gros volume parmi ceux où l'URL était en top 10**

| KW Principal | URL | Ancienne pos | Nouvelle pos | Diff | Volume | Nb KW perdus |
|--------------|-----|--------------|--------------|------|--------|--------------|
"""
            for _, row in df_pertes_limited.iterrows():
                mc = str(row.get('mot_cle', 'N/A'))[:50]
                url = str(row.get('url', 'N/A'))
                anc = row.get('ancienne_pos', 0)
                anc = 0 if pd.isna(anc) else int(anc)
                dern = row.get('derniere_pos', 0)
                dern = 0 if pd.isna(dern) else int(dern)
                diff = row.get('diff_pos', 0)
                diff = 0 if pd.isna(diff) else int(diff)
                vol = row.get('volume', 0)
                vol = 0 if pd.isna(vol) else int(vol)
                nb_kw = row.get('nb_kw_perdus', 1)
                nb_kw = 1 if pd.isna(nb_kw) else int(nb_kw)
                report += f"| {mc} | {url} | {anc} | {dern} | {diff} | {vol:,} | {nb_kw} |\n"
            
            # Info sur les URLs non affichées
            nb_autres_urls = len(df_pertes_par_url) - len(df_pertes_limited)
            if nb_autres_urls > 0:
                report += f"\n_+ {nb_autres_urls:,} autres URLs avec des pertes ≥ 5 positions (non affichées)_\n"

            # Filtrer les KW avec gains significatifs (≥ 5 positions)
            df_gains_significatifs = df_gains_rapport[df_gains_rapport['diff_pos'] >= 5].copy()
            
            # Grouper par URL et garder le KW principal :
            # = celui avec le plus gros volume PARMI ceux où l'URL ranke bien maintenant (derniere_pos ≤ 10)
            if 'volume' in df_gains_significatifs.columns and 'url' in df_gains_significatifs.columns and len(df_gains_significatifs) > 0:
                # Filtrer les KW où l'URL ranke vraiment bien maintenant (top 10)
                df_bien_ranke = df_gains_significatifs[df_gains_significatifs['derniere_pos'] <= 10].copy()
                
                if len(df_bien_ranke) > 0:
                    idx_kw_principal = df_bien_ranke.groupby('url')['volume'].idxmax()
                    df_gains_par_url = df_bien_ranke.loc[idx_kw_principal].copy()
                else:
                    idx_kw_principal = df_gains_significatifs.groupby('url')['volume'].idxmax()
                    df_gains_par_url = df_gains_significatifs.loc[idx_kw_principal].copy()
                
                kw_count = df_gains_significatifs.groupby('url').size().rename('nb_kw_gains')
                df_gains_par_url = df_gains_par_url.merge(kw_count, on='url', how='left')
                
                df_gains_par_url = df_gains_par_url.sort_values('diff_pos', ascending=False)
            else:
                df_gains_par_url = df_gains_significatifs.sort_values('diff_pos', ascending=False)
            
            df_gains_limited = df_gains_par_url.head(max_kw_rapport)
            
            report += f"""

---

# 5. GAINS SIGNIFICATIFS — TOP {len(df_gains_limited):,} URLs (gains ≥ 5 positions)

**✅ Ce qui fonctionne — KW principal = plus gros volume parmi ceux en top 10 actuel**

| KW Principal | URL | Ancienne pos | Nouvelle pos | Diff | Volume | Nb KW gagnés |
|--------------|-----|--------------|--------------|------|--------|--------------|
"""
            for _, row in df_gains_limited.iterrows():
                mc = str(row.get('mot_cle', 'N/A'))[:50]
                url = str(row.get('url', 'N/A'))
                anc = row.get('ancienne_pos', 0)
                anc = 0 if pd.isna(anc) else int(anc)
                dern = row.get('derniere_pos', 0)
                dern = 0 if pd.isna(dern) else int(dern)
                diff = row.get('diff_pos', 0)
                diff = 0 if pd.isna(diff) else int(diff)
                vol = row.get('volume', 0)
                vol = 0 if pd.isna(vol) else int(vol)
                nb_kw = row.get('nb_kw_gains', 1)
                nb_kw = 1 if pd.isna(nb_kw) else int(nb_kw)
                report += f"| {mc} | {url} | {anc} | {dern} | +{diff} | {vol:,} | {nb_kw} |\n"
            
            nb_autres_urls = len(df_gains_par_url) - len(df_gains_limited)
            if nb_autres_urls > 0:
                report += f"\n_+ {nb_autres_urls:,} autres URLs avec des gains ≥ 5 positions (non affichées)_\n"

            report += f"""

---

# 6. RECOMMANDATIONS POUR L'ÉQUIPE ÉDITO

## 🔴 Actions immédiates (cette semaine)
"""
            if has_leads_merged:
                report += """1. **PRIORITÉ ABSOLUE : URLs avec leads + pertes SEO** — Ces pages génèrent du business ET perdent en visibilité
2. **Auditer le contenu** des 10 premières URLs critiques
3. **Vérifier le maillage interne** vers ces pages stratégiques
"""
            else:
                report += """1. **Auditer les 10 premières URLs critiques** — Vérifier : contenu à jour ? maillage interne ? balises optimisées ?
2. **Identifier les KW à fort volume perdus** — Filtrer les pertes avec volume > 1000
3. **Vérifier la concurrence** — Les concurrents ont-ils amélioré leur contenu ?
"""

            report += """
## 🟠 Actions court terme (ce mois)
1. **Mettre à jour les contenus des pages critiques** — Enrichir, actualiser, ajouter des sections
2. **Renforcer le maillage interne** vers les pages en perte
3. **Créer du contenu de support** pour les thématiques en baisse

## 🟡 Actions moyen terme (ce trimestre)
1. **Audit technique** — Vérifier Core Web Vitals des pages impactées
2. **Analyse des backlinks** — Les pages ont-elles perdu des liens ?
3. **Stratégie de contenu** — Planifier les mises à jour récurrentes

---

# 7. MÉTRIQUES DE SUIVI

Refaire cette analyse dans 1 mois pour mesurer :
- [ ] Réduction du nombre de KW en perte
- [ ] Récupération des positions sur les KW prioritaires
- [ ] Amélioration du volume de recherche capté
"""
            if has_leads_merged:
                report += """- [ ] Stabilisation ou hausse des leads sur les URLs retravaillées
"""

            report += f"""
---

_Rapport généré automatiquement — Haloscan SEO Diff Analyzer_
_Données : {len(df):,} mots-clés analysés"""
            
            if has_leads_merged:
                report += f" | {len(leads_df):,} URLs avec données leads"
            
            report += "_\n"
            
            st.session_state['report'] = report
            st.success("✅ Rapport généré !")
        
        if 'report' in st.session_state:
            st.markdown(st.session_state['report'])
            
            st.divider()
            
            # === ANALYSE IA ===
            st.subheader("🤖 Analyse IA et TODO")
            
            if anthropic_api_key:
                if st.button("🤖 Générer l'analyse IA", type="secondary"):
                    with st.spinner("Claude Opus 4.5 analyse vos données... (peut prendre 30-60 secondes)"):
                        try:
                            import anthropic
                            
                            client = anthropic.Anthropic(api_key=anthropic_api_key)
                            
                            # Préparer les données pour le LLM
                            # 1. Métriques globales
                            metrics_globales = {
                                "total_kw": total,
                                "kw_en_perte": pertes,
                                "kw_en_gain": gains,
                                "kw_stables": stables,
                                "volume_perdu": vol_perdu,
                                "volume_gagne": vol_gagne,
                                "bilan_volume": vol_gagne - vol_perdu
                            }
                            
                            # 2. Top 50 URLs critiques (en perte)
                            df_pertes_ia = df_f[df_f['diff_pos'] < 0].copy()
                            if 'url' in df_pertes_ia.columns:
                                urls_critiques_ia = df_pertes_ia.groupby('url').agg({
                                    'diff_pos': ['count', 'mean'],
                                    'volume': 'sum'
                                }).reset_index()
                                urls_critiques_ia.columns = ['url', 'nb_kw_perdus', 'diff_moyenne', 'volume_total']
                                
                                # Ajouter leads si disponible
                                if has_leads_merged:
                                    leads_by_url = df_pertes_ia.groupby('url').agg({
                                        'leads_total': 'first',
                                        'leads_evolution': 'first'
                                    }).reset_index()
                                    urls_critiques_ia = urls_critiques_ia.merge(leads_by_url, on='url', how='left')
                                
                                urls_critiques_ia = urls_critiques_ia.sort_values('volume_total', ascending=False).head(50)
                                urls_critiques_list = urls_critiques_ia.to_dict('records')
                            else:
                                urls_critiques_list = []
                            
                            # 3. Top 30 KW en perte (les plus impactants)
                            top_kw_pertes = df_pertes_ia.nlargest(30, 'volume')[['mot_cle', 'url', 'diff_pos', 'volume', 'ancienne_pos', 'derniere_pos']].to_dict('records') if 'volume' in df_pertes_ia.columns else []
                            
                            # 4. Données multi-périodes si disponibles
                            tendances_multi = {}
                            if has_dual_haloscan and 'tendance_multi' in df_f.columns:
                                tendances_multi = df_f['tendance_multi'].value_counts().to_dict()
                                # Top 20 en chute continue
                                df_chute = df_f[df_f['tendance_multi'] == '📉📉 Chute continue'].head(20)
                                if len(df_chute) > 0:
                                    tendances_multi['top_chute_continue'] = df_chute[['mot_cle', 'url', 'diff_pos', 'volume']].to_dict('records')
                            
                            # 5. Données GSC si disponibles
                            gsc_data = {}
                            if has_gsc and gsc_pages_df is not None:
                                gsc_data['total_clics'] = int(gsc_pages_df['Clics'].sum())
                                gsc_data['total_impressions'] = int(gsc_pages_df['Impressions'].sum())
                                gsc_data['ctr_moyen'] = round(gsc_pages_df['CTR'].mean(), 2)
                                # Top 20 pages par clics
                                gsc_data['top_pages'] = gsc_pages_df.nlargest(20, 'Clics')[['url', 'Clics', 'Impressions', 'CTR', 'Position']].to_dict('records')
                            
                            # 6. Cannibalisations
                            cannibalisations = []
                            if 'mot_cle' in df_f.columns:
                                df_canni_ia = df_f[['mot_cle', 'url', 'diff_pos', 'volume']].copy()
                                df_pertes_c = df_canni_ia[df_canni_ia['diff_pos'] < 0]
                                df_gains_c = df_canni_ia[df_canni_ia['diff_pos'] > 0]
                                kw_perte = set(df_pertes_c['mot_cle'].unique())
                                kw_gain = set(df_gains_c['mot_cle'].unique())
                                kw_canni = kw_perte & kw_gain
                                if len(kw_canni) > 0:
                                    for kw in list(kw_canni)[:20]:
                                        url_perte = df_pertes_c[df_pertes_c['mot_cle'] == kw].iloc[0]['url'] if len(df_pertes_c[df_pertes_c['mot_cle'] == kw]) > 0 else None
                                        url_gain = df_gains_c[df_gains_c['mot_cle'] == kw].iloc[0]['url'] if len(df_gains_c[df_gains_c['mot_cle'] == kw]) > 0 else None
                                        if url_perte and url_gain:
                                            cannibalisations.append({'mot_cle': kw, 'url_perte': url_perte, 'url_gain': url_gain})
                            
                            # Construire le contexte JSON
                            context_data = {
                                "metriques_globales": metrics_globales,
                                "urls_critiques": urls_critiques_list,
                                "top_kw_en_perte": top_kw_pertes,
                                "tendances_multi_periodes": tendances_multi,
                                "donnees_gsc": gsc_data,
                                "cannibalisations_detectees": cannibalisations,
                                "has_leads": has_leads_merged,
                                "has_gsc": has_gsc,
                                "has_dual_period": has_dual_haloscan
                            }
                            
                            # Prompt système
                            system_prompt = """Tu es un expert SEO senior spécialisé dans l'analyse de données et la stratégie de contenu.

Tu reçois des données SEO complètes d'un site et tu dois produire :

1. **ANALYSE STRATÉGIQUE** (5-10 lignes max)
- Diagnostic clair et direct de la situation
- Identification des patterns (saisonnalité, problème technique, cannibalisation...)
- Points d'alerte majeurs

2. **TODO POUR L'ÉQUIPE CONTENT** 
Une liste d'actions CONCRÈTES et PRIORISÉES. Chaque action doit être :
- Précise (pas de "améliorer le contenu" mais "ajouter une section FAQ avec les questions X, Y, Z")
- Assignable (une personne peut la prendre et la faire)
- Avec l'URL exacte concernée
- Avec l'impact attendu (estimation)

Format de la TODO :
```
## 🔴 PRIORITÉ HAUTE (à faire cette semaine)
- [ ] **[Action précise]** - URL: [url complète] - Impact: [estimation] - Raison: [pourquoi]

## 🟠 PRIORITÉ MOYENNE (à faire ce mois)
- [ ] **[Action précise]** - URL: [url complète] - Impact: [estimation] - Raison: [pourquoi]

## 🟡 PRIORITÉ BASSE (à planifier)
- [ ] **[Action précise]** - URL: [url complète] - Impact: [estimation] - Raison: [pourquoi]
```

3. **ALERTES** (si applicable)
- Risques identifiés
- Dépendances dangereuses
- Tendances inquiétantes

Sois direct, pragmatique, et orienté action. Pas de blabla corporate. L'équipe content doit pouvoir prendre cette TODO et l'exécuter immédiatement."""

                            # Appel à Claude
                            message = client.messages.create(
                                model="claude-opus-4-5-20251101",
                                max_tokens=4096,
                                system=system_prompt,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": f"""Voici les données SEO à analyser :

```json
{json.dumps(context_data, ensure_ascii=False, indent=2, default=str)}
```

Génère ton analyse stratégique et la TODO priorisée pour l'équipe content."""
                                    }
                                ]
                            )
                            
                            # Extraire la réponse
                            ai_analysis = message.content[0].text
                            
                            # Stocker dans session_state
                            st.session_state['ai_analysis'] = ai_analysis
                            
                            st.success("✅ Analyse IA générée !")
                            
                        except anthropic.AuthenticationError:
                            st.error("❌ Clé API invalide. Vérifiez votre clé Anthropic.")
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'analyse IA : {str(e)}")
                
                # Afficher l'analyse IA si disponible
                if 'ai_analysis' in st.session_state:
                    st.divider()
                    st.markdown("## 🤖 Analyse IA et TODO")
                    st.markdown(st.session_state['ai_analysis'])
                    
                    # Bouton pour télécharger le rapport complet (avec IA)
                    rapport_complet = st.session_state['report'] + "\n\n---\n\n# 🤖 ANALYSE IA ET TODO\n\n" + st.session_state['ai_analysis']
                    st.download_button(
                        "📥 Télécharger le rapport COMPLET avec IA (Markdown)",
                        rapport_complet,
                        "rapport_seo_complet_avec_ia.md",
                        "text/markdown"
                    )
            else:
                st.info("👆 Entrez votre clé API Anthropic dans la sidebar pour activer l'analyse IA")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Télécharger le rapport (Markdown)", 
                    st.session_state['report'], 
                    "rapport_seo_complet.md",
                    "text/markdown"
                )
            with col2:
                # Export aussi en CSV les données brutes
                df_export = df_f[df_f['diff_pos'] < 0].sort_values('diff_pos', ascending=True)
                cols_export = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'tendance_seo', 'volume', 'leads_total', 'leads_avant', 'leads_apres', 'leads_evolution', 'tendance_leads'] if c in df_export.columns]
                csv_export = df_export[cols_export].to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "📥 Télécharger les données (CSV)",
                    csv_export,
                    "pertes_completes.csv",
                    "text/csv"
                )

else:
    st.info("👆 Charge un fichier CSV pour commencer")
