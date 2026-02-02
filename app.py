"""
Haloscan SEO Diff Analyzer
Version corrigée pour le format exact du fichier Baptiste
Avec intégration des données de leads par URL
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

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
    # Retirer le slash final
    url = url.rstrip('/')
    return url

# =============================================================================
# INTERFACE
# =============================================================================

st.title("📊 Haloscan SEO Diff Analyzer")

with st.sidebar:
    st.header("📁 Import des données")
    
    uploaded_file = st.file_uploader("1️⃣ CSV Haloscan (positions)", type=['csv'])
    
    uploaded_leads = st.file_uploader("2️⃣ Excel Leads par URL (optionnel)", type=['xlsx', 'xls'], 
                                       help="Fichier avec colonnes: url, puis une colonne par mois (YYYY_MM)")

# Variables globales pour les leads
leads_df = None
has_leads = False
month_cols = []
periode_avant = []
periode_apres = []

if uploaded_leads:
    leads_df_raw = pd.read_excel(uploaded_leads)
    
    # Identifier les colonnes de mois
    month_cols = [col for col in leads_df_raw.columns if col != 'url' and '_' in str(col)]
    month_cols_sorted = sorted(month_cols)
    
    has_leads = True
    
    with st.sidebar:
        st.subheader("📅 Périodes à comparer")
        st.caption("Sélectionnez les mois correspondant à votre export Haloscan")
        
        # Calculer les valeurs par défaut
        default_avant = [c for c in month_cols_sorted if c.startswith('2025_09')]
        if not default_avant:
            default_avant = month_cols_sorted[-6:-3] if len(month_cols_sorted) >= 6 else month_cols_sorted[:3]
        
        default_apres = [c for c in month_cols_sorted if c.startswith('2025_11') or c.startswith('2026')]
        if not default_apres:
            default_apres = month_cols_sorted[-3:] if len(month_cols_sorted) >= 3 else month_cols_sorted[-1:]
        
        # Période AVANT (ancienne position)
        st.markdown("**Période AVANT** (ex: sept 2025)")
        periode_avant = st.multiselect(
            "Mois période avant",
            options=month_cols_sorted,
            default=default_avant,
            key="avant"
        )
        
        # Période APRÈS (position actuelle)
        st.markdown("**Période APRÈS** (ex: fév 2026)")
        periode_apres = st.multiselect(
            "Mois période après", 
            options=month_cols_sorted,
            default=default_apres,
            key="apres"
        )
    
    # Calculer les métriques leads sur les bonnes périodes
    leads_df = leads_df_raw.copy()
    
    leads_df['leads_total'] = leads_df[month_cols].sum(axis=1)
    leads_df['leads_avant'] = leads_df[periode_avant].sum(axis=1) if periode_avant else 0
    leads_df['leads_apres'] = leads_df[periode_apres].sum(axis=1) if periode_apres else 0
    leads_df['leads_evolution'] = leads_df['leads_apres'] - leads_df['leads_avant']
    leads_df['leads_evolution_pct'] = ((leads_df['leads_apres'] - leads_df['leads_avant']) / leads_df['leads_avant'].replace(0, 1) * 100).round(1)
    
    leads_df['url_normalized'] = leads_df['url'].apply(normalize_url)
    
    st.sidebar.success(f"✅ {len(leads_df):,} URLs avec données leads")
    if periode_avant and periode_apres:
        st.sidebar.info(f"Avant: {', '.join(periode_avant)}\nAprès: {', '.join(periode_apres)}")

if uploaded_file:
    df = load_data(uploaded_file)
    
    # Croiser avec les données leads si disponibles
    if has_leads and 'url' in df.columns:
        df['url_normalized'] = df['url'].apply(normalize_url)
        df = df.merge(
            leads_df[['url_normalized', 'leads_total', 'leads_avant', 'leads_apres', 'leads_evolution', 'leads_evolution_pct']], 
            on='url_normalized', 
            how='left'
        )
        # Score de priorité enrichi : intègre les leads
        df['priority_score_business'] = df['priority_score'] * (1 + df['leads_total'].fillna(0) / 100)
        st.success(f"✅ {len(df):,} mots-clés chargés — Données leads croisées !")
        has_leads_merged = True
    else:
        df['leads_total'] = 0
        df['leads_avant'] = 0
        df['leads_apres'] = 0
        df['leads_evolution'] = 0
        df['priority_score_business'] = df['priority_score']
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔴 Pertes", "📁 Par URL", "🟢 Gains", "📝 Rapport"])
    
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
            # Leads sur les URLs en perte
            urls_en_perte = df_f[df_f['diff_pos'] < 0]['url'].unique() if 'url' in df_f.columns else []
            leads_urls_perte = df_f[df_f['url'].isin(urls_en_perte)]['leads_total'].fillna(0).sum()
            c3.metric("⚠️ Leads sur URLs en perte", f"{int(leads_urls_perte):,}")
            
            leads_evol = df_f[df_f['diff_pos'] < 0]['leads_evolution'].fillna(0).sum()
            delta_color = "inverse" if leads_evol < 0 else "normal"
            c4.metric("📊 Évol. leads (période)", f"{int(leads_evol):+,}", delta_color=delta_color)
        
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
            if has_leads_merged and len(df_pertes_temp) > 0 and 'priority_score_business' in df_pertes_temp.columns and 'leads_total' in df_pertes_temp.columns:
                st.subheader("🎯 URLs critiques : Pertes SEO + Impact Business")
                
                # Construire l'agrégation dynamiquement selon les colonnes disponibles
                agg_dict_dash = {'diff_pos': ('diff_pos', 'count')}
                if 'volume' in df_pertes_temp.columns:
                    agg_dict_dash['volume_perdu'] = ('volume', 'sum')
                if 'leads_total' in df_pertes_temp.columns:
                    agg_dict_dash['leads_total'] = ('leads_total', 'first')
                if 'leads_evolution' in df_pertes_temp.columns:
                    agg_dict_dash['leads_evolution'] = ('leads_evolution', 'first')
                if 'priority_score_business' in df_pertes_temp.columns:
                    agg_dict_dash['score'] = ('priority_score_business', 'sum')
                
                df_perte_urls = df_pertes_temp.groupby('url').agg(**agg_dict_dash).reset_index()
                df_perte_urls = df_perte_urls.rename(columns={'diff_pos': 'kw_perdus'})
                
                if 'score' in df_perte_urls.columns:
                    df_perte_urls = df_perte_urls.sort_values('score', ascending=False).head(15)
                else:
                    df_perte_urls = df_perte_urls.sort_values('kw_perdus', ascending=False).head(15)
                
                st.dataframe(df_perte_urls, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible d'afficher les URLs critiques: {e}")
    
    # TAB 2: PERTES
    with tab2:
        st.header("🔴 Pertes critiques")
        df_pertes = df_f[df_f['diff_pos'] < 0].sort_values('priority_score', ascending=False)
        st.info(f"**{len(df_pertes):,}** mots-clés en perte")
        
        cols = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume', 'priority_score'] if c in df_pertes.columns]
        st.dataframe(df_pertes[cols].head(500), use_container_width=True, height=600)
        
        csv = df_pertes[cols].to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("📥 Export CSV", csv, "pertes.csv")
    
    # TAB 3: PAR URL
    with tab3:
        st.header("📁 Analyse par URL")
        if 'url' in df_f.columns:
            try:
                # Construire l'agrégation dynamiquement
                agg_funcs = {
                    'diff_pos': ['count', lambda x: (x < 0).sum(), lambda x: (x > 0).sum(), 'mean'],
                }
                if 'volume' in df_f.columns:
                    agg_funcs['volume'] = 'sum'
                if 'priority_score' in df_f.columns:
                    agg_funcs['priority_score'] = 'sum'
                if has_leads_merged:
                    if 'leads_total' in df_f.columns:
                        agg_funcs['leads_total'] = 'first'
                    if 'leads_avant' in df_f.columns:
                        agg_funcs['leads_avant'] = 'first'
                    if 'leads_apres' in df_f.columns:
                        agg_funcs['leads_apres'] = 'first'
                    if 'leads_evolution' in df_f.columns:
                        agg_funcs['leads_evolution'] = 'first'
                    if 'priority_score_business' in df_f.columns:
                        agg_funcs['priority_score_business'] = 'sum'
                
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
                    'diff_pos_mean': 'diff_moyen'
                }
                url_stats = url_stats.rename(columns=rename_dict)
                
                # Calculer le % santé si possible
                if 'total_kw' in url_stats.columns and 'kw_perte' in url_stats.columns:
                    url_stats['sante_pct'] = ((url_stats['total_kw'] - url_stats['kw_perte']) / url_stats['total_kw'] * 100).round(1)
                
                # Tri
                if 'priority_score_business' in url_stats.columns:
                    url_stats = url_stats.sort_values('priority_score_business', ascending=False)
                elif 'priority_score' in url_stats.columns:
                    url_stats = url_stats.sort_values('priority_score', ascending=False)
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
                    c1, c2, c3, c4 = st.columns(4)
                    leads_t = df_url['leads_total'].iloc[0] if len(df_url) > 0 else 0
                    leads_av = df_url['leads_avant'].iloc[0] if len(df_url) > 0 and 'leads_avant' in df_url.columns else 0
                    leads_ap = df_url['leads_apres'].iloc[0] if len(df_url) > 0 and 'leads_apres' in df_url.columns else 0
                    leads_e = df_url['leads_evolution'].iloc[0] if len(df_url) > 0 and 'leads_evolution' in df_url.columns else 0
                    c1.metric("📊 Leads total", f"{int(leads_t or 0):,}")
                    c2.metric("📊 Leads AVANT", f"{int(leads_av or 0):,}")
                    c3.metric("📊 Leads APRÈS", f"{int(leads_ap or 0):,}")
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
        df_gains = df_f[df_f['diff_pos'] > 0].sort_values('priority_score', ascending=False)
        st.success(f"**{len(df_gains):,}** mots-clés en gain")
        
        cols = [c for c in ['mot_cle', 'url', 'diff_pos', 'volume', 'derniere_pos', 'ancienne_pos'] if c in df_gains.columns]
        st.dataframe(df_gains[cols], use_container_width=True, height=600)
        
        csv_gains = df_gains[cols].to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("📥 Exporter TOUS les gains (CSV)", csv_gains, "gains_complet.csv")
    
    # TAB 5: RAPPORT
    with tab5:
        st.header("📝 Rapport complet pour l'équipe édito")
        
        if st.button("🔄 Générer le rapport complet", type="primary"):
            
            # Calculs pour le rapport
            df_pertes_rapport = df_f[df_f['diff_pos'] < 0].sort_values('priority_score_business' if has_leads_merged else 'priority_score', ascending=False)
            df_gains_rapport = df_f[df_f['diff_pos'] > 0].sort_values('priority_score', ascending=False)
            
            # URLs les plus impactées
            if 'url' in df_f.columns:
                agg_url = {
                    'diff_pos': 'count',
                    'volume': 'sum',
                    'priority_score': 'sum',
                }
                if has_leads_merged:
                    agg_url['leads_total'] = 'first'
                    agg_url['leads_avant'] = 'first'
                    agg_url['leads_apres'] = 'first'
                    agg_url['leads_evolution'] = 'first'
                    agg_url['priority_score_business'] = 'sum'
                
                urls_critiques = df_pertes_rapport.groupby('url').agg(agg_url).reset_index()
                urls_critiques.columns = ['url', 'nb_kw_perdus', 'volume_impacte', 'score_seo'] + \
                                        (['leads_total', 'leads_avant', 'leads_apres', 'leads_evolution', 'score_business'] if has_leads_merged else [])
                
                sort_col = 'score_business' if has_leads_merged else 'score_seo'
                urls_critiques = urls_critiques.sort_values(sort_col, ascending=False)
            
            # Calcul impact leads
            if has_leads_merged:
                total_leads_perte = int(df_pertes_rapport['leads_total'].fillna(0).sum())
                total_leads_avant_perte = int(df_pertes_rapport['leads_avant'].fillna(0).sum())
                total_leads_apres_perte = int(df_pertes_rapport['leads_apres'].fillna(0).sum())
                leads_evolution_total = int(df_f[df_f['diff_pos'] < 0]['leads_evolution'].fillna(0).sum())
            
            report = f"""# 📊 RAPPORT D'ANALYSE SEO COMPLET
## Période : Septembre 2025 → Février 2026
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

            report += """---

# 2. DIAGNOSTIC

"""
            if pertes > gains:
                report += f"""⚠️ **SITUATION PRÉOCCUPANTE** : Le site perd plus de positions qu'il n'en gagne.
- Ratio pertes/gains : {pertes/gains:.1f}x plus de pertes
- Action recommandée : **Audit urgent des contenus impactés**

"""
            else:
                report += f"""✅ **SITUATION POSITIVE** : Le site gagne plus de positions qu'il n'en perd.
- Ratio gains/pertes : {gains/pertes:.1f}x plus de gains

"""

            report += f"""---

# 3. TOUTES LES PAGES À TRAITER ({len(urls_critiques):,} URLs)

"""
            if has_leads_merged:
                report += """**Triées par SCORE BUSINESS (SEO × Leads)** — Les URLs avec pertes SEO ET leads historiques sont en priorité maximale.

| Priorité | URL | KW perdus | Volume | Leads total | Leads avant | Leads après | Évol. | Score |
|----------|-----|-----------|--------|-------------|-------------|-------------|-------|-------|
"""
                for i, row in urls_critiques.iterrows():
                    score = row.get('score_business', row.get('score_seo', 0))
                    prio = "🔴 CRITIQUE" if row.get('leads_total', 0) > 100 and row['nb_kw_perdus'] > 5 else \
                           "🟠 URGENT" if score > urls_critiques[sort_col].quantile(0.9) else \
                           "🟡 MOYEN" if score > urls_critiques[sort_col].quantile(0.5) else "⚪ FAIBLE"
                    report += f"| {prio} | {row['url']} | {int(row['nb_kw_perdus'])} | {int(row.get('volume_impacte', 0)):,} | {int(row.get('leads_total', 0) or 0):,} | {int(row.get('leads_avant', 0) or 0):,} | {int(row.get('leads_apres', 0) or 0):,} | {int(row.get('leads_evolution', 0) or 0):+,} | {int(score):,} |\n"
            else:
                report += """**Triées par score de priorité (volume × perte)**

| Priorité | URL | KW perdus | Volume impacté | Score |
|----------|-----|-----------|----------------|-------|
"""
                for i, row in urls_critiques.iterrows():
                    prio = "🔴 URGENT" if row['score_seo'] > urls_critiques['score_seo'].quantile(0.9) else "🟠 MOYEN" if row['score_seo'] > urls_critiques['score_seo'].quantile(0.5) else "🟡 FAIBLE"
                    report += f"| {prio} | {row['url']} | {int(row['nb_kw_perdus'])} | {int(row.get('volume_impacte', 0)):,} | {int(row['score_seo']):,} |\n"

            report += f"""

---

# 4. MOTS-CLÉS EN PERTE — LISTE COMPLÈTE ({len(df_pertes_rapport):,} KW)

**Triés par score de priorité"""
            
            if has_leads_merged:
                report += " (intégrant l'impact business)"
            
            report += """

| Mot-clé | URL | Ancienne pos | Nouvelle pos | Diff | Volume | Score |
|---------|-----|--------------|--------------|------|--------|-------|
"""
            for _, row in df_pertes_rapport.iterrows():
                mc = str(row.get('mot_cle', 'N/A'))[:50]
                url = str(row.get('url', 'N/A'))[:60]
                anc = int(row.get('ancienne_pos', 0) or 0)
                dern = int(row.get('derniere_pos', 0) or 0)
                diff = int(row.get('diff_pos', 0) or 0)
                vol = int(row.get('volume', 0) or 0)
                score = int(row.get('priority_score_business' if has_leads_merged else 'priority_score', 0) or 0)
                report += f"| {mc} | {url} | {anc} | {dern} | {diff} | {vol:,} | {score:,} |\n"

            report += f"""

---

# 5. MOTS-CLÉS EN GAIN — LISTE COMPLÈTE ({len(df_gains_rapport):,} KW)

**Ce qui fonctionne bien — à analyser pour répliquer**

| Mot-clé | URL | Ancienne pos | Nouvelle pos | Diff | Volume |
|---------|-----|--------------|--------------|------|--------|
"""
            for _, row in df_gains_rapport.iterrows():
                mc = str(row.get('mot_cle', 'N/A'))[:50]
                url = str(row.get('url', 'N/A'))[:60]
                anc = int(row.get('ancienne_pos', 0) or 0)
                dern = int(row.get('derniere_pos', 0) or 0)
                diff = int(row.get('diff_pos', 0) or 0)
                vol = int(row.get('volume', 0) or 0)
                report += f"| {mc} | {url} | {anc} | {dern} | +{diff} | {vol:,} |\n"

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
                df_export = df_f[df_f['diff_pos'] < 0].sort_values('priority_score_business' if has_leads_merged else 'priority_score', ascending=False)
                cols_export = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume', 'leads_total', 'leads_avant', 'leads_apres', 'leads_evolution', 'priority_score', 'priority_score_business'] if c in df_export.columns]
                csv_export = df_export[cols_export].to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "📥 Télécharger les données (CSV)",
                    csv_export,
                    "pertes_completes.csv",
                    "text/csv"
                )

else:
    st.info("👆 Charge un fichier CSV pour commencer")
