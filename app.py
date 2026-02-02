"""
Haloscan SEO Diff Analyzer
Version corrigée pour le format exact du fichier Baptiste
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

# =============================================================================
# INTERFACE
# =============================================================================

st.title("📊 Haloscan SEO Diff Analyzer")

with st.sidebar:
    st.header("📁 Import")
    uploaded_file = st.file_uploader("Charger le CSV Haloscan", type=['csv'])

if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"✅ {len(df):,} mots-clés chargés")
    
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
        
        c1, c2 = st.columns(2)
        c1.metric("📉 Volume perdu", f"{vol_perdu:,}")
        c2.metric("📈 Volume gagné", f"{vol_gagne:,}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=[pertes, gains, stables], names=['Pertes', 'Gains', 'Stables'],
                        color_discrete_sequence=['#EF4444', '#22C55E', '#6B7280'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df_f, x='diff_pos', nbins=50)
            st.plotly_chart(fig, use_container_width=True)
    
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
            url_stats = df_f.groupby('url').agg(
                total_kw=('diff_pos', 'count'),
                kw_perte=('diff_pos', lambda x: (x < 0).sum()),
                volume=('volume', 'sum') if 'volume' in df_f.columns else ('diff_pos', 'count'),
                score=('priority_score', 'sum')
            ).reset_index().sort_values('score', ascending=False)
            
            st.dataframe(url_stats.head(200), use_container_width=True, height=400)
            
            # Détail URL
            st.subheader("🔍 Détail")
            url_sel = st.selectbox("URL", url_stats['url'].head(50).tolist())
            if url_sel:
                df_url = df_f[df_f['url'] == url_sel]
                cols = [c for c in ['mot_cle', 'diff_pos', 'volume', 'derniere_pos'] if c in df_url.columns]
                st.dataframe(df_url[cols], use_container_width=True)
        else:
            st.warning("Colonne 'url' non trouvée")
    
    # TAB 4: GAINS
    with tab4:
        st.header("🟢 Gains")
        df_gains = df_f[df_f['diff_pos'] > 0].sort_values('priority_score', ascending=False)
        st.success(f"**{len(df_gains):,}** mots-clés en gain")
        
        cols = [c for c in ['mot_cle', 'url', 'diff_pos', 'volume'] if c in df_gains.columns]
        st.dataframe(df_gains[cols].head(500), use_container_width=True, height=600)
    
    # TAB 5: RAPPORT
    with tab5:
        st.header("📝 Rapport")
        if st.button("Générer"):
            report = f"""# Rapport SEO — {datetime.now().strftime('%d/%m/%Y')}

## Résumé
- Total: {total:,} KW
- Pertes: {pertes:,}
- Gains: {gains:,}
- Volume perdu: {vol_perdu:,}

## Top 10 Pertes
"""
            for _, r in df_f[df_f['diff_pos'] < 0].nlargest(10, 'priority_score').iterrows():
                report += f"- {r.get('mot_cle', 'N/A')} (vol:{int(r.get('volume', 0) or 0)}, diff:{int(r.get('diff_pos', 0) or 0)})\n"
            
            st.markdown(report)
            st.download_button("📥 Télécharger", report, "rapport.md")

else:
    st.info("👆 Charge un fichier CSV pour commencer")
