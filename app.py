"""
Haloscan SEO Diff Analyzer
Analyse des différentiels de positions SEO entre deux périodes
Conçu pour traiter des fichiers volumineux (250k+ lignes)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Haloscan SEO Diff Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Charge et parse le fichier CSV avec détection automatique du séparateur"""
    
    # Lire le fichier
    try:
        df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
        if len(df.columns) < 5:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep=',', encoding='latin-1')
            if len(df.columns) < 5:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',', encoding='cp1252')
    
    # Nettoyage des noms de colonnes
    df.columns = (df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace(';', '', regex=False)
        .str.replace(':', '', regex=False)
        .str.replace('é', 'e', regex=False)
        .str.replace('è', 'e', regex=False)
        .str.replace('ê', 'e', regex=False)
        .str.replace('à', 'a', regex=False)
    )
    
    # Mapping des colonnes
    column_mapping = {
        'mot-cle_(mc)': 'mot_cle',
        'mot_cle_(mc)': 'mot_cle',
        'mot-cle': 'mot_cle',
        'mc': 'mot_cle',
        'keyword': 'mot_cle',
        'derniere_pos': 'derniere_pos',
        'position': 'derniere_pos',
        'plus_vieille_pos': 'ancienne_pos',
        'vieille_pos': 'ancienne_pos',
        'meilleure_pos': 'meilleure_pos',
        'diff_pos': 'diff_pos',
        'diff': 'diff_pos',
        'volume': 'volume',
        'trafic': 'trafic',
        'traffic': 'trafic',
        'url': 'url',
        'statut': 'statut',
        'cpc': 'cpc',
        'comp': 'competition',
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Conversion des colonnes numériques
    numeric_cols = ['derniere_pos', 'ancienne_pos', 'meilleure_pos', 'diff_pos', 'volume', 'trafic', 'cpc']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def safe_get(df, col, default=0):
    """Récupère une colonne de manière sécurisée"""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def calculate_priority_score(df):
    """Calcule le score de priorité pour tout le dataframe"""
    volume = safe_get(df, 'volume', 0)
    diff = safe_get(df, 'diff_pos', 0).abs()
    ancienne_pos = safe_get(df, 'ancienne_pos', 100)
    
    # Facteur de position
    facteur = pd.Series([1.0] * len(df), index=df.index)
    facteur = facteur.where(ancienne_pos > 20, 1.5)
    facteur = facteur.where(ancienne_pos > 10, 2.0)
    facteur = facteur.where(ancienne_pos > 3, 3.0)
    
    return volume * diff * facteur


def calculate_recovery_potential(df):
    """Calcule le potentiel de récupération"""
    volume = safe_get(df, 'volume', 0)
    meilleure_pos = safe_get(df, 'meilleure_pos', 100).replace(0, 1)
    return volume / meilleure_pos


# =============================================================================
# INTERFACE PRINCIPALE
# =============================================================================

st.title("📊 Haloscan SEO Diff Analyzer")
st.markdown("Analyse des différentiels de positions SEO • Conçu pour fichiers volumineux (250k+ lignes)")

# Sidebar - Upload
with st.sidebar:
    st.header("📁 Import des données")
    uploaded_file = st.file_uploader(
        "Charger le fichier CSV Haloscan",
        type=['csv'],
        help="Export Haloscan avec colonnes : mot-clé, url, positions, diff, volume..."
    )

# =============================================================================
# TRAITEMENT DES DONNÉES
# =============================================================================

if uploaded_file:
    with st.spinner("⏳ Chargement et analyse des données..."):
        df = load_data(uploaded_file)
        
        # Calcul des scores
        df['priority_score'] = calculate_priority_score(df)
        df['recovery_potential'] = calculate_recovery_potential(df)
        
    st.success(f"✅ {len(df):,} mots-clés chargés")
    
    # Debug : colonnes détectées
    with st.sidebar:
        with st.expander("🔍 Colonnes détectées", expanded=True):
            st.write(list(df.columns))
    
    # Vérification des colonnes critiques
    has_diff = 'diff_pos' in df.columns
    has_volume = 'volume' in df.columns
    has_url = 'url' in df.columns
    has_mot_cle = 'mot_cle' in df.columns
    has_derniere_pos = 'derniere_pos' in df.columns
    has_ancienne_pos = 'ancienne_pos' in df.columns
    has_meilleure_pos = 'meilleure_pos' in df.columns
    has_trafic = 'trafic' in df.columns
    
    if not has_diff:
        st.error("❌ Colonne 'diff_pos' non trouvée ! Vérifiez votre fichier.")
        st.info(f"Colonnes disponibles : {', '.join(df.columns.tolist())}")
        st.stop()
    
    # ==========================================================================
    # FILTRES
    # ==========================================================================
    
    with st.sidebar:
        st.header("🎛️ Filtres")
        
        # Filtre par type de variation
        variation_filter = st.multiselect(
            "Type de variation",
            options=['Pertes', 'Gains', 'Stables'],
            default=['Pertes', 'Gains', 'Stables']
        )
        
        # Filtre par volume
        if has_volume:
            vol_min = int(df['volume'].min() or 0)
            vol_max = int(df['volume'].max() or 10000)
            volume_range = st.slider("Volume de recherche", vol_min, vol_max, (vol_min, vol_max))
        else:
            volume_range = None
        
        # Filtre par diff
        diff_min = int(df['diff_pos'].min() or -100)
        diff_max = int(df['diff_pos'].max() or 100)
        diff_range = st.slider("Différentiel de position", diff_min, diff_max, (diff_min, diff_max))
        
        # Filtre par position
        if has_derniere_pos:
            position_filter = st.selectbox(
                "Tranche de position actuelle",
                options=['Toutes', 'Top 3', 'Top 10', 'Top 20', 'Page 2 (11-20)', 'Page 3+ (21+)']
            )
        else:
            position_filter = 'Toutes'
        
        # Recherche textuelle
        search_kw = st.text_input("🔎 Rechercher un mot-clé", "")
        search_url = st.text_input("🔎 Filtrer par URL (contient)", "")
    
    # Application des filtres
    df_filtered = df.copy()
    
    # Filtre variation
    conditions = []
    if 'Pertes' in variation_filter:
        conditions.append(df_filtered['diff_pos'] < 0)
    if 'Gains' in variation_filter:
        conditions.append(df_filtered['diff_pos'] > 0)
    if 'Stables' in variation_filter:
        conditions.append(df_filtered['diff_pos'] == 0)
    
    if conditions:
        mask = conditions[0]
        for cond in conditions[1:]:
            mask = mask | cond
        df_filtered = df_filtered[mask]
    
    # Filtre volume
    if volume_range and has_volume:
        df_filtered = df_filtered[
            (df_filtered['volume'] >= volume_range[0]) & 
            (df_filtered['volume'] <= volume_range[1])
        ]
    
    # Filtre diff
    df_filtered = df_filtered[
        (df_filtered['diff_pos'] >= diff_range[0]) & 
        (df_filtered['diff_pos'] <= diff_range[1])
    ]
    
    # Filtre position
    if position_filter != 'Toutes' and has_derniere_pos:
        if position_filter == 'Top 3':
            df_filtered = df_filtered[df_filtered['derniere_pos'] <= 3]
        elif position_filter == 'Top 10':
            df_filtered = df_filtered[df_filtered['derniere_pos'] <= 10]
        elif position_filter == 'Top 20':
            df_filtered = df_filtered[df_filtered['derniere_pos'] <= 20]
        elif position_filter == 'Page 2 (11-20)':
            df_filtered = df_filtered[(df_filtered['derniere_pos'] >= 11) & (df_filtered['derniere_pos'] <= 20)]
        elif position_filter == 'Page 3+ (21+)':
            df_filtered = df_filtered[df_filtered['derniere_pos'] >= 21]
    
    # Filtre recherche
    if search_kw and has_mot_cle:
        df_filtered = df_filtered[df_filtered['mot_cle'].astype(str).str.contains(search_kw, case=False, na=False)]
    
    if search_url and has_url:
        df_filtered = df_filtered[df_filtered['url'].astype(str).str.contains(search_url, case=False, na=False)]
    
    # ==========================================================================
    # CALCUL DES KPIs
    # ==========================================================================
    
    total_kw = len(df_filtered)
    pertes = len(df_filtered[df_filtered['diff_pos'] < 0])
    gains = len(df_filtered[df_filtered['diff_pos'] > 0])
    stables = len(df_filtered[df_filtered['diff_pos'] == 0])
    
    if has_derniere_pos:
        sortis = len(df_filtered[(df_filtered['derniere_pos'] > 100) | (df_filtered['derniere_pos'].isna())])
    else:
        sortis = 0
    
    if has_volume:
        volume_perdu = int(df_filtered[df_filtered['diff_pos'] < 0]['volume'].fillna(0).sum())
        volume_gagne = int(df_filtered[df_filtered['diff_pos'] > 0]['volume'].fillna(0).sum())
    else:
        volume_perdu = volume_gagne = 0
    
    if has_trafic:
        trafic_perdu = int(df_filtered[df_filtered['diff_pos'] < 0]['trafic'].fillna(0).sum())
        trafic_gagne = int(df_filtered[df_filtered['diff_pos'] > 0]['trafic'].fillna(0).sum())
    else:
        trafic_perdu = trafic_gagne = 0
    
    pct_pertes = (pertes / total_kw * 100) if total_kw > 0 else 0
    pct_gains = (gains / total_kw * 100) if total_kw > 0 else 0
    pct_stables = (stables / total_kw * 100) if total_kw > 0 else 0
    
    # ==========================================================================
    # ONGLETS
    # ==========================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dashboard",
        "🔴 Pertes critiques",
        "📁 Par URL",
        "⚡ Quick wins",
        "❌ Sortis",
        "🟢 Gains",
        "📝 Rapport"
    ])
    
    # ==========================================================================
    # TAB 1 : DASHBOARD
    # ==========================================================================
    
    with tab1:
        st.header("Vue d'ensemble")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total KW", f"{total_kw:,}")
        col2.metric("🔴 Pertes", f"{pertes:,}", f"{pct_pertes:.1f}%")
        col3.metric("🟢 Gains", f"{gains:,}", f"{pct_gains:.1f}%")
        col4.metric("⚪ Stables", f"{stables:,}", f"{pct_stables:.1f}%")
        col5.metric("🟠 Sortis", f"{sortis:,}")
        
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📉 Volume perdu", f"{volume_perdu:,}")
        col2.metric("📈 Volume gagné", f"{volume_gagne:,}")
        col3.metric("🚫 Trafic perdu", f"{trafic_perdu:,}")
        col4.metric("✅ Trafic gagné", f"{trafic_gagne:,}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Répartition par statut")
            fig_pie = px.pie(
                values=[pertes, gains, stables, sortis],
                names=['Pertes', 'Gains', 'Stables', 'Sortis'],
                color_discrete_sequence=['#EF4444', '#22C55E', '#6B7280', '#F97316']
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Distribution des différentiels")
            fig_hist = px.histogram(df_filtered, x='diff_pos', nbins=50, color_discrete_sequence=['#667eea'])
            fig_hist.update_layout(xaxis_title="Différentiel", yaxis_title="Nombre de KW", height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Top URLs impactées
        if has_url and has_volume:
            st.subheader("Top 10 URLs les plus impactées")
            df_pertes_url = df_filtered[df_filtered['diff_pos'] < 0].copy()
            if len(df_pertes_url) > 0:
                url_stats = df_pertes_url.groupby('url').agg(
                    nb_kw=('diff_pos', 'count'),
                    volume_perdu=('volume', lambda x: x.fillna(0).sum())
                ).sort_values('volume_perdu', ascending=False).head(10).reset_index()
                
                fig_bar = px.bar(url_stats, x='volume_perdu', y='url', orientation='h', color_discrete_sequence=['#EF4444'])
                fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # ==========================================================================
    # TAB 2 : PERTES CRITIQUES
    # ==========================================================================
    
    with tab2:
        st.header("🔴 Pertes critiques")
        st.markdown("Mots-clés triés par **score de priorité** (volume × diff × facteur position)")
        
        df_pertes = df_filtered[df_filtered['diff_pos'] < 0].sort_values('priority_score', ascending=False)
        
        st.info(f"**{len(df_pertes):,}** mots-clés en perte de position")
        
        # Colonnes à afficher
        cols_display = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume', 'trafic', 'priority_score'] if c in df_pertes.columns]
        
        st.dataframe(df_pertes[cols_display].head(500), use_container_width=True, height=600)
        
        # Export
        csv = df_pertes[cols_display].to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("📥 Exporter (CSV)", csv, "pertes_critiques.csv", "text/csv")
    
    # ==========================================================================
    # TAB 3 : PAR URL
    # ==========================================================================
    
    with tab3:
        st.header("📁 Analyse par URL")
        
        if not has_url:
            st.warning("Colonne 'url' non détectée dans le fichier")
        else:
            # Stats par URL
            url_agg = df_filtered.groupby('url').agg(
                total_kw=('diff_pos', 'count'),
                kw_en_perte=('diff_pos', lambda x: (x < 0).sum()),
                diff_moyen=('diff_pos', 'mean'),
                volume_total=('volume', lambda x: x.fillna(0).sum()) if has_volume else ('diff_pos', 'count'),
                score_priorite=('priority_score', 'sum')
            ).reset_index()
            
            if not has_volume:
                url_agg = url_agg.drop(columns=['volume_total'], errors='ignore')
            
            url_agg['sante_pct'] = ((url_agg['total_kw'] - url_agg['kw_en_perte']) / url_agg['total_kw'] * 100).round(1)
            url_agg = url_agg.sort_values('score_priorite', ascending=False)
            
            st.info(f"**{len(url_agg):,}** URLs analysées")
            st.dataframe(url_agg.head(200), use_container_width=True, height=400)
            
            # Détail d'une URL
            st.subheader("🔍 Détail d'une URL")
            url_list = url_agg['url'].head(100).tolist()
            if url_list:
                url_selectionnee = st.selectbox("Sélectionner une URL", url_list)
                
                df_url = df_filtered[df_filtered['url'] == url_selectionnee]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("KW total", len(df_url))
                col2.metric("KW en perte", len(df_url[df_url['diff_pos'] < 0]))
                if has_volume:
                    col3.metric("Volume total", f"{int(df_url['volume'].fillna(0).sum()):,}")
                
                cols_url = [c for c in ['mot_cle', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume'] if c in df_url.columns]
                st.dataframe(df_url[cols_url], use_container_width=True)
    
    # ==========================================================================
    # TAB 4 : QUICK WINS
    # ==========================================================================
    
    with tab4:
        st.header("⚡ Quick wins — Opportunités de récupération")
        st.markdown("KW qui étaient **top 10**, ont chuté, mais ont un **potentiel de récupération**")
        
        if not has_meilleure_pos or not has_derniere_pos:
            st.warning("Colonnes 'meilleure_pos' et/ou 'derniere_pos' non détectées")
        else:
            mask = (
                (df_filtered['meilleure_pos'] <= 10) &
                (df_filtered['derniere_pos'] > 10)
            )
            if has_volume:
                mask = mask & (df_filtered['volume'] >= 100)
            
            df_qw = df_filtered[mask].sort_values('recovery_potential', ascending=False)
            
            st.success(f"**{len(df_qw):,}** opportunités identifiées")
            
            cols_qw = [c for c in ['mot_cle', 'url', 'meilleure_pos', 'derniere_pos', 'diff_pos', 'volume', 'recovery_potential'] if c in df_qw.columns]
            st.dataframe(df_qw[cols_qw].head(500), use_container_width=True, height=600)
            
            csv = df_qw[cols_qw].to_csv(index=False, sep=';').encode('utf-8')
            st.download_button("📥 Exporter (CSV)", csv, "quick_wins.csv", "text/csv")
    
    # ==========================================================================
    # TAB 5 : SORTIS
    # ==========================================================================
    
    with tab5:
        st.header("❌ Mots-clés sortis des SERPs")
        
        if not has_derniere_pos:
            st.warning("Colonne 'derniere_pos' non détectée")
        else:
            df_sortis = df_filtered[(df_filtered['derniere_pos'] > 100) | (df_filtered['derniere_pos'].isna())]
            
            if has_volume:
                df_sortis = df_sortis.sort_values('volume', ascending=False)
            
            st.warning(f"**{len(df_sortis):,}** mots-clés ont disparu des SERPs")
            
            cols_sortis = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'volume', 'trafic'] if c in df_sortis.columns]
            st.dataframe(df_sortis[cols_sortis].head(500), use_container_width=True, height=600)
            
            csv = df_sortis[cols_sortis].to_csv(index=False, sep=';').encode('utf-8')
            st.download_button("📥 Exporter (CSV)", csv, "kw_sortis.csv", "text/csv")
    
    # ==========================================================================
    # TAB 6 : GAINS
    # ==========================================================================
    
    with tab6:
        st.header("🟢 Gains de position")
        
        df_gains = df_filtered[df_filtered['diff_pos'] > 0].sort_values('priority_score', ascending=False)
        
        st.success(f"**{len(df_gains):,}** mots-clés en progression")
        
        cols_gains = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume', 'trafic'] if c in df_gains.columns]
        st.dataframe(df_gains[cols_gains].head(500), use_container_width=True, height=600)
    
    # ==========================================================================
    # TAB 7 : RAPPORT
    # ==========================================================================
    
    with tab7:
        st.header("📝 Rapport pour l'équipe contenu")
        
        if st.button("🔄 Générer le rapport", type="primary"):
            report = f"""# Rapport d'Analyse SEO — {datetime.now().strftime('%d/%m/%Y')}

## 📊 Situation globale

- **Mots-clés analysés** : {total_kw:,}
- **En perte** : {pertes:,} ({pct_pertes:.1f}%)
- **En gain** : {gains:,} ({pct_gains:.1f}%)
- **Stables** : {stables:,} ({pct_stables:.1f}%)
- **Sortis des SERPs** : {sortis:,}

### Impact business
- **Volume perdu** : {volume_perdu:,} recherches/mois
- **Trafic perdu** : {trafic_perdu:,} visites/mois
- **Volume gagné** : {volume_gagne:,} recherches/mois

---

## 🚨 Top 10 Pertes critiques

"""
            df_top_pertes = df_filtered[df_filtered['diff_pos'] < 0].nlargest(10, 'priority_score')
            for i, row in df_top_pertes.iterrows():
                mc = row.get('mot_cle', 'N/A')
                vol = int(row.get('volume', 0) or 0)
                diff = int(row.get('diff_pos', 0) or 0)
                url = row.get('url', 'N/A')
                report += f"- **{mc}** (vol: {vol:,}, diff: {diff}) — {url}\n"
            
            report += """

---

## 📋 Recommandations

1. **Prioriser les quick wins** : KW anciennement top 10, récupérables avec mise à jour contenu + maillage
2. **Auditer les pages critiques** : Les URLs avec le plus de pertes nécessitent un audit complet
3. **Surveiller la concurrence** : Vérifier si pertes dues à l'algo ou aux concurrents

---
_Rapport généré automatiquement — Haloscan SEO Diff Analyzer_
"""
            st.session_state['report'] = report
        
        if 'report' in st.session_state:
            st.markdown(st.session_state['report'])
            st.download_button("📥 Télécharger (Markdown)", st.session_state['report'], "rapport_seo.md", "text/markdown")

else:
    st.info("👆 Charge un fichier CSV Haloscan dans la sidebar pour commencer")
    
    st.markdown("""
    ### 📋 Format attendu
    
    Colonnes requises :
    - `mot-clé (mc)` — le mot-clé
    - `url` — l'URL positionnée  
    - `diff_pos` — différentiel de position
    - `volume` — volume de recherche
    
    ### 🚀 Capacité
    
    L'outil gère **300 000+ lignes** sans problème.
    """)
