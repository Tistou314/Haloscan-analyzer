"""
Haloscan SEO Diff Analyzer
Analyse des différentiels de positions SEO entre deux périodes
Conçu pour traiter des fichiers volumineux (250k+ lignes)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Haloscan SEO Diff Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .loss { color: #EF4444; }
    .gain { color: #22C55E; }
    .stable { color: #6B7280; }
    .out { color: #F97316; }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(uploaded_file):
    """Charge et parse le fichier CSV avec détection automatique du séparateur"""
    try:
        # Essai avec point-virgule d'abord (format Haloscan habituel)
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        if len(df.columns) < 5:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
            if len(df.columns) < 5:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', encoding='latin-1')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', encoding='cp1252')
    
    # Normalisation des noms de colonnes
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Renommage des colonnes courantes Haloscan
    column_mapping = {
        'mot-clé_(mc)': 'mot_cle',
        'mot-clé': 'mot_cle',
        'mc': 'mot_cle',
        'keyword': 'mot_cle',
        'dernière_pos': 'derniere_pos',
        'derniere_pos': 'derniere_pos',
        'position': 'derniere_pos',
        'vieille_pos': 'ancienne_pos',
        'plus_vieille_pos': 'ancienne_pos',
        'old_pos': 'ancienne_pos',
        'meilleure_pos': 'meilleure_pos',
        'best_pos': 'meilleure_pos',
        'pos_perdues': 'pos_perdues',
        'diff_pos': 'diff_pos',
        'diff': 'diff_pos',
        'volume': 'volume',
        'vol': 'volume',
        'volumeh': 'volumeh',
        'statut': 'statut',
        'status': 'statut',
        'trafic': 'trafic',
        'traffic': 'trafic',
        'url': 'url',
        'cpc': 'cpc',
        'comp': 'competition',
        'competition': 'competition'
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Conversion des colonnes numériques
    numeric_cols = ['derniere_pos', 'ancienne_pos', 'meilleure_pos', 'diff_pos', 'pos_perdues', 'volume', 'volumeh', 'trafic', 'cpc']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def calculate_priority_score(row):
    """Calcule le score de priorité pour un mot-clé"""
    volume = row.get('volume', 0) or 0
    diff = abs(row.get('diff_pos', 0) or 0)
    ancienne_pos = row.get('ancienne_pos', 100) or 100
    
    # Facteur de position : plus on était haut, plus c'est grave de perdre
    if ancienne_pos <= 3:
        facteur = 3
    elif ancienne_pos <= 10:
        facteur = 2
    elif ancienne_pos <= 20:
        facteur = 1.5
    else:
        facteur = 1
    
    return volume * diff * facteur


def calculate_recovery_potential(row):
    """Calcule le potentiel de récupération"""
    volume = row.get('volume', 0) or 0
    meilleure_pos = row.get('meilleure_pos', 100) or 100
    if meilleure_pos == 0:
        meilleure_pos = 1
    return volume / meilleure_pos


def get_status_color(statut):
    """Retourne la couleur associée au statut"""
    statut = str(statut).lower()
    if 'perd' in statut or 'lost' in statut or 'down' in statut:
        return '#EF4444'
    elif 'gagn' in statut or 'gain' in statut or 'up' in statut:
        return '#22C55E'
    elif 'stable' in statut:
        return '#6B7280'
    elif 'sort' in statut or 'out' in statut:
        return '#F97316'
    elif 'nouveau' in statut or 'new' in statut:
        return '#3B82F6'
    return '#6B7280'


def generate_report(df, df_filtered, kpis):
    """Génère le rapport textuel pour l'équipe contenu"""
    
    # Top 5 URLs impactées
    if 'url' in df_filtered.columns:
        url_impact = df_filtered[df_filtered['diff_pos'] < 0].groupby('url').agg({
            'mot_cle': 'count',
            'volume': 'sum',
            'trafic': 'sum'
        }).sort_values('volume', ascending=False).head(5)
    else:
        url_impact = pd.DataFrame()
    
    # Quick wins
    quick_wins = df_filtered[
        (df_filtered.get('meilleure_pos', pd.Series([100]*len(df_filtered))) <= 10) &
        (df_filtered.get('derniere_pos', pd.Series([0]*len(df_filtered))) > 10) &
        (df_filtered.get('volume', pd.Series([0]*len(df_filtered))) >= 100)
    ].nlargest(10, 'volume') if 'meilleure_pos' in df_filtered.columns else pd.DataFrame()
    
    # Top pertes
    top_pertes = df_filtered[df_filtered['diff_pos'] < 0].nlargest(10, 'priority_score')
    
    # KW sortis
    if 'statut' in df_filtered.columns:
        statut_str = df_filtered['statut'].astype(str).str.lower()
        kw_sortis = df_filtered[statut_str.str.contains('sort|out', na=False)].nlargest(10, 'volume')
    else:
        kw_sortis = df_filtered[df_filtered['derniere_pos'] > 100].nlargest(10, 'volume') if 'derniere_pos' in df_filtered.columns else pd.DataFrame()
    
    report = f"""# Rapport d'Analyse SEO — {datetime.now().strftime('%d/%m/%Y')}

## 📊 Situation globale

- **Mots-clés analysés** : {kpis['total']:,}
- **En perte** : {kpis['pertes']:,} ({kpis['pct_pertes']:.1f}%)
- **En gain** : {kpis['gains']:,} ({kpis['pct_gains']:.1f}%)
- **Stables** : {kpis['stables']:,} ({kpis['pct_stables']:.1f}%)
- **Sortis des SERPs** : {kpis['sortis']:,}

### Impact business
- **Volume de recherche impacté (pertes)** : {kpis['volume_perdu']:,} recherches/mois
- **Trafic estimé perdu** : {kpis['trafic_perdu']:,} visites/mois
- **Volume gagné** : {kpis['volume_gagne']:,} recherches/mois

---

## 🚨 Pages critiques (Top 5 URLs impactées)

"""
    
    if not url_impact.empty:
        for i, (url, row) in enumerate(url_impact.iterrows(), 1):
            report += f"{i}. **{url}**\n   - {int(row['mot_cle'])} KW en perte\n   - Volume impacté : {int(row['volume']):,}\n\n"
    else:
        report += "_Données URL non disponibles_\n\n"
    
    report += """---

## ⚡ Actions prioritaires

### 🔴 Urgence haute — Quick wins (récupération rapide possible)

Ces mots-clés étaient en top 10 et ont chuté. Le potentiel de récupération est élevé :

"""
    
    if not quick_wins.empty:
        for _, row in quick_wins.head(5).iterrows():
            mc = row.get('mot_cle', 'N/A')
            vol = int(row.get('volume', 0))
            best = int(row.get('meilleure_pos', 0))
            current = int(row.get('derniere_pos', 0))
            url = row.get('url', 'N/A')
            report += f"- **{mc}** (vol: {vol:,}) : était #{best} → maintenant #{current}\n  URL : {url}\n\n"
    else:
        report += "_Aucun quick win identifié avec les critères actuels_\n\n"
    
    report += """
### 🟠 Urgence moyenne — Top pertes par impact

Ces mots-clés représentent les plus grosses pertes pondérées (volume × chute × position d'origine) :

"""
    
    if not top_pertes.empty:
        for _, row in top_pertes.head(5).iterrows():
            mc = row.get('mot_cle', 'N/A')
            vol = int(row.get('volume', 0))
            diff = int(row.get('diff_pos', 0))
            url = row.get('url', 'N/A')
            report += f"- **{mc}** (vol: {vol:,}, diff: {diff})\n  URL : {url}\n\n"
    
    report += """
### 🟡 À surveiller — Mots-clés sortis des SERPs

"""
    
    if not kw_sortis.empty:
        for _, row in kw_sortis.head(5).iterrows():
            mc = row.get('mot_cle', 'N/A')
            vol = int(row.get('volume', 0))
            report += f"- **{mc}** (vol: {vol:,})\n"
    else:
        report += "_Aucun mot-clé sorti identifié_\n"
    
    report += f"""

---

## 📋 Recommandations générales

1. **Prioriser les quick wins** : Ces KW étaient bien positionnés, une mise à jour du contenu + renforcement du maillage interne peut suffire.

2. **Auditer les pages critiques** : Les {min(5, len(url_impact))} URLs les plus impactées nécessitent un audit complet (contenu, technique, backlinks).

3. **Surveiller la concurrence** : Vérifier si les pertes sont dues à des mises à jour algo ou à des concurrents qui ont amélioré leur contenu.

4. **Planifier les mises à jour** : Créer un calendrier éditorial pour retravailler les contenus impactés par ordre de priorité.

---

_Rapport généré automatiquement — Haloscan SEO Diff Analyzer_
"""
    
    return report


# =============================================================================
# INTERFACE PRINCIPALE
# =============================================================================

st.title("📊 Haloscan SEO Diff Analyzer")
st.markdown("Analyse des différentiels de positions SEO • Conçu pour fichiers volumineux (250k+ lignes)")

# Sidebar - Upload et filtres
with st.sidebar:
    st.header("📁 Import des données")
    uploaded_file = st.file_uploader(
        "Charger le fichier CSV Haloscan",
        type=['csv'],
        help="Export Haloscan avec colonnes : mot-clé, url, positions, diff, volume, statut..."
    )
    
    if uploaded_file:
        st.success(f"✅ Fichier chargé : {uploaded_file.name}")

# Chargement des données
if uploaded_file:
    with st.spinner("⏳ Chargement et analyse des données..."):
        df = load_data(uploaded_file)
        
        # Calcul du score de priorité
        df['priority_score'] = df.apply(calculate_priority_score, axis=1)
        
        # Calcul du potentiel de récupération si possible
        if 'meilleure_pos' in df.columns:
            df['recovery_potential'] = df.apply(calculate_recovery_potential, axis=1)
        
        st.success(f"✅ {len(df):,} mots-clés chargés")
    
    # Affichage des colonnes détectées
    with st.sidebar:
        with st.expander("🔍 Colonnes détectées"):
            st.write(list(df.columns))
    
    # ==========================================================================
    # FILTRES
    # ==========================================================================
    
    with st.sidebar:
        st.header("🎛️ Filtres")
        
        # Filtre par statut
        if 'statut' in df.columns:
            statuts_disponibles = df['statut'].astype(str).dropna().unique().tolist()
            statuts_selectionnes = st.multiselect(
                "Statut",
                options=statuts_disponibles,
                default=statuts_disponibles
            )
        else:
            statuts_selectionnes = None
        
        # Filtre par volume
        if 'volume' in df.columns:
            vol_min, vol_max = int(df['volume'].min() or 0), int(df['volume'].max() or 10000)
            volume_range = st.slider(
                "Volume de recherche",
                min_value=vol_min,
                max_value=vol_max,
                value=(vol_min, vol_max)
            )
        else:
            volume_range = None
        
        # Filtre par diff
        if 'diff_pos' in df.columns:
            diff_min, diff_max = int(df['diff_pos'].min() or -100), int(df['diff_pos'].max() or 100)
            diff_range = st.slider(
                "Différentiel de position",
                min_value=diff_min,
                max_value=diff_max,
                value=(diff_min, diff_max)
            )
        else:
            diff_range = None
        
        # Filtre par position
        if 'derniere_pos' in df.columns:
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
    
    if statuts_selectionnes is not None and 'statut' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['statut'].isin(statuts_selectionnes)]
    
    if volume_range and 'volume' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['volume'] >= volume_range[0]) & 
            (df_filtered['volume'] <= volume_range[1])
        ]
    
    if diff_range and 'diff_pos' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['diff_pos'] >= diff_range[0]) & 
            (df_filtered['diff_pos'] <= diff_range[1])
        ]
    
    if position_filter != 'Toutes' and 'derniere_pos' in df_filtered.columns:
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
    
    if search_kw and 'mot_cle' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['mot_cle'].str.contains(search_kw, case=False, na=False)]
    
    if search_url and 'url' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['url'].str.contains(search_url, case=False, na=False)]
    
    # ==========================================================================
    # CALCUL DES KPIs
    # ==========================================================================
    
    total_kw = len(df_filtered)
    
    # Détection des pertes/gains selon la colonne disponible
    if 'statut' in df_filtered.columns:
        # Conversion en string pour éviter les erreurs sur colonnes mixtes
        statut_str = df_filtered['statut'].astype(str).str.lower()
        pertes = len(df_filtered[statut_str.str.contains('perd|lost|down', na=False)])
        gains = len(df_filtered[statut_str.str.contains('gagn|gain|up', na=False)])
        stables = len(df_filtered[statut_str.str.contains('stable', na=False)])
        sortis = len(df_filtered[statut_str.str.contains('sort|out', na=False)])
    elif 'diff_pos' in df_filtered.columns:
        pertes = len(df_filtered[df_filtered['diff_pos'] < 0])
        gains = len(df_filtered[df_filtered['diff_pos'] > 0])
        stables = len(df_filtered[df_filtered['diff_pos'] == 0])
        sortis = len(df_filtered[df_filtered['derniere_pos'] > 100]) if 'derniere_pos' in df_filtered.columns else 0
    else:
        pertes = gains = stables = sortis = 0
    
    # Calculs de volume/trafic
    if 'volume' in df_filtered.columns and 'diff_pos' in df_filtered.columns:
        volume_perdu = int(df_filtered[df_filtered['diff_pos'] < 0]['volume'].sum())
        volume_gagne = int(df_filtered[df_filtered['diff_pos'] > 0]['volume'].sum())
    else:
        volume_perdu = volume_gagne = 0
    
    if 'trafic' in df_filtered.columns and 'diff_pos' in df_filtered.columns:
        trafic_perdu = int(df_filtered[df_filtered['diff_pos'] < 0]['trafic'].sum())
        trafic_gagne = int(df_filtered[df_filtered['diff_pos'] > 0]['trafic'].sum())
    else:
        trafic_perdu = trafic_gagne = 0
    
    kpis = {
        'total': total_kw,
        'pertes': pertes,
        'gains': gains,
        'stables': stables,
        'sortis': sortis,
        'pct_pertes': (pertes / total_kw * 100) if total_kw > 0 else 0,
        'pct_gains': (gains / total_kw * 100) if total_kw > 0 else 0,
        'pct_stables': (stables / total_kw * 100) if total_kw > 0 else 0,
        'volume_perdu': volume_perdu,
        'volume_gagne': volume_gagne,
        'trafic_perdu': trafic_perdu,
        'trafic_gagne': trafic_gagne
    }
    
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
        
        # KPIs principaux
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total KW", f"{total_kw:,}")
        with col2:
            st.metric("🔴 Pertes", f"{pertes:,}", f"{kpis['pct_pertes']:.1f}%")
        with col3:
            st.metric("🟢 Gains", f"{gains:,}", f"{kpis['pct_gains']:.1f}%")
        with col4:
            st.metric("⚪ Stables", f"{stables:,}", f"{kpis['pct_stables']:.1f}%")
        with col5:
            st.metric("🟠 Sortis", f"{sortis:,}")
        
        st.divider()
        
        # Impact business
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📉 Volume perdu", f"{volume_perdu:,}")
        with col2:
            st.metric("📈 Volume gagné", f"{volume_gagne:,}")
        with col3:
            st.metric("🚫 Trafic perdu", f"{trafic_perdu:,}")
        with col4:
            st.metric("✅ Trafic gagné", f"{trafic_gagne:,}")
        
        st.divider()
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Répartition par statut")
            if 'statut' in df_filtered.columns:
                statut_counts = df_filtered['statut'].astype(str).value_counts()
                fig_pie = px.pie(
                    values=statut_counts.values,
                    names=statut_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_pie.update_layout(height=350)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                # Pie chart basé sur diff_pos
                labels = ['Pertes', 'Gains', 'Stables']
                values = [pertes, gains, stables]
                fig_pie = px.pie(values=values, names=labels, color_discrete_sequence=['#EF4444', '#22C55E', '#6B7280'])
                fig_pie.update_layout(height=350)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Distribution des différentiels")
            if 'diff_pos' in df_filtered.columns:
                fig_hist = px.histogram(
                    df_filtered,
                    x='diff_pos',
                    nbins=50,
                    color_discrete_sequence=['#667eea']
                )
                fig_hist.update_layout(
                    xaxis_title="Différentiel de position",
                    yaxis_title="Nombre de KW",
                    height=350
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # Top URLs impactées
        if 'url' in df_filtered.columns and 'volume' in df_filtered.columns:
            st.subheader("Top 10 URLs les plus impactées (en volume perdu)")
            url_impact = df_filtered[df_filtered['diff_pos'] < 0].groupby('url').agg({
                'mot_cle': 'count',
                'volume': 'sum'
            }).rename(columns={'mot_cle': 'nb_kw_perdus', 'volume': 'volume_impacte'})
            url_impact = url_impact.sort_values('volume_impacte', ascending=False).head(10)
            
            fig_bar = px.bar(
                url_impact.reset_index(),
                x='volume_impacte',
                y='url',
                orientation='h',
                color_discrete_sequence=['#EF4444']
            )
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
        cols_display = ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume', 'trafic', 'priority_score']
        cols_display = [c for c in cols_display if c in df_pertes.columns]
        
        st.dataframe(
            df_pertes[cols_display].head(500),
            use_container_width=True,
            height=600
        )
        
        # Export
        csv_pertes = df_pertes[cols_display].to_csv(index=False, sep=';').encode('utf-8')
        st.download_button(
            "📥 Exporter les pertes critiques (CSV)",
            csv_pertes,
            "pertes_critiques.csv",
            "text/csv"
        )
    
    # ==========================================================================
    # TAB 3 : PAR URL
    # ==========================================================================
    
    with tab3:
        st.header("📁 Analyse par URL")
        
        if 'url' in df_filtered.columns:
            # Agrégation par URL
            url_stats = df_filtered.groupby('url').agg({
                'mot_cle': 'count',
                'diff_pos': ['sum', 'mean'],
                'volume': 'sum',
                'trafic': 'sum',
                'priority_score': 'sum'
            }).reset_index()
            
            url_stats.columns = ['url', 'total_kw', 'diff_total', 'diff_moyen', 'volume_total', 'trafic_total', 'score_priorite']
            
            # KW en perte par URL
            pertes_par_url = df_filtered[df_filtered['diff_pos'] < 0].groupby('url').size().reset_index(name='kw_en_perte')
            url_stats = url_stats.merge(pertes_par_url, on='url', how='left')
            url_stats['kw_en_perte'] = url_stats['kw_en_perte'].fillna(0).astype(int)
            
            # Score de santé
            url_stats['sante_pct'] = ((url_stats['total_kw'] - url_stats['kw_en_perte']) / url_stats['total_kw'] * 100).round(1)
            
            url_stats = url_stats.sort_values('score_priorite', ascending=False)
            
            st.info(f"**{len(url_stats):,}** URLs analysées")
            
            st.dataframe(
                url_stats.head(200),
                use_container_width=True,
                height=500
            )
            
            # Détail d'une URL
            st.subheader("🔍 Détail d'une URL")
            url_selectionnee = st.selectbox("Sélectionner une URL", url_stats['url'].head(100).tolist())
            
            if url_selectionnee:
                df_url = df_filtered[df_filtered['url'] == url_selectionnee]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("KW total", len(df_url))
                with col2:
                    st.metric("KW en perte", len(df_url[df_url['diff_pos'] < 0]))
                with col3:
                    st.metric("Volume total", f"{int(df_url['volume'].sum()):,}")
                
                cols_url = ['mot_cle', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume', 'statut']
                cols_url = [c for c in cols_url if c in df_url.columns]
                st.dataframe(df_url[cols_url], use_container_width=True)
        else:
            st.warning("Colonne 'url' non détectée dans le fichier")
    
    # ==========================================================================
    # TAB 4 : QUICK WINS
    # ==========================================================================
    
    with tab4:
        st.header("⚡ Quick wins — Opportunités de récupération")
        st.markdown("KW qui étaient **top 10**, ont chuté, mais ont un **potentiel de récupération**")
        
        if 'meilleure_pos' in df_filtered.columns and 'derniere_pos' in df_filtered.columns:
            df_quickwins = df_filtered[
                (df_filtered['meilleure_pos'] <= 10) &
                (df_filtered['derniere_pos'] > 10) &
                (df_filtered['volume'] >= 100)
            ].copy()
            
            df_quickwins = df_quickwins.sort_values('recovery_potential', ascending=False)
            
            st.success(f"**{len(df_quickwins):,}** opportunités de récupération identifiées")
            
            cols_qw = ['mot_cle', 'url', 'meilleure_pos', 'derniere_pos', 'diff_pos', 'volume', 'recovery_potential']
            cols_qw = [c for c in cols_qw if c in df_quickwins.columns]
            
            st.dataframe(
                df_quickwins[cols_qw].head(500),
                use_container_width=True,
                height=600
            )
            
            csv_qw = df_quickwins[cols_qw].to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                "📥 Exporter les quick wins (CSV)",
                csv_qw,
                "quick_wins.csv",
                "text/csv"
            )
        else:
            st.warning("Colonnes 'meilleure_pos' et/ou 'derniere_pos' non détectées")
    
    # ==========================================================================
    # TAB 5 : SORTIS
    # ==========================================================================
    
    with tab5:
        st.header("❌ Mots-clés sortis des SERPs")
        
        if 'statut' in df_filtered.columns:
            statut_str = df_filtered['statut'].astype(str).str.lower()
            df_sortis = df_filtered[statut_str.str.contains('sort|out', na=False)]
        elif 'derniere_pos' in df_filtered.columns:
            df_sortis = df_filtered[df_filtered['derniere_pos'] > 100]
        else:
            df_sortis = pd.DataFrame()
        
        if not df_sortis.empty:
            df_sortis = df_sortis.sort_values('volume', ascending=False)
            
            st.warning(f"**{len(df_sortis):,}** mots-clés ont disparu des SERPs")
            
            cols_sortis = ['mot_cle', 'url', 'ancienne_pos', 'volume', 'trafic']
            cols_sortis = [c for c in cols_sortis if c in df_sortis.columns]
            
            st.dataframe(
                df_sortis[cols_sortis].head(500),
                use_container_width=True,
                height=600
            )
            
            csv_sortis = df_sortis[cols_sortis].to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                "📥 Exporter les KW sortis (CSV)",
                csv_sortis,
                "kw_sortis.csv",
                "text/csv"
            )
        else:
            st.info("Aucun mot-clé sorti détecté")
    
    # ==========================================================================
    # TAB 6 : GAINS
    # ==========================================================================
    
    with tab6:
        st.header("🟢 Gains de position")
        
        df_gains = df_filtered[df_filtered['diff_pos'] > 0].sort_values('priority_score', ascending=False)
        
        st.success(f"**{len(df_gains):,}** mots-clés en progression")
        
        cols_gains = ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume', 'trafic']
        cols_gains = [c for c in cols_gains if c in df_gains.columns]
        
        st.dataframe(
            df_gains[cols_gains].head(500),
            use_container_width=True,
            height=600
        )
    
    # ==========================================================================
    # TAB 7 : RAPPORT
    # ==========================================================================
    
    with tab7:
        st.header("📝 Rapport pour l'équipe contenu")
        
        if st.button("🔄 Générer le rapport", type="primary"):
            with st.spinner("Génération du rapport..."):
                report = generate_report(df, df_filtered, kpis)
                st.session_state['report'] = report
        
        if 'report' in st.session_state:
            st.markdown(st.session_state['report'])
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Télécharger en Markdown",
                    st.session_state['report'],
                    "rapport_seo.md",
                    "text/markdown"
                )
            
            with col2:
                st.code(st.session_state['report'], language='markdown')

else:
    # État initial - pas de fichier chargé
    st.info("👆 Charge un fichier CSV Haloscan dans la sidebar pour commencer l'analyse")
    
    st.markdown("""
    ### 📋 Format attendu
    
    Le fichier doit contenir au minimum ces colonnes :
    - `mot-clé (mc)` ou `keyword` — le mot-clé tracké
    - `url` — l'URL positionnée
    - `diff_pos` — différentiel de position (négatif = perte)
    - `volume` — volume de recherche mensuel
    
    Colonnes optionnelles mais recommandées :
    - `dernière_pos` — position actuelle
    - `vieille_pos` / `ancienne_pos` — position de la période précédente
    - `meilleure_pos` — meilleure position historique
    - `statut` — perdu, gagné, stable, sorti...
    - `trafic` — estimation du trafic
    
    ### 🚀 Capacité
    
    L'outil peut traiter des fichiers jusqu'à **300 000+ lignes** sans problème.
    """)
