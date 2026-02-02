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
                kw_gain=('diff_pos', lambda x: (x > 0).sum()),
                diff_moyen=('diff_pos', 'mean'),
                volume=('volume', 'sum') if 'volume' in df_f.columns else ('diff_pos', 'count'),
                score=('priority_score', 'sum')
            ).reset_index()
            url_stats['sante_pct'] = ((url_stats['total_kw'] - url_stats['kw_perte']) / url_stats['total_kw'] * 100).round(1)
            url_stats = url_stats.sort_values('score', ascending=False)
            
            st.info(f"**{len(url_stats):,}** URLs analysées — Affichage complet")
            st.dataframe(url_stats, use_container_width=True, height=500)
            
            # Export complet
            csv_urls = url_stats.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button("📥 Exporter TOUTES les URLs (CSV)", csv_urls, "analyse_urls_complete.csv")
            
            st.divider()
            
            # Détail URL
            st.subheader("🔍 Détail d'une URL")
            url_sel = st.selectbox("Sélectionner une URL", url_stats['url'].tolist())
            if url_sel:
                df_url = df_f[df_f['url'] == url_sel]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total KW", len(df_url))
                c2.metric("En perte", len(df_url[df_url['diff_pos'] < 0]))
                c3.metric("En gain", len(df_url[df_url['diff_pos'] > 0]))
                if 'volume' in df_url.columns:
                    c4.metric("Volume total", f"{int(df_url['volume'].fillna(0).sum()):,}")
                
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
            df_pertes_rapport = df_f[df_f['diff_pos'] < 0].sort_values('priority_score', ascending=False)
            df_gains_rapport = df_f[df_f['diff_pos'] > 0].sort_values('priority_score', ascending=False)
            
            # URLs les plus impactées
            if 'url' in df_f.columns:
                urls_critiques = df_pertes_rapport.groupby('url').agg(
                    nb_kw_perdus=('diff_pos', 'count'),
                    volume_impacte=('volume', 'sum') if 'volume' in df_f.columns else ('diff_pos', 'count'),
                    diff_moyen=('diff_pos', 'mean'),
                    score_total=('priority_score', 'sum')
                ).reset_index().sort_values('score_total', ascending=False)
            
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

---

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

Ces URLs sont triées par score de priorité (volume × chute de position).
**L'équipe édito doit traiter ces pages dans l'ordre.**

| Priorité | URL | KW perdus | Volume impacté | Diff moyen | Score |
|----------|-----|-----------|----------------|------------|-------|
"""
            if 'url' in df_f.columns:
                for i, row in urls_critiques.iterrows():
                    prio = "🔴 URGENT" if row['score_total'] > urls_critiques['score_total'].quantile(0.9) else "🟠 MOYEN" if row['score_total'] > urls_critiques['score_total'].quantile(0.5) else "🟡 FAIBLE"
                    report += f"| {prio} | {row['url']} | {int(row['nb_kw_perdus'])} | {int(row.get('volume_impacte', 0)):,} | {row['diff_moyen']:.1f} | {int(row['score_total']):,} |\n"

            report += f"""

---

# 4. MOTS-CLÉS EN PERTE — LISTE COMPLÈTE ({len(df_pertes_rapport):,} KW)

**Triés par score de priorité (volume × perte de position)**

| Mot-clé | URL | Ancienne pos | Nouvelle pos | Diff | Volume | Score priorité |
|---------|-----|--------------|--------------|------|--------|----------------|
"""
            for _, row in df_pertes_rapport.iterrows():
                mc = str(row.get('mot_cle', 'N/A'))[:50]
                url = str(row.get('url', 'N/A'))[:60]
                anc = int(row.get('ancienne_pos', 0) or 0)
                dern = int(row.get('derniere_pos', 0) or 0)
                diff = int(row.get('diff_pos', 0) or 0)
                vol = int(row.get('volume', 0) or 0)
                score = int(row.get('priority_score', 0) or 0)
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

## Actions immédiates (cette semaine)
1. **Auditer les 10 premières URLs critiques** — Vérifier : contenu à jour ? maillage interne ? balises optimisées ?
2. **Identifier les KW à fort volume perdus** — Filtrer les pertes avec volume > 1000
3. **Vérifier la concurrence** — Les concurrents ont-ils amélioré leur contenu ?

## Actions court terme (ce mois)
1. **Mettre à jour les contenus des pages critiques** — Enrichir, actualiser, ajouter des sections
2. **Renforcer le maillage interne** vers les pages en perte
3. **Créer du contenu de support** pour les thématiques en baisse

## Actions moyen terme (ce trimestre)
1. **Audit technique** — Vérifier Core Web Vitals des pages impactées
2. **Analyse des backlinks** — Les pages ont-elles perdu des liens ?
3. **Stratégie de contenu** — Planifier les mises à jour récurrentes

---

# 7. MÉTRIQUES DE SUIVI

Refaire cette analyse dans 1 mois pour mesurer :
- [ ] Réduction du nombre de KW en perte
- [ ] Récupération des positions sur les KW prioritaires
- [ ] Amélioration du volume de recherche capté

---

_Rapport généré automatiquement — Haloscan SEO Diff Analyzer_
_Données : {len(df):,} mots-clés analysés_
"""
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
                df_export = df_f[df_f['diff_pos'] < 0].sort_values('priority_score', ascending=False)
                cols_export = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume', 'priority_score'] if c in df_export.columns]
                csv_export = df_export[cols_export].to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "📥 Télécharger les données (CSV)",
                    csv_export,
                    "pertes_completes.csv",
                    "text/csv"
                )

else:
    st.info("👆 Charge un fichier CSV pour commencer")
