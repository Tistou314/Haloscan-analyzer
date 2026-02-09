"""
tabs_gsc.py — Onglet Search Console
"""

import streamlit as st
import pandas as pd
from data_loader import normalize_url


def render(df_f, has_gsc, gsc_pages_df, gsc_queries_df):
    st.header("🔍 Données Search Console")
    
    if not has_gsc:
        st.warning("👆 Uploadez un fichier ZIP Search Console pour voir les données de trafic réel")
        st.info("**Comment obtenir l'export :**\n"
                "1. Allez sur [Google Search Console](https://search.google.com/search-console)\n"
                "2. Sélectionnez votre propriété\n"
                "3. Allez dans \"Performances\" > \"Résultats de recherche\"\n"
                "4. Cliquez sur \"Exporter\" > \"Télécharger au format ZIP\"")
        return
    
    st.info("**Données réelles Google** : Clics, impressions, CTR et positions moyennes des 12 derniers mois")
    gsc_tab1, gsc_tab2, gsc_tab3 = st.tabs(["🚨 URLs en danger", "💡 Opportunités CTR", "📊 Vue globale"])
    
    # URLs en danger
    with gsc_tab1:
        st.subheader("🚨 URLs en danger : Perte SEO + Trafic réel")
        st.caption("URLs qui perdent des positions Haloscan ET qui ont beaucoup de clics GSC → Perte de trafic réelle")
        if gsc_pages_df is not None and 'url' in df_f.columns:
            df_haloscan_urls = df_f.groupby('url').agg({'diff_pos': ['mean', 'sum', 'count'], 'volume': 'sum'}).reset_index()
            df_haloscan_urls.columns = ['url', 'diff_pos_mean', 'diff_pos_sum', 'nb_kw', 'volume_total']
            df_haloscan_urls['url_normalized'] = df_haloscan_urls['url'].apply(normalize_url)
            df_danger = df_haloscan_urls.merge(
                gsc_pages_df[['url_normalized', 'Clics', 'Impressions', 'CTR', 'Position']],
                on='url_normalized', how='inner')
            df_danger = df_danger[df_danger['diff_pos_mean'] < 0].copy()
            df_danger['score_danger'] = df_danger['Clics'] * df_danger['diff_pos_mean'].abs()
            df_danger = df_danger.sort_values('score_danger', ascending=False)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("URLs en danger", f"{len(df_danger):,}")
            col2.metric("Clics totaux menacés", f"{int(df_danger['Clics'].sum()):,}")
            col3.metric("Impressions menacées", f"{int(df_danger['Impressions'].sum()):,}")
            
            if len(df_danger) > 0:
                df_danger_display = df_danger[['url', 'Clics', 'Impressions', 'CTR', 'Position', 'diff_pos_mean', 'nb_kw', 'volume_total']].copy()
                df_danger_display = df_danger_display.rename(columns={
                    'Clics': '🖱️ Clics GSC', 'Impressions': '👁️ Impressions', 'CTR': '📊 CTR %',
                    'Position': '📍 Pos GSC', 'diff_pos_mean': '📉 Δ Haloscan', 'nb_kw': 'Nb KW', 'volume_total': 'Vol. total'})
                df_danger_display['📉 Δ Haloscan'] = df_danger_display['📉 Δ Haloscan'].round(1)
                st.dataframe(df_danger_display.head(50), use_container_width=True, height=400)
                st.error("**🚨 ACTION REQUISE** : Ces URLs perdent des positions ET génèrent du trafic réel.\n→ Prioriser leur réoptimisation")
                csv_danger = df_danger.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button("📥 Exporter les URLs en danger (CSV)", csv_danger, "urls_danger_gsc.csv")
            else:
                st.success("✅ Aucune URL en danger détectée !")
        else:
            st.warning("Données Pages GSC ou URLs Haloscan non disponibles")
    
    # Opportunités CTR
    with gsc_tab2:
        st.subheader("💡 Opportunités CTR : Bien positionné mais peu cliqué")
        st.caption("URLs en Top 10 avec CTR < 5% → Title et meta description à optimiser")
        if gsc_pages_df is not None:
            df_ctr_opps = gsc_pages_df[
                (gsc_pages_df['Position'] <= 10) & (gsc_pages_df['CTR'] < 5) & (gsc_pages_df['Impressions'] >= 100)
            ].copy()
            df_ctr_opps['ctr_potentiel'] = 5.0
            df_ctr_opps['clics_potentiels'] = (df_ctr_opps['Impressions'] * df_ctr_opps['ctr_potentiel'] / 100).astype(int)
            df_ctr_opps['clics_supplementaires'] = df_ctr_opps['clics_potentiels'] - df_ctr_opps['Clics']
            df_ctr_opps = df_ctr_opps.sort_values('clics_supplementaires', ascending=False)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("URLs à optimiser", f"{len(df_ctr_opps):,}")
            col2.metric("Clics actuels", f"{int(df_ctr_opps['Clics'].sum()):,}")
            col3.metric("Potentiel clics en +", f"+{int(df_ctr_opps['clics_supplementaires'].sum()):,}")
            
            if len(df_ctr_opps) > 0:
                df_ctr_display = df_ctr_opps[['url', 'Position', 'CTR', 'Clics', 'Impressions', 'clics_supplementaires']].copy()
                df_ctr_display = df_ctr_display.rename(columns={
                    'Position': '📍 Position', 'CTR': '📊 CTR actuel %', 'Clics': '🖱️ Clics',
                    'Impressions': '👁️ Impressions', 'clics_supplementaires': '🎯 Potentiel clics +'})
                df_ctr_display['📍 Position'] = df_ctr_display['📍 Position'].round(1)
                st.dataframe(df_ctr_display.head(50), use_container_width=True, height=400)
                st.warning("**💡 OPTIMISATION RECOMMANDÉE** :\n"
                           "- Revoir les **titles** pour les rendre plus attractifs\n"
                           "- Améliorer les **meta descriptions** avec des CTA\n"
                           "- Ajouter des **données structurées** pour enrichir les snippets")
                csv_ctr = df_ctr_opps.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button("📥 Exporter les opportunités CTR (CSV)", csv_ctr, "opportunites_ctr.csv")
            else:
                st.success("✅ Toutes les URLs en Top 10 ont un bon CTR !")
        else:
            st.warning("Données Pages GSC non disponibles")
    
    # Vue globale
    with gsc_tab3:
        st.subheader("📊 Vue globale Search Console")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔍 Top Requêtes (par clics)**")
            if gsc_queries_df is not None:
                st.dataframe(gsc_queries_df[['query', 'Clics', 'Impressions', 'CTR', 'Position']].head(20), use_container_width=True, height=400)
            else:
                st.info("Données requêtes non disponibles")
        with col2:
            st.markdown("**📄 Top Pages (par clics)**")
            if gsc_pages_df is not None:
                df_pages_display = gsc_pages_df[['url', 'Clics', 'Impressions', 'CTR', 'Position']].head(20).copy()
                df_pages_display['url'] = df_pages_display['url'].str.replace('https://www.ootravaux.fr', '...')
                st.dataframe(df_pages_display, use_container_width=True, height=400)
            else:
                st.info("Données pages non disponibles")
        st.divider()
        st.markdown("**📈 Statistiques globales GSC**")
        col1, col2, col3, col4 = st.columns(4)
        if gsc_pages_df is not None:
            col1.metric("Total Clics", f"{int(gsc_pages_df['Clics'].sum()):,}")
            col2.metric("Total Impressions", f"{int(gsc_pages_df['Impressions'].sum()):,}")
            col3.metric("CTR moyen", f"{gsc_pages_df['CTR'].mean():.2f}%")
            col4.metric("Position moyenne", f"{gsc_pages_df['Position'].mean():.1f}")
