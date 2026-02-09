"""
tabs_urls.py — Onglet Analyse par URL
"""

import streamlit as st
import pandas as pd
from data_loader import tendance_seo_url


def render(df_f, df, has_leads_merged):
    st.header("📁 Analyse par URL")
    if 'url' not in df_f.columns:
        st.warning("Colonne 'url' non trouvée")
        return
    
    try:
        agg_funcs = {
            'diff_pos': ['count', lambda x: (x < 0).sum(), lambda x: (x > 0).sum(), 'sum'],
        }
        if 'volume' in df_f.columns:
            agg_funcs['volume'] = 'sum'
        if has_leads_merged:
            for col in ['leads_total', 'leads_avant', 'leads_apres', 'leads_evolution', 'tendance_leads']:
                if col in df_f.columns:
                    agg_funcs[col] = 'first'
        
        url_stats = df_f.groupby('url').agg(agg_funcs).reset_index()
        
        # Aplatir les colonnes multi-index
        new_cols = ['url']
        for col in url_stats.columns[1:]:
            if isinstance(col, tuple):
                new_cols.append(f"{col[0]}_{col[1]}" if col[1] != 'sum' and col[1] != 'first' else col[0])
            else:
                new_cols.append(col)
        url_stats.columns = new_cols
        
        rename_dict = {
            'diff_pos_count': 'total_kw',
            'diff_pos_<lambda_0>': 'kw_perte',
            'diff_pos_<lambda_1>': 'kw_gain',
            'diff_pos_sum': 'diff_total'
        }
        url_stats = url_stats.rename(columns=rename_dict)
        
        if 'diff_total' in url_stats.columns:
            url_stats['📊 SEO'] = url_stats['diff_total'].apply(tendance_seo_url)
        if 'tendance_leads' in url_stats.columns:
            url_stats = url_stats.rename(columns={'tendance_leads': '📊 LEADS'})
        
        if 'leads_evolution' in url_stats.columns:
            url_stats = url_stats.sort_values('leads_evolution', ascending=True)
        elif 'kw_perte' in url_stats.columns:
            url_stats = url_stats.sort_values('kw_perte', ascending=False)
        else:
            url_stats = url_stats.sort_values('total_kw', ascending=False)
        
        st.info(f"**{len(url_stats):,}** URLs analysées")
        st.dataframe(url_stats, use_container_width=True, height=500)
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
        
        if has_leads_merged and 'leads_total' in df_url.columns:
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
        csv_url_detail = df_url[cols].to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("📥 Exporter les KW de cette URL", csv_url_detail, "detail_url.csv")
