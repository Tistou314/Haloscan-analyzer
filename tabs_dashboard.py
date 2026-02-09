"""
tabs_dashboard.py — Onglet Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render(df_f, has_leads_merged, has_dual_haloscan, df, label_debut_p1, label_fin_p1, label_fin_p2,
           total, pertes, gains, stables, vol_perdu, vol_gagne):
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{total:,}")
    c2.metric("🔴 Pertes", f"{pertes:,}")
    c3.metric("🟢 Gains", f"{gains:,}")
    c4.metric("⚪ Stables", f"{stables:,}")
    st.divider()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📉 Volume perdu", f"{vol_perdu:,}")
    c2.metric("📈 Volume gagné", f"{vol_gagne:,}")
    
    if has_leads_merged:
        df_pertes_dash = df_f[df_f['diff_pos'] < 0]
        df_urls_perte_unique = df_pertes_dash.drop_duplicates(subset=['url']) if 'url' in df_pertes_dash.columns else df_pertes_dash
        leads_urls_perte = df_urls_perte_unique['leads_total'].fillna(0).sum()
        c3.metric("⚠️ Leads sur URLs en perte", f"{int(leads_urls_perte):,}")
        leads_evol = df_urls_perte_unique['leads_evolution'].fillna(0).sum()
        delta_color = "inverse" if leads_evol < 0 else "normal"
        c4.metric("📊 Évol. leads (période)", f"{int(leads_evol):+,}", delta_color=delta_color)
    
    # Multi-périodes
    if has_dual_haloscan and 'tendance_multi' in df_f.columns:
        st.divider()
        st.subheader(f"📈 Analyse multi-périodes ({label_debut_p1} → {label_fin_p1} → {label_fin_p2})")
        tendances = df_f['tendance_multi'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📉📉 Chute continue", f"{tendances.get('📉📉 Chute continue', 0):,}", help="Perte P1 ET perte P2")
        col2.metric("📈📉 Rebond puis rechute", f"{tendances.get('📈📉 Rebond puis rechute', 0):,}", help="Gain P1 puis perte P2")
        col3.metric("📉📈 Récupération", f"{tendances.get('📉📈 Récupération', 0):,}", help="Perte P1 puis gain P2")
        col4.metric("📈📈 Hausse continue", f"{tendances.get('📈📈 Hausse continue', 0):,}", help="Gain P1 ET gain P2")
        
        df_chute_continue = df_f[df_f['tendance_multi'] == '📉📉 Chute continue'].copy()
        if len(df_chute_continue) > 0:
            st.error(f"🚨 **{len(df_chute_continue):,}** mots-clés en CHUTE CONTINUE — Problème structurel à traiter !")
            cols_multi = [c for c in ['mot_cle', 'url', 'pos_debut_p1', 'pos_fin_p1', 'diff_p1', 'pos_fin_p2', 'diff_p2', 'diff_pos', 'volume'] if c in df_chute_continue.columns]
            df_chute_display = df_chute_continue[cols_multi].head(50).copy()
            rename_map = {
                'pos_debut_p1': f'Pos {label_debut_p1}', 'pos_fin_p1': f'Pos {label_fin_p1}',
                'diff_p1': 'Δ P1', 'pos_fin_p2': f'Pos {label_fin_p2}',
                'diff_p2': 'Δ P2', 'diff_pos': 'Δ TOTAL', 'volume': 'Volume'
            }
            df_chute_display = df_chute_display.rename(columns=rename_map)
            st.dataframe(df_chute_display.sort_values('Δ TOTAL', ascending=True), use_container_width=True, height=300)
    
    # Double peine
    if has_leads_merged and 'double_peine' in df_f.columns:
        df_double_peine = df_f[df_f['double_peine'] == True]
        if len(df_double_peine) > 0:
            st.divider()
            st.subheader("🚨 ALERTE : URLs en DOUBLE PEINE (perte SEO + perte leads)")
            st.error(f"**{df_double_peine['url'].nunique()}** URLs perdent à la fois des positions ET des leads !")
            p_avant = df.attrs.get('periode_avant_label', 'AVANT')
            p_apres = df.attrs.get('periode_apres_label', 'APRÈS')
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
            if 'Évol. Leads' in df_dp_urls.columns:
                df_dp_urls = df_dp_urls.sort_values('Évol. Leads', ascending=True)
            elif 'Diff total' in df_dp_urls.columns:
                df_dp_urls = df_dp_urls.sort_values('Diff total', ascending=True)
            st.dataframe(df_dp_urls.head(20), use_container_width=True, hide_index=True)
    
    # Charts
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=[pertes, gains, stables], names=['Pertes', 'Gains', 'Stables'],
                     color_discrete_sequence=['#EF4444', '#22C55E', '#6B7280'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df_f, x='diff_pos', nbins=50)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top URLs impactées
    try:
        df_pertes_temp = df_f[df_f['diff_pos'] < 0]
        if has_leads_merged and len(df_pertes_temp) > 0 and 'leads_total' in df_pertes_temp.columns:
            st.subheader("🎯 URLs critiques : Pertes SEO + Impact Business")
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
            if 'leads_evolution' in df_perte_urls.columns:
                df_perte_urls = df_perte_urls.sort_values('leads_evolution', ascending=True).head(15)
            else:
                df_perte_urls = df_perte_urls.sort_values('kw_perdus', ascending=False).head(15)
            st.dataframe(df_perte_urls, use_container_width=True)
    except Exception as e:
        st.warning(f"Impossible d'afficher les URLs critiques: {e}")
