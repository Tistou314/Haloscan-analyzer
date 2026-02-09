"""
tabs_pertes_gains.py — Onglets Pertes critiques et Gains
"""

import streamlit as st
import pandas as pd


def render_pertes(df_f):
    st.header("🔴 Pertes critiques")
    df_pertes = df_f[df_f['diff_pos'] < 0].sort_values('diff_pos', ascending=True)
    st.info(f"**{len(df_pertes):,}** mots-clés en perte")
    cols = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'tendance_seo', 'volume'] if c in df_pertes.columns]
    st.dataframe(df_pertes[cols], use_container_width=True, height=600)
    csv = df_pertes[cols].to_csv(index=False, sep=';').encode('utf-8')
    st.download_button("📥 Export CSV", csv, "pertes.csv")


def render_gains(df_f):
    st.header("🟢 Gains")
    df_gains = df_f[df_f['diff_pos'] > 0].sort_values('diff_pos', ascending=False)
    st.success(f"**{len(df_gains):,}** mots-clés en gain")
    cols = [c for c in ['mot_cle', 'url', 'diff_pos', 'tendance_seo', 'volume', 'derniere_pos', 'ancienne_pos'] if c in df_gains.columns]
    st.dataframe(df_gains[cols], use_container_width=True, height=600)
    csv_gains = df_gains[cols].to_csv(index=False, sep=';').encode('utf-8')
    st.download_button("📥 Exporter TOUS les gains (CSV)", csv_gains, "gains_complet.csv")
