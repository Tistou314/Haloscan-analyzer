"""
tabs_cannibalisation.py — Onglet détection cannibalisation interne
"""

import streamlit as st
import pandas as pd


def render(df):
    st.header("🔄 Détection de cannibalisation interne")
    st.info("**Objectif** : Identifier les KW où une URL perd des positions tandis qu'une autre URL du site en gagne. Avant de réoptimiser une page en perte, vérifiez qu'une autre page n'a pas pris le relais !")
    
    if 'mot_cle' not in df.columns or 'url' not in df.columns:
        st.warning("Colonnes 'mot_cle' et 'url' nécessaires pour l'analyse de cannibalisation")
        return
    
    with st.spinner("Analyse des cannibalisations en cours..."):
        df_canni = df[['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'volume']].copy()
        df_pertes_canni = df_canni[df_canni['diff_pos'] < 0].copy()
        df_gains_canni = df_canni[df_canni['diff_pos'] > 0].copy()
        
        kw_en_perte = set(df_pertes_canni['mot_cle'].unique())
        kw_en_gain = set(df_gains_canni['mot_cle'].unique())
        kw_cannibalisation = kw_en_perte & kw_en_gain
        
        st.metric("🔄 KW avec cannibalisation potentielle", f"{len(kw_cannibalisation):,}")
        
        if len(kw_cannibalisation) == 0:
            st.success("✅ Aucune cannibalisation détectée ! Chaque KW n'a qu'une seule URL qui bouge.")
            return
        
        resultats_canni = []
        for kw in kw_cannibalisation:
            urls_perte = df_pertes_canni[df_pertes_canni['mot_cle'] == kw].sort_values('diff_pos', ascending=True)
            urls_gain = df_gains_canni[df_gains_canni['mot_cle'] == kw].sort_values('diff_pos', ascending=False)
            if len(urls_perte) > 0 and len(urls_gain) > 0:
                perte = urls_perte.iloc[0]
                gain = urls_gain.iloc[0]
                vol = max(perte.get('volume', 0) or 0, gain.get('volume', 0) or 0)
                resultats_canni.append({
                    'mot_cle': kw, 'volume': vol,
                    'url_perte': perte['url'],
                    'ancienne_pos_perte': perte.get('ancienne_pos', 0),
                    'nouvelle_pos_perte': perte.get('derniere_pos', 0),
                    'diff_perte': perte.get('diff_pos', 0),
                    'url_gain': gain['url'],
                    'ancienne_pos_gain': gain.get('ancienne_pos', 0),
                    'nouvelle_pos_gain': gain.get('derniere_pos', 0),
                    'diff_gain': gain.get('diff_pos', 0),
                })
        
        if not resultats_canni:
            return
        
        df_resultats = pd.DataFrame(resultats_canni).sort_values('volume', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            vol_min_canni = st.number_input("Volume minimum", min_value=0, value=0, step=50, key="vol_canni")
        with col2:
            diff_min_canni = st.number_input("Perte minimum (positions)", min_value=0, value=0, step=1, key="diff_canni")
        
        df_resultats_f = df_resultats[
            (df_resultats['volume'].fillna(0) >= vol_min_canni) &
            (df_resultats['diff_perte'].fillna(0).abs() >= diff_min_canni)
        ]
        
        st.success(f"**{len(df_resultats_f):,}** cas de cannibalisation détectés (sur {len(df_resultats):,} total)")
        st.subheader("⚠️ KW à risque — Vérifier avant réoptimisation")
        
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
        
        cols_display = ['mot_cle', 'Volume', '📉 URL en perte', 'Était pos', '→ Maintenant', 'Diff',
                        '📈 URL en hausse', 'Était pos ', '→ Maintenant ', 'Diff ']
        st.dataframe(df_display[cols_display].head(100), use_container_width=True, height=500)
        
        st.warning("**⚠️ ATTENTION avant de réoptimiser une URL en perte :**\n"
                    "1. Vérifiez si l'URL en hausse répond mieux à l'intention de recherche\n"
                    "2. Si oui → renforcez l'URL en hausse plutôt que l'ancienne\n"
                    "3. Si non → vérifiez le maillage interne pour éviter la cannibalisation\n"
                    "4. Envisagez une redirection 301 si l'ancienne URL n'a plus de raison d'être")
        
        csv_canni = df_resultats_f.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("📥 Exporter les cannibalisations (CSV)", csv_canni, "cannibalisations.csv")
