"""
tabs_rapport.py — Onglet Rapport complet + Analyse IA
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
from data_loader import normalize_url


def render(df_f, df, has_leads_merged, has_dual_haloscan, has_gsc, gsc_pages_df,
           label_debut_p1, label_fin_p1, label_fin_p2,
           periode_avant, periode_apres, anthropic_api_key,
           total, pertes, gains, stables, vol_perdu, vol_gagne):
    
    st.header("📝 Rapport complet pour l'équipe édito")
    
    if st.button("🔄 Générer le rapport complet", type="primary"):
        df_pertes_rapport = df_f[df_f['diff_pos'] < 0].sort_values('diff_pos', ascending=True)
        df_gains_rapport = df_f[df_f['diff_pos'] > 0].sort_values('diff_pos', ascending=False)
        
        # URLs critiques
        urls_critiques = pd.DataFrame()
        if 'url' in df_f.columns and len(df_pertes_rapport) > 0:
            agg_url = {'diff_pos': 'count'}
            if 'volume' in df_pertes_rapport.columns:
                agg_url['volume'] = 'sum'
            if has_leads_merged:
                for col in ['leads_total', 'leads_avant', 'leads_apres', 'leads_evolution', 'tendance_leads']:
                    if col in df_pertes_rapport.columns:
                        agg_url[col] = 'first'
            try:
                urls_critiques = df_pertes_rapport.groupby('url').agg(agg_url).reset_index()
                urls_critiques = urls_critiques.rename(columns={'diff_pos': 'nb_kw_perdus', 'volume': 'volume_impacte'})
                if 'leads_evolution' in urls_critiques.columns:
                    urls_critiques = urls_critiques.sort_values('leads_evolution', ascending=True)
                else:
                    urls_critiques = urls_critiques.sort_values('nb_kw_perdus', ascending=False)
            except Exception as e:
                st.warning(f"Erreur agrégation URLs: {e}")
        
        # Impact leads
        total_leads_perte = total_leads_avant_perte = total_leads_apres_perte = leads_evolution_total = 0
        if has_leads_merged:
            df_urls_perte_unique = df_pertes_rapport.drop_duplicates(subset=['url'])
            total_leads_perte = int(df_urls_perte_unique['leads_total'].fillna(0).sum())
            total_leads_avant_perte = int(df_urls_perte_unique['leads_avant'].fillna(0).sum())
            total_leads_apres_perte = int(df_urls_perte_unique['leads_apres'].fillna(0).sum())
            leads_evolution_total = int(df_urls_perte_unique['leads_evolution'].fillna(0).sum())
        
        periode_rapport = f"{label_debut_p1} → {label_fin_p2}" if has_dual_haloscan else "Période analysée"
        
        # --- Construction du rapport ---
        report = _build_report(
            df_f, df, df_pertes_rapport, df_gains_rapport, urls_critiques,
            has_leads_merged, has_dual_haloscan, has_gsc, gsc_pages_df,
            label_debut_p1, label_fin_p1, label_fin_p2,
            periode_avant, periode_apres, periode_rapport,
            total, pertes, gains, stables, vol_perdu, vol_gagne,
            total_leads_perte, total_leads_avant_perte, total_leads_apres_perte, leads_evolution_total
        )
        
        st.session_state['report'] = report
        st.success("✅ Rapport généré !")
    
    if 'report' in st.session_state:
        st.markdown(st.session_state['report'])
        st.divider()
        
        # Analyse IA
        st.subheader("🤖 Analyse IA et TODO")
        if anthropic_api_key:
            if st.button("🤖 Générer l'analyse IA", type="secondary"):
                _generate_ai_analysis(df_f, df, has_leads_merged, has_dual_haloscan, has_gsc, gsc_pages_df,
                                      anthropic_api_key, total, pertes, gains, stables, vol_perdu, vol_gagne)
            if 'ai_analysis' in st.session_state:
                st.divider()
                st.markdown("## 🤖 Analyse IA et TODO")
                st.markdown(st.session_state['ai_analysis'])
                rapport_complet = st.session_state['report'] + "\n\n---\n\n# 🤖 ANALYSE IA ET TODO\n\n" + st.session_state['ai_analysis']
                st.download_button("📥 Télécharger le rapport COMPLET avec IA (Markdown)", rapport_complet, "rapport_seo_complet_avec_ia.md", "text/markdown")
        else:
            st.info("👆 Entrez votre clé API Anthropic dans la sidebar pour activer l'analyse IA")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Télécharger le rapport (Markdown)", st.session_state['report'], "rapport_seo_complet.md", "text/markdown")
        with col2:
            df_export = df_f[df_f['diff_pos'] < 0].sort_values('diff_pos', ascending=True)
            cols_export = [c for c in ['mot_cle', 'url', 'ancienne_pos', 'derniere_pos', 'diff_pos', 'tendance_seo', 'volume',
                                       'leads_total', 'leads_avant', 'leads_apres', 'leads_evolution', 'tendance_leads'] if c in df_export.columns]
            csv_export = df_export[cols_export].to_csv(index=False, sep=';').encode('utf-8')
            st.download_button("📥 Télécharger les données (CSV)", csv_export, "pertes_completes.csv", "text/csv")


def _safe_int(val, default=0):
    """Convertit en int en gérant NaN"""
    if pd.isna(val):
        return default
    return int(val)


def _build_report(df_f, df, df_pertes_rapport, df_gains_rapport, urls_critiques,
                  has_leads_merged, has_dual_haloscan, has_gsc, gsc_pages_df,
                  label_debut_p1, label_fin_p1, label_fin_p2,
                  periode_avant, periode_apres, periode_rapport,
                  total, pertes, gains, stables, vol_perdu, vol_gagne,
                  total_leads_perte, total_leads_avant_perte, total_leads_apres_perte, leads_evolution_total):
    
    report = f"""# 📊 RAPPORT D'ANALYSE SEO COMPLET
## Période : {periode_rapport}
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
    
    # Leads
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
    
    # Multi-périodes
    if has_dual_haloscan and 'tendance_multi' in df_f.columns:
        tendances = df_f['tendance_multi'].value_counts()
        report += f"""---

## 📈 ANALYSE MULTI-PÉRIODES ({label_debut_p1} → {label_fin_p1} → {label_fin_p2})

| Tendance | Nombre de KW | Signification |
|----------|--------------|---------------|
| 📉📉 **Chute continue** | {tendances.get('📉📉 Chute continue', 0):,} | Perte sur P1 ET P2 — **Problème structurel** |
| 📈📉 Rebond puis rechute | {tendances.get('📈📉 Rebond puis rechute', 0):,} | Gain sur P1 puis perte sur P2 |
| 📉📈 Récupération | {tendances.get('📉📈 Récupération', 0):,} | Perte sur P1 puis gain sur P2 |
| 📈📈 Hausse continue | {tendances.get('📈📈 Hausse continue', 0):,} | Gain sur P1 ET P2 |

"""
        df_chute = df_f[df_f['tendance_multi'] == '📉📉 Chute continue'].copy().sort_values('diff_pos', ascending=True)
        if len(df_chute) > 0:
            report += f"### 🚨 TOP 100 KW en CHUTE CONTINUE\n\n| Mot-clé | URL | Pos {label_debut_p1} | Pos {label_fin_p1} | Δ P1 | Pos {label_fin_p2} | Δ P2 | Δ TOTAL | Volume |\n|---------|-----|-----|-----|-----|-----|-----|---------|--------|\n"
            for _, row in df_chute.head(100).iterrows():
                report += f"| {str(row.get('mot_cle', 'N/A'))[:40]} | {row.get('url', 'N/A')} | {_safe_int(row.get('pos_debut_p1'))} | {_safe_int(row.get('pos_fin_p1'))} | {_safe_int(row.get('diff_p1'))} | {_safe_int(row.get('pos_fin_p2'))} | {_safe_int(row.get('diff_p2'))} | {_safe_int(row.get('diff_pos'))} | {_safe_int(row.get('volume')):,} |\n"
            if len(df_chute) > 100:
                report += f"\n_+ {len(df_chute) - 100:,} autres KW en chute continue_\n"
    
    # GSC
    if has_gsc and gsc_pages_df is not None and 'url' in df_f.columns:
        df_h = df_f.groupby('url').agg({'diff_pos': 'mean', 'volume': 'sum'}).reset_index()
        df_h['url_normalized'] = df_h['url'].apply(normalize_url)
        df_danger_rpt = df_h.merge(gsc_pages_df[['url_normalized', 'Clics', 'Impressions', 'CTR', 'Position']], on='url_normalized', how='inner')
        df_danger_rpt = df_danger_rpt[df_danger_rpt['diff_pos'] < 0].sort_values('Clics', ascending=False)
        df_ctr_rpt = gsc_pages_df[(gsc_pages_df['Position'] <= 10) & (gsc_pages_df['CTR'] < 5) & (gsc_pages_df['Impressions'] >= 100)].copy()
        df_ctr_rpt['clics_potentiels'] = (df_ctr_rpt['Impressions'] * 5 / 100 - df_ctr_rpt['Clics']).astype(int)
        df_ctr_rpt = df_ctr_rpt.sort_values('clics_potentiels', ascending=False)
        report += f"""---

## 🔍 DONNÉES SEARCH CONSOLE

| Métrique | Valeur |
|----------|--------|
| **Total clics** | {int(gsc_pages_df['Clics'].sum()):,} |
| **URLs en danger** | {len(df_danger_rpt):,} |
| **Opportunités CTR** | {len(df_ctr_rpt):,} |
| **Clics potentiels à gagner** | +{int(df_ctr_rpt['clics_potentiels'].sum()):,} |

"""
        if len(df_danger_rpt) > 0:
            report += "### 🚨 TOP 20 URLs EN DANGER\n\n| URL | Clics GSC | Δ Haloscan | Position GSC | CTR |\n|-----|-----------|------------|--------------|-----|\n"
            for _, row in df_danger_rpt.head(20).iterrows():
                report += f"| {row.get('url', 'N/A')} | {int(row.get('Clics', 0)):,} | {round(row.get('diff_pos', 0), 1)} | {round(row.get('Position', 0), 1)} | {round(row.get('CTR', 0), 2)}% |\n"
        if len(df_ctr_rpt) > 0:
            report += "\n### 💡 TOP 20 OPPORTUNITÉS CTR\n\n| URL | Position | CTR actuel | Impressions | Potentiel clics + |\n|-----|----------|------------|-------------|-------------------|\n"
            for _, row in df_ctr_rpt.head(20).iterrows():
                report += f"| {row.get('url', 'N/A')} | {round(row.get('Position', 0), 1)} | {round(row.get('CTR', 0), 2)}% | {int(row.get('Impressions', 0)):,} | +{int(row.get('clics_potentiels', 0)):,} |\n"
    
    # Diagnostic
    report += "\n---\n\n# 2. DIAGNOSTIC\n\n"
    if gains == 0:
        report += f"⚠️ **SITUATION CRITIQUE** : Aucun gain. {pertes:,} mots-clés en perte. Action : **Audit urgent**\n\n"
    elif pertes > gains:
        report += f"⚠️ **SITUATION PRÉOCCUPANTE** : Ratio pertes/gains = {pertes/gains:.1f}x. Action : **Audit urgent**\n\n"
    elif pertes == 0:
        report += f"✅ **SITUATION EXCELLENTE** : Aucune perte ! {gains:,} mots-clés en gain.\n\n"
    else:
        report += f"✅ **SITUATION POSITIVE** : Ratio gains/pertes = {gains/pertes:.1f}x.\n\n"
    
    # Pages à traiter
    if len(urls_critiques) > 0:
        report += f"---\n\n# 3. PAGES À TRAITER ({len(urls_critiques):,} URLs)\n\n"
        if has_leads_merged:
            p_av = df.attrs.get('periode_avant_label', 'AVANT')
            p_ap = df.attrs.get('periode_apres_label', 'APRÈS')
            report += f"| Priorité | URL | KW perdus | Volume | Leads {p_av} | Leads {p_ap} | 📊 TENDANCE |\n|----------|-----|-----------|--------|------|------|-------------|\n"
            for _, row in urls_critiques.iterrows():
                le = _safe_int(row.get('leads_evolution', 0))
                t = row.get('tendance_leads', '➡️ N/A')
                prio = "🔴 CRITIQUE" if le < -100 else "🟠 URGENT" if le < -20 else "🟡 MOYEN" if le < 0 else "⚪ STABLE"
                report += f"| {prio} | {row['url']} | {_safe_int(row.get('nb_kw_perdus'))} | {_safe_int(row.get('volume_impacte')):,} | {_safe_int(row.get('leads_avant')):,} | {_safe_int(row.get('leads_apres')):,} | {t} |\n"
        else:
            report += "| Priorité | URL | KW perdus | Volume |\n|----------|-----|-----------|--------|\n"
            for _, row in urls_critiques.iterrows():
                nk = row['nb_kw_perdus']
                prio = "🔴 URGENT" if nk > 50 else "🟠 MOYEN" if nk > 10 else "🟡 FAIBLE"
                report += f"| {prio} | {row['url']} | {int(nk)} | {_safe_int(row.get('volume_impacte')):,} |\n"
    
    # Pertes critiques
    df_pc = df_pertes_rapport[df_pertes_rapport['diff_pos'] <= -5].copy()
    if 'volume' in df_pc.columns and 'url' in df_pc.columns and len(df_pc) > 0:
        df_br = df_pc[df_pc['ancienne_pos'] <= 10].copy()
        if len(df_br) > 0:
            idx = df_br.groupby('url')['volume'].idxmax()
            df_pu = df_br.loc[idx].copy()
        else:
            idx = df_pc.groupby('url')['volume'].idxmax()
            df_pu = df_pc.loc[idx].copy()
        kw_count = df_pc.groupby('url').size().rename('nb_kw_perdus')
        df_pu = df_pu.merge(kw_count, on='url', how='left').sort_values('diff_pos', ascending=True)
        df_pl = df_pu.head(500)
        report += f"\n---\n\n# 4. PERTES CRITIQUES — TOP {len(df_pl):,} URLs\n\n| KW Principal | URL | Anc. pos | Nouv. pos | Diff | Volume | Nb KW |\n|---|---|---|---|---|---|---|\n"
        for _, r in df_pl.iterrows():
            report += f"| {str(r.get('mot_cle', 'N/A'))[:50]} | {r.get('url', 'N/A')} | {_safe_int(r.get('ancienne_pos'))} | {_safe_int(r.get('derniere_pos'))} | {_safe_int(r.get('diff_pos'))} | {_safe_int(r.get('volume')):,} | {_safe_int(r.get('nb_kw_perdus'), 1)} |\n"
        if len(df_pu) > 500:
            report += f"\n_+ {len(df_pu) - 500:,} autres URLs_\n"
    
    # Gains significatifs
    df_gs = df_gains_rapport[df_gains_rapport['diff_pos'] >= 5].copy()
    if 'volume' in df_gs.columns and 'url' in df_gs.columns and len(df_gs) > 0:
        df_br = df_gs[df_gs['derniere_pos'] <= 10].copy()
        if len(df_br) > 0:
            idx = df_br.groupby('url')['volume'].idxmax()
            df_gu = df_br.loc[idx].copy()
        else:
            idx = df_gs.groupby('url')['volume'].idxmax()
            df_gu = df_gs.loc[idx].copy()
        kw_count = df_gs.groupby('url').size().rename('nb_kw_gains')
        df_gu = df_gu.merge(kw_count, on='url', how='left').sort_values('diff_pos', ascending=False)
        df_gl = df_gu.head(500)
        report += f"\n---\n\n# 5. GAINS SIGNIFICATIFS — TOP {len(df_gl):,} URLs\n\n| KW Principal | URL | Anc. pos | Nouv. pos | Diff | Volume | Nb KW |\n|---|---|---|---|---|---|---|\n"
        for _, r in df_gl.iterrows():
            report += f"| {str(r.get('mot_cle', 'N/A'))[:50]} | {r.get('url', 'N/A')} | {_safe_int(r.get('ancienne_pos'))} | {_safe_int(r.get('derniere_pos'))} | +{_safe_int(r.get('diff_pos'))} | {_safe_int(r.get('volume')):,} | {_safe_int(r.get('nb_kw_gains'), 1)} |\n"
    
    # Recommandations
    report += "\n---\n\n# 6. RECOMMANDATIONS\n\n"
    if has_leads_merged:
        report += "## 🔴 Actions immédiates\n1. **PRIORITÉ ABSOLUE : URLs avec leads + pertes SEO**\n2. **Auditer le contenu** des 10 premières URLs critiques\n3. **Vérifier le maillage interne** vers ces pages stratégiques\n\n"
    else:
        report += "## 🔴 Actions immédiates\n1. **Auditer les 10 premières URLs critiques**\n2. **Identifier les KW à fort volume perdus**\n3. **Vérifier la concurrence**\n\n"
    report += "## 🟠 Court terme\n1. Mettre à jour les contenus des pages critiques\n2. Renforcer le maillage interne\n3. Créer du contenu de support\n\n"
    report += "## 🟡 Moyen terme\n1. Audit technique (Core Web Vitals)\n2. Analyse des backlinks\n3. Stratégie de contenu récurrente\n\n"
    report += f"\n---\n\n_Rapport généré automatiquement — {len(df_f):,} mots-clés analysés_\n"
    
    return report


def _generate_ai_analysis(df_f, df, has_leads_merged, has_dual_haloscan, has_gsc, gsc_pages_df,
                          api_key, total, pertes, gains, stables, vol_perdu, vol_gagne):
    with st.spinner("Claude Opus 4.5 analyse vos données... (30-60 secondes)"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            metrics = {"total_kw": total, "kw_en_perte": pertes, "kw_en_gain": gains, "volume_perdu": vol_perdu, "volume_gagne": vol_gagne}
            
            df_pertes_ia = df_f[df_f['diff_pos'] < 0].copy()
            urls_critiques_list = []
            if 'url' in df_pertes_ia.columns:
                u = df_pertes_ia.groupby('url').agg({'diff_pos': ['count', 'mean'], 'volume': 'sum'}).reset_index()
                u.columns = ['url', 'nb_kw_perdus', 'diff_moyenne', 'volume_total']
                if has_leads_merged:
                    l = df_pertes_ia.groupby('url').agg({'leads_total': 'first', 'leads_evolution': 'first'}).reset_index()
                    u = u.merge(l, on='url', how='left')
                urls_critiques_list = u.sort_values('volume_total', ascending=False).head(50).to_dict('records')
            
            top_kw = df_pertes_ia.nlargest(30, 'volume')[['mot_cle', 'url', 'diff_pos', 'volume', 'ancienne_pos', 'derniere_pos']].to_dict('records') if 'volume' in df_pertes_ia.columns else []
            
            context_data = {"metriques_globales": metrics, "urls_critiques": urls_critiques_list, "top_kw_en_perte": top_kw, "has_leads": has_leads_merged}
            
            system_prompt = """Tu es un expert SEO senior. Produis :
1. **ANALYSE STRATÉGIQUE** (5-10 lignes) — Diagnostic clair, patterns, alertes
2. **TODO POUR L'ÉQUIPE CONTENT** — Actions CONCRÈTES et PRIORISÉES avec URL exacte et impact estimé :
## 🔴 PRIORITÉ HAUTE (cette semaine)
- [ ] **[Action]** - URL: [url] - Impact: [estimation] - Raison: [pourquoi]
## 🟠 PRIORITÉ MOYENNE (ce mois)
## 🟡 PRIORITÉ BASSE (à planifier)
3. **ALERTES** si applicable. Sois direct et pragmatique."""
            
            message = client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Données SEO :\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2, default=str)}\n```\nGénère l'analyse et la TODO."}]
            )
            
            st.session_state['ai_analysis'] = message.content[0].text
            st.success("✅ Analyse IA générée !")
            
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
