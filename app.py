"""
Haloscan SEO Diff Analyzer — Point d'entrée
"""
import streamlit as st
import pandas as pd
import re
from data_loader import load_data, normalize_url, tendance_seo
import tabs_dashboard, tabs_pertes_gains, tabs_urls, tabs_cannibalisation, tabs_gsc, tabs_rapport

st.set_page_config(page_title="Haloscan SEO Diff Analyzer", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Haloscan SEO Diff Analyzer")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📁 Import des données")
    st.subheader("📊 Fichiers Haloscan")
    uploaded_file_p1 = st.file_uploader("1️⃣ CSV Haloscan Période 1", type=['csv'], key="haloscan_p1")
    uploaded_file_p2 = st.file_uploader("2️⃣ CSV Haloscan Période 2", type=['csv'], key="haloscan_p2")
    if uploaded_file_p1 and uploaded_file_p2:
        st.caption("📅 Nommez vos périodes :")
        c1, c2 = st.columns(2)
        with c1:
            label_debut_p1 = st.text_input("Début P1", value="Jan 2025", key="ldp1")
            label_fin_p1 = st.text_input("Fin P1 / Début P2", value="Sept 2025", key="lfp1")
        with c2:
            label_fin_p2 = st.text_input("Fin P2", value="Fév 2026", key="lfp2")
    else:
        label_debut_p1, label_fin_p1, label_fin_p2 = "Début P1", "Fin P1", "Fin P2"
    st.subheader("💰 Données business")
    uploaded_leads = st.file_uploader("3️⃣ Excel Leads par URL (optionnel)", type=['xlsx', 'xls'])
    st.subheader("🔍 Search Console")
    uploaded_gsc = st.file_uploader("4️⃣ ZIP Search Console (optionnel)", type=['zip'])
    st.subheader("🤖 Analyse IA")
    anthropic_api_key = st.text_input("Clé API Anthropic", type="password")

# --- VARIABLES ---
leads_df = None; has_leads = False; month_cols = []; periode_avant = []; periode_apres = []
periode_avant_label = 'N/A'; periode_apres_label = 'N/A'
has_dual_haloscan = False; gsc_queries_df = None; gsc_pages_df = None; has_gsc = False

def label_to_month(label):
    mm = {'jan':'01','fev':'02','fév':'02','mar':'03','avr':'04','apr':'04','mai':'05','may':'05','jun':'06','jui':'07','jul':'07','aou':'08','aoû':'08','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12','déc':'12'}
    ll = label.lower().strip()
    ym = re.search(r'20\d{2}', ll)
    y = ym.group() if ym else None
    m = next((v for k, v in mm.items() if k in ll), None)
    return f"{y}_{m}" if y and m else None

# --- LEADS ---
if uploaded_leads:
    try:
        xlsx = pd.ExcelFile(uploaded_leads)
        st.sidebar.caption(f"Feuilles : {xlsx.sheet_names}")
        ls = next((s for s in xlsx.sheet_names if 'lead' in s.lower()), None)
        if ls:
            leads_df_raw = pd.read_excel(xlsx, sheet_name=ls)
        elif len(xlsx.sheet_names) > 1:
            leads_df_raw = pd.read_excel(xlsx, sheet_name=1)
        else:
            leads_df_raw = pd.read_excel(xlsx, sheet_name=0)
        mc_chk = [c for c in leads_df_raw.columns if '2025' in str(c) or '2024' in str(c)]
        if mc_chk:
            mv = leads_df_raw[mc_chk].mean().mean()
            if mv > 500: st.sidebar.error(f"⚠️ Moyenne={mv:.0f} — Probablement les VISITES")
            else: st.sidebar.info(f"✅ Moyenne={mv:.1f} — Données leads OK")
    except Exception as e:
        leads_df_raw = pd.read_excel(uploaded_leads)
    month_cols = sorted([c for c in leads_df_raw.columns if c != 'url' and '_' in str(c)])
    has_leads = True
    auto_det = False
    if uploaded_file_p1 and uploaded_file_p2:
        d1, f1, f2 = label_to_month(label_debut_p1), label_to_month(label_fin_p1), label_to_month(label_fin_p2)
        if d1 and f1 and f2:
            da = [m for m in month_cols if d1 <= m < f1]
            dp = [m for m in month_cols if f1 <= m <= f2]
            if da and dp: auto_det = True; st.sidebar.success("🎯 Périodes auto-détectées")
    if not auto_det:
        da = [c for c in month_cols if c.startswith('2025_09')] or (month_cols[-6:-3] if len(month_cols) >= 6 else month_cols[:3])
        dp = [c for c in month_cols if c.startswith('2025_11') or c.startswith('2026')] or (month_cols[-3:] if len(month_cols) >= 3 else month_cols[-1:])
    with st.sidebar:
        st.subheader("📅 Périodes leads")
        periode_avant = st.multiselect("Période AVANT", options=month_cols, default=da, key="avant")
        periode_apres = st.multiselect("Période APRÈS", options=month_cols, default=dp, key="apres")
    leads_df = leads_df_raw.copy()
    for c in month_cols:
        if c in leads_df.columns: leads_df[c] = pd.to_numeric(leads_df[c], errors='coerce').fillna(0)
    periode_avant_label = '+'.join(periode_avant) if periode_avant else 'N/A'
    periode_apres_label = '+'.join(periode_apres) if periode_apres else 'N/A'
    if periode_avant and periode_apres:
        aa = periode_avant + periode_apres
        pc = [m for m in month_cols if min(aa) <= m <= max(aa)]
        leads_df['leads_total'] = leads_df[pc].sum(axis=1) if pc else 0
    elif month_cols: leads_df['leads_total'] = leads_df[month_cols].sum(axis=1)
    else: leads_df['leads_total'] = 0
    leads_df['leads_avant'] = leads_df[periode_avant].sum(axis=1) if periode_avant else 0
    leads_df['leads_apres'] = leads_df[periode_apres].sum(axis=1) if periode_apres else 0
    leads_df['leads_evolution'] = leads_df['leads_apres'] - leads_df['leads_avant']
    leads_df['leads_evolution_pct'] = ((leads_df['leads_apres'] - leads_df['leads_avant']) / leads_df['leads_avant'].replace(0, 1) * 100).round(1)
    leads_df['url_normalized'] = leads_df['url'].apply(normalize_url)
    st.sidebar.success(f"✅ {len(leads_df):,} URLs avec données leads")

# --- GSC ---
if uploaded_gsc:
    import zipfile
    try:
        with zipfile.ZipFile(uploaded_gsc, 'r') as z:
            qf = next((f for f in z.namelist() if any(k in f for k in ['Requêtes','Queries','requetes'])), None)
            pf = next((f for f in z.namelist() if 'Pages' in f or 'pages' in f.lower()), None)
            if qf:
                with z.open(qf) as q: gsc_queries_df = pd.read_csv(q)
                gsc_queries_df.columns = gsc_queries_df.columns.str.strip()
                gsc_queries_df = gsc_queries_df.rename(columns={gsc_queries_df.columns[0]: 'query'})
                if 'CTR' in gsc_queries_df.columns: gsc_queries_df['CTR'] = gsc_queries_df['CTR'].astype(str).str.replace('%','').str.replace(',','.').astype(float)
            if pf:
                with z.open(pf) as p: gsc_pages_df = pd.read_csv(p)
                gsc_pages_df.columns = gsc_pages_df.columns.str.strip()
                gsc_pages_df = gsc_pages_df.rename(columns={gsc_pages_df.columns[0]: 'url'})
                if 'CTR' in gsc_pages_df.columns: gsc_pages_df['CTR'] = gsc_pages_df['CTR'].astype(str).str.replace('%','').str.replace(',','.').astype(float)
                gsc_pages_df['url_normalized'] = gsc_pages_df['url'].apply(normalize_url)
            if gsc_queries_df is not None or gsc_pages_df is not None:
                has_gsc = True
                st.sidebar.success(f"🔍 GSC chargé")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur ZIP GSC: {e}")

# --- HALOSCAN ---
uploaded_file = None
if uploaded_file_p1 and uploaded_file_p2:
    has_dual_haloscan = True; st.sidebar.success("📊 Mode double période")
elif uploaded_file_p1: uploaded_file = uploaded_file_p1
elif uploaded_file_p2: uploaded_file = uploaded_file_p2

if has_dual_haloscan:
    df_p1 = load_data(uploaded_file_p1); df_p2 = load_data(uploaded_file_p2)
    # Debug colonnes pour diagnostic
    with st.sidebar.expander("🔍 Debug colonnes CSV", expanded=True):
        st.write("P1:", list(df_p1.columns))
        st.write("P2:", list(df_p2.columns))
    
    # Vérifier que les colonnes essentielles existent
    for required_col, label in [('mot_cle', 'Mot-clé'), ('url', 'URL')]:
        if required_col not in df_p1.columns:
            st.error(f"❌ Colonne '{required_col}' ({label}) non trouvée dans P1. Colonnes disponibles : {list(df_p1.columns)}")
            st.stop()
        if required_col not in df_p2.columns:
            st.error(f"❌ Colonne '{required_col}' ({label}) non trouvée dans P2. Colonnes disponibles : {list(df_p2.columns)}")
            st.stop()
    
    # Renommer les colonnes de position avec suffixes _p1 et _p2
    for col_orig, col_new in [('ancienne_pos','pos_debut'), ('derniere_pos','pos_fin'), ('diff_pos','diff')]:
        if col_orig in df_p1.columns:
            df_p1 = df_p1.rename(columns={col_orig: f'{col_new}_p1'})
        if col_orig in df_p2.columns:
            df_p2 = df_p2.rename(columns={col_orig: f'{col_new}_p2'})
    
    # Colonnes P2 à garder pour le merge
    p2_pos_cols = [c for c in ['pos_debut_p2', 'pos_fin_p2', 'diff_p2'] if c in df_p2.columns]
    p2_extra_cols = [c for c in df_p2.columns if c not in ['mot_cle', 'url'] + p2_pos_cols]
    
    # Merge sur mot_cle + url
    df = df_p1.merge(df_p2[['mot_cle', 'url'] + p2_pos_cols], on=['mot_cle', 'url'], how='outer', suffixes=('', '_dup'))
    
    # Recalculer les colonnes globales
    col_debut_p1 = 'pos_debut_p1' if 'pos_debut_p1' in df.columns else None
    col_fin_p1 = 'pos_fin_p1' if 'pos_fin_p1' in df.columns else None
    col_fin_p2 = 'pos_fin_p2' if 'pos_fin_p2' in df.columns else None
    col_debut_p2 = 'pos_debut_p2' if 'pos_debut_p2' in df.columns else None
    
    df['ancienne_pos'] = (df[col_debut_p1] if col_debut_p1 else pd.Series(0, index=df.index)).fillna(df[col_debut_p2] if col_debut_p2 else 0)
    df['derniere_pos'] = (df[col_fin_p2] if col_fin_p2 else pd.Series(0, index=df.index)).fillna(df[col_fin_p1] if col_fin_p1 else 0)
    df['diff_pos'] = df['ancienne_pos'] - df['derniere_pos']
    def ctm(row):
        d1 = 0 if pd.isna(row.get('diff_p1',0)) else (row.get('diff_p1',0) or 0)
        d2 = 0 if pd.isna(row.get('diff_p2',0)) else (row.get('diff_p2',0) or 0)
        if d1<-5 and d2<-5: return "📉📉 Chute continue"
        elif d1>5 and d2<-5: return "📈📉 Rebond puis rechute"
        elif d1<-5 and d2>5: return "📉📈 Récupération"
        elif d1>5 and d2>5: return "📈📈 Hausse continue"
        elif abs(d1)<=5 and abs(d2)<=5: return "➡️ Stable"
        elif d1<0 or d2<0: return "📉 Baisse"
        else: return "📈 Hausse"
    df['tendance_multi'] = df.apply(ctm, axis=1)
    if 'volume' not in df.columns and 'volumeh' in df.columns: df['volume'] = df['volumeh']
    df['priority_score'] = df.get('volume', pd.Series(0, index=df.index)).fillna(0) * df['diff_pos'].abs().fillna(0)
    st.sidebar.info(f"🔗 {len(df):,} KW fusionnés")
elif uploaded_file:
    df = load_data(uploaded_file)

# --- CROISEMENT LEADS + FILTRES + ONGLETS ---
if (has_dual_haloscan or uploaded_file) and 'df' in dir():
    has_leads_merged = False
    if has_leads and 'url' in df.columns:
        df['url_normalized'] = df['url'].apply(normalize_url)
        df = df.merge(leads_df[['url_normalized','leads_total','leads_avant','leads_apres','leads_evolution','leads_evolution_pct']], on='url_normalized', how='left')
        df.attrs['periode_avant_label'] = periode_avant_label
        df.attrs['periode_apres_label'] = periode_apres_label
        def tl(row):
            e = row.get('leads_evolution',0) or 0; p = row.get('leads_evolution_pct',0) or 0
            if e<-10 or p<-20: return "🔻🔻 CHUTE"
            elif e<0: return "🔻 Baisse"
            elif e==0: return "➡️ Stable"
            elif e>10 or p>20: return "🔺🔺 BOOM"
            else: return "🔺 Hausse"
        df['tendance_leads'] = df.apply(tl, axis=1)
        df['priority_score_business'] = df['priority_score'] * (1+df['leads_total'].fillna(0)/100) * (1+df['leads_evolution'].fillna(0).clip(upper=0).abs()/100)
        df['double_peine'] = (df['diff_pos']<0) & (df['leads_evolution']<0)
        df['tendance_seo'] = df['diff_pos'].apply(tendance_seo)
        st.success(f"✅ {len(df):,} mots-clés — Données leads croisées !")
        st.info(f"📊 {df[df['leads_total'].notna()&(df['leads_total']>0)]['url'].nunique()} URLs avec leads | ⚠️ {df[df['double_peine']==True]['url'].nunique()} en double peine")
        has_leads_merged = True
    else:
        for c in ['leads_total','leads_avant','leads_apres','leads_evolution']: df[c] = 0
        df['tendance_leads'] = "➡️ N/A"
        df['tendance_seo'] = df['diff_pos'].apply(tendance_seo)
        st.success(f"✅ {len(df):,} mots-clés chargés")
    with st.sidebar:
        with st.expander("🔍 Colonnes", expanded=True): st.write(list(df.columns))
    if 'diff_pos' not in df.columns: st.error("❌ Colonne 'diff_pos' non trouvée"); st.stop()
    with st.sidebar:
        st.header("🎛️ Filtres")
        variation = st.multiselect("Variation", ['Pertes','Gains','Stables'], default=['Pertes','Gains','Stables'])
        vol_range = st.slider("Volume", int(df['volume'].min() or 0), int(df['volume'].max() or 10000), (int(df['volume'].min() or 0), int(df['volume'].max() or 10000))) if 'volume' in df.columns else None
        search_kw = st.text_input("🔎 Mot-clé"); search_url = st.text_input("🔎 URL contient")
    df_f = df.copy()
    masks = []
    if 'Pertes' in variation: masks.append(df_f['diff_pos']<0)
    if 'Gains' in variation: masks.append(df_f['diff_pos']>0)
    if 'Stables' in variation: masks.append(df_f['diff_pos']==0)
    if masks:
        c = masks[0]
        for m in masks[1:]: c = c | m
        df_f = df_f[c]
    if vol_range and 'volume' in df_f.columns: df_f = df_f[(df_f['volume']>=vol_range[0])&(df_f['volume']<=vol_range[1])]
    if search_kw and 'mot_cle' in df_f.columns: df_f = df_f[df_f['mot_cle'].astype(str).str.contains(search_kw, case=False, na=False)]
    if search_url and 'url' in df_f.columns: df_f = df_f[df_f['url'].astype(str).str.contains(search_url, case=False, na=False)]
    total=len(df_f); pertes=len(df_f[df_f['diff_pos']<0]); gains=len(df_f[df_f['diff_pos']>0]); stables=len(df_f[df_f['diff_pos']==0])
    vol_perdu = int(df_f[df_f['diff_pos']<0]['volume'].fillna(0).sum()) if 'volume' in df_f.columns else 0
    vol_gagne = int(df_f[df_f['diff_pos']>0]['volume'].fillna(0).sum()) if 'volume' in df_f.columns else 0
    t1,t2,t3,t4,t5,t6,t7 = st.tabs(["📊 Dashboard","🔴 Pertes","📁 Par URL","🟢 Gains","🔄 Cannibalisation","🔍 Search Console","📝 Rapport"])
    with t1: tabs_dashboard.render(df_f, has_leads_merged, has_dual_haloscan, df, label_debut_p1, label_fin_p1, label_fin_p2, total, pertes, gains, stables, vol_perdu, vol_gagne)
    with t2: tabs_pertes_gains.render_pertes(df_f)
    with t3: tabs_urls.render(df_f, df, has_leads_merged)
    with t4: tabs_pertes_gains.render_gains(df_f)
    with t5: tabs_cannibalisation.render(df)
    with t6: tabs_gsc.render(df_f, has_gsc, gsc_pages_df, gsc_queries_df)
    with t7: tabs_rapport.render(df_f, df, has_leads_merged, has_dual_haloscan, has_gsc, gsc_pages_df, label_debut_p1, label_fin_p1, label_fin_p2, periode_avant, periode_apres, anthropic_api_key, total, pertes, gains, stables, vol_perdu, vol_gagne)
else:
    st.info("👆 Charge un fichier CSV pour commencer")
