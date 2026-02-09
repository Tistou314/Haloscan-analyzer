"""
data_loader.py — Chargement et normalisation des données Haloscan
"""

import streamlit as st
import pandas as pd


@st.cache_data
def load_data(uploaded_file):
    """Charge le CSV avec détection automatique du séparateur et de l'encodage"""
    
    # Lire les premiers octets pour détecter séparateur et encodage
    raw = uploaded_file.read(8192)
    uploaded_file.seek(0)
    
    # Détecter l'encodage
    detected_encoding = 'utf-8'
    for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            raw.decode(enc)
            detected_encoding = enc
            break
        except (UnicodeDecodeError, LookupError):
            continue
    
    # Détecter le séparateur sur la première ligne
    sample = raw.decode(detected_encoding, errors='replace')
    first_line = sample.split('\n')[0]
    sep_counts = {
        ';': first_line.count(';'),
        ',': first_line.count(','),
        '\t': first_line.count('\t'),
    }
    detected_sep = max(sep_counts, key=sep_counts.get)
    if sep_counts[detected_sep] == 0:
        detected_sep = ','
    
    # Charger avec les paramètres détectés
    df = None
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=detected_sep,
            encoding=detected_encoding,
            on_bad_lines='skip',
            low_memory=False
        )
    except Exception:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine='python',
                encoding='latin-1',
                on_bad_lines='skip'
            )
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',', encoding='latin-1')
    
    # Si moins de 3 colonnes, réessayer avec un autre séparateur
    if len(df.columns) < 3:
        for alt_sep in [';', ',', '\t']:
            if alt_sep == detected_sep:
                continue
            try:
                uploaded_file.seek(0)
                df_alt = pd.read_csv(
                    uploaded_file,
                    sep=alt_sep,
                    encoding=detected_encoding,
                    on_bad_lines='skip',
                    low_memory=False
                )
                if len(df_alt.columns) >= 3:
                    df = df_alt
                    break
            except Exception:
                continue
    
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
    
    # Créer colonne 'volume' à partir de 'volumeh' si absente
    if 'volume' not in df.columns and 'volumeh' in df.columns:
        df['volume'] = df['volumeh']
    
    # Conversion numérique
    for col in ['derniere_pos', 'ancienne_pos', 'meilleure_pos', 'diff_pos', 'volume', 'volumeh', 'trafic', 'cpc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calcul du score de priorité
    vol = df['volume'].fillna(0) if 'volume' in df.columns else 0
    diff = df['diff_pos'].fillna(0).abs() if 'diff_pos' in df.columns else 0
    df['priority_score'] = vol * diff
    
    return df


def normalize_url(url):
    """Normalise une URL pour la comparaison"""
    if pd.isna(url):
        return ""
    url = str(url).lower().strip()
    url = url.replace('https://', '').replace('http://', '')
    url = url.replace('www.', '')
    while '//' in url:
        url = url.replace('//', '/')
    url = url.rstrip('/')
    url = url.lstrip('/')
    return url


def tendance_seo(diff):
    """Indicateur visuel de tendance SEO"""
    if pd.isna(diff):
        return "➡️ N/A"
    diff = int(diff)
    if diff <= -20:
        return "🔻🔻 CHUTE"
    elif diff < 0:
        return "🔻 Baisse"
    elif diff == 0:
        return "➡️ Stable"
    elif diff >= 20:
        return "🔺🔺 BOOM"
    else:
        return "🔺 Hausse"


def tendance_seo_url(diff):
    """Indicateur visuel de tendance SEO pour les URLs (seuils plus élevés)"""
    if pd.isna(diff):
        return "➡️ N/A"
    diff = int(diff)
    if diff <= -50:
        return "🔻🔻 CHUTE"
    elif diff < 0:
        return "🔻 Baisse"
    elif diff == 0:
        return "➡️ Stable"
    elif diff >= 50:
        return "🔺🔺 BOOM"
    else:
        return "🔺 Hausse"
