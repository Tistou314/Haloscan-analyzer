# Haloscan SEO Diff Analyzer

Outil d'analyse des différentiels de positions SEO entre deux périodes.  
Conçu pour traiter des fichiers volumineux (250k+ lignes).

## 🚀 Installation locale

```bash
# Cloner ou télécharger les fichiers
cd haloscan_analyzer

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

L'app s'ouvre automatiquement dans le navigateur sur `http://localhost:8501`

## ☁️ Déploiement sur Streamlit Cloud (gratuit)

1. **Créer un repo GitHub** avec les fichiers `app.py` et `requirements.txt`

2. **Aller sur [share.streamlit.io](https://share.streamlit.io)**

3. **Connecter ton compte GitHub**

4. **Déployer** en sélectionnant ton repo

5. **Partager l'URL** avec ton équipe

## 📊 Fonctionnalités

### Dashboard
- KPIs globaux (pertes, gains, stables, sortis)
- Impact en volume et trafic
- Visualisations (pie chart, histogramme, top URLs)

### Pertes critiques
- Tri par score de priorité (volume × diff × facteur position)
- Export CSV

### Analyse par URL
- Agrégation des KW par page
- Score de santé par URL
- Vue détaillée par URL

### Quick wins
- KW qui étaient top 10 et ont chuté
- Potentiel de récupération calculé

### KW sortis
- Liste des mots-clés disparus des SERPs
- Tri par volume

### Rapport
- Génération automatique d'un rapport structuré
- Recommandations actionnables
- Export Markdown

## 🎛️ Filtres disponibles

- Par statut (perdu, gagné, stable, sorti)
- Par volume de recherche (min/max)
- Par différentiel de position
- Par tranche de position (top 3, top 10, etc.)
- Recherche textuelle sur mot-clé
- Filtre par URL

## 📁 Format de fichier attendu

Export CSV Haloscan avec colonnes :
- `mot-clé (mc)` — mot-clé tracké
- `url` — URL positionnée
- `diff_pos` — différentiel de position
- `volume` — volume de recherche
- `dernière_pos` — position actuelle
- `vieille_pos` — ancienne position
- `meilleure_pos` — meilleure position historique
- `statut` — état du mot-clé
- `trafic` — estimation du trafic

## ⚡ Performance

L'outil utilise Pandas et peut traiter **300 000+ lignes** sans problème.  
Les calculs sont optimisés et mis en cache.

## 📝 License

Usage libre — créé pour Easy Content Flow
