# **Author Clustering & Recommendation System**

## 📌 Overview
This project uses **author behavior** + **content analysis** to:
- Group authors into meaningful clusters.
- Identify content themes per cluster.
- Recommend authors to companies based on **industry, size, and target topics**.
- Provide **clear visualizations** for analysis.

Everything runs from **`src.py`**.

---

## 🚀 Features
- **Data Cleaning & Merging**
- **TF-IDF Keyword Extraction**
- **Behavioral Metrics** (engagement, posting frequency, consistency)
- **KMeans Clustering** with silhouette optimization
- **Cluster Labeling** and **Theme Extraction**
- **Company-to-Author Matching** based on weighted scoring
- **Visualizations** for:
  - PCA clustering
  - Engagement vs. influence
  - Cluster distributions & keyword trends

---

## 📂 Structure

├── src.py # All functions & pipeline
├── README.md # Documentation
├── requirements.txt # Dependencies
└── data/
├── authors.csv
├── companies.csv
└── posts.csv


---

## ⚙️ Install & Run
```bash
# Clone repo
git clone https://github.com/yourusername/author-clustering.git
cd author-clustering

# Install dependencies
pip install -r requirements.txt

# Run the script
python src.py

## 🛠 Usage in Python

from src import (
    prepare_data,
    extract_content_features,
    create_enhanced_features,
    enhanced_clustering_with_content,
    analyze_cluster_content_themes,
    detailed_cluster_analysis_with_content,
    content_based_recommendations,
    display_recommendations,
    create_enhanced_visualizations
)

import pandas as pd

# Load data
authors = pd.read_csv('data/authors.csv')
companies = pd.read_csv('data/companies.csv')
posts = pd.read_csv('data/posts.csv')

# Pipeline
authors_df, companies_df, posts_df, authors_posts_df = prepare_data(authors, companies, posts)
tfidf_df, author_texts, feature_names = extract_content_features(authors_posts_df)
df = create_enhanced_features(authors_posts_df)

behavioral_features = [...]  # Your numeric feature list
df, model, combined_features, labels, scalers = enhanced_clustering_with_content(df, behavioral_features, tfidf_df)

themes = analyze_cluster_content_themes(df, tfidf_df, feature_names)
analysis = detailed_cluster_analysis_with_content(df, themes, author_texts)

recs = content_based_recommendations(companies_df, df, analysis, themes)
display_recommendations(recs, companies_df)

create_enhanced_visualizations(df, combined_features, labels, analysis, themes)


## 📊 Example Output

Cluster: Top Influencers
High engagement + high influence
- Size: 14 authors
- Influence: 95.2
- Engagement: 88.1
- Keywords: software • development • AI
- Featured Authors: Alice | Bob | Charlie

## 🔄 Workflow Diagram

flowchart TD
    A[Load CSV Data<br>(authors.csv, companies.csv, posts.csv)] --> B[prepare_data()]
    B --> C[extract_content_features()]
    C --> D[create_enhanced_features()]
    D --> E[enhanced_clustering_with_content()]
    E --> F[analyze_cluster_content_themes()]
    F --> G[detailed_cluster_analysis_with_content()]
    G --> H[content_based_recommendations()]
    H --> I[display_recommendations()]
    E --> J[create_enhanced_visualizations()]

