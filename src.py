import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import time


def prepare_data(authors_df, companies_df, posts_df):
    # Columns to snake case
    for df in [authors_df, companies_df, posts_df]:
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
    
    # Data processing
    posts_df = posts_df.dropna(subset=['title', 'content'])
    posts_df['date'] = pd.to_datetime(posts_df['date'], errors='coerce')
    posts_df['year'] = posts_df['date'].dt.year
    posts_df['month'] = posts_df['date'].dt.month
    
    # Fill missing values
    companies_df['year_founded'] = companies_df['year_founded'].fillna(companies_df['year_founded'].median())
    companies_df['country'] = companies_df['country'].fillna('Unknown')
    
    # Join data
    authors_posts_df = posts_df.merge(authors_df, left_on="blogger's_id", right_on="author_id", how='inner')
    
    return authors_df, companies_df, posts_df, authors_posts_df


def extract_content_features(df):
    """Extract content-based features from posts with progress tracking"""
    print("Extracting content features...")
    
    # Prepare combined text for each author
    print("   • Preparing text data...")
    df['combined_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    
    # Create author-level text aggregation with progress bar
    print("   • Aggregating text by author...")
    unique_authors = df['author_id'].unique()
    
    author_texts_list = []
    for author_id in tqdm(unique_authors, desc="Processing authors", leave=False):
        author_data = df[df['author_id'] == author_id]
        combined_text = ' '.join(author_data['combined_text'])
        post_count = len(author_data)
        content_text = ' '.join(author_data['content'].fillna(''))
        
        author_texts_list.append({
            'author_id': author_id,
            'combined_text': combined_text,
            'title': post_count,
            'content': content_text
        })
    
    author_texts = pd.DataFrame(author_texts_list)
    
    # TF-IDF for content analysis
    print("   • Computing TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=100,  # Top 100 most important words
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Fit TF-IDF on author texts
    tfidf_matrix = tfidf.fit_transform(author_texts['combined_text'])
    feature_names = tfidf.get_feature_names_out()
    
    # Convert to DataFrame for easier handling
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=[f'tfidf_{name}' for name in feature_names]
    )
    tfidf_df['author_id'] = author_texts['author_id'].values
    
    # Additional content metrics with progress
    print("   • Computing content metrics...")
    
    # Calculate metrics with progress tracking
    metrics_data = []
    for author_id in tqdm(unique_authors, desc="Computing metrics", leave=False):
        author_data = df[df['author_id'] == author_id]
        
        avg_title_length = author_data['title'].str.len().mean()
        avg_content_length = author_data['content'].str.len().mean()
        
        # Vocabulary diversity
        all_words = ' '.join(author_data['combined_text']).split()
        unique_words = set(all_words)
        vocab_diversity = len(unique_words) / max(len(all_words), 1)
        
        metrics_data.append({
            'author_id': author_id,
            'avg_title_length': avg_title_length,
            'avg_content_length': avg_content_length,
            'vocabulary_diversity': vocab_diversity
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    author_texts = author_texts.merge(metrics_df, on='author_id', how='left')
    
    print("Content feature extraction completed!")
    return tfidf_df, author_texts, feature_names


def create_enhanced_features(df):
    """Create enhanced features including content-based ones"""
    df['word_efficiency'] = (
        df['average_number_of_words_in_posts_(without_stopwords)'] / 
        df['average_number_of_words_in_posts']
    )
    df['influence_quality_ratio'] = df['meibi'] / (df['meibix'] + 1)
    
    # Aggregated features by authors
    author_stats = df.groupby('author_id').agg({
        'number_of_comments': ['mean', 'std', 'sum'],
        'number_of_retrieved_inlinks': ['mean', 'sum'],
        'number_of_retrieved_comments': ['mean', 'sum'],
        'post_id': 'count'
    }).round(2)
    
    author_stats.columns = ['_'.join(col).strip() for col in author_stats.columns]
    author_stats = author_stats.reset_index()
    
    # Add new features
    author_stats['avg_engagement'] = (
        author_stats['number_of_comments_mean'] + 
        author_stats['number_of_retrieved_comments_mean']
    ) / 2
    
    author_stats['consistency'] = 1 / (1 + author_stats['number_of_comments_std'].fillna(0))
    author_stats['productivity'] = author_stats['post_id_count']
    
    # Join with main dataframe
    df = df.merge(author_stats, on='author_id', how='left')

    df['days_since_first_post'] = (df['date'].max() - df.groupby('author_id')['date'].transform('min')).dt.days
    df['posting_frequency'] = df['productivity'] / (df['days_since_first_post'] + 1)
    
    return df


def enhanced_clustering_with_content(df, behavioral_features, content_features_df, 
                                   content_weight=0.3, behavioral_weight=0.7):
    """Enhanced clustering combining behavioral and content features"""
    
    # Get unique authors
    unique_authors = df.drop_duplicates('author_id')[['author_id'] + behavioral_features]
    
    # Merge with content features
    unique_authors = unique_authors.merge(content_features_df, on='author_id', how='left')
    
    # Separate behavioral and content features
    behavioral_data = unique_authors[behavioral_features].fillna(0).values
    content_columns = [col for col in content_features_df.columns if col.startswith('tfidf_')]
    content_data = unique_authors[content_columns].fillna(0).values
    
    # Scale features separately
    behavioral_scaler = RobustScaler()
    content_scaler = StandardScaler()
    
    behavioral_scaled = behavioral_scaler.fit_transform(behavioral_data)
    content_scaled = content_scaler.fit_transform(content_data)
    
    # Combine features with weights
    combined_features = np.hstack([
        behavioral_scaled * behavioral_weight,
        content_scaled * content_weight
    ])
    
    # Find optimal number of clusters
    best_score = -1
    best_model = None
    best_labels = None
    best_k = None
    
    silhouette_scores = []
    k_range = range(3, 12)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(combined_features)
        
        sil_score = silhouette_score(combined_features, labels)
        silhouette_scores.append(sil_score)
        
        if sil_score > best_score:
            best_score = sil_score
            best_model = kmeans
            best_labels = labels
            best_k = k
    
    print(f"Optimal number of clusters: {best_k}")
    print(f"Silhouette Score: {best_score:.3f}")
    
    # Add cluster labels
    unique_authors['cluster'] = best_labels
    
    # Add clusters to main dataframe
    df = df.merge(unique_authors[['author_id', 'cluster']], on='author_id', how='left')
    
    return df, best_model, combined_features, best_labels, (behavioral_scaler, content_scaler)


def analyze_cluster_content_themes(df, tfidf_df, feature_names):
    """Analyze content themes for each cluster"""
    
    cluster_themes = {}
    
    # Merge cluster info with TF-IDF features
    cluster_content = df[['author_id', 'cluster']].drop_duplicates().merge(
        tfidf_df, on='author_id', how='left'
    )
    
    print("\nCONTENT THEMES BY CLUSTER")
    print("=" * 50)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = cluster_content[cluster_content['cluster'] == cluster_id]
        
        # Calculate mean TF-IDF scores for this cluster
        tfidf_cols = [col for col in cluster_data.columns if col.startswith('tfidf_')]
        mean_scores = cluster_data[tfidf_cols].mean()
        
        # Get top keywords
        top_keywords = mean_scores.nlargest(10)
        keywords = [keyword.replace('tfidf_', '') for keyword in top_keywords.index]
        
        cluster_themes[cluster_id] = {
            'keywords': keywords,
            'scores': top_keywords.values.tolist()
        }
        
        print(f"\nCluster {cluster_id} - Primary Topics:")
        print(f"   Key themes: {' | '.join(keywords[:5])}")
        top_3_with_scores = [(kw, score) for kw, score in zip(keywords[:3], top_keywords.values[:3])]
        for kw, score in top_3_with_scores:
            print(f"   • {kw}: {score:.3f}")
    
    return cluster_themes


def detailed_cluster_analysis_with_content(df, cluster_themes, author_content_stats):
    """Enhanced cluster analysis including content characteristics with better formatting"""
    
    cluster_analysis = {}
    
    # Merge content stats
    df_with_content = df.merge(
        author_content_stats[['author_id', 'avg_title_length', 'avg_content_length', 'vocabulary_diversity']], 
        on='author_id', how='left'
    )
    
    print("\nDETAILED CLUSTER ANALYSIS")
    print("=" * 60)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df_with_content[df_with_content['cluster'] == cluster_id]
        unique_authors = cluster_data.drop_duplicates('author_id')
        
        analysis = {
            'size': len(unique_authors),
            'avg_meibi': cluster_data['meibi'].mean(),
            'avg_meibix': cluster_data['meibix'].mean(),
            'avg_posts_per_author': cluster_data.groupby('author_id').size().mean(),
            'avg_engagement': cluster_data['avg_engagement'].mean(),
            'avg_word_efficiency': cluster_data['word_efficiency'].mean(),
            'avg_consistency': cluster_data['consistency'].mean(),
            'posting_frequency': cluster_data['posting_frequency'].mean(),
            'avg_title_length': cluster_data['avg_title_length'].mean(),
            'avg_content_length': cluster_data['avg_content_length'].mean(),
            'vocabulary_diversity': cluster_data['vocabulary_diversity'].mean(),
            'content_themes': cluster_themes.get(cluster_id, {}).get('keywords', [])[:5],
            'sample_authors': unique_authors['name'].head(3).tolist()
        }
        
        # Enhanced cluster characterization
        if analysis['avg_meibi'] > df['meibi'].quantile(0.8):
            if analysis['avg_engagement'] > df['avg_engagement'].quantile(0.7):
                cluster_type = "Top Influencers"
                description = "High influence and engagement leaders"
            else:
                cluster_type = "Influential Authors"
                description = "Strong influence with moderate engagement"
        elif analysis['avg_engagement'] > df['avg_engagement'].quantile(0.8):
            cluster_type = "Engaging Bloggers"
            description = "Masters of audience engagement"
        elif analysis['posting_frequency'] > df['posting_frequency'].quantile(0.8):
            cluster_type = "Active Authors"
            description = "High-frequency content creators"
        elif analysis['vocabulary_diversity'] > df_with_content['vocabulary_diversity'].quantile(0.7):
            cluster_type = "Diverse Content Creators"
            description = "Rich vocabulary and varied topics"
        elif analysis['avg_word_efficiency'] > df['word_efficiency'].quantile(0.7):
            cluster_type = "Quality Content"
            description = "Efficient and high-quality writing"
        else:
            cluster_type = "Emerging Bloggers"
            description = "Growing authors with potential"
        
        analysis['cluster_type'] = cluster_type
        analysis['description'] = description
        cluster_analysis[cluster_id] = analysis
        
        # Formatted output
        print(f"\n{cluster_type}")
        print(f"{description}")
        print(f"├─ Size: {analysis['size']} authors")
        print(f"├─ Influence Score: {analysis['avg_meibi']:.1f}")
        print(f"├─ Engagement Level: {analysis['avg_engagement']:.1f}")
        print(f"├─ Content Quality: {analysis['avg_word_efficiency']:.3f}")
        print(f"├─ Vocabulary Richness: {analysis['vocabulary_diversity']:.3f}")
        print(f"├─ Posting Frequency: {analysis['posting_frequency']:.4f} posts/day")
        print(f"├─ Content Themes: {' • '.join(analysis['content_themes'])}")
        print(f"└─ Featured Authors: {' | '.join(analysis['sample_authors'])}")
    
    return cluster_analysis


def content_based_recommendations(companies_df, authors_df, cluster_analysis, cluster_themes, n_recs=3):
    """Content-aware recommendation system with progress tracking and better formatting"""
    
    print("\nGENERATING CONTENT-AWARE RECOMMENDATIONS")
    print("=" * 55)
    
    # Industry to content keywords mapping
    industry_keywords = {
        'information technology and services': ['software', 'technology', 'development', 'digital', 'data'],
        'computer software': ['software', 'programming', 'development', 'code', 'application'],
        'internet': ['web', 'online', 'digital', 'internet', 'platform'],
        'telecommunications': ['communication', 'network', 'mobile', 'technology'],
        'health, wellness and fitness': ['health', 'wellness', 'fitness', 'medical', 'care'],
        'hospital & health care': ['health', 'medical', 'hospital', 'patient', 'care'],
        'automotive': ['car', 'automotive', 'vehicle', 'driving', 'transport'],
        'food & beverages': ['food', 'restaurant', 'cooking', 'recipe', 'beverage'],
        'retail': ['retail', 'shopping', 'store', 'customer', 'product'],
        'financial services': ['finance', 'money', 'investment', 'banking', 'financial'],
        'marketing and advertising': ['marketing', 'advertising', 'brand', 'campaign', 'promotion'],
        'education management': ['education', 'learning', 'student', 'teaching', 'school']
    }
    
    recommendations = {}
    
    # Process companies with progress bar
    for _, company in tqdm(companies_df.iterrows(), total=len(companies_df), desc="Processing companies"):
        company_name = company['name']
        industry = company['industry']
        company_size = company['size_range']
        
        # Get relevant keywords for this industry
        relevant_keywords = industry_keywords.get(industry, [])
        
        # Find clusters with relevant content
        relevant_clusters = []
        for cluster_id, themes in cluster_themes.items():
            keyword_overlap = len(set(themes['keywords'][:10]) & set(relevant_keywords))
            if keyword_overlap > 0:
                relevant_clusters.append((cluster_id, keyword_overlap))
        
        # Sort by keyword overlap
        relevant_clusters.sort(key=lambda x: x[1], reverse=True)
        target_cluster_ids = [cid for cid, _ in relevant_clusters[:3]]  # Top 3 relevant clusters
        
        if not target_cluster_ids:
            target_cluster_ids = list(cluster_analysis.keys())
        
        # Size-based strategy
        if '1 - 10' in str(company_size):
            # Small companies - high engagement, niche content
            suitable_authors = authors_df[
                (authors_df['cluster'].isin(target_cluster_ids)) &
                (authors_df['avg_engagement'] > authors_df['avg_engagement'].quantile(0.6))
            ].copy()
            strategy = "High-engagement niche experts"
        elif '11 - 50' in str(company_size):
            # Medium companies - balanced approach
            suitable_authors = authors_df[
                (authors_df['cluster'].isin(target_cluster_ids)) &
                (authors_df['meibi'] > authors_df['meibi'].quantile(0.5))
            ].copy()
            strategy = "Balanced influence and relevance"
        else:
            # Large companies - top influencers in relevant topics
            suitable_authors = authors_df[
                (authors_df['cluster'].isin(target_cluster_ids)) &
                (authors_df['meibi'] > authors_df['meibi'].quantile(0.7))
            ].copy()
            strategy = "Top industry influencers"
        
        if suitable_authors.empty:
            suitable_authors = authors_df[authors_df['cluster'].isin(target_cluster_ids)].copy()
        
        # Enhanced scoring including content relevance
        suitable_authors['content_relevance'] = suitable_authors['cluster'].map(
            lambda x: dict(relevant_clusters).get(x, 0)
        )
        
        suitable_authors['score'] = (
            suitable_authors['meibi'] * 0.25 +
            suitable_authors['avg_engagement'] * 0.25 +
            suitable_authors['word_efficiency'] * 100 * 0.15 +
            suitable_authors['consistency'] * 50 * 0.15 +
            suitable_authors['content_relevance'] * 10 * 0.2  # Content relevance weight
        )
        
        # Top recommendations
        top_recommendations = (
            suitable_authors
            .drop_duplicates('author_id')
            .nlargest(n_recs, 'score')
            [['name', 'meibi', 'avg_engagement', 'content_relevance', 'score']]
        )
        
        recommendations[company_name] = {
            'authors': top_recommendations.values.tolist(),
            'relevant_themes': [cluster_themes[cid]['keywords'][:3] for cid in target_cluster_ids[:2]],
            'strategy': strategy
        }
    
    return recommendations


def display_recommendations(recommendations, companies_df, top_n=5):
    """Display recommendations in a beautiful, human-readable format"""
    
    print(f"\nTOP {top_n} COMPANY RECOMMENDATIONS")
    print("=" * 65)
    
    # Get top companies by some criteria (e.g., alphabetical or by size)
    displayed_companies = list(recommendations.keys())[:top_n]
    
    for i, company_name in enumerate(displayed_companies, 1):
        rec_data = recommendations[company_name]
        
        # Get company info
        company_info = companies_df[companies_df['name'] == company_name].iloc[0]
        
        print(f"\n{'='*60}")
        print(f"{i}. {company_name}")
        print(f"{'='*60}")
        print(f"Industry: {company_info['industry']}")
        print(f"Size: {company_info['size_range']} employees")
        print(f"Strategy: {rec_data['strategy']}")
        
        # Display relevant themes
        themes = rec_data['relevant_themes']
        if themes:
            flat_themes = [theme for theme_list in themes for theme in theme_list]
            print(f"Relevant Topics: {' • '.join(flat_themes[:5])}")
        
        print(f"\nRECOMMENDED AUTHORS:")
        print("-" * 40)
        
        for j, author_data in enumerate(rec_data['authors'], 1):
            name, meibi, engagement, content_rel, score = author_data
            
            # Create rating display based on score
            rating = "*" * min(5, int(score / 20))
            
            print(f"\n{j}. {name} {rating}")
            print(f"   Influence Score: {meibi:.1f}")
            print(f"   Engagement: {engagement:.1f}")
            print(f"   Content Match: {content_rel}/10")
            print(f"   Overall Score: {score:.1f}")
            
            # Add recommendation reason
            if meibi > 80:
                reason = "Top-tier influencer with massive reach"
            elif engagement > 50:
                reason = "Highly engaging content creator"
            elif content_rel >= 3:
                reason = "Perfect content-industry alignment"
            else:
                reason = "Well-balanced author profile"
            
            print(f"   Why: {reason}")


def create_enhanced_visualizations(df, combined_features, cluster_labels, cluster_analysis, cluster_themes):
    """Enhanced visualizations including content analysis"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. PCA visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(combined_features)
    
    scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0,0].set_title('Author Clusters (Behavioral + Content Features)')
    axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.colorbar(scatter, ax=axes[0,0])
    
    # 2. Cluster sizes
    cluster_sizes = [info['size'] for info in cluster_analysis.values()]
    axes[0,1].bar(range(len(cluster_sizes)), cluster_sizes, color='skyblue')
    axes[0,1].set_title('Cluster Sizes')
    axes[0,1].set_xlabel('Cluster')
    axes[0,1].set_ylabel('Number of Authors')
    
    # 3. MEIBI vs Engagement colored by cluster
    unique_data = df.drop_duplicates('author_id')
    scatter2 = axes[0,2].scatter(unique_data['meibi'], unique_data['avg_engagement'], 
                                c=unique_data['cluster'], cmap='viridis', alpha=0.7)
    axes[0,2].set_title('Influence vs Engagement')
    axes[0,2].set_xlabel('MEIBI')
    axes[0,2].set_ylabel('Average Engagement')
    
    # 4. Content characteristics by cluster
    if 'avg_content_length' in cluster_analysis[0]:
        content_lengths = [info.get('avg_content_length', 0) for info in cluster_analysis.values()]
        axes[1,0].bar(range(len(content_lengths)), content_lengths, color='lightcoral')
        axes[1,0].set_title('Average Content Length by Cluster')
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('Average Content Length')
    
    # 5. Vocabulary diversity
    if 'vocabulary_diversity' in cluster_analysis[0]:
        vocab_diversity = [info.get('vocabulary_diversity', 0) for info in cluster_analysis.values()]
        axes[1,1].bar(range(len(vocab_diversity)), vocab_diversity, color='lightgreen')
        axes[1,1].set_title('Vocabulary Diversity by Cluster')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Vocabulary Diversity')
    
    # 6. Word efficiency by cluster
    word_efficiency = [info['avg_word_efficiency'] for info in cluster_analysis.values()]
    axes[1,2].bar(range(len(word_efficiency)), word_efficiency, color='orange')
    axes[1,2].set_title('Word Efficiency by Cluster')
    axes[1,2].set_xlabel('Cluster')
    axes[1,2].set_ylabel('Word Efficiency')
    
    # 7. Posting frequency
    posting_freq = [info['posting_frequency'] for info in cluster_analysis.values()]
    axes[2,0].bar(range(len(posting_freq)), posting_freq, color='purple')
    axes[2,0].set_title('Posting Frequency by Cluster')
    axes[2,0].set_xlabel('Cluster')
    axes[2,0].set_ylabel('Posts per Day')
    
    # 8. Cluster types distribution
    cluster_types = [info['cluster_type'] for info in cluster_analysis.values()]
    type_counts = Counter(cluster_types)
    axes[2,1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
    axes[2,1].set_title('Cluster Types Distribution')
    
    # 9. Content themes word cloud simulation (bar chart of most common words)
    all_keywords = []
    for themes in cluster_themes.values():
        all_keywords.extend(themes.get('keywords', [])[:5])
    
    keyword_counts = Counter(all_keywords)
    if keyword_counts:
        top_keywords = dict(keyword_counts.most_common(10))
        axes[2,2].barh(list(top_keywords.keys()), list(top_keywords.values()), color='teal')
        axes[2,2].set_title('Most Common Content Keywords')
        axes[2,2].set_xlabel('Frequency')
    
    plt.tight_layout()
    plt.show()