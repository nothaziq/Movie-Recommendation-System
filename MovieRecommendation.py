# Importing the libraries

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings
import os 

warnings.filterwarnings('ignore')
class HybridRecommender:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.user_item_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.knn_model = None
        self.memory_efficient = memory_efficient

    def load_data(self):
         movies = pd.read_csv(Dataset/movies.csv, dtype={'movieId': 'int32'})

         if sample_ratings:
            print(f"Sampling {sample_ratings:,} ratings from dataset...")
            chunk_size = 1000000
            chunks = []
            total_rows = 0
            
            for chunk in pd.read_csv(Dataset/ratings.csv, chunksize=chunk_size, 
                                     dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}):
                if total_rows >= sample_ratings:
                    break
                chunks.append(chunk)
                total_rows += len(chunk)
            
            ratings = pd.concat(chunks, ignore_index=True).head(sample_ratings)
            del chunks
                else:
                    ratings = pd.read_csv(Dataset/ratings.csv, dtype={
                        'userId': 'int32',
                        'movieId': 'int32',
                        'rating': 'float32'
                    })
        
        if 'timestamp' in ratings.columns:
            ratings = ratings.drop('timestamp', axis=1)

        gc.collect()
        # Load tags (sample if too large)
        try:
            tags = pd.read_csv(Dataset/tags.csv, dtype={'movieId': 'int32', 'userId': 'int32'})
            if 'timestamp' in tags.columns:
                tags = tags.drop('timestamp', axis=1)

            # Sample tags if too many
            if len(tags) > 500000:
                tags = tags.sample(n=500000, random_state=42)
                print(f"Sampled 500,000 tags from dataset")
        except:
            print("Error loading tags, creating empty dataframe")
            tags = pd.DataFrame(columns=['movieId', 'tag'])
        
        # Clean tags
        tags['tag'] = tags['tag'].fillna('').str.lower().str.strip()
        
        # Load links
        print("Loading links...")
        links = pd.read_csv(Dataset/links.csv, dtype={'movieId': 'int32'})
        
        # Aggregate data efficiently
        print("Aggregating ratings...")
        avg_ratings = ratings.groupby('movieId', as_index=False).agg({
            'rating': ['mean', 'count']
        })
        avg_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
        avg_ratings['avg_rating'] = avg_ratings['avg_rating'].astype('float32')
        avg_ratings['rating_count'] = avg_ratings['rating_count'].astype('int32')
        
        # Aggregate tags efficiently
        print("Aggregating tags...")
        movie_tags = tags.groupby('movieId')['tag'].apply(
            lambda x: ' '.join(list(set(list(x)[:20])))  # Limit to 20 tags per movie
        ).reset_index()
        movie_tags.rename(columns={'tag': 'tags_text'}, inplace=True)
        
        # Free memory
        del tags
        gc.collect()
        
        # Merge everything
        print("Merging datasets...")
        self.movies_df = movies.merge(avg_ratings, on='movieId', how='left') \
                               .merge(movie_tags, on='movieId', how='left') \
                               .merge(links[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')
        
        # Handle missing values
        self.movies_df['avg_rating'] = self.movies_df['avg_rating'].fillna(0)
        self.movies_df['rating_count'] = self.movies_df['rating_count'].fillna(0)
        self.movies_df['tags_text'] = self.movies_df['tags_text'].fillna('')

         # Create content field
        self.movies_df['genres_text'] = self.movies_df['genres'].str.replace('|', ' ')
        self.movies_df['content'] = (
            self.movies_df['genres_text'] + ' ' + 
            self.movies_df['tags_text']
        )
        
        # Keep only essential columns
        self.movies_df = self.movies_df[['movieId', 'title', 'genres', 'content', 
                                         'avg_rating', 'rating_count']]
        
        self.ratings_df = ratings
        
        # Free memory
        del movies, avg_ratings, movie_tags, links
        gc.collect()
        
        return self

    def build_content_based(self, max_features=2000):

        # TF-IDF vectorization with reduced features for 32M dataset
          tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 1),  # Unigrams only to save memory
            dtype=np.float32,
            min_df=2  # Ignore rare terms
        )

        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])

        # Free memory
        gc.collect()
        
        return self

    def build_collaborative_filtering(self, min_ratings=50, max_users=20000, max_movies=5000):
        
        # Get most popular movies
        popular_movies = self.movies_df.nlargest(max_movies, 'rating_count')['movieId'].values

        # Filter ratings for popular movies only
        filtered_ratings = self.ratings_df[
            self.ratings_df['movieId'].isin(popular_movies)
        ]

        #Get the most Active users
        user_counts = filtered_ratings['userId'].value_counts()
        top_users = user_counts.head(max_users).index
        
        filtered_ratings = filtered_ratings[filtered_ratings['userId'].isin(top_users)]

        # Create user-item matrix with sparse format
        print("Creating sparse user-item matrix...")
        self.user_item_matrix = filtered_ratings.pivot_table(
            index='movieId',
            columns='userId',
            values='rating',
            fill_value=0
        ).astype(np.float32)

        # Create mappings
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(self.user_item_matrix.index)
        }
        self.idx_to_movie_id = {
            idx: mid for mid, idx in self.movie_id_to_idx.items()
        }

        # Convert to sparse matrix
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        sparsity = 100 * (1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]))

        # Train KNN model
        self.knn_model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=min(20, len(self.user_item_matrix) - 1),
            n_jobs=-1  # Use all CPU cores
        )
        self.knn_model.fit(sparse_matrix)

        # Free memory
        del filtered_ratings, sparse_matrix, user_counts
        gc.collect()

        return self

    def get_content_recommendations(self, movie_id, n=10):
       try:
            idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]

            # Compute similarity only for this movie (memory efficient)
            movie_vector = self.tfidf_matrix[idx]
            sim_scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
            
            # Get top N similar movies
            top_indices = np.argsort(sim_scores)[::-1][1:n+1]  # Exclude the movie itself
            
            recommendations = self.movies_df.iloc[top_indices].copy()
            recommendations['similarity_score'] = sim_scores[top_indices]
            
            return recommendations[['movieId', 'title', 'genres', 'avg_rating', 'similarity_score']]

       except IndexError:
            return pd.DataFrame()

    def get_collaborative_recommendations(self, movie_id, n=10):
        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame()

        idx = self.movie_id_to_idx[movie_id]
        distances, indices = self.knn_model.kneighbors(
            self.user_item_matrix.iloc[idx].values.reshape(1, -1),
            n_neighbors=n+1
        )

        # Skip first result (the movie itself)
        similar_indices = indices.flatten()[1:]
        similar_distances = distances.flatten()[1:]
        
        similar_movie_ids = [self.idx_to_movie_id[i] for i in similar_indices]
        
        recommendations = self.movies_df[
            self.movies_df['movieId'].isin(similar_movie_ids)
        ].copy()

        # Add similarity scores
        score_dict = dict(zip(similar_movie_ids, 1 - similar_distances))
        recommendations['similarity_score'] = recommendations['movieId'].map(score_dict)
        recommendations = recommendations.sort_values('similarity_score', ascending=False)
        
        return recommendations[['movieId', 'title', 'genres', 'avg_rating', 'similarity_score']]

    def get_hybrid_recommendations(self, movie_id, n=10, content_weight=0.5, collab_weight=0.5):

        # Get recommendations from both methods
        content_recs = self.get_content_recommendations(movie_id, n=20)
        collab_recs = self.get_collaborative_recommendations(movie_id, n=20)
        
        if content_recs.empty and collab_recs.empty:
            return pd.DataFrame()
        
        # Normalize weights
        total_weight = content_weight + collab_weight
        content_weight /= total_weight
        collab_weight /= total_weight
        
        # Combine recommendations
        all_movies = pd.concat([
            content_recs[['movieId', 'title', 'genres', 'avg_rating']],
            collab_recs[['movieId', 'title', 'genres', 'avg_rating']]
        ]).drop_duplicates(subset='movieId')
        
        # Calculate hybrid scores
        hybrid_scores = []
        for _, movie in all_movies.iterrows():
            mid = movie['movieId']
            
            # Get scores from both methods
            content_score = 0
            if not content_recs.empty:
                content_match = content_recs[content_recs['movieId'] == mid]
                if not content_match.empty:
                    content_score = content_match['similarity_score'].values[0]
            
            collab_score = 0
            if not collab_recs.empty:
                collab_match = collab_recs[collab_recs['movieId'] == mid]
                if not collab_match.empty:
                    collab_score = collab_match['similarity_score'].values[0]
            
            # Calculate weighted hybrid score
            hybrid_score = (content_weight * content_score) + (collab_weight * collab_score)
            hybrid_scores.append(hybrid_score)
        
        all_movies['hybrid_score'] = hybrid_scores
        
        # Sort and return top N
        recommendations = all_movies.sort_values('hybrid_score', ascending=False).head(n)
        
        return recommendations[['movieId', 'title', 'genres', 'avg_rating', 'hybrid_score']]
    
    def get_recommendations_by_title(self, title, n=10, method='hybrid', 
                                    content_weight=0.5, collab_weight=0.5):
        # Find movie by title (case-insensitive partial match)
        matches = self.movies_df[
            self.movies_df['title'].str.contains(title, case=False, na=False)
        ]
        
        if matches.empty:
            print(f"No movies found matching: {title}")
            return pd.DataFrame()
        
        if len(matches) > 1:
            print(f"\nMultiple matches found for '{title}':")
            for idx, row in matches.head(5).iterrows():
                print(f"  - {row['title']} (ID: {row['movieId']})")
            print("\nUsing the first match. Please be more specific if needed.")
        
        movie_id = matches.iloc[0]['movieId']
        movie_title = matches.iloc[0]['title']
        
        print(f"\nGetting recommendations for: {movie_title}")
        print("-" * 60)
        
        if method == 'content':
            return self.get_content_recommendations(movie_id, n)
        elif method == 'collaborative':
            return self.get_collaborative_recommendations(movie_id, n)
        else:  # hybrid
            return self.get_hybrid_recommendations(movie_id, n, content_weight, collab_weight)
 