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