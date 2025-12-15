# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings
import gc
import os

# Nobody likes warning messages cluttering up the output
warnings.filterwarnings('ignore')

class HybridRecommender:
    """
    A hybrid movie recommendation system that combines content-based and collaborative filtering.
    
    Think of it as having two friends helping you pick movies:
    - One friend knows all about movie genres and themes (content-based)
    - The other friend knows what similar users enjoyed (collaborative filtering)
    
    Together they give you better recommendations than either would alone!
    """
    
    def __init__(self, memory_efficient=True, use_full_dataset=True):
        # Setting up all our data containers - they start empty
        self.movies_df = None              # All movie info (titles, genres, etc.)
        self.ratings_df = None             # User ratings data
        self.tfidf_matrix = None           # Text features for content-based filtering
        self.user_item_matrix = None       # User-movie ratings matrix for collaborative filtering
        self.movie_id_to_idx = {}          # Quick lookup: movie ID -> matrix index
        self.idx_to_movie_id = {}          # Quick lookup: matrix index -> movie ID
        self.knn_model = None              # KNN model for finding similar movies
        self.memory_efficient = memory_efficient
        self.use_full_dataset = use_full_dataset

    def load_data(self, data_path='Dataset'):
        """
        Loads all the movie data from CSV files.
        This is where we read in movies, ratings, tags, and links.
        """
        
        # Load the movies file - pretty straightforward
        movies = pd.read_csv(f'{data_path}/movies.csv', dtype={'movieId': 'int32'})

        # Load ratings - this is the big one! Can have millions of rows
        # Using specific dtypes saves a ton of memory (int32 vs default int64)
        ratings = pd.read_csv(f'{data_path}/ratings.csv', dtype={
            'userId': 'int32',
            'movieId': 'int32',
            'rating': 'float32'
        })
        
        # We don't really need timestamps for recommendations, so toss them
        if 'timestamp' in ratings.columns:
            ratings = ratings.drop('timestamp', axis=1)

        # Force Python to clean up unused memory - helps with large datasets
        gc.collect()
        
        # Try to load tags - not all datasets have these, so we catch errors
        try:
            tags = pd.read_csv(f'{data_path}/tags.csv', dtype={'movieId': 'int32', 'userId': 'int32'})
            if 'timestamp' in tags.columns:
                tags = tags.drop('timestamp', axis=1)
        except:
            # If tags file doesn't exist, just make an empty dataframe
            tags = pd.DataFrame(columns=['movieId', 'tag'])
        
        # Clean up the tags - lowercase and trim whitespace for consistency
        tags['tag'] = tags['tag'].fillna('').str.lower().str.strip()
        
        # Try to load the links file (IMDb and TMDb IDs)
        try:
            links = pd.read_csv(f'{data_path}/links.csv', dtype={'movieId': 'int32'})
        except:
            links = pd.DataFrame(columns=['movieId', 'imdbId', 'tmdbId'])
        
        # Calculate average ratings and rating counts for each movie
        # This tells us how popular a movie is and how well-rated
        avg_ratings = ratings.groupby('movieId', as_index=False).agg({
            'rating': ['mean', 'count']
        })
        avg_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
        avg_ratings['avg_rating'] = avg_ratings['avg_rating'].astype('float32')
        avg_ratings['rating_count'] = avg_ratings['rating_count'].astype('int32')
        
        # Combine all tags for each movie into one text string
        # We convert to a set first to remove duplicates
        movie_tags = tags.groupby('movieId')['tag'].apply(
            lambda x: ' '.join(list(set(list(x))))
        ).reset_index()
        movie_tags.rename(columns={'tag': 'tags_text'}, inplace=True)
        
        # Clean up tags from memory - we don't need it anymore
        del tags
        gc.collect()
        
        # Start building our main movies dataframe by merging in the ratings data
        self.movies_df = movies.merge(avg_ratings, on='movieId', how='left')
        
        # Add tags if we have them
        if not movie_tags.empty:
            self.movies_df = self.movies_df.merge(movie_tags, on='movieId', how='left')
        else:
            self.movies_df['tags_text'] = ''
        
        # Add IMDb/TMDb links if available    
        if not links.empty and 'imdbId' in links.columns:
            self.movies_df = self.movies_df.merge(links[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')
        
        # Fill in missing values - better than having NaN everywhere
        self.movies_df['avg_rating'] = self.movies_df['avg_rating'].fillna(0)
        self.movies_df['rating_count'] = self.movies_df['rating_count'].fillna(0)
        self.movies_df['tags_text'] = self.movies_df['tags_text'].fillna('')

        # Create a combined content field for content-based filtering
        # We replace the | separators in genres with spaces so it's easier to process
        self.movies_df['genres_text'] = self.movies_df['genres'].str.replace('|', ' ')
        self.movies_df['content'] = (
            self.movies_df['genres_text'] + ' ' + 
            self.movies_df['tags_text']
        )
        
        # Keep only the columns we actually need - saves memory
        essential_cols = ['movieId', 'title', 'genres', 'content', 'avg_rating', 'rating_count']
        self.movies_df = self.movies_df[essential_cols]
        
        # Store the ratings for later use in collaborative filtering
        self.ratings_df = ratings
        
        # Clean up all the temporary dataframes we don't need anymore
        del movies, avg_ratings, movie_tags, links
        gc.collect()
        
        return self

    def build_content_based(self, max_features=5000):
        """
        Builds the content-based recommendation model using TF-IDF.
        
        TF-IDF is fancy math that figures out which words are important for each movie.
        For example, "space" is important for Star Wars but not for The Notebook.
        """
        
        # Set up the TF-IDF vectorizer with some smart settings
        tfidf = TfidfVectorizer(
            max_features=max_features,      # Limit features to save memory
            stop_words='english',           # Ignore common words like "the", "a", "is"
            ngram_range=(1, 2),             # Look at both single words and word pairs
            dtype=np.float32,               # Use less memory than float64
            min_df=2                        # Ignore super rare terms that only appear once
        )

        # Transform all movie content into numerical vectors
        # This is where the magic happens!
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])
        
        # Clean up memory again
        gc.collect()
        return self

    def build_collaborative_filtering(self, min_user_ratings=20, min_movie_ratings=50, max_users=None, max_movies=None):
        """
        Builds the collaborative filtering model using K-Nearest Neighbors.
        
        This method finds movies that similar users enjoyed. If you liked movie X and
        another user also liked X, you'll probably like other movies they enjoyed too!
        """
        
        # First, filter down to movies that have enough ratings to be meaningful
        # A movie with only 2 ratings isn't very reliable
        popular_movies = self.movies_df[
            self.movies_df['rating_count'] >= min_movie_ratings
        ]['movieId'].values
        
        # If we want to limit the number of movies (for memory), take the most popular ones
        if max_movies:
            popular_movies = self.movies_df.nlargest(max_movies, 'rating_count')['movieId'].values

        # Filter our ratings to only include these popular movies
        filtered_ratings = self.ratings_df[
            self.ratings_df['movieId'].isin(popular_movies)
        ]

        # Now filter users - we only want "active" users who've rated enough movies
        # Someone who only rated 3 movies isn't very helpful for recommendations
        user_counts = filtered_ratings['userId'].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index
        
        # If we want to limit users (for memory), take the most active ones
        if max_users:
            active_users = user_counts.head(max_users).index
        
        # Filter ratings again to only include our active users
        filtered_ratings = filtered_ratings[filtered_ratings['userId'].isin(active_users)]
        
        # Create the user-item matrix - rows are movies, columns are users
        # This is basically a giant spreadsheet of who rated what
        self.user_item_matrix = filtered_ratings.pivot_table(
            index='movieId',
            columns='userId',
            values='rating',
            fill_value=0  # If a user hasn't rated a movie, put 0
        ).astype(np.float32)

        # Create lookup dictionaries for quick conversions
        # These help us go from movie IDs to matrix positions and back
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(self.user_item_matrix.index)
        }
        self.idx_to_movie_id = {
            idx: mid for mid, idx in self.movie_id_to_idx.items()
        }

        # Convert to sparse matrix - most entries are 0, so this saves tons of memory
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Just for fun, calculate how sparse our matrix is
        # Usually it's over 99% sparse (most users haven't rated most movies)
        sparsity = 100 * (1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]))
        
        # Train the KNN model to find similar movies based on user ratings
        # We use cosine similarity - it's perfect for comparing rating patterns
        self.knn_model = NearestNeighbors(
            metric='cosine',               # Measure similarity using cosine distance
            algorithm='brute',             # Check everything (slower but more accurate)
            n_neighbors=min(50, len(self.user_item_matrix) - 1),  # Find up to 50 neighbors
            n_jobs=-1                      # Use all CPU cores for speed
        )
        self.knn_model.fit(sparse_matrix)

        # Clean up temporary data structures
        del filtered_ratings, sparse_matrix, user_counts
        gc.collect()
        
        return self

    def get_content_recommendations(self, movie_id, n=10):
        """
        Get recommendations based on movie content (genres, tags, etc.)
        
        This finds movies with similar descriptions/genres to the input movie.
        """
        try:
            # Find where this movie is in our dataframe
            idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
            
            # Get this movie's TF-IDF vector
            movie_vector = self.tfidf_matrix[idx]
            
            # Calculate how similar this movie is to ALL other movies
            # Returns a similarity score between 0 (totally different) and 1 (identical)
            sim_scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
            
            # Sort by similarity and grab the top N (excluding the movie itself)
            top_indices = np.argsort(sim_scores)[::-1][1:n+1]
            
            # Build our recommendations dataframe
            recommendations = self.movies_df.iloc[top_indices].copy()
            recommendations['similarity_score'] = sim_scores[top_indices]
            
            return recommendations[['movieId', 'title', 'genres', 'avg_rating', 'similarity_score']]

        except IndexError:
            # Movie not found - return empty dataframe
            return pd.DataFrame()

    def get_collaborative_recommendations(self, movie_id, n=10):
        """
        Get recommendations based on what similar users liked.
        
        This uses the "people who liked this also liked..." approach.
        """
        
        # Check if this movie is in our collaborative filtering model
        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame()

        # Get the matrix index for this movie
        idx = self.movie_id_to_idx[movie_id]
        
        # Use KNN to find the nearest neighbors (similar movies based on ratings)
        distances, indices = self.knn_model.kneighbors(
            self.user_item_matrix.iloc[idx].values.reshape(1, -1),
            n_neighbors=n+1  # +1 because the first result is the movie itself
        )

        # Skip the first result (the input movie) and get the rest
        similar_indices = indices.flatten()[1:]
        similar_distances = distances.flatten()[1:]
        
        # Convert matrix indices back to movie IDs
        similar_movie_ids = [self.idx_to_movie_id[i] for i in similar_indices]
        
        # Look up the movie details for these recommendations
        recommendations = self.movies_df[
            self.movies_df['movieId'].isin(similar_movie_ids)
        ].copy()

        # Add similarity scores (convert distance to similarity: similarity = 1 - distance)
        score_dict = dict(zip(similar_movie_ids, 1 - similar_distances))
        recommendations['similarity_score'] = recommendations['movieId'].map(score_dict)
        recommendations = recommendations.sort_values('similarity_score', ascending=False)
        
        return recommendations[['movieId', 'title', 'genres', 'avg_rating', 'similarity_score']]

    def get_hybrid_recommendations(self, movie_id, n=10, content_weight=0.5, collab_weight=0.5):
        """
        Combines content-based and collaborative filtering for better recommendations!
        
        This is like asking two different friends for movie suggestions and then
        combining their advice. Usually gives better results than either method alone.
        """
        
        # Get recommendations from both methods (ask for 20 to have a bigger pool)
        content_recs = self.get_content_recommendations(movie_id, n=20)
        collab_recs = self.get_collaborative_recommendations(movie_id, n=20)
        
        # If both methods failed, we're out of luck
        if content_recs.empty and collab_recs.empty:
            return pd.DataFrame()
        
        # Normalize the weights so they add up to 1
        total_weight = content_weight + collab_weight
        content_weight /= total_weight
        collab_weight /= total_weight
        
        # Combine all recommended movies (removing duplicates)
        all_movies = pd.concat([
            content_recs[['movieId', 'title', 'genres', 'avg_rating']],
            collab_recs[['movieId', 'title', 'genres', 'avg_rating']]
        ]).drop_duplicates(subset='movieId')
        
        # Calculate hybrid scores for each movie
        hybrid_scores = []
        for _, movie in all_movies.iterrows():
            mid = movie['movieId']
            
            # Get the content-based score (0 if not in content recommendations)
            content_score = 0
            if not content_recs.empty:
                content_match = content_recs[content_recs['movieId'] == mid]
                if not content_match.empty:
                    content_score = content_match['similarity_score'].values[0]
            
            # Get the collaborative filtering score (0 if not in collab recommendations)
            collab_score = 0
            if not collab_recs.empty:
                collab_match = collab_recs[collab_recs['movieId'] == mid]
                if not collab_match.empty:
                    collab_score = collab_match['similarity_score'].values[0]
            
            # Combine the scores using the weights
            # This weighted average gives us the best of both worlds!
            hybrid_score = (content_weight * content_score) + (collab_weight * collab_score)
            hybrid_scores.append(hybrid_score)
        
        # Add scores and sort by them
        all_movies['hybrid_score'] = hybrid_scores
        recommendations = all_movies.sort_values('hybrid_score', ascending=False).head(n)
        
        return recommendations[['movieId', 'title', 'genres', 'avg_rating', 'hybrid_score']]
    
    def get_recommendations_by_title(self, title, n=10, method='hybrid', 
                                    content_weight=0.5, collab_weight=0.5):
        """
        The main interface method - just give it a movie title and get recommendations!
        
        This is probably what you'll use most often. Just type in a movie name
        and it'll find similar movies using your chosen method.
        """
        
        # Search for movies matching the title (case-insensitive)
        matches = self.movies_df[
            self.movies_df['title'].str.contains(title, case=False, na=False)
        ]
        
        # No matches found - maybe check your spelling?
        if matches.empty:
            return pd.DataFrame()
        
        # Use the first match (usually the most relevant one)
        movie_id = matches.iloc[0]['movieId']
        movie_title = matches.iloc[0]['title']
        
        # Route to the appropriate recommendation method
        if method == 'content':
            return self.get_content_recommendations(movie_id, n)
        elif method == 'collaborative':
            return self.get_collaborative_recommendations(movie_id, n)
        else:  # hybrid is the default and usually the best choice
            return self.get_hybrid_recommendations(movie_id, n, content_weight, collab_weight)