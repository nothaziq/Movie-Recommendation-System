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
        