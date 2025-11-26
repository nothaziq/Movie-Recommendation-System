# Importing the libraries

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os 

#---DATA LOADING---
names = pd.read_csv('Dataset/movies.csv')
ratings = pd.read_csv('Dataset/ratings.csv')
tags = pd.read_csv('Dataset/tags.csv')
links = pd.read_csv('Dataset/links.csv')

#---DATA CLEANING---
tags['tag'] = tags['tag'].fillna('').str.lower().str.strip()

user_movies = ratings.merge(names, on='movieId', how='left')
print("User-level dataset:", user_movies.shape)

avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index() 

movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: list(set(x))).reset_index()
movie_tags.rename(columns={'tag': 'tags_list'}, inplace=True)

clean_movies = names.merge(avg_ratings, on='movieId', how='left') \
                    .merge(movie_tags, on='movieId', how='left') \
                    .merge(links, on='movieId', how='left')

clean_movies['genres'] = clean_movies['genres'].str.split('|')

print("Movie-level dataset:", clean_movies.shape)
print(clean_movies.head())

clean_movies['features'] = (
    clean_movies['genres'].apply(lambda g: ' '.join(g) if isinstance(g, list) else '') + ' ' +
    clean_movies['tags_list'].apply(lambda t: ' '.join(t) if isinstance(t, list) else '')
).str.strip()

#---SAVING THE CLEANED DATASETS---
# user_movies.to_csv('Dataset/user_movies_clean.csv', index=False)
# clean_movies.to_csv('Dataset/clean_movies_agg.csv', index=False)

#---Content-Based Recommender (CBF)---

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(clean_movies['features'])

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
title_to_index = {t:i for i, t in enumerate(clean_movies['title'])}

def content_based_recommend(title, top_k=10):
    if title not in title_to_index:
        return []
    idx = title_to_index[title]
    sims = cos_sim[idx]
    order = sims.argsort()[::-1]
    recs = []
    for j in order:
        if len(recs) >= top_k+1: break
        recs.append((clean_movies.iloc[j]['title'], sims[j]))
    return [(t,s) for t,s in recs if t != title][:top_k]

