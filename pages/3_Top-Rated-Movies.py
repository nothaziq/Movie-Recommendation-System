import streamlit as st
import pandas as pd 

names = pd.read_csv('Dataset/movies.csv')
ratings = pd.read_csv('Dataset/ratings.csv')

TopGenres = ratings.groupby('movieId')['rating'].mean().reset_index()

top_genres = TopGenres.sort_values('rating', ascending=False).head(25)
top_genres = top_genres.merge(names, on='movieId', how='left')
top_genres = top_genres[['title', 'rating' , 'genres']]
st.title("Top 25 HighestRated Movies")
st.dataframe(top_genres)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;                 /* full page width */
        background-color: #0e1117;
        text-align: center;          /* center the text inside */
        padding: 10px 20px;
        font-size: 14px;
        color: #fff;
        border-top: 1px solid #444;  /* divider line across the whole page */
    }
    .footer a {
        color: #fff;
        text-decoration: none;
        margin: 0 8px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        Movie Recommendation System <br>
        Designed for <b>Introduction to Artificial Intelligence</b> course <br>
        Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> |
        <a href="https://github.com/nothaziq/Movie-Recommendation-System" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)