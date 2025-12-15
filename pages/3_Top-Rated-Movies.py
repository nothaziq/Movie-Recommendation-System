import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Top Rated Movies",
    page_icon="⭐",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .movie-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffd700;
        margin-bottom: 0.5rem;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateX(5px);
    }
    .movie-rank {
        font-size: 2rem;
        font-weight: bold;
        color: #ffd700;
    }
    .movie-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;
    }
    .movie-genres {
        color: #888;
        font-style: italic;
    }
    .rating-badge {
        background-color: #ffd700;
        color: #000;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⭐ Top Rated Movies")
st.markdown("### Discover the highest-rated movies in our database")

# Load data
@st.cache_data
def load_movie_data():
    movies = pd.read_csv('Dataset/movies.csv')
    ratings = pd.read_csv('Dataset/ratings.csv')
    
    # Calculate average ratings and count
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
    
    # Merge with movie info
    top_movies = movie_stats.merge(movies, on='movieId', how='left')
    
    return top_movies, ratings

top_movies, ratings = load_movie_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Minimum ratings filter
min_ratings = st.sidebar.slider(
    "Minimum number of ratings:",
    min_value=1,
    max_value=1000,
    value=100,
    step=50,
    help="Filter movies that have been rated at least this many times"
)

# Genre filter
all_genres = set()
for genres in top_movies['genres'].dropna():
    all_genres.update(genres.split('|'))
all_genres = sorted([g for g in all_genres if g != '(no genres listed)'])

selected_genres = st.sidebar.multiselect(
    "Filter by genres:",
    options=all_genres,
    help="Select one or more genres to filter"
)

# Number of movies to show
num_movies = st.sidebar.slider(
    "Number of movies to display:",
    min_value=10,
    max_value=100,
    value=25,
    step=5
)

# Apply filters
filtered_movies = top_movies[top_movies['rating_count'] >= min_ratings].copy()

if selected_genres:
    # Filter movies that have at least one of the selected genres
    filtered_movies = filtered_movies[
        filtered_movies['genres'].apply(
            lambda x: any(genre in str(x) for genre in selected_genres)
        )
    ]

# Sort and get top N
top_n = filtered_movies.nlargest(num_movies, 'avg_rating')

# Display stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Movies", f"{len(filtered_movies):,}")

with col2:
    st.metric("⭐ Avg Rating", f"{top_n['avg_rating'].mean():.2f}")

with col3:
    st.metric("🎬 Showing", f"{len(top_n)}")

with col4:
    st.metric("📈 Total Ratings", f"{filtered_movies['rating_count'].sum():,}")

st.divider()

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📋 List View", "📊 Charts", "📈 Statistics"])

with tab1:
    st.markdown("### Top Rated Movies")
    
    # Display movies in a nice format
    for idx, (_, movie) in enumerate(top_n.iterrows(), 1):
        col1, col2, col3 = st.columns([0.5, 4, 1])
        
        with col1:
            st.markdown(f"<div class='movie-rank'>#{idx}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{movie['title']}**")
            st.markdown(f"*{movie['genres']}*")
            st.caption(f"Based on {int(movie['rating_count']):,} ratings")
        
        with col3:
            st.markdown(
                f"<span class='rating-badge'>⭐ {movie['avg_rating']:.2f}</span>", 
                unsafe_allow_html=True
            )
        
        st.divider()
    
    # Download button
    csv = top_n[['title', 'genres', 'avg_rating', 'rating_count']].to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="top_rated_movies.csv",
        mime="text/csv"
    )

with tab2:
    st.markdown("### Visualizations")
    
    # Top 15 for better visibility in charts
    top_15 = top_n.head(15).copy()
    
    # Bar chart of ratings
    fig1 = px.bar(
        top_15,
        x='avg_rating',
        y='title',
        orientation='h',
        title='Top 15 Movies by Rating',
        labels={'avg_rating': 'Average Rating', 'title': 'Movie'},
        color='avg_rating',
        color_continuous_scale='viridis'
    )
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Rating distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = px.histogram(
            filtered_movies,
            x='avg_rating',
            nbins=20,
            title='Rating Distribution',
            labels={'avg_rating': 'Average Rating', 'count': 'Number of Movies'},
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Genre distribution in top movies
        genre_counts = {}
        for genres in top_n['genres']:
            for genre in str(genres).split('|'):
                if genre != '(no genres listed)':
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
        genre_df = genre_df.sort_values('Count', ascending=False).head(10)
        
        fig3 = px.pie(
            genre_df,
            values='Count',
            names='Genre',
            title='Genre Distribution in Top Movies'
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Rating Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Mean Rating', 'Median Rating', 'Std Deviation', 'Min Rating', 'Max Rating'],
            'Value': [
                f"{top_n['avg_rating'].mean():.3f}",
                f"{top_n['avg_rating'].median():.3f}",
                f"{top_n['avg_rating'].std():.3f}",
                f"{top_n['avg_rating'].min():.3f}",
                f"{top_n['avg_rating'].max():.3f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        st.markdown("#### 🎬 Rating Count Statistics")
        count_stats_df = pd.DataFrame({
            'Metric': ['Mean Count', 'Median Count', 'Total Ratings', 'Most Rated Movie'],
            'Value': [
                f"{top_n['rating_count'].mean():.0f}",
                f"{top_n['rating_count'].median():.0f}",
                f"{top_n['rating_count'].sum():,}",
                f"{top_n.loc[top_n['rating_count'].idxmax(), 'title']}"
            ]
        })
        st.dataframe(count_stats_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏆 Top 10 Most Rated Movies")
        most_rated = top_n.nlargest(10, 'rating_count')[['title', 'rating_count', 'avg_rating']]
        most_rated.columns = ['Movie', 'Number of Ratings', 'Avg Rating']
        st.dataframe(most_rated, hide_index=True, use_container_width=True)
        
        st.markdown("#### 🎭 Movies by Genre")
        genre_counts_full = {}
        for genres in top_n['genres']:
            for genre in str(genres).split('|'):
                if genre != '(no genres listed)':
                    genre_counts_full[genre] = genre_counts_full.get(genre, 0) + 1
        
        genre_full_df = pd.DataFrame(
            list(genre_counts_full.items()), 
            columns=['Genre', 'Count']
        ).sort_values('Count', ascending=False)
        st.dataframe(genre_full_df, hide_index=True, use_container_width=True)

# Footer
st.markdown("<br>" * 3, unsafe_allow_html=True)
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117;
        text-align: center;
        padding: 10px 20px;
        font-size: 14px;
        color: #fafafa;
        border-top: 1px solid #444;
        z-index: 999;
    }
    .footer a {
        color: #1f77b4;
        text-decoration: none;
        margin: 0 8px;
        font-weight: 500;
    }
    .footer a:hover {
        text-decoration: underline;
        color: #4a9eff;
    }
    .main .block-container {
        padding-bottom: 100px;
    }
    </style>
    <div class="footer">
        🎬 <b>Movie Recommendation System</b> | Top Rated Movies<br>
        Designed for <b>Introduction to Artificial Intelligence</b> course<br>
        Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> | 
        <a href="https://github.com/nothaziq/Movie-Recommendation-System" target="_blank">GitHub Repository</a>
    </div>
""", unsafe_allow_html=True)