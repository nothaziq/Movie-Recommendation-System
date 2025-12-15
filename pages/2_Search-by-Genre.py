import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Browse by Genre",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .genre-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        cursor: pointer;
        transition: transform 0.3s;
        margin-bottom: 1rem;
    }
    .genre-card:hover {
        transform: scale(1.05);
    }
    .genre-name {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .genre-count {
        font-size: 1rem;
        opacity: 0.9;
    }
    .movie-item {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border-left: 3px solid #667eea;
    }
    .movie-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .movie-info {
        color: #888;
        font-size: 0.9rem;
    }
    .search-box {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎭 Browse Movies by Genre")
st.markdown("### Explore movies across different genres")

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('Dataset/movies.csv')
    ratings = pd.read_csv('Dataset/ratings.csv')
    
    # Calculate average ratings
    movie_ratings = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    movie_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
    
    # Merge
    movies_with_ratings = movies.merge(movie_ratings, on='movieId', how='left')
    movies_with_ratings['avg_rating'] = movies_with_ratings['avg_rating'].fillna(0)
    movies_with_ratings['rating_count'] = movies_with_ratings['rating_count'].fillna(0)
    
    return movies_with_ratings

movies_df = load_data()

# Extract all unique genres
all_genres = set()
for genres in movies_df['genres'].dropna():
    all_genres.update(genres.split('|'))
all_genres = sorted([g for g in all_genres if g != '(no genres listed)'])

# Sidebar - Genre Selection
st.sidebar.header("🎬 Select Genre")

# Search box for genres
genre_search = st.sidebar.text_input("🔍 Search genres:", placeholder="Type to search...")

# Filter genres based on search
if genre_search:
    filtered_genres = [g for g in all_genres if genre_search.lower() in g.lower()]
else:
    filtered_genres = all_genres

# Genre selection
selected_genre = st.sidebar.radio(
    "Choose a genre:",
    options=filtered_genres,
    help="Select a genre to explore movies"
)

st.sidebar.divider()

# Filters
st.sidebar.header("⚙️ Filters")

min_rating = st.sidebar.slider(
    "Minimum average rating:",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5
)

min_rating_count = st.sidebar.slider(
    "Minimum number of ratings:",
    min_value=0,
    max_value=500,
    value=10,
    step=10,
    help="Filter movies with at least this many ratings"
)

sort_by = st.sidebar.selectbox(
    "Sort by:",
    options=["Average Rating", "Number of Ratings", "Title (A-Z)", "Recent First"],
    index=0
)

movies_per_page = st.sidebar.slider(
    "Movies per page:",
    min_value=10,
    max_value=100,
    value=20,
    step=10
)

# Filter movies by selected genre
genre_movies = movies_df[
    movies_df['genres'].str.contains(selected_genre, case=False, na=False)
].copy()

# Apply filters
genre_movies = genre_movies[
    (genre_movies['avg_rating'] >= min_rating) &
    (genre_movies['rating_count'] >= min_rating_count)
]

# Sort movies
if sort_by == "Average Rating":
    genre_movies = genre_movies.sort_values('avg_rating', ascending=False)
elif sort_by == "Number of Ratings":
    genre_movies = genre_movies.sort_values('rating_count', ascending=False)
elif sort_by == "Title (A-Z)":
    genre_movies = genre_movies.sort_values('title')
else:  # Recent First - using movieId as proxy
    genre_movies = genre_movies.sort_values('movieId', ascending=False)

# Display genre info and stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎬 Genre", selected_genre)

with col2:
    st.metric("📊 Total Movies", f"{len(genre_movies):,}")

with col3:
    if len(genre_movies) > 0:
        st.metric("⭐ Avg Rating", f"{genre_movies['avg_rating'].mean():.2f}")
    else:
        st.metric("⭐ Avg Rating", "N/A")

with col4:
    if len(genre_movies) > 0:
        st.metric("📈 Total Ratings", f"{int(genre_movies['rating_count'].sum()):,}")
    else:
        st.metric("📈 Total Ratings", "0")

st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎬 Movies", "📊 Analytics", "🌐 Genre Overview"])

with tab1:
    if len(genre_movies) == 0:
        st.warning(f"No movies found for genre **{selected_genre}** with the current filters. Try adjusting your filters!")
    else:
        # Pagination
        total_pages = (len(genre_movies) - 1) // movies_per_page + 1
        
        if total_pages > 1:
            page = st.number_input(
                f"Page (1-{total_pages}):",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1
            )
        else:
            page = 1
        
        start_idx = (page - 1) * movies_per_page
        end_idx = start_idx + movies_per_page
        page_movies = genre_movies.iloc[start_idx:end_idx]
        
        st.markdown(f"### Showing {start_idx + 1}-{min(end_idx, len(genre_movies))} of {len(genre_movies)} movies")
        
        # Display movies
        for idx, (_, movie) in enumerate(page_movies.iterrows(), start=start_idx + 1):
            col1, col2, col3 = st.columns([0.5, 5, 1.5])
            
            with col1:
                st.markdown(f"**#{idx}**")
            
            with col2:
                st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='movie-info'>{movie['genres']}</div>", unsafe_allow_html=True)
            
            with col3:
                if movie['rating_count'] > 0:
                    st.metric("⭐ Rating", f"{movie['avg_rating']:.2f}")
                    st.caption(f"{int(movie['rating_count'])} ratings")
                else:
                    st.caption("Not rated yet")
            
            st.divider()
        
        # Download button
        csv = genre_movies[['title', 'genres', 'avg_rating', 'rating_count']].to_csv(index=False)
        st.download_button(
            label=f"📥 Download all {len(genre_movies)} {selected_genre} movies as CSV",
            data=csv,
            file_name=f"{selected_genre.lower()}_movies.csv",
            mime="text/csv"
        )

with tab2:
    if len(genre_movies) == 0:
        st.info("No data to display. Try adjusting your filters.")
    else:
        st.markdown("### Genre Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            fig1 = px.histogram(
                genre_movies,
                x='avg_rating',
                nbins=20,
                title=f'Rating Distribution for {selected_genre}',
                labels={'avg_rating': 'Average Rating', 'count': 'Number of Movies'},
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Top 10 movies
            st.markdown(f"#### 🏆 Top 10 {selected_genre} Movies")
            top_10 = genre_movies.nlargest(10, 'avg_rating')[['title', 'avg_rating', 'rating_count']]
            top_10.columns = ['Movie', 'Rating', 'Num Ratings']
            st.dataframe(top_10, hide_index=True, use_container_width=True)
        
        with col2:
            # Rating count distribution
            fig2 = px.scatter(
                genre_movies.head(100),  # Top 100 for readability
                x='rating_count',
                y='avg_rating',
                hover_data=['title'],
                title=f'Rating vs Popularity for {selected_genre}',
                labels={'rating_count': 'Number of Ratings', 'avg_rating': 'Average Rating'},
                color='avg_rating',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Statistics
            st.markdown(f"#### 📊 {selected_genre} Statistics")
            stats_df = pd.DataFrame({
                'Metric': [
                    'Total Movies',
                    'Average Rating',
                    'Median Rating',
                    'Highest Rated',
                    'Most Popular'
                ],
                'Value': [
                    f"{len(genre_movies):,}",
                    f"{genre_movies['avg_rating'].mean():.2f}",
                    f"{genre_movies['avg_rating'].median():.2f}",
                    f"{genre_movies.loc[genre_movies['avg_rating'].idxmax(), 'title']}" if len(genre_movies) > 0 else "N/A",
                    f"{genre_movies.loc[genre_movies['rating_count'].idxmax(), 'title']}" if len(genre_movies) > 0 else "N/A"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### All Genres Overview")
    
    # Calculate stats for each genre
    genre_stats = []
    for genre in all_genres:
        genre_df = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)]
        if len(genre_df) > 0:
            genre_stats.append({
                'Genre': genre,
                'Movies': len(genre_df),
                'Avg Rating': genre_df['avg_rating'].mean(),
                'Total Ratings': genre_df['rating_count'].sum()
            })
    
    genre_stats_df = pd.DataFrame(genre_stats)
    genre_stats_df = genre_stats_df.sort_values('Movies', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Genre popularity chart
        fig3 = px.bar(
            genre_stats_df.head(15),
            x='Movies',
            y='Genre',
            orientation='h',
            title='Top 15 Genres by Number of Movies',
            labels={'Movies': 'Number of Movies', 'Genre': ''},
            color='Movies',
            color_continuous_scale='viridis'
        )
        fig3.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Genre ratings chart
        fig4 = px.bar(
            genre_stats_df.sort_values('Avg Rating', ascending=False).head(15),
            x='Avg Rating',
            y='Genre',
            orientation='h',
            title='Top 15 Genres by Average Rating',
            labels={'Avg Rating': 'Average Rating', 'Genre': ''},
            color='Avg Rating',
            color_continuous_scale='plasma'
        )
        fig4.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Full genre table
    st.markdown("### 📋 Complete Genre Statistics")
    genre_stats_df = genre_stats_df.round(2)
    st.dataframe(
        genre_stats_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Movies": st.column_config.NumberColumn(format="%d"),
            "Avg Rating": st.column_config.NumberColumn(format="%.2f"),
            "Total Ratings": st.column_config.NumberColumn(format="%d")
        }
    )

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
        🎬 <b>Movie Recommendation System</b> | Browse by Genre<br>
        Designed for <b>Introduction to Artificial Intelligence</b> course<br>
        Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> | 
        <a href="https://github.com/nothaziq/Movie-Recommendation-System" target="_blank">GitHub Repository</a>
    </div>
""", unsafe_allow_html=True)