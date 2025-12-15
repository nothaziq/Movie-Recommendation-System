
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Movie Recommender - Home",
    page_icon="🎬",
    layout="wide"
)

# Hero section
st.markdown("""
    <style>
    .hero-section {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
    }
    .feature-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# Hero banner
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🎬 Movie Recommendation System</div>
        <div class="hero-subtitle">Discover your next favorite movie with AI-powered recommendations</div>
    </div>
""", unsafe_allow_html=True)

# Quick stats
try:
    movies = pd.read_csv('Dataset/movies.csv')
    ratings = pd.read_csv('Dataset/ratings.csv')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(movies):,}</div>
                <div class="stat-label">Movies</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{ratings['userId'].nunique():,}</div>
                <div class="stat-label">Users</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(ratings):,}</div>
                <div class="stat-label">Ratings</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Extract unique genres
        all_genres = movies['genres'].str.split('|').explode().unique()
        genre_count = len([g for g in all_genres if g and g != '(no genres listed)'])
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{genre_count}</div>
                <div class="stat-label">Genres</div>
            </div>
        """, unsafe_allow_html=True)
except:
    pass

st.markdown("<br>", unsafe_allow_html=True)

# Features section
st.markdown("## 🚀 Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 Hybrid Recommendations</div>
            <p>Combines content-based and collaborative filtering for the most accurate suggestions. 
            Get recommendations based on both movie content and what similar users enjoyed!</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔍 Smart Search</div>
            <p>Search by movie title or browse by genre. Our intelligent search finds movies 
            even with partial titles or misspellings.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-title">⭐ Top Rated Movies</div>
            <p>Discover the highest-rated movies in our database. Filter by genre to find 
            the best movies in categories you love.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎨 Genre Explorer</div>
            <p>Browse movies by genre and discover hidden gems. Perfect for when you know 
            what mood you're in but need inspiration!</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# How it works
st.markdown("## 🤖 How It Works")

tab1, tab2, tab3 = st.tabs(["Content-Based Filtering", "Collaborative Filtering", "Hybrid Approach"])

with tab1:
    st.markdown("""
        ### 📚 Content-Based Filtering
        
        This method analyzes the **content** of movies to find similarities:
        
        - **Genres**: Action, Drama, Comedy, etc.
        - **Tags**: User-generated keywords and themes
        - **Movie characteristics**: Style, tone, and themes
        
        **Example**: If you like "The Matrix" (Sci-Fi, Action), you'll get recommendations 
        like "Inception" or "Blade Runner" - movies with similar genres and themes.
        
        **Pros**: ✅ Works for new users, doesn't need rating history  
        **Cons**: ❌ Limited to finding similar content
    """)

with tab2:
    st.markdown("""
        ### 👥 Collaborative Filtering
        
        This method uses **user behavior** to make predictions:
        
        - Analyzes rating patterns across all users
        - Finds users with similar tastes to you
        - Recommends movies those similar users enjoyed
        
        **Example**: If you and another user both loved "Inception" and "Interstellar", 
        and they also loved "Arrival", you'll likely enjoy "Arrival" too!
        
        **Pros**: ✅ Discovers unexpected gems, learns from community  
        **Cons**: ❌ Needs rating data, cold start problem
    """)

with tab3:
    st.markdown("""
        ### 🎯 Hybrid Approach (Best of Both Worlds!)
        
        Our system **combines both methods** for superior recommendations:
        
        1. **Content-based** finds movies with similar characteristics
        2. **Collaborative** finds movies similar users enjoyed
        3. Results are **weighted and merged** for optimal suggestions
        
        **Benefits**:
        - ✅ More diverse recommendations
        - ✅ Better accuracy than either method alone
        - ✅ Handles various use cases
        - ✅ Balances novelty and relevance
        
        You can even adjust the balance between content and collaborative filtering 
        in the advanced settings!
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Getting started
st.markdown("## 🎬 Get Started")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 1️⃣ Choose a Method\nHead to the **Movie Recommender** page to search by title")

with col2:
    st.success("### 2️⃣ Browse Genres\nExplore the **Browse by Genre** page to discover new movies")

with col3:
    st.warning("### 3️⃣ Check Top Rated\nSee what's popular on the **Top Rated Movies** page")

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
        🎬 <b>Movie Recommendation System</b> | Hybrid Filtering Edition<br>
        Designed for <b>Introduction to Artificial Intelligence</b> course<br>
        Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> | 
        <a href="https://scikit-learn.org" target="_blank">scikit-learn</a> | 
        <a href="https://github.com/nothaziq/Movie-Recommendation-System" target="_blank">GitHub Repository</a>
    </div>
""", unsafe_allow_html=True)