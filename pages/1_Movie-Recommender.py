import streamlit as st
import pandas as pd
from MovieRecommendation import HybridRecommender

# Initialize session state for the recommender
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
    st.session_state.loaded = False

# Load model function with full dataset support
@st.cache_resource
def load_recommender(use_full_dataset=True, max_users=50000, max_movies=10000):
    """
    Load and build the recommendation model.
    
    Parameters:
    - use_full_dataset: If True, loads all ratings data
    - max_users: Maximum number of users for collaborative filtering (None = all)
    - max_movies: Maximum number of movies for collaborative filtering (None = all)
    """
    recommender = HybridRecommender(
        memory_efficient=True,
        use_full_dataset=use_full_dataset
    )
    recommender.load_data(data_path='Dataset')
    recommender.build_content_based(max_features=5000)
    recommender.build_collaborative_filtering(
        min_user_ratings=20,
        min_movie_ratings=50,
        max_users=max_users,
        max_movies=max_movies
    )
    return recommender

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    dataset_option = st.radio(
        "Dataset Size:",
        ["Full Dataset (Recommended)", "Medium (Faster)", "Light (Very Fast)"],
        help="Choose based on your system memory"
    )
    
    # Map options to parameters
    config_map = {
        "Full Dataset (Recommended)": {
            "use_full": True,
            "users": 50000,
            "movies": 10000
        },
        "Medium (Faster)": {
            "use_full": True,
            "users": 30000,
            "movies": 5000
        },
        "Light (Very Fast)": {
            "use_full": True,
            "users": 15000,
            "movies": 3000
        }
    }
    
    config = config_map[dataset_option]
    
    st.divider()
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        content_weight = st.slider(
            "Content Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Weight for content-based recommendations"
        )
        collab_weight = 1.0 - content_weight
        st.write(f"Collaborative Weight: {collab_weight:.1f}")
    
    st.divider()
    
    # Dataset info
    st.info(
        f"**Current Config:**\n\n"
        f"👥 Users: {config['users']:,}\n\n"
        f"🎬 Movies: {config['movies']:,}\n\n"
        f"📊 Full Dataset: {'Yes' if config['use_full'] else 'No'}"
    )
st.markdown(
    """
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
        color: #fff;
        border-top: 1px solid #444;
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