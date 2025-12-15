import streamlit as st
import pandas as pd
from MovieRecommendation import HybridRecommender

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

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

# Load the model
if not st.session_state.loaded:
    with st.spinner('🎬 Loading recommendation model... This may take a few minutes...'):
        try:
            st.session_state.recommender = load_recommender(
                use_full_dataset=config['use_full'],
                max_users=config['users'],
                max_movies=config['movies']
            )
            st.session_state.loaded = True
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.stop()

recommender = st.session_state.recommender

# Main content
st.title("🎬 Movie Recommendation System")
st.markdown("### Find movies similar to your favorites!")

# Create two columns for input
col1, col2 = st.columns([2, 1])

with col1:
    movie_title = st.text_input(
        "🔍 Enter a movie title:",
        placeholder="e.g., Inception, The Matrix, Toy Story...",
        help="Type part or full movie name"
    )

with col2:
    recommendation_type = st.selectbox(
        "📊 Recommendation Type:",
        ["Hybrid", "Content-based", "Collaborative"],
        help="Hybrid uses both content and user ratings"
    )

# Additional options
col3, col4 = st.columns(2)

with col3:
    num_recommendations = st.slider(
        "📝 Number of Recommendations:",
        min_value=1,
        max_value=20,
        value=10,
        help="How many movies to recommend"
    )

with col4:
    show_scores = st.checkbox("Show Similarity Scores", value=True)

# Recommend button
if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
    if movie_title.strip() == "":
        st.warning("⚠️ Please enter a movie title.")
    else:
        # Map selection to method parameter
        method_map = {
            "Hybrid": "hybrid",
            "Collaborative": "collaborative",
            "Content-based": "content"
        }
        method = method_map[recommendation_type]
        
        # Get recommendations
        with st.spinner(f"Finding movies similar to '{movie_title}'..."):
            recommendations = recommender.get_recommendations_by_title(
                title=movie_title,
                n=num_recommendations,
                method=method,
                content_weight=content_weight if method == "hybrid" else 0.5,
                collab_weight=collab_weight if method == "hybrid" else 0.5
            )
        
        if recommendations.empty:
            st.error(f"❌ No movies found matching: **{movie_title}**")
            st.info("💡 Try searching with different keywords or check the spelling.")
        else:
            # Show the matched movie
            st.success(f"✅ Showing recommendations based on: **{movie_title}**")
            
            st.divider()
            st.subheader("🎯 Recommended Movies:")
            
            # Display recommendations in a nice format
            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                with st.container():
                    col_num, col_info, col_rating, col_score = st.columns([0.5, 3, 1, 1])
                    
                    with col_num:
                        st.markdown(f"### {idx}")
                    
                    with col_info:
                        st.markdown(f"### {row['title']}")
                        st.markdown(f"*{row['genres']}*")
                    
                    with col_rating:
                        rating = row['avg_rating']
                        if rating > 0:
                            st.metric("⭐ Rating", f"{rating:.2f}/5.0")
                        else:
                            st.metric("⭐ Rating", "N/A")
                    
                    with col_score:
                        if show_scores:
                            if 'hybrid_score' in row and pd.notna(row['hybrid_score']):
                                st.metric("🎯 Score", f"{row['hybrid_score']:.3f}")
                            elif 'similarity_score' in row and pd.notna(row['similarity_score']):
                                st.metric("🎯 Score", f"{row['similarity_score']:.3f}")
                    
                    st.divider()
            
            # Download recommendations
            csv = recommendations.to_csv(index=False)
            st.download_button(
                label="📥 Download Recommendations as CSV",
                data=csv,
                file_name=f"recommendations_{movie_title.replace(' ', '_')}.csv",
                mime="text/csv"
            )

# Add some spacing before footer
st.markdown("<br>" * 3, unsafe_allow_html=True)

# Footer with statistics
if st.session_state.loaded and recommender is not None:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Movies", f"{len(recommender.movies_df):,}")
    
    with col2:
        if recommender.ratings_df is not None:
            st.metric("👥 Total Users", f"{recommender.ratings_df['userId'].nunique():,}")
    
    with col3:
        if recommender.ratings_df is not None:
            st.metric("⭐ Total Ratings", f"{len(recommender.ratings_df):,}")
    
    with col4:
        if recommender.user_item_matrix is not None:
            st.metric("🎬 CF Movies", f"{len(recommender.user_item_matrix):,}")


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