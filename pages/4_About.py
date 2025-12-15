import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="About - Movie Recommender",
    page_icon="📖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .algorithm-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .algorithm-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .feature-list {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .tech-badge {
        background-color: #667eea;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
    .team-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 3rem; margin-bottom: 0.5rem;">📖 About This Project</h1>
        <p style="color: white; font-size: 1.2rem; opacity: 0.9;">AI-Powered Hybrid Movie Recommendation System</p>
    </div>
""", unsafe_allow_html=True)

# Project Overview
st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This project implements a **Hybrid Movie Recommendation System** that combines two powerful 
    approaches to suggest movies you'll love:
    
    - **Content-Based Filtering**: Analyzes movie characteristics (genres, tags, themes)
    - **Collaborative Filtering**: Learns from user rating patterns
    - **Hybrid Approach**: Combines both methods for superior recommendations
    
    Built for the **Introduction to Artificial Intelligence** course, this system demonstrates 
    practical applications of machine learning, natural language processing, and data analysis.
    
    ### 🎓 Course Information
    - **Course**: Introduction to Artificial Intelligence
    - **Institution**: Air University
    - **Semester**: Fall 2024
    - **Instructor**: Ma'am Aisha Sattar
    """)

with col2:
    # Load dataset stats
    try:
        movies = pd.read_csv('Dataset/movies.csv')
        ratings = pd.read_csv('Dataset/ratings.csv')
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("📊 Total Movies", f"{len(movies):,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("👥 Total Users", f"{ratings['userId'].nunique():,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("⭐ Total Ratings", f"{len(ratings):,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        avg_rating = ratings['rating'].mean()
        st.metric("📈 Avg Rating", f"{avg_rating:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    except:
        st.info("Dataset statistics unavailable")

# System Architecture
st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)

st.markdown("""
Our recommendation system uses a three-stage pipeline:

1. **Data Processing**: Load and preprocess MovieLens dataset
2. **Model Building**: Train content-based and collaborative filtering models
3. **Hybrid Recommendation**: Combine predictions with weighted averaging
""")

# Architecture diagram (text-based)
st.code("""
┌─────────────────┐
│  MovieLens Data │
│ (Movies+Ratings)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Data Preprocessing              │
│  • Clean & aggregate ratings        │
│  • Extract genres & tags            │
│  • Build user-item matrix           │
└────────┬────────────────────────────┘
         │
         ▼
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│Content- │ │Collaborative │
│ Based   │ │  Filtering   │
│TF-IDF + │ │   KNN +      │
│Cosine   │ │   Cosine     │
└────┬────┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            ▼
    ┌───────────────┐
    │Hybrid System  │
    │Weighted Merge │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │Recommendations│
    └───────────────┘
""", language="text")

# Algorithms Explained
st.markdown('<div class="section-header">🤖 Algorithms Explained</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Content-Based Filtering", "Collaborative Filtering", "Hybrid System"])

with tab1:
    st.markdown("""
    ### 📚 Content-Based Filtering
    
    This approach recommends movies based on their **content characteristics**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### How It Works:
        
        1. **Feature Extraction**
           - Extract movie genres (Action, Drama, etc.)
           - Collect user-generated tags
           - Combine into text description
        
        2. **TF-IDF Vectorization**
           - Convert text to numerical vectors
           - Weight words by importance
           - Create 5000-dimensional feature space
        
        3. **Similarity Calculation**
           - Use cosine similarity
           - Compare movie vectors
           - Find most similar movies
        """)
    
    with col2:
        st.markdown("""
        #### Technical Details:
        
        ```python
        TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        ```
        
        **Advantages:**
        - ✅ Works for new users
        - ✅ No cold start problem
        - ✅ Explainable results
        
        **Limitations:**
        - ❌ Limited diversity
        - ❌ Requires good metadata
        - ❌ No user preference learning
        """)
    
    # Example visualization
    st.markdown("#### 📊 Example: Finding Similar Movies")
    st.info("""
    **Input Movie:** The Matrix (Action, Sci-Fi)
    
    **Content Features Extracted:**
    - Genres: Action, Sci-Fi
    - Tags: dystopia, virtual reality, philosophy, cyberpunk
    
    **Similar Movies Found:**
    1. Inception (Action, Sci-Fi) - Similarity: 0.92
    2. Blade Runner (Action, Sci-Fi) - Similarity: 0.89
    3. The Terminator (Action, Sci-Fi) - Similarity: 0.85
    """)

with tab2:
    st.markdown("""
    ### 👥 Collaborative Filtering
    
    This approach learns from **user behavior patterns** to make predictions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### How It Works:
        
        1. **User-Item Matrix Creation**
           - Build matrix of users × movies
           - Fill with rating values
           - Use sparse format (most cells empty)
        
        2. **K-Nearest Neighbors (KNN)**
           - Find movies with similar rating patterns
           - Use cosine distance metric
           - Identify top K neighbors (K=50)
        
        3. **Prediction**
           - Aggregate ratings from similar movies
           - Weight by similarity scores
           - Generate recommendations
        """)
    
    with col2:
        st.markdown("""
        #### Technical Details:
        
        ```python
        NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=50,
            n_jobs=-1
        )
        ```
        
        **Advantages:**
        - ✅ Learns user preferences
        - ✅ Discovers unexpected gems
        - ✅ Community-driven insights
        
        **Limitations:**
        - ❌ Cold start for new items
        - ❌ Requires rating history
        - ❌ Sparsity challenges
        """)
    
    # Example visualization
    st.markdown("#### 📊 Example: Pattern-Based Recommendations")
    st.info("""
    **Scenario:** You loved "Inception"
    
    **System Analysis:**
    - Finds users who also rated Inception highly
    - Looks at what else they enjoyed
    - Discovers patterns: users who liked Inception also loved Interstellar (95% overlap)
    
    **Recommendations Based on User Patterns:**
    1. Interstellar - Many Inception fans rated 5/5
    2. The Prestige - Same director, similar audience
    3. Arrival - Sci-fi fans with similar taste profile
    """)

with tab3:
    st.markdown("""
    ### 🎯 Hybrid Approach
    
    Combines the **best of both worlds** for superior recommendations.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### How It Works:
        
        1. **Parallel Processing**
           - Run content-based filtering
           - Run collaborative filtering
           - Get recommendations from both
        
        2. **Score Normalization**
           - Normalize similarity scores (0-1)
           - Apply configurable weights
           - Default: 50% content + 50% collaborative
        
        3. **Intelligent Merging**
           - Combine unique movies from both lists
           - Calculate weighted hybrid scores
           - Rank by final scores
        """)
    
    with col2:
        st.markdown("""
        #### Why Hybrid is Better:
        
        **Synergy Benefits:**
        - 📈 More accurate predictions
        - 🎨 Diverse recommendations
        - 🎯 Balanced approach
        - 🔧 Configurable weights
        
        **Formula:**
        ```
        hybrid_score = 
            (w₁ × content_score) + 
            (w₂ × collab_score)
        
        where w₁ + w₂ = 1
        ```
        
        **Adjustable in UI:**
        Users can change weights based on preference!
        """)
    
    # Comparison visualization
    st.markdown("#### 📊 Algorithm Comparison")
    
    comparison_df = pd.DataFrame({
        'Feature': ['Accuracy', 'Diversity', 'Cold Start', 'Explainability', 'Scalability'],
        'Content-Based': ['⭐⭐⭐', '⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐'],
        'Collaborative': ['⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐', '⭐⭐', '⭐⭐⭐'],
        'Hybrid': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐']
    })
    
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

# Technology Stack
st.markdown('<div class="section-header">💻 Technology Stack</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Backend & ML")
    st.markdown("""
    <span class="tech-badge">Python 3.10+</span>
    <span class="tech-badge">Pandas</span>
    <span class="tech-badge">NumPy</span>
    <span class="tech-badge">scikit-learn</span>
    <span class="tech-badge">SciPy</span>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Frontend & UI")
    st.markdown("""
    <span class="tech-badge">Streamlit</span>
    <span class="tech-badge">Plotly</span>
    <span class="tech-badge">HTML/CSS</span>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("### Data & Tools")
    st.markdown("""
    <span class="tech-badge">MovieLens Dataset</span>
    <span class="tech-badge">Git/GitHub</span>
    <span class="tech-badge">VS Code</span>
    """, unsafe_allow_html=True)

# Dataset Information
st.markdown('<div class="section-header">📊 Dataset Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### MovieLens 32M Dataset
    
    Our system uses the MovieLens 32M dataset, a widely-used benchmark for recommendation systems.
    
    **Contents:**
    - 📽️ **Movies**: ~87,500 movies with titles and genres
    - ⭐ **Ratings**: ~32 million ratings from users
    - 👥 **Users**: ~200,000 unique users
    - 🏷️ **Tags**: User-generated keywords and themes
    - 🔗 **Links**: IMDb and TMDb identifiers
    
    **Rating Scale:** 0.5 to 5.0 stars (0.5 increments)
    
    **Time Period:** 1995-2015
    """)

with col2:
    st.markdown("""
    ### Data Processing Pipeline
    
    1. **Loading**
       - Read CSV files efficiently
       - Use optimized data types (int32, float32)
       - Handle missing values
    
    2. **Aggregation**
       - Calculate average ratings per movie
       - Count total ratings for popularity
       - Combine tags into text features
    
    3. **Filtering**
       - Remove movies with too few ratings
       - Filter inactive users
       - Clean and normalize text
    
    4. **Optimization**
       - Sparse matrix storage
       - Memory-efficient processing
       - Strategic caching
    """)

# Features
st.markdown('<div class="section-header">✨ Key Features</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-list">
        <h3>🎯 Recommendation Features</h3>
        <ul>
            <li>Three recommendation modes (Hybrid, Content, Collaborative)</li>
            <li>Adjustable algorithm weights</li>
            <li>Title-based movie search</li>
            <li>Similar movie discovery</li>
            <li>Personalized suggestions</li>
        </ul>
    </div>
    
    <div class="feature-list">
        <h3>🔍 Search & Browse</h3>
        <ul>
            <li>Browse by genre</li>
            <li>Top-rated movies</li>
            <li>Advanced filtering options</li>
            <li>Sort by multiple criteria</li>
            <li>Pagination for large results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-list">
        <h3>📊 Analytics & Visualization</h3>
        <ul>
            <li>Interactive Plotly charts</li>
            <li>Rating distributions</li>
            <li>Genre analytics</li>
            <li>Real-time statistics</li>
            <li>Comparison visualizations</li>
        </ul>
    </div>
    
    <div class="feature-list">
        <h3>⚙️ Technical Features</h3>
        <ul>
            <li>Memory-efficient processing</li>
            <li>Full dataset support (20M+ ratings)</li>
            <li>Configurable performance settings</li>
            <li>CSV export functionality</li>
            <li>Responsive design</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Performance & Optimization
st.markdown('<div class="section-header">⚡ Performance & Optimization</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Memory Optimization
    - Sparse matrix storage (CSR format)
    - Efficient data types (int32/float32)
    - Strategic garbage collection
    - Configurable dataset size
    - Selective feature loading
    """)

with col2:
    st.markdown("""
    ### Computational Efficiency
    - Vectorized operations
    - Parallel processing (n_jobs=-1)
    - Cached computations
    - Batch processing
    - Optimized algorithms
    """)

with col3:
    st.markdown("""
    ### User Experience
    - Fast page loads (<2s)
    - Smooth interactions
    - Loading indicators
    - Progressive rendering
    - Responsive design
    """)

# Team Information
st.markdown('<div class="section-header">👥 Team Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="team-card">
        <h3>🎓 Muhammad Haziq Shahid</h3>
        <p>Student ID: 242242</p>
        <p>Email: 242242@students.au.edu.pk</p>
        <p><strong>Role:</strong> Lead Developer</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="team-card">
        <h3>🎓 Hareem Shakeel</h3>
        <p>Student ID: 242204</p>
        <p>Email: 242204@students.au.edu.pk</p>
        <p><strong>Role:</strong> UI/UX Design</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="team-card">
        <h3>🎓 Malik Muhammad Uzaif</h3>
        <p>Student ID: 242178</p>
        <p>Email: 242178@students.au.edu.pk</p>
        <p><strong>Role:</strong> Algorithm Design</p>
    </div>
    """, unsafe_allow_html=True)

# References
st.markdown('<div class="section-header">📚 References & Resources</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Academic Papers
    
    1. **Collaborative Filtering**
       - Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems"
    
    2. **Content-Based Filtering**
       - Lops, P., de Gemmis, M., & Semeraro, G. (2011). "Content-based Recommender Systems"
    
    3. **Hybrid Systems**
       - Burke, R. (2002). "Hybrid Recommender Systems: Survey and Experiments"
    
    4. **TF-IDF**
       - Salton, G., & McGill, M. J. (1986). "Introduction to Modern Information Retrieval"
    """)

with col2:
    st.markdown("""
    ### Tools & Libraries
    
    - **scikit-learn**: Pedregosa et al. (2011)
    - **Streamlit**: Streamlit Inc. (2024)
    - **Plotly**: Plotly Technologies Inc.
    - **MovieLens Dataset**: GroupLens Research
    
    ### Online Resources
    
    - [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
    - [scikit-learn Documentation](https://scikit-learn.org/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [GitHub Repository](https://github.com/nothaziq/Movie-Recommendation-System)
    """)

# Acknowledgments
st.markdown('<div class="section-header">🙏 Acknowledgments</div>', unsafe_allow_html=True)

st.markdown("""
We would like to thank:

- **Ma'am Aisha Sattar** for guidance and support throughout this project
- **GroupLens Research** for providing the MovieLens dataset
- **The open-source community** for excellent tools and libraries

This project was developed as part of the Introduction to Artificial Intelligence course at Air University Islamabad.
""")

# Footer
st.markdown("<br>" * 2, unsafe_allow_html=True)
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
        🎬 <b>Movie Recommendation System</b> | About & Documentation<br>
        Designed for <b>Introduction to Artificial Intelligence</b> course<br>
        Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> | 
        <a href="https://scikit-learn.org" target="_blank">scikit-learn</a> | 
        <a href="https://github.com/nothaziq/Movie-Recommendation-System" target="_blank">GitHub Repository</a>
    </div>
""", unsafe_allow_html=True)