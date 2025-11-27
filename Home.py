import sys
import streamlit as st
#import MovieRecommendation as mr

st.title("Movie Recommendation System")
st.subheader("Hybrid Filtering: Content + Collaborative")

st.write("""
Welcome to the Movie Recommendation System!  
Enter a movie title and get personalized recommendations using a hybrid approach:  
- **Content-based filtering** (genres, keywords, descriptions)  
- **Collaborative filtering** (user–movie rating patterns)  
""")

st.markdown("""
**How to use:**
1. Type a movie title in the box below.  
2. Click 'Get Recommendations'.  
3. Explore the suggested movies!  
""")

# st.write("Dataset Preview:")
# st.dataframe(mr.names.head())

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