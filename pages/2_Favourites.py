import streamlit as st

st.title("Favourite Movies")
st.write("Here are your favourite movies:")

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