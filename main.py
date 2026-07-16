import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Config
# ---------------------------
DATA_PATH = r"E:\All DataSets\movies.csv"

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# Cinematic CSS
# ---------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ── */
.stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(180deg, #0a0a0a 0%, #0f0f1a 50%, #0a0a0a 100%);
    color: #e5e5e5;
}

/* ── Hide default elements ── */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Scrollbar ── */
::-webkit-scrollbar {width: 6px;}
::-webkit-scrollbar-track {background: #0a0a0a;}
::-webkit-scrollbar-thumb {background: #333; border-radius: 3px;}
::-webkit-scrollbar-thumb:hover {background: #555;}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 40%, #16213e 70%, #0a0a0a 100%);
    border-radius: 16px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(229, 9, 20, 0.15);
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(229, 9, 20, 0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(245, 197, 24, 0.05) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 3rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #ffffff 0%, #e5e5e5 50%, #f5c518 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem !important;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1.15rem !important;
    color: #888 !important;
    font-weight: 300 !important;
    margin-bottom: 0 !important;
}
.hero .accent {
    color: #e50914;
    font-weight: 600;
}

/* ── Search Section ── */
.search-section {
    background: rgba(26, 26, 46, 0.6);
    border: 1px solid rgba(229, 9, 20, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.section-header h2 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 !important;
}
.section-header .line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(229, 9, 20, 0.5) 0%, transparent 100%);
}
.section-header .count {
    background: rgba(229, 9, 20, 0.15);
    color: #e50914;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── Movie Cards ── */
.movie-card {
    background: linear-gradient(145deg, #141422 0%, #1a1a2e 100%);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    height: 100%;
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #e50914, #f5c518);
    opacity: 0;
    transition: opacity 0.35s ease;
}
.movie-card:hover {
    transform: translateY(-6px);
    border-color: rgba(229, 9, 20, 0.3);
    box-shadow: 0 12px 40px rgba(229, 9, 20, 0.12), 0 4px 15px rgba(0, 0, 0, 0.4);
}
.movie-card:hover::before {
    opacity: 1;
}
.movie-card .title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.6rem;
    line-height: 1.3;
}
.movie-card .score-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: linear-gradient(135deg, rgba(229, 9, 20, 0.2), rgba(245, 197, 24, 0.15));
    color: #f5c518;
    padding: 0.3rem 0.7rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid rgba(245, 197, 24, 0.2);
}
.movie-card .genres {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.7rem;
}
.movie-card .genre-tag {
    background: rgba(255, 255, 255, 0.06);
    color: #aaa;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 500;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.movie-card .rank {
    position: absolute;
    top: 1rem;
    right: 1rem;
    font-size: 2rem;
    font-weight: 900;
    color: rgba(229, 9, 20, 0.12);
    line-height: 1;
}

/* ── Selected Movie Highlight ── */
.selected-movie {
    background: linear-gradient(135deg, rgba(229, 9, 20, 0.1), rgba(26, 26, 46, 0.8));
    border: 1px solid rgba(229, 9, 20, 0.3);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.selected-movie .icon {
    font-size: 3rem;
    filter: drop-shadow(0 0 10px rgba(229, 9, 20, 0.3));
}
.selected-movie .info h3 {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 !important;
}
.selected-movie .info p {
    color: #888 !important;
    font-size: 0.9rem !important;
    margin: 0.25rem 0 0 0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e50914 0%, #b20710 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    box-shadow: 0 6px 25px rgba(229, 9, 20, 0.4) !important;
    transform: translateY(-2px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    background-color: #141422 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}
.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div:focus {
    border-color: rgba(229, 9, 20, 0.5) !important;
    box-shadow: 0 0 0 2px rgba(229, 9, 20, 0.1) !important;
}
.stSlider > div > div > div > div {
    background-color: #e50914 !important;
}

/* ── Footer ── */
.cinema-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #444;
    font-size: 0.8rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 3rem;
}
.cinema-footer span {
    color: #e50914;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #555;
}
.empty-state .icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}
.empty-state h3 {
    color: #666 !important;
    font-weight: 600 !important;
}
.empty-state p {
    color: #444 !important;
}

/* ── Spinner customization ── */
.stSpinner > div {
    border-top-color: #e50914 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)
    return df


@st.cache_resource
def build_similarity(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["genres_clean"])
    return cosine_similarity(tfidf_matrix)


df = load_data()
similarity_matrix = build_similarity(df)
movie_list = sorted(df["title"].tolist())

# ---------------------------
# Hero Banner
# ---------------------------
st.markdown(
    """
<div class="hero">
    <h1>🎬 Movie Recommender</h1>
    <p>Discover your next favorite film. Powered by <span class="accent">genre intelligence</span>.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Search & Controls
# ---------------------------
st.markdown('<div class="search-section">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    search_query = st.text_input("🔍", placeholder="Search for a movie...", label_visibility="collapsed")

if search_query:
    filtered_movies = [m for m in movie_list if search_query.lower() in m.lower()]
else:
    filtered_movies = movie_list

with col2:
    if filtered_movies:
        selected_movie = st.selectbox("Select a movie", filtered_movies, label_visibility="collapsed")
    else:
        st.warning("No movies found matching your search.")
        selected_movie = None

with col3:
    num_recs = st.slider("Recs", 3, 10, 5, label_visibility="collapsed")
    recommend_clicked = st.button("🎬  Get Recommendations", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Results
# ---------------------------
if recommend_clicked and selected_movie:
    idx = df[df["title"] == selected_movie].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = scores[1 : num_recs + 1]

    # Selected movie display
    movie_genres = df.iloc[idx]["genres"].split("|")
    genres_html = "".join(
        f'<span class="genre-tag">{g.strip()}</span>' for g in movie_genres[:5]
    )

    st.markdown(
        f"""
    <div class="selected-movie">
        <div class="icon">🎬</div>
        <div class="info">
            <h3>{selected_movie}</h3>
            <p>Based on genres: {genres_html}</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Section header
    st.markdown(
        f"""
    <div class="section-header">
        <h2>Recommended For You</h2>
        <div class="line"></div>
        <div class="count">{len(top_scores)} movies</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Movie cards grid
    cols = st.columns(3)
    for display_idx, (movie_idx, score) in enumerate(top_scores):
        col = cols[display_idx % 3]
        movie = df.iloc[movie_idx]
        match_pct = round(score * 100)
        movie_genres = movie["genres"].split("|")[:4]
        genres_html = "".join(
            f'<span class="genre-tag">{g.strip()}</span>' for g in movie_genres
        )

        with col:
            st.markdown(
                f"""
            <div class="movie-card">
                <div class="rank">#{display_idx + 1}</div>
                <div class="title">{movie['title']}</div>
                <div class="score-badge">⚡ {match_pct}% match</div>
                <div class="genres">{genres_html}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

elif recommend_clicked and not selected_movie:
    st.warning("Please select a movie first!")

else:
    st.markdown(
        """
    <div class="empty-state">
        <div class="icon">🍿</div>
        <h3>Your next movie awaits</h3>
        <p>Search for a movie above and click "Get Recommendations" to discover similar films.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
<div class="cinema-footer">
    Built with <span>❤</span> using Streamlit &bull; TF-IDF Genre Intelligence
</div>
""",
    unsafe_allow_html=True,
)
