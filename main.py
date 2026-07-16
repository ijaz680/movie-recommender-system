import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

* { font-family: 'DM Sans', sans-serif; }

#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Metric Cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0.25rem 0;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #aaa;
}

/* ── Movie Result Card ── */
.result-card {
    background: #fff;
    border: 1px solid #eee;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: box-shadow 0.2s;
}
.result-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.result-card .rank-badge {
    display: inline-block;
    background: #1a1a2e;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
}
.result-card .movie-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0.3rem 0;
}
.result-card .genre-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.5rem;
}
.result-card .chip {
    background: #f0f0f5;
    color: #555;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 500;
}

/* ── Score Bar ── */
.score-bar-bg {
    background: #eee;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 0.6rem;
}
.score-bar-fill {
    height: 8px;
    border-radius: 6px;
    background: linear-gradient(90deg, #1a1a2e, #4a6cf7);
}

/* ── Selected Movie Banner ── */
.selected-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #2d3a6e 100%);
    border-radius: 12px;
    padding: 2rem;
    color: #fff;
    margin-bottom: 1.5rem;
}
.selected-banner h2 {
    color: #fff !important;
    margin: 0 !important;
    font-size: 1.6rem !important;
}
.selected-banner p {
    color: rgba(255,255,255,0.7) !important;
    margin: 0.3rem 0 0 0 !important;
}
.selected-banner .genre-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.8rem;
}
.selected-banner .chip {
    background: rgba(255,255,255,0.15);
    color: #fff;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Data Functions
# ---------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)
    return df


@st.cache_resource
def build_similarity(_df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(_df["genres_clean"])
    return cosine_similarity(tfidf_matrix)


def get_recommendations(df, sim_matrix, movie_title, n=5):
    idx = df[df["title"] == movie_title].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : n + 1]
    return scores


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/movie.png", width=64)
    st.title("Movie Recommender")
    st.caption("Genre-based recommendation engine")
    st.divider()

    st.markdown("### 📂 Dataset")
    data_path = st.text_input(
        "movies.csv path",
        value="movies.csv",
        help="Full path to your movies.csv file",
    )

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption("Uses TF-IDF vectorization on movie genres to find similar films via cosine similarity.")
    st.caption("CSV must have `title` and `genres` columns.")

# ---------------------------
# Load Data
# ---------------------------
df = None
similarity_matrix = None
movie_list = []

if data_path and os.path.isfile(data_path):
    try:
        df = load_data(data_path)
        similarity_matrix = build_similarity(df)
        movie_list = sorted(df["title"].tolist())
        all_genres = sorted(set(g.strip() for gs in df["genres"] for g in gs.split("|")))
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.warning("Enter a valid path to `movies.csv` in the sidebar to get started.")

# ---------------------------
# Main App (Tabs)
# ---------------------------
if df is not None:
    tab_discover, tab_analytics, tab_browse = st.tabs(["🎬 Discover", "📊 Analytics", "📋 Browse Data"])

    # ═══════════════════════════════
    # TAB 1: DISCOVER
    # ═══════════════════════════════
    with tab_discover:
        st.subheader("Find Movies Like Your Favorites")

        col_search, col_count = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("🔍 Search movies", placeholder="Type a movie name...")
        with col_count:
            num_recs = st.slider("How many?", 3, 10, 5)

        if search_query:
            filtered_movies = [m for m in movie_list if search_query.lower() in m.lower()]
        else:
            filtered_movies = movie_list

        if filtered_movies:
            selected_movie = st.selectbox("Pick a movie", filtered_movies, index=0)
        else:
            st.info("No movies match your search.")
            selected_movie = None

        if selected_movie:
            idx = df[df["title"] == selected_movie].index[0]
            movie_genres = df.iloc[idx]["genres"].split("|")
            genres_html = "".join(f'<span class="chip">{g.strip()}</span>' for g in movie_genres)

            st.markdown(
                f"""
            <div class="selected-banner">
                <h2>🎬 {selected_movie}</h2>
                <p>Genres</p>
                <div class="genre-chips">{genres_html}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if st.button("Get Recommendations", type="primary", use_container_width=True) and selected_movie:
            scores = get_recommendations(df, similarity_matrix, selected_movie, num_recs)

            st.markdown(f"### Top {len(scores)} Similar Movies")

            for rank, (movie_idx, score) in enumerate(scores, 1):
                movie = df.iloc[movie_idx]
                match_pct = round(score * 100)
                movie_genres = movie["genres"].split("|")[:5]
                genres_html = "".join(f'<span class="chip">{g.strip()}</span>' for g in movie_genres)

                st.markdown(
                    f"""
                <div class="result-card">
                    <span class="rank-badge">#{rank}</span>
                    <div class="movie-title">{movie['title']}</div>
                    <div class="genre-chips">{genres_html}</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width: {match_pct}%;"></div>
                    </div>
                    <div style="text-align:right; font-size:0.8rem; color:#888; margin-top:0.3rem;">
                        {match_pct}% match
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        elif selected_movie is None:
            st.info("Select a movie above to get recommendations.")

    # ═══════════════════════════════
    # TAB 2: ANALYTICS
    # ═══════════════════════════════
    with tab_analytics:
        st.subheader("Dataset Overview")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Movies", f"{len(df):,}")
        m2.metric("Unique Genres", len(all_genres))
        m3.metric("Avg Genres/Movie", round(df["genres"].str.split("|").str.len().mean(), 1))

        st.divider()
        st.markdown("### Genre Distribution")

        genre_counts = (
            df["genres"].str.split("|").explode().str.strip().value_counts().head(20)
        )

        st.bar_chart(genre_counts, horizontal=True, height=400, color="#1a1a2e")

        st.divider()
        st.markdown("### Top 10 Most Common Genre Combinations")
        top_combos = df["genres"].value_counts().head(10)
        for combo, count in top_combos.items():
            st.markdown(f"**{combo}** — {count} movies")

    # ═══════════════════════════════
    # TAB 3: BROWSE DATA
    # ═══════════════════════════════
    with tab_browse:
        st.subheader("Full Dataset")

        browse_search = st.text_input("🔍 Filter by title", placeholder="Search...", key="browse_search")
        if browse_search:
            display_df = df[df["title"].str.contains(browse_search, case=False, na=False)]
        else:
            display_df = df

        st.dataframe(
            display_df[["title", "genres"]].reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

        st.caption(f"Showing {len(display_df)} of {len(df)} movies")

else:
    # No data loaded — show empty state
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://img.icons8.com/fluency/150/movie.png", width=100)
        st.markdown("### Movie Recommender")
        st.info("Configure your dataset path in the **sidebar** to get started.")
        st.markdown("""
        **Requirements:**
        - A `movies.csv` file with at least:
          - `title` — Movie name
          - `genres` — Pipe-separated genres (e.g., `Action|Adventure|Sci-Fi`)
        """)
