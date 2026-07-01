import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Step 1: Load data
# ---------------------------
df = pd.read_csv("E:\\All DataSets\\movies.csv")

# ---------------------------
# Step 2: Clean genres (| -> space)
# ---------------------------
df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)

# ---------------------------
# Step 3: TF-IDF vectorization
# ---------------------------
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["genres_clean"])

# ---------------------------
# Step 4: Cosine similarity matrix
# ---------------------------
similarity_matrix = cosine_similarity(tfidf_matrix)

# ---------------------------
# Step 6: Streamlit UI
# ---------------------------
import streamlit as st

# 1. Setup the page layout
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    section[data-testid="stSidebar"] {
        background-color: #1A1D24;
    }
    h1, h2, h3, h4, p, span, label {
        color: #FAFAFA !important;
    }
    div[data-testid="stContainer"] {
        background-color: #1A1D24;
        border-color: #2D3340 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender")
st.write("Search or pick a movie you like, and get similar recommendations.")

# 2. Put inputs in the sidebar to keep the main screen clean
with st.sidebar:
    st.header("⚙️ Settings")
    
    movie_list = df["title"].tolist()
    search_query = st.text_input("Search for a movie:")
    
    # Filter logic
    if search_query:
        filtered_movies = [m for m in movie_list if search_query.lower() in m.lower()]
    else:
        filtered_movies = sorted(movie_list)
        
    # Selection
    if filtered_movies:
        selected_movie = st.selectbox("Choose a movie:", filtered_movies)
    else:
        st.warning("No movies found.")
        selected_movie = None
        
    num_recs = st.slider("How many recommendations?", 3, 10, 5)
    
    # Using type="primary" makes the button stand out natively
    trigger = st.button("Recommend", type="primary", use_container_width=True)

# 3. Main display area for the results
if trigger and selected_movie:
    st.subheader(f"Top picks based on **{selected_movie}**")
    st.divider() # A clean horizontal line
    
    # Calculate scores
    idx = df[df["title"] == selected_movie].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = scores[1:num_recs + 1]
    
    # Create a simple 3-column grid
    cols = st.columns(3)
    
    for display_idx, (movie_idx, score) in enumerate(top_scores):
        col = cols[display_idx % 3] # Distribute cards evenly across the 3 columns
        
        with col:
            # Built-in Streamlit card look
            with st.container(border=True):
                st.markdown(f"#### {df.iloc[movie_idx]['title']}")
                st.caption(f"Match: {round(score, 2)}")