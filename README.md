# 🎬 Movie Recommender

A cinematic movie recommendation app built with Streamlit. Pick a movie you love and discover similar films powered by TF-IDF vectorization and cosine similarity.

## Features

- **Cinematic Dark UI** — Netflix/IMDb-inspired design with gradient backgrounds and glow effects
- **Smart Search** — Instantly filter through thousands of movies
- **Genre-Based Recommendations** — Uses TF-IDF and cosine similarity on movie genres
- **Responsive Card Grid** — Beautiful movie cards with match scores and genre tags
- **Customizable Results** — Choose how many recommendations you want (3–10)

## Tech Stack

- **Python 3.11**
- **Streamlit** — Web framework
- **pandas** — Data manipulation
- **scikit-learn** — TF-IDF vectorization & cosine similarity

## Installation

### Using `uv` (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/movieRecommender.git
cd movieRecommender

# Install dependencies
uv sync

# Run the app
uv run streamlit run main.py
```

### Using `pip`

```bash
# Clone the repository
git clone https://github.com/yourusername/movieRecommender.git
cd movieRecommender

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

## Usage

1. Launch the app with the command above
2. Use the **search bar** in the hero section to find a movie by title
3. Select a movie from the dropdown
4. Adjust the number of recommendations using the slider
5. Click **Get Recommendations** to discover similar movies

## Project Structure

```
movieRecommender/
├── .streamlit/
│   └── config.toml      # Streamlit dark theme config
├── main.py               # Main application
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
├── .python-version       # Python version (3.11)
└── README.md             # This file
```

## Dataset

The app expects a `movies.csv` file with at least the following columns:
- `title` — Movie title
- `genres` — Pipe-separated genres (e.g., `Action|Adventure|Sci-Fi`)

Update the `DATA_PATH` variable in `main.py` to point to your dataset location.

## License

MIT
