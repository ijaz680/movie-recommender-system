# 🎬 Movie Recommender System

A modern **Movie Recommendation System** built with **Python** and **Streamlit** that recommends movies based on genre similarity using **TF-IDF Vectorization** and **Cosine Similarity**. Simply select a movie and discover similar titles through a clean, Netflix-inspired interface.

---

## 📌 Features

* 🎬 Smart movie recommendations
* 🔍 Fast movie search
* 🧠 TF-IDF Vectorization for feature extraction
* 📊 Cosine Similarity recommendation engine
* 🌙 Modern dark-themed Streamlit UI
* 📱 Responsive movie cards
* 🎯 Adjustable number of recommendations (3–10)
* ⚡ Fast and lightweight application

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Streamlit**
* **Pandas**
* **Scikit-learn**
* **NumPy**

---

## 📂 Project Structure

```text
movie-recommender-system/
│
├── .streamlit/
│   └── config.toml
│
├── movies.csv
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ijaz680/movie-recommender-system.git
```

### 2. Navigate to the Project Folder

```bash
cd movie-recommender-system
```

### 3. Create a Virtual Environment (Optional)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run main.py
```

---

## 💻 Usage

1. Launch the Streamlit application.
2. Search or select your favorite movie.
3. Choose the number of recommendations.
4. Click **Get Recommendations**.
5. Explore similar movies instantly.

---

## 📊 Recommendation Method

The recommendation engine works by:

* Cleaning movie genre data
* Converting genres into TF-IDF vectors
* Computing similarity using Cosine Similarity
* Returning the most similar movies based on genre

---

## 📁 Dataset

The project uses a **movies.csv** dataset containing at least the following columns:

| Column | Description                                  |
| ------ | -------------------------------------------- |
| title  | Movie title                                  |
| genres | Movie genres (e.g., Action|Adventure|Sci-Fi) |

Example:

```csv
title,genres
Avatar,Action|Adventure|Sci-Fi
Inception,Action|Sci-Fi|Thriller
Titanic,Drama|Romance
```

---

## 📸 Application Preview

> Add screenshots after running the application.

```text
README.md
images/
    screenshot.png
```

Then add:

```markdown
![Movie Recommender](images/screenshot.png)
```

---

## 🔮 Future Improvements

* Movie posters using TMDB API
* Content-based recommendations using movie overview
* Hybrid recommendation system
* User authentication
* Favorite movies list
* Watchlist support
* Movie ratings integration
* Deploy on Streamlit Cloud

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Ijaz Ur Rahman**

* GitHub: https://github.com/ijaz680

If you found this project helpful, consider giving it a ⭐ on GitHub.
