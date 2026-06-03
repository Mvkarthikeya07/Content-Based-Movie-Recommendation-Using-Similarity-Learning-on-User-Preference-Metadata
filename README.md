<div align="center">

<h1>🎬 Movie Recommendation System</h1>
<h3>Content-Based Filtering Using Cosine Similarity on User–Movie Rating Metadata</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Cosine%20Similarity-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-MovieLens-E50914?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Movies-9%2C742-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Ratings-100%2C836-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Approach-Item--Based%20Collaborative%20Filtering-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Published-IJIREEICE-red?style=flat-square"/>
</p>

> A **real-time movie recommendation engine** built on the MovieLens dataset — 9,742 movies, 100,836 ratings, and cosine similarity across a full user–movie matrix, deployed as an interactive Flask web app with fuzzy title matching.

</div>

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Live Screenshots](#-live-screenshots)
- [System Architecture](#-system-architecture)
- [How the Recommendation Engine Works](#-how-the-recommendation-engine-works)
- [Dataset Details](#-dataset-details)
- [Model Comparison](#-model-comparison)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Routes](#-api-routes)
- [Engineering Highlights](#-engineering-highlights)
- [Research Publication](#-research-publication)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Problem Statement

With thousands of movies across every genre, users face a discovery problem — finding a film similar to one they loved is either left to chance or locked behind opaque, black-box streaming algorithms. This project builds a **transparent, explainable recommendation engine** grounded in mathematical similarity on real rating behaviour data.

---

## 💡 Solution Overview

The system constructs a **user–movie rating matrix** from 100,836 real ratings, transposes it to compute **item–item cosine similarity** across all 9,742 movies, and returns the top-N most similar titles for any input — in real time, through a clean Flask web interface. Fuzzy title matching via `difflib.get_close_matches` handles spelling variations and partial queries gracefully.

---

## 🖥️ Live Screenshots

### 🏠 Home Page — Movie Search Interface

<img width="1366" height="768" alt="Home Page" src="https://github.com/user-attachments/assets/fce18394-5089-436f-92ce-4a7fdb40a58c"/>

> Users enter any movie title into the search bar. The fuzzy matcher resolves partial or misspelled input against the 9,742-title catalog automatically.

---

### 🎥 Recommendation Results Page

<img width="1366" height="768" alt="Recommendations Page" src="https://github.com/user-attachments/assets/89c5578c-d082-4f6d-88e6-805ad5f1c56e"/>

> The engine returns the top-5 most similar movies ranked by descending cosine similarity score, displayed in a responsive card grid layout.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (Client)                          │
│              index.html  →  POST /recommend  →  recommend.html   │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTP POST (movie_name)
┌─────────────────────────────▼────────────────────────────────────┐
│                     Flask Application (app.py)                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              recommend_movies(movie_name, num=5)         │    │
│  │                                                         │    │
│  │  1. get_close_matches() → fuzzy title resolution        │    │
│  │  2. movie_similarity_df[best_match]                     │    │
│  │  3. sort_values(ascending=False).iloc[1:num+1]          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                  Similarity Matrix (in-memory)                    │
│                                                                  │
│  ratings.csv + movies.csv                                        │
│       │                                                          │
│       ▼                                                          │
│  pivot_table(index=userId, columns=title, values=rating)         │
│  fillna(0)   →   shape: [610 users × 9,742 movies]              │
│       │                                                          │
│       ▼                                                          │
│  cosine_similarity(ratings_matrix.T)                             │
│  shape: [9,742 × 9,742]   ←  movie_similarity_df               │
└──────────────────────────────────────────────────────────────────┘
```

**Everything loads at startup** — the similarity matrix is computed once and held in memory. Every subsequent recommendation request is an O(1) column lookup + sort — sub-millisecond inference per query.

---

## 🔄 How the Recommendation Engine Works

### Step 1 — Data Ingestion & Merge

```python
movies  = pd.read_csv('movies.csv')   # movieId, title, genres
ratings = pd.read_csv('ratings.csv')  # userId, movieId, rating, timestamp

movie_data = pd.merge(ratings, movies, on='movieId')
```

### Step 2 — User–Movie Rating Matrix

```python
ratings_matrix = movie_data.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)
ratings_matrix.fillna(0, inplace=True)
# Shape: 610 users × 9,742 movies
```

| | Toy Story (1995) | Jumanji (1995) | Heat (1995) | ... |
|---|---|---|---|---|
| **User 1** | 4.0 | 0.0 | 0.0 | ... |
| **User 2** | 0.0 | 3.0 | 5.0 | ... |
| **User 3** | 5.0 | 0.0 | 4.0 | ... |

### Step 3 — Item–Item Cosine Similarity

```python
movie_similarity = cosine_similarity(ratings_matrix.T)
# ratings_matrix.T shape: 9,742 movies × 610 users
# movie_similarity shape: 9,742 × 9,742
movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=ratings_matrix.columns,
    columns=ratings_matrix.columns
)
```

**Why cosine similarity?** Each movie is a vector of 610 user ratings. Cosine similarity measures the angle between two vectors — capturing shared rating patterns regardless of scale. Two movies rated similarly by the same users will have a score close to 1.0.

### Step 4 — Fuzzy Title Matching + Top-N Lookup

```python
def recommend_movies(movie_name, num=5):
    # Fuzzy match against all 9,742 titles (handles typos, partial input)
    close_matches = get_close_matches(movie_name, movie_list, n=1, cutoff=0.3)
    best_match = close_matches[0]

    # Column lookup + sort — O(n log n) on 9,742 entries
    similar_scores = movie_similarity_df[best_match].sort_values(ascending=False)
    return list(similar_scores.iloc[1:num + 1].index)  # Skip self (score = 1.0)
```

---

## 📊 Dataset Details

Source: **MovieLens (ml-latest-small)** by GroupLens Research, University of Minnesota

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `movies.csv` | 9,742 | `movieId`, `title`, `genres` | Full movie catalog with pipe-delimited genres |
| `ratings.csv` | 100,836 | `userId`, `movieId`, `rating`, `timestamp` | Real user ratings (0.5 – 5.0 stars) |
| `train.csv` | — | Same as ratings | Offline training split used by `model.py` |

**Key stats:**
- **610** unique users
- **9,742** unique movies
- **100,836** ratings (avg ~165 ratings per user)
- Rating scale: **0.5 to 5.0** in 0.5 increments
- Matrix sparsity: **~98.3%** (most users rated a small fraction of the catalog)

---

## ⚖️ Model Comparison

The research paper accompanying this project (**IJIREEICE**) benchmarks three core recommendation approaches. This table summarizes the comparative analysis:

| Approach | Technique | Similarity Metric | Cold Start | Scalability | Interpretability | This Project |
|---|---|---|---|---|---|---|
| **User-Based CF** | Find users similar to the query user; recommend what they liked | Cosine / Pearson on user vectors | ❌ Severe (new users) | ⚠️ Poor (user matrix grows) | ✅ High | ❌ |
| **Item-Based CF** | Find movies similar to the input movie; recommend those | **Cosine on item vectors** | ✅ Mild (movie-side) | ✅ Good (stable item matrix) | ✅ High | ✅ **Used** |
| **Content-Based** | Recommend based on movie metadata (genre, cast, plot) | TF-IDF + Cosine on text | ✅ None | ✅ Excellent | ✅ High | ❌ |
| **Matrix Factorization (SVD)** | Decompose rating matrix into latent user/item factors | Dot product in latent space | ⚠️ Moderate | ✅ Excellent | ❌ Low | ❌ |
| **Deep Learning (NCF)** | Neural collaborative filtering with embeddings | Learned similarity | ⚠️ Moderate | ✅ Excellent | ❌ Very Low | ❌ |

### Why Item-Based CF was chosen

| Criterion | Reasoning |
|---|---|
| **Stability** | Item similarity is computed once and cached. The movie catalog changes far less frequently than user behaviour |
| **Scalability** | A 9,742 × 9,742 matrix fits comfortably in memory (~360MB float64). User-based matrices scale with user count — unbounded |
| **No retraining** | New users can receive recommendations immediately without recomputing the similarity matrix |
| **Interpretability** | "We recommend X because users who liked Y also liked X" — directly explainable |
| **Performance** | Sub-millisecond inference per query (in-memory column lookup + sort) |

### Cosine vs Pearson Similarity

| Metric | Formula | Behaviour | Best For |
|---|---|---|---|
| **Cosine** | `dot(A,B) / (‖A‖ × ‖B‖)` | Measures angle between vectors; unaffected by magnitude | Sparse matrices with many zeros — **used here** |
| **Pearson** | Cosine on mean-centered vectors | Accounts for rating bias (harsh vs lenient raters) | Dense matrices where user rating styles vary significantly |

> **Note:** Because the rating matrix is ~98.3% sparse (zero = unrated, not a bad rating), cosine similarity is the correct choice. Pearson correlation would compute spurious correlations across the many shared zeros.

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---------|---------|------|
| **Python** | 3.10+ | Core language |
| **Flask** | Latest | Web server, routing, Jinja2 templating |
| **Pandas** | Latest | CSV ingestion, pivot table construction |
| **NumPy** | Latest | Numerical array operations |
| **Scikit-learn** | Latest | `cosine_similarity()` on transposed rating matrix |
| **difflib** | stdlib | `get_close_matches()` fuzzy title resolution |
| **Joblib** | Latest | Model artifact serialization (`model.py` offline pipeline) |

---

## 📁 Project Structure

```
movie_recommendation_app/
│
├── app.py                    # Flask routes + in-memory similarity engine
├── model.py                  # Offline training pipeline → .pkl artifacts
│
├── movies.csv                # 9,742 movies: movieId, title, genres
├── ratings.csv               # 100,836 ratings: userId, movieId, rating, timestamp
├── train.csv                 # Offline training split
│
├── static/
│   └── style.css             # Dark cinema UI — Poppins font, card grid, overlay
│
├── templates/
│   ├── index.html            # Home — search bar with cinematic background
│   ├── recommend.html        # Results — movie card grid
│   └── error.html            # Error — graceful 404/miss display
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mvkarthikeya07/advance_movie_recommendation_app.git
cd advance_movie_recommendation_app/movie_recommendation_app

# 2. Create and activate a virtual environment
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
python app.py
```

### Access

```
http://127.0.0.1:5000
```

> The similarity matrix (~9,742 × 9,742) is computed at startup. Expect a 3–8 second initialization on first launch — all subsequent queries are instant.

### Optional — Serialize the Model

```bash
# Pre-compute and save similarity matrix as .pkl for faster cold starts
python model.py
# Outputs: ratings_matrix.pkl, movie_similarity.pkl
```

---

## 🔌 API Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Render home page with search form |
| `POST` | `/recommend` | Accept `movie_name` form field → return top-5 recommendations |

### Request / Response

**POST `/recommend`**

```
Form body: movie_name=Inception
```

**Response:** Renders `recommend.html` with:
- `movie_title` — the original query string
- `recommendations` — list of matched title + top-5 similar movies

**Error case** (no fuzzy match found):
```
recommendations = ["❌ 'XYZ' not found in database. Try another movie."]
```

---

## 🔬 Engineering Highlights

| Highlight | Detail |
|-----------|--------|
| **Startup-time matrix build** | Merge → pivot → cosine_similarity computed once at `app.py` import; zero per-request recomputation |
| **Fuzzy title resolution** | `difflib.get_close_matches(cutoff=0.3)` — handles partial titles, minor typos, and case differences across 9,742 entries |
| **Self-exclusion** | `.iloc[1:num+1]` skips index 0 (the query movie itself, similarity = 1.0) cleanly |
| **Sparse-aware similarity** | `fillna(0)` on the pivot + cosine similarity correctly treats missing ratings as neutral (not negative) |
| **Dual pipeline** | `app.py` computes in-memory at startup for live use; `model.py` serializes to `.pkl` for offline/production deployment |
| **Cinematic UI** | Full-viewport cinematic background with CSS overlay, Poppins typography, card-based results grid |

---

## 📄 Research Publication

The theoretical foundation and experimental benchmarking underpinning this project is peer-reviewed and published:

> **Title:** Comparative Analysis of User-Based and Item-Based Collaborative Filtering Using the MovieLens Dataset
>
> **Journal:** International Journal of Innovative Research in Electrical, Electronics, Instrumentation and Control Engineering (IJIREEICE)
>
> **Link:** [ijireeice.com — View Publication](https://ijireeice.com/papers/comparative-analysis-of-user-based-and-item-based-collaborative-filtering-using-the-movielens-dataset/)

The paper provides quantitative comparison of user-based vs. item-based CF on the same MovieLens dataset, directly informing the architectural decisions in this implementation.

---

## 🔮 Future Roadmap

- [ ] **Hybrid Recommender** — Combine item-based CF with content-based genre/metadata similarity for cold-start coverage
- [ ] **Matrix Factorization (SVD)** — `surprise` library integration for latent-factor recommendations
- [ ] **Deep Learning (NCF)** — Neural Collaborative Filtering with PyTorch embeddings
- [ ] **Confidence Scores** — Display cosine similarity percentage alongside each recommendation
- [ ] **User Personalization** — Login + rating history for session-aware recommendations
- [ ] **REST API** — JSON endpoint for mobile or third-party integration
- [ ] **TMDB Metadata** — Fetch poster images and descriptions via The Movie Database API
- [ ] **Cloud Deployment** — Docker + AWS/Render production hosting

---

## 👤 Author

**M V Karthikeya**
Computer Science Engineer — Machine Learning, AI Systems & Data Science

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=flat-square&logo=github)](https://github.com/Mvkarthikeya07)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

---

<div align="center">

**Find your next favourite film — powered by mathematics, not marketing.**

*Content-Based Filtering · Cosine Similarity · MovieLens · Flask · Real-Time Inference*

© 2026 M V Karthikeya

</div>
