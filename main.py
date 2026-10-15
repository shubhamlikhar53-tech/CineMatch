
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
# ==========================================
# LOAD MOVIE RELEASE YEARS
# ==========================================


YEAR_FILE = Path(__file__).resolve().parent / "movie_years.csv"

release_year_map = {}

if YEAR_FILE.exists():
    year_df = pd.read_csv(YEAR_FILE)

    year_df["id"] = pd.to_numeric(
        year_df["id"], errors="coerce"
    )
    year_df["release_year"] = pd.to_numeric(
        year_df["release_year"], errors="coerce"
    )

    year_df = year_df.dropna(
        subset=["id", "release_year"]
    ).drop_duplicates("id")

    release_year_map = dict(
        zip(
            year_df["id"].astype(int),
            year_df["release_year"].astype(int)
        )
    )

    print("Release years loaded:", len(release_year_map))
else:
    print("movie_years.csv not found; years unavailable.")
# ==========================================
# 1. MODEL DIRECTORY
# ==========================================

MODEL_DIR = Path(__file__).resolve().parent / "cinematch_model1"


# ==========================================
# 2. LOAD SAVED DATAFRAMES
# ==========================================

experiment_df = pd.read_pickle(
    MODEL_DIR / "experiment_df.pkl"
)

hybrid_df = pd.read_pickle(
    MODEL_DIR / "hybrid_df.pkl"
)


# ==========================================
# 3. LOAD SPARSE MATRICES
# ==========================================

overview_vectors = sparse.load_npz(
    MODEL_DIR / "overview_vectors.npz"
)

keyword_vectors_hybrid = sparse.load_npz(
    MODEL_DIR / "keyword_vectors_hybrid.npz"
)

weighted_vectors = sparse.load_npz(
    MODEL_DIR / "weighted_vectors.npz"
)

tfidf_matrix = sparse.load_npz(
    MODEL_DIR / "tfidf_matrix.npz"
)


# ==========================================
# 4. LOAD GENRE VECTORS
# ==========================================

genre_vectors = np.load(
    MODEL_DIR / "genre_vectors.npy"
)


# ==========================================
# 5. VERIFY MODEL COMPONENTS
# ==========================================

print("=" * 50)
print("CINEMATCH MODEL LOADING TEST")
print("=" * 50)

print("Experiment DataFrame:", experiment_df.shape)
print("Hybrid DataFrame:", hybrid_df.shape)

print("Overview vectors:", overview_vectors.shape)
print("Keyword vectors:", keyword_vectors_hybrid.shape)
print("Genre vectors:", genre_vectors.shape)

print("Weighted vectors:", weighted_vectors.shape)
print("TF-IDF matrix:", tfidf_matrix.shape)

print("\nAll seven saved components loaded successfully!")
# ==========================================
# 6. TEST CINEMATCH RECOMMENDATIONS
# ==========================================

def test_recommendation(movie_title, top_n=5):

    matches = experiment_df[
        experiment_df["title"].str.strip().str.casefold()
        == movie_title.strip().casefold()
    ]

    if matches.empty:
        print(f"\n❌ Movie not found: {movie_title}")
        return

    movie_index = matches.index[0]

    scores = cosine_similarity(
        weighted_vectors[movie_index],
        weighted_vectors
    ).ravel()

    # Exclude the input movie itself
    scores[movie_index] = -1

    # Find the most similar movies
    top_indices = np.argsort(scores)[::-1][:top_n]

    print("\n" + "=" * 55)
    print(f"INPUT MOVIE: {movie_title}")
    print("=" * 55)

    result = experiment_df.iloc[top_indices][
        ["id", "title", "genres", "vote_average"]
    ].copy()

    result["release_year"] = result["id"].map(release_year_map)

    result["similarity"] = scores[top_indices]

    result = result[
        ["title", "release_year", "genres", "vote_average", "similarity"]
    ]

    result["similarity"] = scores[top_indices]

    print(result.to_string(index=False))

# Check available movie metadata
print("\nAvailable columns in experiment_df:")
print(experiment_df.columns.tolist())

print("\nAvailable columns in hybrid_df:")
print(hybrid_df.columns.tolist())

# Test three movies
test_recommendation("Iron Man")
test_recommendation("The Dark Knight")
test_recommendation("Avatar")
