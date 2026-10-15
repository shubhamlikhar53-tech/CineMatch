from pathlib import Path
from html import escape
from datetime import date

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from movie_ai import explain_movie
import urllib.parse
import streamlit as st

# Google Movie Search
search_query = f"{selected_title} movie plot cast release year"

google_url = (
    "https://www.google.com/search?q="
    + urllib.parse.quote_plus(search_query)
)

st.link_button(
    "🔎 Search Movie on Google",
    google_url,
    use_container_width=True
)
# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CineMatch | Discover Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 2. CINEMATIC STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #070a10;
    color: #f4f5fb;
}

header[data-testid="stHeader"] {
    background: rgba(7, 10, 16, 0.95);
}

.block-container {
    max-width: 1550px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

section[data-testid="stMain"] {
    color: #f4f5fb;
}

.brand {
    font-size: 30px;
    font-weight: 900;
    letter-spacing: -1.2px;
    margin-bottom: 2px;
}

.brand span {
    color: #ff416c;
}

.tagline {
    color: #a9afc1;
    font-size: 12px;
    margin-top: -8px;
    margin-bottom: 22px;
}

.hero {
    min-height: 300px;
    padding: 48px 36px;
    border: 1px solid #24283a;
    border-radius: 22px;

    background:
        linear-gradient(
            90deg,
            rgba(5, 8, 17, 0.96) 0%,
            rgba(5, 8, 17, 0.78) 48%,
            rgba(5, 8, 17, 0.25) 100%
        ),
        url("https://images.unsplash.com/photo-1446776811953-b23d57bd21aa")
        center 45% / cover no-repeat;

    margin: 12px 0 28px;
}

.hero-kicker {
    color: #ff6385;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero h1 {
    color: white;
    font-size: clamp(34px, 4vw, 62px);
    line-height: 1.05;
    font-weight: 900;
    margin: 12px 0;
}

.hero h1 span {
    background: linear-gradient(90deg, #ff674b, #ff2d77);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #c5cada;
    max-width: 560px;
    font-size: 16px;
    line-height: 1.7;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    border-left: 5px solid #ff416c;
    padding-left: 12px;
    margin: 30px 0 18px;
}

.movie-card {
    position: relative;
    isolation: isolate;
    background: linear-gradient(155deg, rgba(20, 25, 42, .98), rgba(9, 12, 21, .98));
    border: 1px solid #252b3b;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 14px;
    transform: perspective(1000px) translateZ(0);
    transform-style: preserve-3d;
    box-shadow: 0 7px 18px rgba(0, 0, 0, .20);
    transition:
        transform .32s cubic-bezier(.2, .8, .2, 1),
        border-color .28s ease,
        box-shadow .32s ease,
        filter .32s ease;
    will-change: transform;
}

.movie-card:hover {
    border-color: rgba(255, 65, 108, .88);
    transform: perspective(1000px) rotateX(2deg) rotateY(-2deg) translateY(-7px) scale(1.025);
    box-shadow:
        0 22px 38px rgba(0, 0, 0, .38),
        0 0 0 1px rgba(255, 65, 108, .14),
        0 0 26px rgba(255, 45, 115, .13);
    filter: saturate(1.08);
}

.poster-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 2 / 3;
    background: linear-gradient(145deg, #1c2740, #10131c);
    overflow: hidden;
    transform: translateZ(12px);
}

.poster-wrap::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(
        145deg,
        rgba(255, 255, 255, .16) 0%,
        rgba(255, 255, 255, .035) 20%,
        transparent 48%,
        rgba(0, 0, 0, .25) 100%
    );
    opacity: .55;
    transition: opacity .28s ease;
}

.movie-card:hover .poster-wrap::after {
    opacity: .9;
}

.poster-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform .45s cubic-bezier(.2, .8, .2, 1), filter .3s ease;
    transform: scale(1.001) translateZ(0);
}

.movie-card:hover .poster-wrap img {
    transform: scale(1.075);
    filter: contrast(1.04) brightness(1.06);
}

.movie-card .rating {
    transform: translateZ(24px);
    box-shadow: 0 5px 14px rgba(0, 0, 0, .28);
}

.movie-card .movie-info {
    position: relative;
    transform: translateZ(8px);
    background: linear-gradient(180deg, rgba(15, 19, 31, .96), rgba(10, 13, 22, 1));
}

@media (prefers-reduced-motion: reduce) {
    .movie-card,
    .poster-wrap img,
    .poster-wrap::after {
        transition: none !important;
        transform: none !important;
    }
}

.rating {
    position: absolute;
    top: 9px;
    left: 8px;
    background: rgba(4, 7, 12, .88);
    border: 1px solid #33384a;
    border-radius: 8px;
    padding: 5px 8px;
    font-size: 12px;
    font-weight: 800;
    color: #ffd45b;
}

.poster-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    min-height: 150px;
    font-size: 45px;
}

.movie-info {
    padding: 10px;
}

.movie-title {
    font-size: 14px;
    line-height: 1.4;
    font-weight: 800;
    color: #f7f7fb;
    min-height: 39px;
}

.movie-meta {
    font-size: 11px;
    line-height: 1.5;
    color: #a9b1c4;
    margin-top: 6px;
}

div.stButton > button {
    border-radius: 24px;
    border: 1px solid #ff416c;
    background: linear-gradient(100deg, #ff654d, #ff286e);
    color: white;
    font-weight: 800;
}

div.stButton > button:hover {
    border-color: #ff7c98;
    color: white;
    background: #e92b61;
}

div[data-baseweb="input"] input {
    color: white;
}

div[data-baseweb="select"] > div {
    background: #111725;
    border-color: #30364a;
}

div[data-testid="stTabs"] button {
    color: #b9bfd0;
    font-weight: 700;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ff6385;
}

div[data-testid="stAlert"] {
    background: #111725;
}

.small-note {
    color: #858da2;
    font-size: 12px;
}
/* ==========================================
   CINEMATCH UI POLISH
   Preserves the existing design
========================================== */

/* Smoother page rendering */
.stApp {
    -webkit-font-smoothing: antialiased;
}

/* Refined hero section */
.hero {
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.18);
    transition: border-color 0.25s ease;
}

.hero:hover {
    border-color: #383d52;
}

/* Refined movie cards */
.movie-card {
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}

.movie-card:hover {
    transform: translateY(-5px);
    border-color: #ff416c;
    box-shadow: 0 12px 28px rgba(255, 65, 108, 0.12);
}

/* Better search input */
div[data-baseweb="input"] > div {
    border-radius: 10px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

div[data-baseweb="input"] > div:focus-within {
    border-color: #ff416c;
    box-shadow: 0 0 0 1px rgba(255, 65, 108, 0.25);
}

/* Polished dropdown */
div[data-baseweb="select"] > div {
    border-radius: 10px;
    transition: border-color 0.2s ease;
}

/* Refined buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    border-radius: 24px;
    font-weight: 750;
    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        background 0.2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 18px rgba(255, 65, 108, 0.18);
}

/* Cleaner navigation tabs */
div[data-testid="stTabs"] button {
    transition: color 0.2s ease;
}

/* Cleaner similarity score panel */
div[data-testid="stExpander"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Responsive spacing on smaller screens */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 30px 22px;
        min-height: 250px;
    }

    .hero p {
        font-size: 14px;
    }
}

/* ==========================================
   STEP 2: PREMIUM UI POLISH
   Keeps the existing CineMatch layout
========================================== */

/* Smoother typography and page rendering */
.stApp {
    -webkit-font-smoothing: antialiased;
}

/* Hero: depth and subtle hover treatment */
.hero {
    box-shadow: 0 14px 42px rgba(0, 0, 0, 0.24);
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
    background-color: #0b1220;
}

.hero:hover {
    border-color: #38415a;
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.30);
}

/* Cards: polished hover and consistent edges */
.movie-card {
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.14);
    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}

.movie-card:hover {
    transform: translateY(-5px);
    border-color: #ff416c;
    box-shadow: 0 12px 30px rgba(255, 65, 108, 0.15);
}

/* Search input focus */
div[data-baseweb="input"] > div {
    border-radius: 11px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

div[data-baseweb="input"] > div:focus-within {
    border-color: #ff416c;
    box-shadow: 0 0 0 1px rgba(255, 65, 108, 0.24);
}

/* Dropdowns */
div[data-baseweb="select"] > div {
    border-radius: 11px;
    transition: border-color 0.2s ease;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    border-radius: 24px;
    font-weight: 800;
    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        background 0.2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 18px rgba(255, 65, 108, 0.20);
}

/* Similarity score expander */
div[data-testid="stExpander"] {
    border: 1px solid #292f41;
    border-radius: 13px;
    overflow: hidden;
}

/* Better mobile spacing */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 30px 22px;
        min-height: 250px;
    }

    .hero p {
        font-size: 14px;
    }

    .movie-title {
        font-size: 13px;
    }
}



/* NEXT-LEVEL DISCOVERY BAR + AI PANEL */
.discovery-panel {
    margin: 12px 0 20px;
    padding: 18px 20px;
    border: 1px solid #24324b;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(15, 23, 40, .98), rgba(10, 13, 25, .98));
    box-shadow: 0 12px 34px rgba(0,0,0,.16);
}
.discovery-heading { font-size: 21px; font-weight: 850; color: #f7f8ff; }
.discovery-subtitle { color: #a9b5cc; font-size: 12px; margin-top: 4px; }
.ai-panel {
    padding: 18px;
    border: 1px solid #49305d;
    border-radius: 18px;
    background: radial-gradient(circle at 100% 0%, rgba(255, 42, 130, .13), transparent 44%), linear-gradient(145deg, #171326, #0d1422);
    min-height: 235px;
}
.ai-title { font-size: 20px; font-weight: 900; color: #fff; }
.ai-title span { color: #ff4e9a; }
.ai-copy { font-size: 12px; color: #b8c1d5; margin: 4px 0 12px; }
.mood-label { color: #aebbd1; font-size: 12px; font-weight: 700; padding-top: 8px; }
div[data-testid="stHorizontalBlock"] { gap: 0.75rem; }
@media (max-width: 768px) {
    .discovery-panel { padding: 14px; }
    .ai-panel { min-height: unset; }
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD SAVED MODEL
# ==========================================

MODEL_DIR = Path(__file__).resolve().parent / "cinematch_model1"

@st.cache_resource
def load_model():
    df = pd.read_pickle(MODEL_DIR / "experiment_df.pkl")
    vectors = sparse.load_npz(MODEL_DIR / "weighted_vectors.npz")

    if len(df) != vectors.shape[0]:
        raise ValueError(
            "Movie DataFrame aur weighted vectors ke row counts match nahi karte."
        )

    return df.reset_index(drop=True), vectors

try:
    movies, weighted_vectors = load_model()
except Exception as error:
    st.error("CineMatch model load nahi ho paya.")
    st.code(str(error))
    st.info(
        "Check karo ki cinematch_model1 folder app.py ke saath "
        "same project root mein hai aur saved model files usmein hain."
    )
    st.stop()

# ==========================================
# LOAD MOVIE RELEASE YEARS
# ==========================================

YEAR_FILE = Path(__file__).resolve().parent / "movie_years.csv"

@st.cache_data(show_spinner="Loading movie release years...")
def load_release_years(year_file):
    year_df = pd.read_csv(year_file)

    year_df["id"] = pd.to_numeric(year_df["id"], errors="coerce")
    year_df["release_year"] = pd.to_numeric(
        year_df["release_year"], errors="coerce"
    )

    year_df = year_df.dropna(
        subset=["id", "release_year"]
    ).drop_duplicates(subset=["id"])

    return dict(
        zip(
            year_df["id"].astype(int),
            year_df["release_year"].astype(int)
        )
    )

try:
    if not YEAR_FILE.exists():
        raise FileNotFoundError(f"File not found: {YEAR_FILE}")

    release_year_map = load_release_years(str(YEAR_FILE))

    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies["release_year"] = movies["id"].map(release_year_map)

    print("Release years matched:", movies["release_year"].notna().sum())

except Exception as error:
    st.warning(f"Release years could not be loaded: {error}")
# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def get_value(row, column, default=""):
    """Return a safe value even if a column is missing."""
    if column not in row.index or pd.isna(row[column]):
        return default
    return row[column]

def get_poster_url(row):
    path = str(get_value(row, "poster_path", "")).strip()

    if not path or path.lower() in {"nan", "none", "null"}:
        return None

    # TMDB poster paths generally start with /
    if path.startswith("/"):
        return f"https://image.tmdb.org/t/p/w500{path}"

    # Already a complete URL
    if path.startswith("https://") or path.startswith("http://"):
        return path

    return None

def find_movie(title):
    titles = movies["title"].fillna("").astype(str).str.strip()
    matches = titles.str.casefold() == title.strip().casefold()

    positions = np.flatnonzero(matches.to_numpy())

    if len(positions) == 0:
        return None

    return int(positions[0])

def recommend_movies(movie_title, top_n=5):
    """Use the saved weighted vectors to find similar movies."""
    movie_position = find_movie(movie_title)

    if movie_position is None:
        return pd.DataFrame()

    scores = cosine_similarity(
        weighted_vectors[movie_position],
        weighted_vectors,
    ).ravel()

    # Never recommend the input movie itself
    scores[movie_position] = -1

    top_positions = np.argsort(scores)[::-1][:top_n]

    results = movies.iloc[top_positions].copy()
    results["similarity"] = scores[top_positions]

    return results.reset_index(drop=True)

def render_movie_card(row):
    title = escape(str(get_value(row, "title", "Untitled movie")))
    genres = escape(str(get_value(row, "genres", "Genre unavailable")))
    year = escape(str(get_value(row, "release_year", "")))

    if not year:
        release_date = str(get_value(row, "release_date", ""))
        if len(release_date) >= 4:
            year = escape(release_date[:4])

    try:
        rating = float(get_value(row, "vote_average", 0))
        rating_text = f"{rating:.1f}"
    except (ValueError, TypeError):
        rating_text = "N/A"

    poster_url = get_poster_url(row)

    if poster_url:
        poster_html = (
            f'<img src="{escape(poster_url, quote=True)}" '
            f'alt="{title}" loading="lazy">'
        )
    else:
        poster_html = '<div class="poster-placeholder">🎬</div>'

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="poster-wrap">
                {poster_html}
                <div class="rating">⭐ {rating_text}</div>
            </div>
            <div class="movie-info">
                <div class="movie-title">{title}</div>
                <div class="movie-meta">
                    {year if year else "Year unavailable"}
                    <br>
                    {genres}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_movie_grid(data, columns_per_row=5):
    if data.empty:
        st.info("Is section ke liye abhi movies available nahi hain.")
        return

    for start in range(0, len(data), columns_per_row):
        batch = data.iloc[start:start + columns_per_row]
        columns = st.columns(columns_per_row)

        for column, (_, row) in zip(columns, batch.iterrows()):
            with column:
                render_movie_card(row)

def section_title(title):
    st.markdown(
        f'<div class="section-title">{escape(title)}</div>',
        unsafe_allow_html=True,
    )

# ==========================================
# 5. HEADER + HERO
# ==========================================

st.markdown(
    """
    <div class="brand">▶ Cine<span>Match</span></div>
    <div class="tagline">DISCOVER MOVIES YOU'LL LOVE</div>

    <div class="hero">
        <div class="hero-kicker">YOUR PERSONAL MOVIE UNIVERSE</div>
        <h1>Find Your Next<br>
        <span>Favourite Movie.</span></h1>
        <p>
            Discover movies with your AI-powered recommendation engine.
            Search, explore genres and find your next favourite.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 6. SMART DISCOVERY + CINE MATCH ASSISTANT
# ==========================================

st.markdown(
    """
    <div class="discovery-panel">
        <div class="discovery-heading">🔎 What are you in the mood for? <span style="color:#ff4e9a">✦</span></div>
        <div class="discovery-subtitle">Search a movie title below, or start with a mood.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

mood_columns = st.columns(6)
mood_options = [
    ("🧠 Mind-bending", "science fiction"),
    ("😊 Feel-good", "comedy"),
    ("🚀 Sci-fi", "science fiction"),
    ("🔥 Thriller", "thriller"),
    ("💗 Romance", "romance"),
    ("👻 Horror", "horror"),
]
for col, (label, query) in zip(mood_columns, mood_options):
    with col:
        if st.button(label, key=f"mood_{query}_{label}", use_container_width=True):
            st.session_state["cine_search_input"] = query
            st.session_state["mood_selected"] = label
            st.session_state["active_mood_genre"] = query
            st.session_state.pop("selected_for_recommendations", None)
            st.rerun()

# Mood buttons show results immediately; users do not need to press Search Movies.
active_mood_genre = st.session_state.get("active_mood_genre")
if active_mood_genre:
    mood_results = movies.copy()
    if "genres" in mood_results.columns:
        mood_results = mood_results[
            mood_results["genres"].fillna("").astype(str).str.contains(
                str(active_mood_genre), case=False, regex=False
            )
        ].copy()

        if "vote_average" in mood_results.columns:
            mood_results["_mood_rating"] = pd.to_numeric(
                mood_results["vote_average"], errors="coerce"
            ).fillna(0)
            if "vote_count" in mood_results.columns:
                mood_results["_mood_votes"] = pd.to_numeric(
                    mood_results["vote_count"], errors="coerce"
                ).fillna(0)
                mood_results = mood_results[mood_results["_mood_votes"] >= 10]
                mood_results = mood_results.sort_values(
                    ["_mood_rating", "_mood_votes"], ascending=False
                )
            else:
                mood_results = mood_results.sort_values(
                    "_mood_rating", ascending=False
                )

        st.markdown(f"### 🎭 {escape(str(st.session_state.get('mood_selected', 'Mood')))} movies")
        st.caption(f"Showing movies matching: {active_mood_genre}")
        render_movie_grid(mood_results.head(10), columns_per_row=5)

        if st.button("✖ Clear mood results", key="clear_mood_results"):
            st.session_state.pop("active_mood_genre", None)
            st.session_state.pop("mood_selected", None)
            st.rerun()
    else:
        st.info("Genre information is unavailable in the movie dataset.")

# A lightweight, honest assistant using the metadata already available in the dataset.
# It does not claim to be an LLM; it helps users find movies by genre or title.
assistant_left, assistant_right = st.columns([1.55, 1], gap="large")
with assistant_left:
    st.markdown("### 🎬 Find your next favourite")
    st.caption("Choose a mood above or search for a movie title to discover similar films.")
    st.markdown(
        "<div class='small-note'>Powered by CineMatch's saved content-similarity vectors.</div>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------
    # AI MOVIE OF THE DAY
    # Select one well-rated movie deterministically for each date.
    # -----------------------------------------
    featured_pool = movies.copy()

    if "vote_average" in featured_pool.columns:
        featured_pool["_featured_rating"] = pd.to_numeric(
            featured_pool["vote_average"], errors="coerce"
        ).fillna(0)
        featured_pool = featured_pool[featured_pool["_featured_rating"] >= 6.5]

    if "vote_count" in featured_pool.columns:
        featured_pool["_featured_votes"] = pd.to_numeric(
            featured_pool["vote_count"], errors="coerce"
        ).fillna(0)
        featured_pool = featured_pool[featured_pool["_featured_votes"] >= 25]

    featured_pool = featured_pool[
        featured_pool["title"].notna()
        & featured_pool["title"].astype(str).str.strip().ne("")
    ]

    if not featured_pool.empty:
        featured_pool = featured_pool.sort_values("title").reset_index(drop=True)
        featured_index = date.today().toordinal() % len(featured_pool)
        featured_movie = featured_pool.iloc[featured_index]

        featured_title = str(get_value(featured_movie, "title", "Movie of the Day"))
        featured_year = str(get_value(featured_movie, "release_year", ""))
        featured_genres = str(get_value(featured_movie, "genres", "Genres unavailable"))
        featured_overview = str(get_value(featured_movie, "overview", "Overview unavailable"))
        featured_rating = get_value(featured_movie, "vote_average", "Unavailable")
        featured_poster = get_poster_url(featured_movie)

        st.markdown(
            """
            <div class="ai-panel" style="margin-top:18px; min-height:unset;">
                <div class="ai-title">🌟 AI Movie of the Day</div>
                <div class="ai-copy">A daily pick from your CineMatch movie dataset.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        poster_col, details_col = st.columns([0.8, 1.2], gap="medium")

        with poster_col:
            if featured_poster:
                st.image(featured_poster, use_container_width=True)
            else:
                st.markdown(
                    """
                    <div class="poster-placeholder"
                         style="border:1px solid #30364a;border-radius:12px;">
                        🎬
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

     
        with details_col:
            st.markdown(f"### {escape(featured_title)}")

            year_text = (
                featured_year
                if featured_year not in {"", "nan", "None"}
                else "Year unavailable"
            )
            st.caption(f"📅 {escape(year_text)}")
            st.write(f"**Genre:** {escape(featured_genres)}")

            try:
                st.write(
                    f"⭐ **Rating:** {float(featured_rating):.1f}/10"
                )
            except (TypeError, ValueError):
                st.write("⭐ **Rating:** Unavailable")

            google_query = (
                f"{featured_title} movie plot cast release year"
            )
            google_url = (
                "https://www.google.com/search?q="
                + urllib.parse.quote_plus(google_query)
            )

            st.link_button(
                "🔎 Search Movie on Google",
                google_url,
                use_container_width=True,
            )

        # Keep the existing outer else aligned with its matching if.
        

with assistant_right:
    st.markdown(
        """
        <div class="ai-panel">
            <div class="ai-title">🤖 Ask <span>CineMatch AI</span></div>
            <div class="ai-copy">Ask about genres, ratings, or find a movie by mood.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("cine_assistant_form", clear_on_submit=False):
        assistant_question = st.text_input(
            "Ask about movies",
            placeholder="e.g. Recommend a sci-fi movie",
            label_visibility="collapsed",
        )
        assistant_submit = st.form_submit_button("✨ Ask CineMatch", use_container_width=True)

    if assistant_submit:
        question = assistant_question.strip()

        if not question:
            st.warning("Please type a movie question first.")

        else:
            # -----------------------------------------
            # 1. Check whether the user mentioned a movie
            # -----------------------------------------
            question_lower = question.casefold()

            available_titles = (
                movies["title"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            # Check longer titles first to reduce partial-title matches
            matching_titles = sorted(
                [
                    title for title in available_titles
                    if title.casefold() in question_lower
                ],
                key=len,
                reverse=True,
            )

            # -----------------------------------------
            # 2. If a movie title is found, use Qwen
            # -----------------------------------------
            if matching_titles:
                selected_title = matching_titles[0]

                movie_matches = movies[
                    movies["title"].fillna("").astype(str).str.casefold()
                    == selected_title.casefold()
                    ]

                row = movie_matches.iloc[0]

                movie_overview = get_value(
                    row, "overview", "Overview unavailable"
                )

                movie_genres = get_value(
                    row, "genres", "Genres unavailable"
                )

                movie_rating = get_value(
                    row, "vote_average", "Unavailable"
                )

                movie_year = get_value(
                    row, "release_year", "Unavailable"
                )

                st.markdown("### 🔎 Search Movie Information")

st.caption(
    f"Search Google for information about {selected_title}."
)

google_query = (
    f"{selected_title} movie plot cast release year "
    f"{movie_genres}"
)

google_url = (
    "https://www.google.com/search?q="
    + urllib.parse.quote_plus(google_query)
)

st.link_button(
    "🔎 Search on Google",
    google_url,
    use_container_width=True,
)

                   

            # -----------------------------------------
            # 3. Otherwise, search by genre or mood
            # -----------------------------------------
            else:
                genre_aliases = {
                    "sci-fi": "science fiction",
                    "scifi": "science fiction",
                    "romantic": "romance",
                    "love": "romance",
                    "funny": "comedy",
                    "scary": "horror",
                    "suspense": "thriller",
                    "action-packed": "action",
                }

                detected_genre = next(
                    (
                        genre
                        for keyword, genre in genre_aliases.items()
                        if keyword in question_lower
                    ),
                    None,
                )

                known_genres = sorted({
                    part.strip().casefold()
                    for value in movies.get(
                        "genres", pd.Series(dtype=str)
                    ).dropna().astype(str)
                    for part in value.split(",")
                    if part.strip()
                })

                if detected_genre is None:
                    detected_genre = next(
                        (
                            genre for genre in known_genres
                            if genre in question_lower
                        ),
                        None,
                    )

                if detected_genre and "genres" in movies.columns:
                    picks = movies[
                        movies["genres"].fillna("").str.contains(
                            detected_genre,
                            case=False,
                            regex=False,
                        )
                    ].copy()

                    if "vote_average" in picks.columns:
                        picks["vote_average"] = pd.to_numeric(
                            picks["vote_average"],
                            errors="coerce",
                        )

                        if "vote_count" in picks.columns:
                            picks["vote_count"] = pd.to_numeric(
                                picks["vote_count"],
                                errors="coerce",
                            ).fillna(0)

                            picks = picks[picks["vote_count"] >= 25]

                        picks = picks.sort_values(
                            "vote_average",
                            ascending=False,
                        )

                    st.success(
                        f"Here are some {detected_genre} movies to explore:"
                    )

                    render_movie_grid(
                        picks.head(4),
                        columns_per_row=4,
                    )

                else:
                    st.info(
                        "Try asking about a movie by name, or search for "
                        "a genre such as science fiction, romance, comedy, "
                        "horror, thriller, or action."
                    )

# ==========================================
# 7. SEARCH
# ==========================================

st.markdown("### 🔎 Find a movie")

with st.form("movie_search_form"):
    search_text = st.text_input(
        "Search movies",
        placeholder="Type a movie title, e.g. Iron Man, Avatar...",
        label_visibility="collapsed",
        key="cine_search_input",
    )

    search_clicked = st.form_submit_button(
        "Search Movies",
        use_container_width=True,
    )

if search_clicked:
    cleaned_search = search_text.strip()

    if cleaned_search:
        title_series = movies["title"].fillna("").astype(str)

        matching_titles = (
            movies.loc[
                title_series.str.contains(
                    cleaned_search,
                    case=False,
                    regex=False,
                ),
                "title",
            ]
            .drop_duplicates()
            .head(30)
            .tolist()
        )

        if matching_titles:
            st.session_state["search_results"] = matching_titles
            st.session_state["search_query"] = cleaned_search
        else:
            st.session_state["search_results"] = []
            st.warning("No matching movie found. Try another title.")

    else:
        st.warning("Pehle movie ka naam enter karo.")

if "search_results" in st.session_state:
    query = st.session_state.get("search_query", "")

    if st.session_state["search_results"]:
        st.markdown(f"#### Search results for: {escape(query)}")

        selected_title = st.selectbox(
            "Select the exact movie",
            st.session_state["search_results"],
            key="selected_movie",
        )

        if st.button("✨ Recommend Similar Movies", key="recommend_button"):
            st.session_state["selected_for_recommendations"] = selected_title

    else:
        st.info("Try searching for another movie.")

# ==========================================
# 8. NAVIGATION
# ==========================================

home_tab, movies_tab, genres_tab, top_tab, trending_tab, list_tab = st.tabs(
    ["🏠 Home", "🎬 Movies", "▦ Genres", "⭐ Top Rated", "🔥 Trending", "♡ My List"]
)

# ==========================================
# 9. HOME
# ==========================================

with home_tab:
    selected_title = st.session_state.get(
        "selected_for_recommendations"
    )

    if selected_title:
        section_title(f"Recommended for: {selected_title}")

        recommendations = recommend_movies(selected_title, top_n=5)

        if not recommendations.empty:
            render_movie_grid(recommendations, columns_per_row=5)

            # ==========================================
            # PREMIUM RECOMMENDATION SIMILARITY SCORES
            # ==========================================

            with st.expander(
                "📊 View recommendation similarity scores",
                expanded=False
            ):
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(
                            110deg,
                            rgba(255, 101, 77, 0.10),
                            rgba(255, 40, 110, 0.06)
                        );
                        border: 1px solid #292d3d;
                        border-radius: 14px;
                        padding: 16px;
                        margin-bottom: 18px;
                    ">
                        <div style="
                            font-size: 19px;
                            font-weight: 800;
                            color: #f7f7fb;
                        ">
                            🎬 Movie Similarity Analysis
                        </div>
                        <div style="
                            color: #a9b1c4;
                            font-size: 12px;
                            margin-top: 6px;
                        ">
                            Compare how closely each movie matches
                            your selected movie according to CineMatch.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Similarity scores measure vector similarity. "
                    "They are not probabilities or audience-match percentages."
                )

                for rank, (_, score_row) in enumerate(
                    recommendations.iterrows(),
                    start=1
                ):
                    movie_name = str(
                        get_value(score_row, "title", "Unknown movie")
                    )

                    movie_genres = str(
                        get_value(score_row, "genres", "Genres unavailable")
                    )

                    try:
                        similarity = float(score_row["similarity"])
                    except (KeyError, ValueError, TypeError):
                        similarity = 0.0

                    try:
                        movie_rating = float(
                            get_value(score_row, "vote_average", 0)
                        )
                        rating_text = f"{movie_rating:.1f}/10"
                    except (ValueError, TypeError):
                        rating_text = "N/A"

                    # Keep progress values between 0 and 1
                    progress_value = max(
                        0.0,
                        min(1.0, similarity)
                    )

                    left, middle, right = st.columns(
                        [0.08, 0.57, 0.35]
                    )

                    with left:
                        st.markdown(
                            f"""
                            <div style="
                                background: #171c29;
                                border: 1px solid #30364a;
                                border-radius: 10px;
                                text-align: center;
                                padding: 8px 2px;
                                font-weight: 800;
                                color: #ff6385;
                            ">
                                #{rank}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with middle:
                        st.markdown(
                            f"**{escape(movie_name)}**"
                        )

                        st.caption(
                            f"⭐ {rating_text} · {escape(movie_genres)}"
                        )

                    with right:
                        st.markdown(
                            f"""
                            <div style="
                                text-align: right;
                                font-size: 17px;
                                font-weight: 800;
                                color: #ff6385;
                                margin-bottom: 2px;
                            ">
                                {similarity:.2%}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.progress(progress_value)

                    st.markdown(
                        """
                        <div style="
                            border-bottom: 1px solid #222737;
                            margin: 8px 0 12px;
                        "></div>
                        """,
                        unsafe_allow_html=True,
                    )

        else:
            st.warning(
                "Is title ke liye recommendations nahi milin."
            )

        if st.button("Clear recommendations"):
            st.session_state.pop("selected_for_recommendations", None)
            st.rerun()

    else:
        section_title("Recommended For You")
        st.caption(
            "Start by searching for a movie above to get personalized recommendations."
        )

        # Show highly rated movies as a discovery shelf until a movie is selected
        rating_values = pd.to_numeric(
            movies.get("vote_average", pd.Series(0, index=movies.index)),
            errors="coerce",
        )

        preview = movies.loc[rating_values >= 7.0].copy()

        if "vote_count" in preview.columns:
            preview["_votes"] = pd.to_numeric(
                preview["vote_count"], errors="coerce"
            ).fillna(0)
            preview = preview[preview["_votes"] >= 100]
            preview = preview.sort_values(
                ["vote_average", "_votes"],
                ascending=False,
            )
        else:
            preview = preview.sort_values(
                "vote_average",
                ascending=False,
            )

        render_movie_grid(preview.head(10), columns_per_row=5)


# ==========================================
# 10. MOVIES
# ==========================================

with movies_tab:
    section_title("Explore Movies")

    if "genres" in movies.columns:
        all_genres = sorted({
            genre.strip()
            for value in movies["genres"].dropna().astype(str)
            for genre in value.split(",")
            if genre.strip()
        })
    else:
        all_genres = []

    chosen_genre = st.selectbox(
        "Filter by genre",
        ["All Genres"] + all_genres,
        key="movies_genre_filter",
    )

    movie_results = movies.copy()

    if chosen_genre != "All Genres" and "genres" in movie_results.columns:
        movie_results = movie_results[
            movie_results["genres"].fillna("").str.contains(
                chosen_genre,
                case=False,
                regex=False,
            )
        ]

    st.caption(f"{len(movie_results):,} movies match this filter.")
    render_movie_grid(movie_results.head(30), columns_per_row=5)

# ==========================================
# 11. GENRES
# ==========================================

with genres_tab:
    section_title("Browse Genres")

    if all_genres:
        chosen_genre = st.selectbox(
            "Choose a genre",
            all_genres,
            key="genre_page_filter",
        )

        genre_movies = movies[
            movies["genres"].fillna("").str.contains(
                chosen_genre,
                case=False,
                regex=False,
            )
        ].copy()

        if "vote_average" in genre_movies.columns:
            genre_movies = genre_movies.sort_values(
                "vote_average",
                ascending=False,
            )

        st.caption(f"{len(genre_movies):,} movies in {chosen_genre}.")
        render_movie_grid(genre_movies.head(20), columns_per_row=5)
    else:
        st.info("Genre information is unavailable in this model dataset.")

# ==========================================
# 12. TOP RATED
# ==========================================

with top_tab:
    section_title("Top Rated Movies")

    if "vote_average" in movies.columns:
        top_movies = movies.copy()
        top_movies["vote_average"] = pd.to_numeric(
            top_movies["vote_average"],
            errors="coerce",
        )

        if "vote_count" in top_movies.columns:
            top_movies["vote_count"] = pd.to_numeric(
                top_movies["vote_count"],
                errors="coerce",
            ).fillna(0)
            top_movies = top_movies[top_movies["vote_count"] >= 100]

        top_movies = top_movies.sort_values(
            "vote_average",
            ascending=False,
        )

        render_movie_grid(top_movies.head(20), columns_per_row=5)
    else:
        st.info("Rating information is unavailable.")

# ==========================================
# 13. TRENDING
# ==========================================

with trending_tab:
    section_title("Trending Movies")

    if "popularity" in movies.columns:
        trending_movies = movies.copy()
        trending_movies["popularity"] = pd.to_numeric(
            trending_movies["popularity"],
            errors="coerce",
        ).fillna(0)

        trending_movies = trending_movies.sort_values(
            "popularity",
            ascending=False,
        )
        render_movie_grid(trending_movies.head(20), columns_per_row=5)

    elif "vote_count" in movies.columns:
        st.caption(
            "Popularity column nahi mila, isliye vote count ko discovery proxy "
            "ke roop mein use kar rahe hain."
        )

        trending_movies = movies.copy()
        trending_movies["vote_count"] = pd.to_numeric(
            trending_movies["vote_count"],
            errors="coerce",
        ).fillna(0)

        trending_movies = trending_movies.sort_values(
            "vote_count",
            ascending=False,
        )
        render_movie_grid(trending_movies.head(20), columns_per_row=5)

    else:
        st.info("Trending ke liye popularity data available nahi hai.")

# ==========================================
# 14. MY LIST
# ==========================================

with list_tab:
    section_title("My Watchlist")

    if "my_list" not in st.session_state:
        st.session_state["my_list"] = []

    if not st.session_state["my_list"]:
        st.info(
            "Your watchlist is empty. Search for a movie and add it here "
            "in the next UI upgrade."
        )
    else:
        saved_movies = movies[
            movies["title"].isin(st.session_state["my_list"])
        ]
        render_movie_grid(saved_movies, columns_per_row=5)

# ==========================================
# 15. FOOTER
# ==========================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#828ba0;font-size:12px;">
        🎬 CineMatch · Content-based movie recommendations
        <br>Recommendations are generated using your saved similarity vectors.
    </div>
    """,
    unsafe_allow_html=True,
)
