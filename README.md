# 🎬 CineMatch AI
### Discover Movies You'll Love 🍿

CineMatch is an AI-powered movie recommendation web application
that helps users discover movies based on their interests, genres,
and favourite films.

Built with Python, Streamlit, Machine Learning, and content-based
similarity techniques, CineMatch makes movie discovery simple,
interactive, and enjoyable.

---

## 🚀 Live Demo

🌐 **Try CineMatch:** https://ai-cinematch.streamlit.app/

---

## ✨ Features

- 🎬 **Movie Recommendations:** Discover movies similar to your
  favourite titles.
- 🧠 **Content-Based Filtering:** Uses precomputed movie feature
  vectors and cosine similarity.
- 🎭 **Mood-Based Discovery:** Explore movies by genres such as
  science fiction, comedy, thriller, romance, and horror.
- 🔎 **Movie Search:** Search for a movie by its title.
- 🌟 **Movie of the Day:** Discover a daily movie selection.
- 🖼️ **Movie Posters:** Browse movies with poster images when available.
- ⭐ **Movie Ratings:** View available movie ratings.
- 🤖 **CineMatch AI Assistant:** Ask movie-related questions and
  request movie explanations when the AI service is available.
- 🌌 **Cinematic UI:** Enjoy a dark interface with a modern,
  responsive design.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data processing |
| NumPy | Numerical computing |
| SciPy | Sparse matrix loading and processing |
| Scikit-learn | Cosine similarity and recommendation logic |
| Pickle | Loading saved movie data |
| TMDB Dataset | Movie metadata and information |
| Hugging Face Transformers | Local language-model integration in the current AI implementation |

---

## 🧠 How It Works

CineMatch uses a content-based movie recommendation approach.

### Recommendation Pipeline

1. Load the saved movie dataset.
2. Load precomputed movie feature vectors.
3. Find the selected movie in the dataset.
4. Calculate cosine similarity between the selected movie and
   other movies.
5. Exclude the selected movie from the results.
6. Rank movies by similarity.
7. Display the most similar movies in the application.

### Architecture

```text
          User
           |
           v
    Streamlit Interface
           |
           v
      Movie Search
           |
           v
   Movie Dataset + Vectors
           |
           v
    Cosine Similarity
           |
           v
   Ranked Recommendations
           |
           v
    Movie Recommendation UI




## 💻 Installation & Setup

Follow these steps to run CineMatch locally.

### 1. Clone the Repository

```bash
git clone https://github.com/shubhamlikhar53-tech/CineMatch.git
cd CineMatch

2. Create a Virtual Environment
python -m venv .venv

3. Activate the Environment

Windows:

.venv\Scripts\activate

Linux / macOS:

source .venv/bin/activate
4. Install Dependencies
pip install -r requirements.txt
5. Verify Project Files

Make sure the required files and saved movie vectors are present:

app.py
main.py
movie_ai.py
movie_years.csv
requirements.txt
cinematch_model1/

6. Run CineMatch
streamlit run app.py
