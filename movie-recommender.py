# -*- coding: utf-8 -*-
"""
Movie Recommendation System — Premium Cinema Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark Cinema Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Root Variables ── */
:root {
    --gold:       #E8B84B;
    --gold-dim:   #a07c28;
    --red:        #C0392B;
    --bg:         #0a0a0f;
    --bg2:        #111118;
    --bg3:        #1a1a26;
    --card:       #16161f;
    --card-hover: #1e1e2e;
    --border:     rgba(232,184,75,0.15);
    --text:       #e8e6e0;
    --text-dim:   #888899;
    --font-head:  'Playfair Display', Georgia, serif;
    --font-body:  'DM Sans', sans-serif;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background-color: var(--bg); }

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--font-body);
    font-size: 0.9rem;
    color: var(--text-dim);
    cursor: pointer;
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--gold); }
[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div + label { color: var(--gold); }

/* ── Headings ── */
h1, h2, h3 {
    font-family: var(--font-head);
    color: var(--text);
}

/* ── Gold Divider ── */
hr { border-color: var(--border) !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}
[data-testid="metric-container"] label {
    color: var(--text-dim) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-family: var(--font-head);
    font-size: 2.2rem !important;
}

/* ── Text Input ── */
[data-testid="stTextInput"] input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-body);
    font-size: 1rem;
    padding: 0.65rem 1rem !important;
    transition: border 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(232,184,75,0.12) !important;
}

/* ── Select Box ── */
[data-baseweb="select"] > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--gold) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.04em;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #f5cc6a !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(232,184,75,0.25) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--gold), #d4a035) !important;
}

/* ── Progress Bar ── */
[data-testid="stProgressBar"] > div { background: var(--bg3) !important; border-radius: 99px; }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, var(--gold), #f5cc6a) !important; border-radius: 99px; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; gap: 0.5rem; }
[data-baseweb="tab"] {
    font-family: var(--font-body) !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 1.2rem !important;
    border-radius: 6px 6px 0 0;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: rgba(232,184,75,0.06) !important;
}

/* ── Info / Warning Boxes ── */
[data-testid="stAlert"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p { color: var(--gold) !important; }

/* ── Multiselect ── */
[data-baseweb="tag"] {
    background: rgba(232,184,75,0.15) !important;
    color: var(--gold) !important;
    border-radius: 6px !important;
}

/* ── Matplotlib / Charts ── */
.stPlotlyChart, .stPyplot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_movies_data():
    df = pd.read_csv("movies.csv")
    df['genres'] = df['genres'].fillna("")
    df['genres'] = df['genres'].str.split('|')
    df['title'] = df['title'].apply(lambda x: re.sub("[^a-zA-Z0-9 ]", "", x))
    df = df[~df['genres'].apply(lambda x: '(no genres listed)' in x)]
    return df

@st.cache_data
def load_ratings_data():
    df = pd.read_csv("ratings.csv")
    return df.drop(['timestamp'], axis=1)

@st.cache_data
def prepare_vectors(movies_data):
    vec_title = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_title = vec_title.fit_transform(movies_data['title'])
    movies_data['genres_text'] = movies_data['genres'].apply(lambda x: ' '.join(x))
    vec_genres = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_genres = vec_genres.fit_transform(movies_data['genres_text'])
    return vec_title, tfidf_title, vec_genres, tfidf_genres


# ─────────────────────────────────────────────────────────────
# TMDB POSTER
# ─────────────────────────────────────────────────────────────
TMDB_API_KEY = "f7abbd106ffe7a0b21d4f884ebae6318"

@st.cache_data(show_spinner=False)
def fetch_movie_poster(movie_title):
    try:
        import urllib.parse
        q = urllib.parse.quote(movie_title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={q}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get('results'):
            r = data['results'][0]
            if r.get('poster_path'):
                return f"https://image.tmdb.org/t/p/w500{r['poster_path']}"
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_title):
    """Fetch synopsis, rating, and poster from TMDB by title."""
    try:
        import urllib.parse
        q = urllib.parse.quote(movie_title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={q}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        if not data.get('results'):
            return {}
        r = data['results'][0]
        return {
            'overview': r.get('overview') or '-',
            'vote_average': r.get('vote_average'),
            'vote_count': r.get('vote_count'),
            'release_date': r.get('release_date', ''),
            'poster_url': f"https://image.tmdb.org/t/p/w500{r['poster_path']}" if r.get('poster_path') else None,
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
# INIT DATA
# ─────────────────────────────────────────────────────────────
movies_data    = load_movies_data()
ratings_data   = load_ratings_data()
combined_data  = ratings_data.merge(movies_data, on='movieId')
vec_title, tfidf_title, vec_genres, tfidf_genres = prepare_vectors(movies_data)

all_genres = sorted(movies_data['genres'].explode().unique())
all_genres = [g for g in all_genres if g and g != '']

# ── Session state for movie detail view ──
if 'detail_movie' not in st.session_state:
    st.session_state['detail_movie'] = None   # dict: {title, genres, avg_rating}
if 'detail_from_page' not in st.session_state:
    st.session_state['detail_from_page'] = None
if 'last_recs' not in st.session_state:
    st.session_state['last_recs'] = []
if 'last_recs_title' not in st.session_state:
    st.session_state['last_recs_title'] = ''
if 'last_recs_genres_filter' not in st.session_state:
    st.session_state['last_recs_genres_filter'] = []


# ─────────────────────────────────────────────────────────────
# HELPER: Render a clickable movie card that sets detail state
# ─────────────────────────────────────────────────────────────
def movie_detail_button(title, genres, avg_rating=None, button_key=""):
    """Render a small 'View Details' button that stores movie info in session state."""
    if st.button("🔍 View Details", key=f"det_{button_key}", use_container_width=True):
        st.session_state['detail_movie'] = {
            'title': title,
            'genres': genres,
            'avg_rating': avg_rating,
        }
        st.rerun()


# ─────────────────────────────────────────────────────────────
# RECOMMENDATION LOGIC
# ─────────────────────────────────────────────────────────────
def clean_title(t):
    return re.sub("[^a-zA-Z0-9 ]", "", t)

def search_by_title(title):
    title = clean_title(title)
    qv = vec_title.transform([title])
    sim = cosine_similarity(qv, tfidf_title).flatten()
    idx = np.argpartition(sim, -5)[-5:]
    return movies_data.iloc[idx][::-1]

def search_similar_genres(genres):
    qv = vec_genres.transform([genres])
    sim = cosine_similarity(qv, tfidf_genres).flatten()
    idx = np.argpartition(sim, -10)[-10:]
    return movies_data.iloc[idx][::-1]

def scores_calculator(movie_id):
    if movie_id not in combined_data['movieId'].values:
        return pd.DataFrame(columns=['similar', 'all', 'score'])
    similar_users = combined_data.loc[
        (combined_data['movieId'] == movie_id) & (combined_data['rating'] >= 4), 'userId'
    ].unique()
    if len(similar_users) == 0:
        return pd.DataFrame(columns=['similar', 'all', 'score'])
    sim_recs = combined_data.loc[
        (combined_data['userId'].isin(similar_users)) & (combined_data['rating'] >= 4), 'movieId'
    ].value_counts(normalize=True)
    all_recs = combined_data.loc[
        combined_data['movieId'].isin(sim_recs.index) & (combined_data['rating'] >= 4)
    ]['movieId'].value_counts(normalize=True)
    movie_row = combined_data.loc[combined_data['movieId'] == movie_id]
    if len(movie_row) == 0:
        return pd.DataFrame(columns=['similar', 'all', 'score'])
    sel_genres = movie_row['genres'].iloc[0]
    if isinstance(sel_genres, list):
        sel_genres = " ".join(sel_genres)
    sim_genre_ids = search_similar_genres(sel_genres)['movieId']
    sim_recs.loc[sim_recs.index.isin(sim_genre_ids)] *= 1.5
    all_recs.loc[all_recs.index.isin(sim_genre_ids)] *= 0.9
    scores = pd.DataFrame({'similar': sim_recs, 'all': all_recs}).fillna(0)
    scores['score'] = np.where(scores['all'] > 0, scores['similar'] / scores['all'], 0)
    return scores.sort_values('score', ascending=False)

def recommendation_results(user_input, title_idx=0, genre_filter=None):
    candidates = search_by_title(user_input)
    if len(candidates) == 0:
        return pd.DataFrame(columns=['title', 'score', 'genres'])
    movie_id = candidates.iloc[title_idx]['movieId']
    scores = scores_calculator(movie_id)
    if len(scores) == 0:
        return pd.DataFrame(columns=['title', 'score', 'genres'])
    results = scores.head(30).merge(movies_data, left_index=True, right_on='movieId')[['movieId', 'title', 'score', 'genres']]
    if genre_filter and len(genre_filter) > 0:
        results = results[results['genres'].apply(lambda x: any(g in x for g in genre_filter))]
    return results.head(10)


# ─────────────────────────────────────────────────────────────
# HELPER: Render genre badge HTML
# ─────────────────────────────────────────────────────────────
def genre_badges(genres: list) -> str:
    colors = {
        'Action':'#e74c3c','Adventure':'#e67e22','Animation':'#3498db',
        'Comedy':'#f1c40f','Crime':'#8e44ad','Documentary':'#1abc9c',
        'Drama':'#2980b9','Fantasy':'#9b59b6','Horror':'#c0392b',
        'Mystery':'#16a085','Romance':'#e91e63','Sci-Fi':'#00bcd4',
        'Thriller':'#d35400','War':'#7f8c8d','Western':'#795548',
    }
    badges = ""
    for g in genres[:4]:
        c = colors.get(g, '#555577')
        badges += f'<span style="background:{c}22;color:{c};border:1px solid {c}44;padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:500;margin-right:4px;white-space:nowrap;">{g}</span>'
    return badges

def score_bar_html(score: float, max_score: float = 10.0) -> str:
    pct = min(score / max_score * 100, 100)
    stars = min(int(score / 2) + 1, 5)
    star_str = "★" * stars + "☆" * (5 - stars)
    return f"""
    <div style="margin:6px 0 10px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="flex:1;background:#1a1a26;border-radius:99px;height:6px;overflow:hidden;">
                <div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,#E8B84B,#f5cc6a);border-radius:99px;"></div>
            </div>
            <span style="color:#E8B84B;font-size:0.85rem;font-weight:600;white-space:nowrap;">{score:.2f}</span>
            <span style="color:#E8B84B;font-size:0.85rem;letter-spacing:1px;">{star_str}</span>
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem;text-align:center;">
        <div style="font-size:2.4rem;margin-bottom:0.3rem;">🎬</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:900;
                    background:linear-gradient(135deg,#E8B84B,#f5cc6a);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            CineMatch
        </div>
        <div style="font-size:0.72rem;color:#888899;letter-spacing:0.12em;text-transform:uppercase;margin-top:2px;">
            Smart Movie Recommendations
        </div>
    </div>
    <hr style="border-color:rgba(232,184,75,0.15);margin:0 0 1.2rem;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠  Home", "📊  Analytics", "🎯  Recommend", "🎭  Browse Genre"],
        label_visibility="collapsed"
    )

    # Back button when in detail view
    if st.session_state.get('detail_movie'):
        st.markdown("<hr style='border-color:rgba(232,184,75,0.1);margin:1rem 0;'>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            st.session_state['detail_movie'] = None
            st.rerun()

    st.markdown("""
    <hr style="border-color:rgba(232,184,75,0.1);margin:1.5rem 0 0;">
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: MOVIE DETAIL
# ─────────────────────────────────────────────────────────────
if st.session_state.get('detail_movie'):
    m = st.session_state['detail_movie']
    title    = m['title']
    genres   = m['genres'] if isinstance(m['genres'], list) else []
    avg_r    = m.get('avg_rating')

    with st.spinner("Loading film details…"):
        details = fetch_movie_details(title)

    poster_url   = details.get('poster_url') or fetch_movie_poster(title)
    overview     = details.get('overview') or '-'
    tmdb_score   = details.get('vote_average')
    vote_count   = details.get('vote_count')
    release_date = details.get('release_date', '')
    release_year = release_date[:4] if release_date else ''

    badges = genre_badges(genres)

    # ── Detail layout: poster left / info right ──
    col_poster, col_info = st.columns([1, 2], gap="large")

    with col_poster:
        if poster_url:
            st.markdown(
                f'<img src="{poster_url}" style="width:100%;border-radius:14px;'
                f'box-shadow:0 8px 40px rgba(0,0,0,0.6);display:block;">',
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
            <div style="width:100%;height:380px;background:#16161f;border:1px solid rgba(232,184,75,0.15);
                        border-radius:14px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;gap:0.5rem;">
                <div style="font-size:3rem;opacity:0.2;">🎬</div>
                <div style="font-size:0.8rem;color:#888899;">No Poster Available</div>
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        # Title + year
        st.markdown(f"""
        <div style="margin-bottom:0.6rem;">
            <h1 style="font-family:'Playfair Display',serif;font-size:2.2rem;
                       font-weight:900;margin:0 0 0.3rem;line-height:1.15;">{title}</h1>
            {'<div style="color:#888899;font-size:0.9rem;margin-bottom:0.5rem;">' + release_year + '</div>' if release_year else ''}
        </div>
        """, unsafe_allow_html=True)
        # Genre badges rendered separately to avoid HTML escaping issues
        st.markdown(f'<div style="margin-bottom:1rem;">{badges}</div>', unsafe_allow_html=True)

        # Ratings row
        st.markdown("""
        <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;
                    color:#888899;margin-bottom:0.6rem;">Ratings</div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            if tmdb_score is not None:
                stars = min(int(tmdb_score / 2) + (1 if tmdb_score % 2 >= 0.5 else 0), 5)
                star_str = "★" * stars + "☆" * (5 - stars)
                st.markdown(f"""
                <div style="background:#16161f;border:1px solid rgba(232,184,75,0.15);
                            border-radius:12px;padding:1rem 1.2rem;">
                    <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;
                                color:#888899;margin-bottom:0.3rem;">TMDB Score</div>
                    <div style="color:#E8B84B;font-family:'Playfair Display',serif;
                                font-size:2rem;font-weight:700;">{tmdb_score:.1f}<span style="font-size:1rem;color:#555566;">/10</span></div>
                    <div style="color:#E8B84B;font-size:0.9rem;margin-top:0.2rem;">{star_str}</div>
                    {f'<div style="color:#555566;font-size:0.72rem;margin-top:0.2rem;">{vote_count:,} votes</div>' if vote_count else ''}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#16161f;border:1px solid rgba(232,184,75,0.15);
                            border-radius:12px;padding:1rem 1.2rem;">
                    <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;
                                color:#888899;margin-bottom:0.3rem;">TMDB Score</div>
                    <div style="color:#555566;font-size:1.5rem;">—</div>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            if avg_r is not None and pd.notna(avg_r):
                stars2 = min(int(avg_r / 1) + (1 if avg_r % 1 >= 0.5 else 0), 5)
                star_str2 = "★" * stars2 + "☆" * (5 - stars2)
                st.markdown(f"""
                <div style="background:#16161f;border:1px solid rgba(232,184,75,0.15);
                            border-radius:12px;padding:1rem 1.2rem;">
                    <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;
                                color:#888899;margin-bottom:0.3rem;">User Rating (Dataset)</div>
                    <div style="color:#E8B84B;font-family:'Playfair Display',serif;
                                font-size:2rem;font-weight:700;">{avg_r:.2f}<span style="font-size:1rem;color:#555566;">/5</span></div>
                    <div style="color:#E8B84B;font-size:0.9rem;margin-top:0.2rem;">{star_str2}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#16161f;border:1px solid rgba(232,184,75,0.15);
                            border-radius:12px;padding:1rem 1.2rem;">
                    <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;
                                color:#888899;margin-bottom:0.3rem;">User Rating (Dataset)</div>
                    <div style="color:#555566;font-size:1.5rem;">—</div>
                </div>
                """, unsafe_allow_html=True)

        # Synopsis
        st.markdown("""
        <div style="margin-top:1.4rem;">
            <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;
                        color:#888899;margin-bottom:0.6rem;">Synopsis</div>
        """, unsafe_allow_html=True)

        if overview and overview != '-':
            st.markdown(f"""
            <div style="background:#16161f;border:1px solid rgba(232,184,75,0.1);
                        border-radius:12px;padding:1.1rem 1.3rem;
                        color:#e8e6e0;font-size:0.93rem;line-height:1.75;">
                {overview}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#16161f;border:1px solid rgba(232,184,75,0.1);
                        border-radius:12px;padding:1.1rem 1.3rem;
                        color:#555566;font-size:1.1rem;text-align:center;">
                —
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()  # Don't render any other page content


# ─────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────
if page == "🏠  Home":
    # Hero
    st.markdown("""
    <div style="position:relative;padding:4rem 2rem 3.5rem;margin:-2rem -2.5rem 2rem;
                background:linear-gradient(135deg,#0a0a0f 0%,#12121e 50%,#0f0f18 100%);
                border-bottom:1px solid rgba(232,184,75,0.12);overflow:hidden;">
        <div style="position:absolute;top:-60px;right:-60px;width:320px;height:320px;
                    background:radial-gradient(circle,rgba(232,184,75,0.06) 0%,transparent 70%);
                    border-radius:50%;pointer-events:none;"></div>
        <div style="position:absolute;bottom:-80px;left:10%;width:500px;height:200px;
                    background:radial-gradient(ellipse,rgba(192,57,43,0.05) 0%,transparent 70%);
                    pointer-events:none;"></div>
        <div style="position:relative;max-width:800px;">
            <h1 style="font-family:'Playfair Display',serif;font-size:clamp(2.2rem,4vw,3.6rem);
                       font-weight:900;line-height:1.1;margin:0 0 1rem;
                       background:linear-gradient(135deg,#e8e6e0,#E8B84B 60%,#f5cc6a);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                Discover Your Next<br>Favorite Film
            </h1>
            <p style="color:#888899;font-size:1.05rem;max-width:520px;line-height:1.7;margin:0;">
                CineMatch blends collaborative filtering with content-based intelligence 
                to surface films you'll love — before you even know to look for them.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats Row
    avg_rating = ratings_data['rating'].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎥 Total Films", f"{len(movies_data):,}")
    c2.metric("⭐ Total Ratings", f"{len(ratings_data):,}")
    c3.metric("🎭 Unique Genres", f"{len(all_genres)}")
    c4.metric("📊 Avg Rating", f"{avg_rating:.2f} / 5")

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <h2 style="font-family:'Playfair Display',serif;font-size:1.6rem;margin-bottom:1.5rem;">
        How It Works
    </h2>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "🔍", "Search a Film", "Type any movie title and our search engine finds it instantly using TF-IDF similarity."),
        (col2, "🤝", "Find Your Tribe", "We identify users with similar taste who highly rated the same films."),
        (col3, "✨", "Get Matched", "Genre-weighted scoring surfaces movies that fit your vibe — not just the obvious picks."),
    ]:
        col.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:14px;
                    padding:1.5rem;height:100%;transition:all 0.3s;">
            <div style="font-size:2rem;margin-bottom:0.8rem;">{icon}</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;
                        color:#e8e6e0;margin-bottom:0.5rem;">{title}</div>
            <div style="color:#888899;font-size:0.88rem;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Top-rated sample
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style="font-family:'Playfair Display',serif;font-size:1.6rem;margin-bottom:1.2rem;">
        🔥 Top Rated Films in Dataset
    </h2>
    """, unsafe_allow_html=True)

    top_movies = (
        combined_data.groupby(['movieId', 'title'])['rating']
        .agg(['mean', 'count'])
        .reset_index()
    )
    top_movies = top_movies[top_movies['count'] >= 50].sort_values('mean', ascending=False).head(6)

    cols = st.columns(6)
    for i, (_, row) in enumerate(top_movies.iterrows()):
        with cols[i]:
            movie_genres = movies_data.loc[movies_data['title'] == row['title'], 'genres']
            genres_val = movie_genres.iloc[0] if len(movie_genres) > 0 else []
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
                        padding:1rem 0.8rem;text-align:center;margin-bottom:0.5rem;">
                <div style="font-size:1.8rem;margin-bottom:0.5rem;">🎬</div>
                <div style="font-family:'Playfair Display',serif;font-size:0.82rem;font-weight:700;
                            color:#e8e6e0;line-height:1.3;min-height:40px;">{row['title'][:30]}{'…' if len(row['title'])>30 else ''}</div>
                <div style="color:#E8B84B;font-size:1.1rem;font-weight:700;margin-top:0.5rem;">
                    ★ {row['mean']:.2f}
                </div>
                <div style="color:#555566;font-size:0.72rem;">{int(row['count']):,} ratings</div>
            </div>
            """, unsafe_allow_html=True)
            movie_detail_button(row['title'], genres_val, row['mean'], button_key=f"home_{i}")


# ─────────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────
elif page == "📊  Analytics":
    st.markdown("""
    <h1 style="font-family:'Playfair Display',serif;font-size:2.4rem;margin-bottom:0.3rem;">
        Dataset Analytics
    </h1>
    <p style="color:#888899;margin-bottom:2rem;">Deep-dive into the movie landscape behind the recommendations.</p>
    """, unsafe_allow_html=True)

    DARK_PARAMS = dict(
        facecolor='#111118', text_color='#e8e6e0',
        grid_color=(1.0, 1.0, 1.0, 0.06), spine_color='#1e1e2e', accent='#E8B84B'
    )

    tab1, tab2, tab3 = st.tabs(["🎭 Genre Landscape", "⭐ Rating Distribution", "📈 Genre vs Rating"])

    with tab1:
        genre_counts = pd.Series([g for gl in movies_data['genres'] for g in gl]).value_counts()
        fig, ax = plt.subplots(figsize=(13, 6), facecolor=DARK_PARAMS['facecolor'])
        ax.set_facecolor(DARK_PARAMS['facecolor'])
        bars = ax.bar(genre_counts.index, genre_counts.values,
                      color=[DARK_PARAMS['accent'] if i == 0 else '#2a2a3e' for i in range(len(genre_counts))],
                      edgecolor='none', width=0.7)
        ax.set_title("Number of Films per Genre", color=DARK_PARAMS['text_color'],
                     fontfamily='serif', fontsize=15, fontweight='bold', pad=16)
        ax.set_xlabel("Genre", color=DARK_PARAMS['text_color'], fontsize=10)
        ax.set_ylabel("Films", color=DARK_PARAMS['text_color'], fontsize=10)
        ax.tick_params(colors=DARK_PARAMS['text_color'], labelsize=8.5)
        plt.xticks(rotation=40, ha='right')
        for spine in ax.spines.values():
            spine.set_color(DARK_PARAMS['spine_color'])
        ax.yaxis.grid(True, color=DARK_PARAMS['grid_color'], linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_PARAMS['facecolor'])
        ax.set_facecolor(DARK_PARAMS['facecolor'])
        n, bins, patches = ax.hist(ratings_data['rating'], bins=10,
                                   color=DARK_PARAMS['accent'], edgecolor='#0a0a0f', linewidth=0.5, alpha=0.85)
        ax.set_title("Rating Distribution", color=DARK_PARAMS['text_color'],
                     fontfamily='serif', fontsize=15, fontweight='bold', pad=16)
        ax.set_xlabel("Rating", color=DARK_PARAMS['text_color'], fontsize=10)
        ax.set_ylabel("Frequency", color=DARK_PARAMS['text_color'], fontsize=10)
        ax.tick_params(colors=DARK_PARAMS['text_color'], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(DARK_PARAMS['spine_color'])
        ax.yaxis.grid(True, color=DARK_PARAMS['grid_color'], linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)

    with tab3:
        # Fast explode instead of slow iterrows
        gr_df = (
            combined_data[['movieId', 'rating', 'genres']]
            .copy()
        )
        gr_df['genres'] = gr_df['genres'].apply(lambda x: x if isinstance(x, list) else [x])
        gr_df = gr_df.explode('genres')
        gr_df = gr_df[gr_df['genres'].notna() & (gr_df['genres'] != '')]
        avg_by_genre = gr_df.groupby('genres')['rating'].mean().sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(9, 6), facecolor=DARK_PARAMS['facecolor'])
        ax.set_facecolor(DARK_PARAMS['facecolor'])
        colors_bar = ['#E8B84B' if v == avg_by_genre.max() else '#2a2a3e' for v in avg_by_genre.values]
        ax.barh(avg_by_genre.index, avg_by_genre.values, color=colors_bar, edgecolor='none', height=0.65)
        ax.set_title("Average Rating by Genre (Top 15)", color=DARK_PARAMS['text_color'],
                     fontfamily='serif', fontsize=15, fontweight='bold', pad=16)
        ax.set_xlabel("Average Rating", color=DARK_PARAMS['text_color'], fontsize=10)
        ax.tick_params(colors=DARK_PARAMS['text_color'], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(DARK_PARAMS['spine_color'])
        ax.xaxis.grid(True, color=DARK_PARAMS['grid_color'], linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)


# ─────────────────────────────────────────────────────────────
# PAGE: RECOMMEND
# ─────────────────────────────────────────────────────────────
elif page == "🎯  Recommend":
    st.markdown("""
    <h1 style="font-family:'Playfair Display',serif;font-size:2.4rem;margin-bottom:0.3rem;">
        Find Your Next Film
    </h1>
    <p style="color:#888899;margin-bottom:1.5rem;">
        Enter a movie you love — we'll find what to watch next.
    </p>
    """, unsafe_allow_html=True)

    # Genre filter in sidebar
    with st.sidebar:
        st.markdown("""
        <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;
                    color:#888899;margin-bottom:0.5rem;margin-top:1rem;">Filter by Genre</div>
        """, unsafe_allow_html=True)
        selected_genres_filter = st.multiselect(
            "", options=all_genres, default=[], label_visibility="collapsed"
        )

    # Search bar
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_input("", placeholder="e.g. The Dark Knight, Jumanji, Interstellar…",
                                   label_visibility="collapsed")
    with col_btn:
        search_btn = st.button("Search", use_container_width=True)

    if search_btn or user_input:
        candidates = search_by_title(user_input)
        if len(candidates) == 0:
            st.warning(f"No results for **{user_input}**. Try a different spelling.")
        else:
            st.markdown("""
            <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;
                        color:#888899;margin:1rem 0 0.3rem;">Did you mean?</div>
            """, unsafe_allow_html=True)

            title_options = candidates['title'].tolist()
            selected_title = st.selectbox("", title_options, label_visibility="collapsed")

            st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
            rec_btn = st.button("✨ Get Recommendations", type="primary", use_container_width=False)

            if rec_btn:
                idx = title_options.index(selected_title)
                with st.spinner("Curating your personal watchlist…"):
                    recs = recommendation_results(user_input, idx, selected_genres_filter)
                # Store in session state so results survive View Details rerun
                st.session_state['last_recs'] = recs.to_dict('records') if len(recs) > 0 else []
                st.session_state['last_recs_title'] = selected_title
                st.session_state['last_recs_genres_filter'] = selected_genres_filter

            # Render recommendations from session state (survives rerun)
            recs_data = st.session_state.get('last_recs', [])
            recs_title = st.session_state.get('last_recs_title', '')
            recs_filter = st.session_state.get('last_recs_genres_filter', [])

            if recs_data:
                recs = pd.DataFrame(recs_data)
                if len(recs) == 0:
                    st.warning("No recommendations found. Try removing genre filters or a different film.")
                else:
                    # Header
                    st.markdown(f"""
                    <div style="margin:2rem 0 1.5rem;">
                        <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;
                                    color:#E8B84B;margin-bottom:0.4rem;">Because you like</div>
                        <h2 style="font-family:'Playfair Display',serif;font-size:2rem;margin:0;">
                            {recs_title}
                        </h2>
                        {'<div style="margin-top:0.5rem;font-size:0.83rem;color:#888899;">Filtered by: ' + ', '.join(f'<span style="color:#E8B84B;">{g}</span>' for g in recs_filter) + '</div>' if recs_filter else ''}
                    </div>
                    <hr style="border-color:rgba(232,184,75,0.12);margin-bottom:1.5rem;">
                    """, unsafe_allow_html=True)

                    # Recommendation cards — 2-column grid
                    for pair_start in range(0, len(recs), 2):
                        cols = st.columns(2, gap="medium")
                        for ci, (_, row) in enumerate(recs.iloc[pair_start:pair_start+2].iterrows()):
                            with cols[ci]:
                                poster_url = fetch_movie_poster(row['title'])
                                badges = genre_badges(row['genres'])
                                title_text = row['title']
                                genres_list = row['genres']

                                # ── Card top border ──
                                st.markdown(f"""
                                <div style="background:#16161f;border:1px solid rgba(232,184,75,0.15);
                                            border-radius:12px;overflow:hidden;margin-bottom:0.5rem;">
                                """, unsafe_allow_html=True)

                                # ── Poster ──
                                if poster_url:
                                    st.markdown(
                                        f'<img src="{poster_url}" style="width:100%;height:320px;'
                                        f'object-fit:contain;background:#0d0d18;display:block;">',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown("""
                                    <div style="width:100%;height:220px;background:#f5f5f5;
                                                display:flex;flex-direction:column;
                                                align-items:center;justify-content:center;gap:0.5rem;">
                                        <div style="font-size:2.5rem;opacity:0.25;filter:grayscale(1);">🎬</div>
                                        <div style="font-size:0.8rem;color:#999;font-weight:500;
                                                    letter-spacing:0.05em;">No Image</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # ── Info body ──
                                st.markdown(f"""
                                <div style="padding:1rem 1.1rem 0.4rem;">
                                    <div style="font-family:'Playfair Display',serif;font-size:1rem;
                                                font-weight:700;color:#e8e6e0;line-height:1.35;
                                                margin-bottom:0.55rem;">{title_text}</div>
                                    <div style="font-size:0.68rem;color:#888899;text-transform:uppercase;
                                                letter-spacing:0.09em;margin-bottom:0.25rem;">Match Score</div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Badges rendered separately
                                st.markdown(f'<div style="padding:0 1.1rem 0.4rem;">{badges}</div>', unsafe_allow_html=True)

                                # ── Score bar ──
                                score_pct = min(row['score'] / 10 * 100, 100)
                                stars = min(int(row['score'] / 2) + 1, 5)
                                star_str = "★" * stars + "☆" * (5 - stars)
                                st.markdown(f"""
                                <div style="padding:0 1.1rem 1rem;">
                                    <div style="display:flex;align-items:center;gap:10px;">
                                        <div style="flex:1;background:#1a1a26;border-radius:99px;height:6px;overflow:hidden;">
                                            <div style="width:{score_pct:.1f}%;height:100%;
                                                        background:linear-gradient(90deg,#E8B84B,#f5cc6a);
                                                        border-radius:99px;"></div>
                                        </div>
                                        <span style="color:#E8B84B;font-size:0.85rem;font-weight:600;white-space:nowrap;">{row['score']:.2f}</span>
                                        <span style="color:#E8B84B;font-size:0.85rem;">{star_str}</span>
                                    </div>
                                </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # ── View Details button ──
                                movie_detail_button(
                                    title_text, genres_list,
                                    button_key=f"rec_{pair_start}_{ci}"
                                )


# ─────────────────────────────────────────────────────────────
# PAGE: BROWSE GENRE
# ─────────────────────────────────────────────────────────────
elif page == "🎭  Browse Genre":
    st.markdown("""
    <h1 style="font-family:'Playfair Display',serif;font-size:2.4rem;margin-bottom:0.3rem;">
        Browse by Genre
    </h1>
    <p style="color:#888899;margin-bottom:1.5rem;">Explore the full catalog filtered by your favourite genres.</p>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;
                    color:#888899;margin-bottom:0.5rem;margin-top:1rem;">Select Genres</div>
        """, unsafe_allow_html=True)
        browse_genres = st.multiselect(
            "", options=all_genres,
            default=["Drama"] if "Drama" in all_genres else [],
            label_visibility="collapsed"
        )
        st.markdown("""
        <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;
                    color:#888899;margin-bottom:0.5rem;margin-top:1.2rem;">Options</div>
        """, unsafe_allow_html=True)
        num_movies = st.slider("Films to show", 6, 48, 18, step=6)
        sort_by = st.selectbox("Sort by", ["Rating ↓ (Highest)", "Rating ↑ (Lowest)", "Title A→Z", "Title Z→A"])

    if browse_genres:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;flex-wrap:wrap;">
            <span style="color:#888899;font-size:0.85rem;">Browsing:</span>
            {''.join(f'<span style="background:rgba(232,184,75,0.1);border:1px solid rgba(232,184,75,0.3);color:#E8B84B;padding:3px 12px;border-radius:20px;font-size:0.82rem;">{g}</span>' for g in browse_genres)}
        </div>
        """, unsafe_allow_html=True)

        filtered = movies_data.copy()
        for g in browse_genres:
            filtered = filtered[filtered['genres'].apply(lambda x: g in x)]

        avg_r = combined_data.groupby('movieId')['rating'].mean().reset_index()
        filtered = filtered.merge(avg_r, on='movieId', how='left').rename(columns={'rating': 'avg_rating'})

        sort_map = {
            "Rating ↓ (Highest)": ('avg_rating', False),
            "Rating ↑ (Lowest)": ('avg_rating', True),
            "Title A→Z": ('title', True),
            "Title Z→A": ('title', False),
        }
        col_s, asc_s = sort_map[sort_by]
        filtered = filtered.sort_values(col_s, ascending=asc_s).head(num_movies)

        if len(filtered) == 0:
            st.warning("No films match the selected genres.")
        else:
            st.caption(f"{len(filtered)} films found")
            cols = st.columns(3, gap="medium")
            for i, (_, row) in enumerate(filtered.iterrows()):
                with cols[i % 3]:
                    badges = genre_badges(row['genres'])
                    rating_txt = f"★ {row['avg_rating']:.2f}" if pd.notna(row.get('avg_rating')) else "—"
                    avg_r_val = row.get('avg_rating') if pd.notna(row.get('avg_rating')) else None
                    st.markdown(f"""
                    <div style="background:var(--card);border:1px solid var(--border);
                                border-radius:12px;padding:1.2rem;margin-bottom:0.5rem;">
                        <div style="font-family:'Playfair Display',serif;font-size:0.98rem;
                                    font-weight:700;color:#e8e6e0;margin-bottom:0.6rem;
                                    line-height:1.3;">{row['title']}</div>
                        <div style="margin-bottom:0.7rem;">{badges}</div>
                        <div style="display:flex;align-items:center;justify-content:space-between;">
                            <span style="color:#E8B84B;font-size:1rem;font-weight:600;">{rating_txt}</span>
                            <span style="color:#555566;font-size:0.72rem;">/5.0</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    movie_detail_button(row['title'], row['genres'], avg_r_val, button_key=f"browse_{i}")

    else:
        # Genre discovery grid
        st.markdown("""
        <h3 style="font-family:'Playfair Display',serif;font-size:1.3rem;margin-bottom:1rem;">
            🔥 Popular Genres
        </h3>
        """, unsafe_allow_html=True)
        genre_counts = pd.Series([g for gl in movies_data['genres'] for g in gl]).value_counts().head(16)
        gcols = st.columns(4)
        for i, (genre, count) in enumerate(genre_counts.items()):
            with gcols[i % 4]:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;
                            padding:1rem;margin-bottom:0.8rem;text-align:center;">
                    <div style="font-family:'Playfair Display',serif;font-size:1rem;
                                font-weight:700;color:#e8e6e0;margin-bottom:0.2rem;">{genre}</div>
                    <div style="color:#E8B84B;font-size:0.82rem;">{count:,} films</div>
                </div>
                """, unsafe_allow_html=True)
        st.info("👈 Pick a genre from the sidebar to start browsing!")
