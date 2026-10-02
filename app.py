# =========================================
# SMART BOOK RECOMMENDATION SYSTEM
# Final Beginner-Friendly Version
# =========================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------
# App Configuration
# -----------------------------------------
st.set_page_config(page_title="Book Recommender", layout="wide")

st.title(" Book Recommendation System")
st.write("This system recommends books using popularity, user behaviour, and similarity")

# -----------------------------------------
# Load Saved Data
# -----------------------------------------
@st.cache_resource
def load_data():
    pt = pickle.load(open("pt.pkl", "rb"))                     # Book × User rating matrix
    books = pickle.load(open("books.pkl", "rb"))               # Book details
    similarity = pickle.load(open("similarity_scores.pkl", "rb"))  # Item similarity
    return pt, books, similarity

pt, books_df, similarity_scores = load_data()

# -----------------------------------------
# Get Book Details
# -----------------------------------------
def get_book_info(book_title):
    book = books_df[books_df["Book-Title"] == book_title].iloc[0]
    return book["Book-Author"], book["Image-URL-M"]

# -----------------------------------------
# Explain Recommendation (Human-Friendly)
# -----------------------------------------
def explain_recommendation(rec_type):
    if rec_type == "🔥 Popularity Based":
        return "Recommended because many users rated this book highly."
    elif rec_type == "📘 Item-Based Collaborative":
        return "Recommended because it is similar to the book you selected."
    elif rec_type == "👤 User-Based Collaborative":
        return "Recommended based on users with similar reading preferences."
    else:
        return "Recommended using a combination of popularity and similarity."

# -----------------------------------------
#1️⃣ Popularity-Based Recommendation
# -----------------------------------------
def popularity_based(n):
    rating_count = pt.notnull().sum(axis=1)
    avg_rating = pt.mean(axis=1)

    popular_df = pd.DataFrame({
        "Book-Title": rating_count.index,
        "rating_count": rating_count.values,
        "avg_rating": avg_rating.values
    })

    popular_df = popular_df[popular_df["rating_count"] >= 50]
    popular_df = popular_df.sort_values(
        by=["avg_rating", "rating_count"],
        ascending=False
    )

    return popular_df.head(n)["Book-Title"].values

# -----------------------------------------
# 2️⃣ Item-Based Collaborative Filtering
# -----------------------------------------
def item_based_recommend(book_name, n):
    index = np.where(pt.index == book_name)[0][0]
    distances = similarity_scores[index]

    similar_books = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:n+1]

    return [pt.index[i[0]] for i in similar_books]

# -----------------------------------------
# 3️⃣ User-Based Collaborative Filtering (FIXED + FALLBACK)
# -----------------------------------------
def user_based_recommend(user_id, n):
    # If user has very few ratings → fallback to popularity
    user_ratings = pt[user_id].dropna()
    if len(user_ratings) < 3:
        return popularity_based(n)

    # Create user-item matrix
    user_item_matrix = pt.fillna(0).T

    # Compute cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix)

    user_index = list(user_item_matrix.index).index(user_id)
    similarity_scores_user = user_similarity[user_index]

    similar_users = sorted(
        list(enumerate(similarity_scores_user)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommended_books = set()

    for user, score in similar_users:
        books_rated = user_item_matrix.iloc[user][user_item_matrix.iloc[user] > 0].index
        for book in books_rated:
            if pt.loc[book, user_id] == 0:
                recommended_books.add(book)

    return list(recommended_books)[:n]

# -----------------------------------------
# 4️⃣ Hybrid Recommendation (BEST)
# -----------------------------------------
def hybrid_recommend(book_name, n):
    index = np.where(pt.index == book_name)[0][0]
    similarity_list = similarity_scores[index]

    rating_count = pt.notnull().sum(axis=1)
    avg_rating = pt.mean(axis=1)

    scores = []

    for i, sim in enumerate(similarity_list):
        title = pt.index[i]
        if rating_count[title] >= 50:
            final_score = (0.7 * sim) + (0.3 * (avg_rating[title] / 10))
            scores.append((title, final_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [book[0] for book in scores[1:n+1]]

# -----------------------------------------
# Sidebar Controls
# -----------------------------------------
st.sidebar.header("⚙ Recommendation Settings")

rec_type = st.sidebar.radio(
    "Select Recommendation Type",
    [
        "🔥 Popularity Based",
        "📘 Item-Based Collaborative",
        "👤 User-Based Collaborative",
        "🚀 Hybrid (Best)"
    ]
)

top_n = st.sidebar.slider("Number of recommendations", 3, 20, 10)

# -----------------------------------------
# UI – Popularity Based
# -----------------------------------------
if rec_type == "🔥 Popularity Based":
    st.subheader("🔥 Popular Books")

    books_list = popularity_based(top_n)
    cols = st.columns(top_n)

    for col, book in zip(cols, books_list):
        author, image = get_book_info(book)
        with col:
            st.image(image)
            st.markdown(f"**{book}**")
            st.caption(author)
            st.caption(explain_recommendation(rec_type))

# -----------------------------------------
# UI – Item-Based
# -----------------------------------------
elif rec_type == "📘 Item-Based Collaborative":
    st.subheader("📘 Similar Books")

    selected_book = st.selectbox("Select a book", pt.index)

    if st.button("Recommend"):
        results = item_based_recommend(selected_book, top_n)
        cols = st.columns(top_n)

        for col, book in zip(cols, results):
            author, image = get_book_info(book)
            with col:
                st.image(image)
                st.markdown(f"**{book}**")
                st.caption(author)
                st.caption(explain_recommendation(rec_type))

# -----------------------------------------
# UI – User-Based
# -----------------------------------------
elif rec_type == "👤 User-Based Collaborative":
    st.subheader("👤 User-Based Recommendations")

    user_id = st.selectbox("Select a user", pt.columns)

    if st.button("Recommend for User"):
        results = user_based_recommend(user_id, top_n)
        cols = st.columns(top_n)

        for col, book in zip(cols, results):
            author, image = get_book_info(book)
            with col:
                st.image(image)
                st.markdown(f"**{book}**")
                st.caption(author)
                st.caption(explain_recommendation(rec_type))

# -----------------------------------------
# UI – Hybrid
# -----------------------------------------
else:
    st.subheader("🚀 Hybrid Recommendations")

    selected_book = st.selectbox("Select a book", pt.index)

    if st.button("Get Best Recommendations"):
        results = hybrid_recommend(selected_book, top_n)
        cols = st.columns(top_n)

        for col, book in zip(cols, results):
            author, image = get_book_info(book)
            with col:
                st.image(image)
                st.markdown(f"**{book}**")
                st.caption(author)
                st.caption(explain_recommendation(rec_type))
