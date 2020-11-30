"""
Computes Similar Measures for all movies.
"""

import pandas as pd
from numpy import savetxt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path


def compute_similarity():
    """
    Computes Cosine Similarity between movies.

    Yields
    ------
        cosine_similarity_scores.csv
    """
    movie_data = pd.read_csv("movie_recsys/datasets/movie_data.csv")

    # Compute TF-IDF representation.
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movie_data["story"])

    # Compute Cosine Similarity.
    cosine_sim_scores = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Saving.
    file_path = Path.cwd() / "movie_recsys/datasets/cosine_sim_scores.csv"
    savetxt(file_path, cosine_sim_scores)
    return