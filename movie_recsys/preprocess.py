"""
Pre-process raw data for Content Based Filtering

"""

import pandas as pd
from pathlib import Path


def make_dataset():
    """
    Make dataset for training.

    Yields
    ------
        movie_data: .csv in datasets.
    """
    meta = pd.read_csv("movie_recsys/datasets/bollywood_meta_2010-2019.csv")
    ratings = pd.read_csv("movie_recsys/datasets/bollywood_ratings_2010-2019.csv")
    plot = pd.read_csv("movie_recsys/datasets/bollywood_text_2010-2019.csv")

    # Clean year_of_release.
    meta = meta[meta["year_of_release"] != r"\N"]
    meta["year_of_release"] = meta["year_of_release"].astype("int")    # Change year_of_release to int.
    meta = meta[meta["year_of_release"] >= 2010]    # Movies released after 2010 only.
    movie_data = meta.drop(["original_title", "is_adult"], axis=1)

    # Fixing Movie Titles.
    movie_data["title"] = movie_data["title"].str.lower()
    movie_data["title"] = movie_data["title"].str.split(":", expand=True)[0]        # Removing movie tag lines.

    # Fixing Genres.
    movie_data["genres"] = movie_data["genres"].str.lower()
    movie_data["genres"] = movie_data["genres"].str.split("|", expand=True)[0]

    # Filter Movies Less than 7 IMDb rating.
    movie_data = movie_data.merge(ratings[["imdb_id", "imdb_rating"]], how="left", on="imdb_id")
    movie_data = movie_data[movie_data["imdb_rating"] >= 7.0]

    # Merge Plot.
    movie_data = movie_data.merge(plot[["story", "imdb_id", "summary"]], how="left", on="imdb_id")

    # Drop meta data.
    movie_data.drop(["runtime", "imdb_rating"], axis=1, inplace=True)

    # Remove Duplicates and NaN.
    movie_data.dropna(axis=0, inplace=True)
    movie_data.drop_duplicates(inplace=True)

    file_path = Path.cwd() / "movie_recsys/datasets/movie_data.csv"
    movie_data.to_csv(file_path, index=False)
    return
