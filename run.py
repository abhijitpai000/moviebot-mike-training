"""
Train models and generate required files using Terminal.
"""

# Movie Recommendation.
from movie_recsys.preprocess import make_dataset
from movie_recsys.content_filtering import compute_similarity

# Intent Classifier
from intent_classifier.train import train_model

if __name__ == '__main__':
    # Movie Recommendation.
    make_dataset()
    compute_similarity()

    # Training Intent Classifier.
    train_model()
    print("Training Completed")
