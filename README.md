🎬 User-Based Movie Recommendation System (Python)


A simple yet effective user-based collaborative filtering recommendation system built with Python, pandas, and scikit-learn. This system recommends movies to a user by analyzing the preferences of similar users using cosine similarity.


📌 Features
User-based collaborative filtering
Calculates similarity using cosine similarity
Predicts unseen movie scores using weighted averages
Returns top N personalized movie recommendations
Built entirely with pandas and scikit-learn (no deep learning)


📊 How It Works
A user-item ratings matrix is created (users as rows, movies as columns).
Cosine similarity is computed between the target user and all others.
Ratings from similar users are weighted by their similarity scores.
Already-watched movies are excluded.
The top N movies with the highest predicted ratings are returned as recommendations.


📦 Libraries Used
pandas
scikit-learn (for cosine_similarity)
