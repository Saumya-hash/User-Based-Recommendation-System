import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
ratings_dict = {
    'Movie A': [5, 4, 1, 0, 0],
    'Movie B': [0, 0, 1, 1, 0],
    'Movie C': [0, 0, 0, 5, 5],
    'Movie D': [2, 3, 5, 4, 1],
    'Movie E': [1, 1, 4, 0, 0]
}

users = ['User A', 'User B', 'User C', 'User D', 'User E']

ratings_df = pd.DataFrame(ratings_dict, index=users)

def user_based_recommendation(target_user, ratings, top_n=3):
    sim_matrix = pd.DataFrame(
        cosine_similarity(ratings),
        index=ratings.index,
        columns=ratings.index
    )

    user_similarities = sim_matrix.loc[target_user].drop(target_user)

    weighted_ratings_sum = ratings.loc[user_similarities.index].T.dot(user_similarities)

    sum_similarities = user_similarities.sum()

    if sum_similarities == 0:
        weighted_avg = weighted_ratings_sum
    else:
        weighted_avg = weighted_ratings_sum / sum_similarities

    already_rated = ratings.loc[target_user][ratings.loc[target_user] > 0].index
    rec = weighted_avg.drop(already_rated)

    return rec.sort_values(ascending=False).head(top_n)

if __name__ == "__main__":
    print("User-Item Ratings:\n", ratings_df)
    user = 'User B'
    recommendations = user_based_recommendation(user, ratings_df, top_n=3)
    print(f"\nTop recommendations for {user}:")
    for idx, (item, score) in enumerate(recommendations.items(), 1):
        print(f"{idx}. {item} (Score: {score:.2f})")
