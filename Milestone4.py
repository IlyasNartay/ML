import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# DTO-класс для обработки входных запросов
class RecommendationRequest:
    def __init__(self, gender: str, age: int, country: str, query: str):
        self.gender = gender.lower()
        self.age = age
        self.country = country.lower()
        self.query = query.lower()

class MovieRecommendationModel:
    def __init__(self, movie_data: pd.DataFrame):
        """
        Initialize the recommendation model with movie data.

        Parameters:
            movie_data (pd.DataFrame): A DataFrame containing movie information with columns such as 'Movie Name', 'Genres', 'Overview', etc.
        """
        self.movie_data = movie_data.copy()
        self.movie_data['content'] = (
            self.movie_data['Movie Name'].str.lower().fillna('') + ' ' +
            self.movie_data['Genres'].fillna('') + ' ' +
            self.movie_data['Overview'].str.lower().fillna('')
        )
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movie_data['content'])

    def recommend(self, request: RecommendationRequest, top_n: int = 3):
        """
        Recommend movies based on the user's query and preferences.

        Parameters:
            request (RecommendationRequest): A DTO containing user preferences and search query.
            top_n (int): Number of top recommendations to return.

        Returns:
            pd.DataFrame: A DataFrame containing recommended movies with their details.
        """
        # Extract query keywords
        user_query = f"{request.query} {request.gender} {request.age} {request.country}"
        query_vector = self.vectorizer.transform([user_query])

        # Compute similarity scores
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Filter by genres if keywords are present in query
        genre_keywords = [
            'adventure', 'sci-fi', 'science fiction', 'action', 'fantasy', 'drama', 'horror', 'thriller',
            'comedy', 'romance', 'documentary', 'biography', 'mystery', 'crime', 'war', 'musical', 'animation',
            'family', 'history', 'western', 'superhero', 'psychological', 'epic', 'space', 'alien', 'future',
            'supernatural', 'paranormal', 'detective', 'sports', 'survival'
        ]
        filtered_data = self.movie_data

        for keyword in genre_keywords:
            if keyword in request.query:
                filtered_data = filtered_data[filtered_data['Genres'].str.contains(keyword, case=False, na=False)]
                break  # Use the first matching keyword to filter

        # Recompute similarity scores for the filtered dataset
        if not filtered_data.empty:
            filtered_indices = filtered_data.index
            filtered_similarity_scores = similarity_scores[filtered_indices]
            top_indices = filtered_indices[filtered_similarity_scores.argsort()[-top_n:][::-1]]
        else:
            top_indices = similarity_scores.argsort()[-top_n:][::-1]

        # Retrieve recommendations
        recommendations = self.movie_data.iloc[top_indices].copy()
        recommendations.loc[:, 'similarity'] = similarity_scores[top_indices]

        # Sort by similarity and popularity (if available)
        if 'Popularity' in recommendations.columns:
            recommendations = recommendations.sort_values(by=['similarity', 'Popularity'], ascending=[False, False])

        return recommendations[['Movie Name', 'Poster path', 'Overview']]

# # Load the dataset
# file_path = 'tmdb_movies_cleaned.csv'
# movie_data = pd.read_csv(file_path)
#
# # Initialize the recommendation model
# model = MovieRecommendationModel(movie_data)

# # Create recommendation requests
# request_user1 = RecommendationRequest(
#     gender="male",
#     age=25,
#     country="USA",
#     query="An epic adventure in space with heroic battles and stunning visuals."
# )
#
# request_user2 = RecommendationRequest(
#     gender="female",
#     age=40,
#     country="France",
#     query="An epic adventure in space with heroic battles and stunning visuals."
# )
#
# # Get recommendations for user 1
# recommendations_user1 = model.recommend(request_user1, top_n=3)
# print("Recommendations for User 1:")
# print(recommendations_user1)
#
# # Get recommendations for user 2
# recommendations_user2 = model.recommend(request_user2, top_n=3)
# print("\nRecommendations for User 2:")
# print(recommendations_user2)