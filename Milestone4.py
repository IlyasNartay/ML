from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Класс для модели рекомендаций
class RecommendationModel:
    def __init__(self, data):
        self.movies_df = data.dropna(subset=['Overview'])  # Удаляем фильмы без описаний
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['Overview'])
        self.user_feedback_data = {}

    # Метод для обновления фидбэка пользователя
    def update_feedback(self, selected_movies, feedback_weight=0.1):
        for movie in selected_movies:
            if movie in self.user_feedback_data:
                self.user_feedback_data[movie] += feedback_weight
            else:
                self.user_feedback_data[movie] = feedback_weight

        for movie in selected_movies:
            index = self.movies_df[self.movies_df['Movie Name'] == movie].index
            if not index.empty:
                self.tfidf_matrix[index, :] = self.tfidf_matrix[index, :] * (1 + self.user_feedback_data[movie])

    # Метод для рекомендаций на основе текстового запроса
    def recommend_movies(self, query, gender_preferences=None, age=None, n=5):
        query_vector = self.tfidf_vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Фильтрация фильмов по возрасту
        filtered_movies = self.movies_df.copy()
        if age is not None and age < 18:
            filtered_movies = filtered_movies[filtered_movies['Adult'] == False]

        # Фильтрация по предпочтениям жанров
        if gender_preferences:
            genre_columns = [genre for genre in gender_preferences if genre in filtered_movies.columns]
            if genre_columns:
                filtered_movies = filtered_movies[filtered_movies[genre_columns].sum(axis=1) > 0]

        filtered_indices = filtered_movies.index
        filtered_similarities = cosine_similarities[filtered_indices]

        # Сортировка по убыванию схожести
        top_indices = filtered_indices[filtered_similarities.argsort()[-n:][::-1]]
        recommended_movies = filtered_movies.loc[top_indices, ['Movie Name', 'Overview', 'Vote Average']]
        recommended_movies['Similarity'] = filtered_similarities[top_indices - filtered_indices[0]]
        return recommended_movies

# Предпочтения жанров по гендеру
gender_genre_preferences = {
    "male": ["Action", "Adventure", "Sci-Fi", "Thriller"],
    "female": ["Drama", "Romance", "Comedy", "Fantasy"],
    "other": ["Documentary", "Animation", "Family", "Mystery"]
}

# Пример использования
if __name__ == "__main__":
    # Загрузка данных
    file_path = 'tmdb_movies_cleaned.csv'
    movies_df = pd.read_csv(file_path)

    # Инициализация модели
    model = RecommendationModel(movies_df)

    # Пример запроса
    query = "superhero saving the world with powers"
    gender = "male"
    age = 16
    preferences = gender_genre_preferences.get(gender.lower(), [])
    recommendations = model.recommend_movies(query, gender_preferences=preferences, age=age, n=5)

    # Вывод рекомендаций
    print(recommendations)
