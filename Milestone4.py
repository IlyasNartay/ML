import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Пути к файлам
MOVIES_FILE = 'cleaned_tmdb_movies.csv'
QUERIES_FILE = 'queries.csv'


# Класс для обработки запросов
class Request:
    def __init__(self, gender: str, age: int, country: str, query: str, n: int = 5):
        self.gender = gender
        self.age = age
        self.country = country
        self.query = query
        self.n = n


# Класс для формирования ответа
class Response:
    def __init__(self, recommended_movies, message="Success"):
        self.recommended_movies = recommended_movies
        self.message = message

    def to_dict(self):
        return {
            "message": self.message,
            "recommended_movies": self.recommended_movies.to_dict(orient="records") if not self.recommended_movies.empty else []
        }


# Класс для обработки обратной связи
class UserFeedback:
    def __init__(self, selected_movies):
        """
        :param selected_movies: Список выбранных пользователем фильмов (их названия)
        """
        self.selected_movies = selected_movies


# Загрузка данных о фильмах
def load_movies(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise Exception(f"File {file_path} not found. Please ensure the file exists.")


# Загрузка или создание файла запросов
def load_or_create_queries(file_path):
    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            # Если файл существует, но пуст, создаём заголовки
            df = pd.DataFrame(columns=["gender", "age", "country", "query", "recommended_movies"])
            df.to_csv(file_path, index=False)
            return df
        else:
            return pd.read_csv(file_path)
    else:
        # Создаём новый файл с колонками
        df = pd.DataFrame(columns=["gender", "age", "country", "query", "recommended_movies"])
        df.to_csv(file_path, index=False)
        return df


# Обновление файла запросов
def save_query_to_csv(file_path, request: Request, selected_movies):
    new_query = {
        "gender": request.gender,
        "age": request.age,
        "country": request.country,
        "query": request.query,
        "recommended_movies": ";".join(map(str, selected_movies))
    }
    df = load_or_create_queries(file_path)
    new_query_df = pd.DataFrame([new_query])  # Создаём DataFrame из новой записи
    df = pd.concat([df, new_query_df], ignore_index=True)  # Используем pd.concat для добавления
    df.to_csv(file_path, index=False)


# Функция фильтрации фильмов по возрасту
def filter_movies_by_age(movies, user_age):
    if user_age < 18:
        return movies[movies['Adult'] == False]  # Исключаем фильмы для взрослых
    return movies


def recommend_movies(request: Request, movies):
    # Фильтруем фильмы по возрасту
    filtered_movies = filter_movies_by_age(movies, request.age)

    # Проверяем, есть ли фильмы для обработки
    if filtered_movies.empty:
        return Response(recommended_movies=pd.DataFrame(), message="No movies available for the given criteria")

    # Обрабатываем текстовые данные
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_movies['Overview'].fillna('') + [request.query])

    # Вычисляем сходство между запросом и фильмами
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    similarity_scores = cosine_sim.flatten()

    # Добавляем текстовое сходство в DataFrame
    filtered_movies = filtered_movies.copy()  # Создаём копию, чтобы избежать изменений в оригинале
    filtered_movies['similarity'] = similarity_scores

    # Добавляем комбинированный балл (текстовое сходство + популярность)
    weight_similarity = 0.7  # Вес текстового сходства
    weight_popularity = 0.3  # Вес популярности
    filtered_movies['combined_score'] = (
        weight_similarity * filtered_movies['similarity'] +
        weight_popularity * filtered_movies['Popularity']
    )

    # Сортируем фильмы по комбинированному баллу
    recommendations = filtered_movies.sort_values(by='combined_score', ascending=False).head(request.n)

    return Response(recommended_movies=recommendations)



# Обновление модели на основе обратной связи
def update_model_with_feedback(movies, feedback: UserFeedback):
    """
    Обновляет данные модели на основе выбора пользователя.
    :param movies: DataFrame с фильмами.
    :param feedback: Объект UserFeedback с выбором пользователя.
    :return: Обновлённый DataFrame.
    """
    for movie_name in feedback.selected_movies:
        # Увеличиваем популярность фильмов, выбранных пользователем
        if movie_name in movies['Movie Name'].values:
            movies.loc[movies['Movie Name'] == movie_name, 'Popularity'] += 1
        else:
            raise Exception(f"Movie '{movie_name}' not found in dataset.")
    return movies


# Корректировка модели на основе предыдущих запросов
def adjust_model_with_queries(movies, queries):
    for _, row in queries.iterrows():
        recommended_ids = row['recommended_movies'].split(';')
        for movie_id in recommended_ids:
            if movie_id in movies['Movie Name'].values:
                movies.loc[movies['Movie Name'] == movie_id, 'Popularity'] += 1
    return movies


# Глобальная загрузка данных
movies = load_movies(MOVIES_FILE)
queries = load_or_create_queries(QUERIES_FILE)
movies = adjust_model_with_queries(movies, queries)
