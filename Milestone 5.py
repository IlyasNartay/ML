from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from starlette.middleware.cors import CORSMiddleware
from Milestone4 import recommend_movies, update_model_with_feedback, movies, UserFeedback, Request  # Импортируем из Milestone4

# DTO-класс для API запросов
class APIRecommendationRequest(BaseModel):
    gender: str
    age: int
    country: str
    query: str
    n: int = 5  # Количество фильмов по умолчанию

# DTO-класс для API обратной связи
class FeedbackRequest(BaseModel):
    selected_movies: List[str]  # Список фильмов, которые выбрал пользователь

# DTO-класс для API ответов
class MovieResponse(BaseModel):
    movie_name: str
    poster_path: str
    overview: str
    popularity: float

# Инициализация FastAPI
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Укажите источник или ["*"] для разрешения всех источников
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST, DELETE и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

@app.post("/recommendations", response_model=List[MovieResponse])
def get_recommendations(request: APIRecommendationRequest):
    try:
        # Преобразуем API запрос в объект Request
        recommendation_request = Request(
            gender=request.gender,
            age=request.age,
            country=request.country,
            query=request.query,
            n=request.n
        )

        # Получаем рекомендации
        response = recommend_movies(recommendation_request, movies)

        # Формируем ответ для клиента
        recommendations = [
            MovieResponse(
                movie_name=row['Movie Name'],
                poster_path=row.get('Poster path', ''),
                overview=row['Overview'],
                popularity=row['Popularity']
            ) for _, row in response.recommended_movies.iterrows()
        ]
        return recommendations
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing column in dataset: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def send_feedback(feedback: FeedbackRequest):
    try:
        # Преобразуем API запрос в объект UserFeedback
        user_feedback = UserFeedback(selected_movies=feedback.selected_movies)

        # Обновляем модель на основе выбора пользователя
        global movies
        movies = update_model_with_feedback(movies, user_feedback)

        return {"message": "Feedback processed successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
