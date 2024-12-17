import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from Milestone4 import RecommendationModel, gender_genre_preferences

# Инициализация FastAPI приложения
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка данных
file_path = 'cleaned_tmdb_movies.csv'
movies_df = pd.read_csv(file_path)

# Инициализация модели
model = RecommendationModel(movies_df)

# Pydantic модель для запроса
class RecommendRequest(BaseModel):
    query: str
    gender: str
    age: int
    n: int = 5

# Pydantic модель для фидбэка
class FeedbackRequest(BaseModel):
    selected_movies: List[str]

# Эндпоинт для рекомендаций
@app.post("/recommendations")
def get_recommendations(request: RecommendRequest):
    preferences = gender_genre_preferences.get(request.gender.lower(), [])
    recommendations = model.recommend_movies(
        query=request.query, gender_preferences=preferences, age=request.age, n=request.n
    )
    return {
        "message": "Success",
        "recommendations": recommendations.to_dict(orient="records")
    }

# Эндпоинт для обновления фидбэка
@app.post("/feedback")
def update_user_feedback(feedback: FeedbackRequest):
    model.update_feedback(feedback.selected_movies)
    return {"message": "Feedback successfully updated"}

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
