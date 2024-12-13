from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from Milestone4 import MovieRecommendationModel, RecommendationRequest  # Assuming Milestone_4.py has the model

# DTO-класс для API запросов
class APIRecommendationRequest(BaseModel):
    gender: str
    age: int
    country: str
    query: str

# DTO-класс для API ответов
class MovieResponse(BaseModel):
    movie_name: str
    poster_path: str
    overview: str

# Load dataset
file_path = 'tmdb_movies_cleaned.csv'
movie_data = pd.read_csv(file_path)

# Initialize the recommendation model
model = MovieRecommendationModel(movie_data)

# Initialize FastAPI
app = FastAPI()

@app.post("/recommendations", response_model=List[MovieResponse])
def get_recommendations(request: APIRecommendationRequest):
    try:
        recommendation_request = RecommendationRequest(
            gender=request.gender,
            age=request.age,
            country=request.country,
            query=request.query
        )
        recommendations = model.recommend(recommendation_request)
        response = [
            MovieResponse(
                movie_name=row['Movie Name'],
                poster_path=row['Poster path'],
                overview=row['Overview']
            ) for _, row in recommendations.iterrows()
        ]
        return response
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing column in dataset: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

