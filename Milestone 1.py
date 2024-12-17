import pandas as pd
import requests

# API URL (без указания страницы, чтобы можно было менять параметр page)
base_url = 'https://api.themoviedb.org/3/discover/movie'
headers = {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJhODhhMmE1ZDYxMWQ5MjFjYTQ5YzNmNmQ5MTVkODJkYiIsIm5iZiI6MTczMzc2MTc2Ny4yODE5OTk4LCJzdWIiOiI2NzU3MWFlNzliYzIwZWFhNDk2ODBlMjQiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.XcXgrJAq0R5SNZ7bm1rOxDqS_0WPEHMGHNk7ICpO6Nk'
    # Замените на ваш API токен
}

# Списки для хранения данных
movie_name = []
original_title = []
year = []
language = []
overview = []
popularity = []
vote_average = []
vote_count = []
genres = []
adults = []
poster_path = []

# Запрашиваем первые 50 страниц (по 20 фильмов на каждой)
for page in range(1, 4001):  # 50 страниц = 1000 фильмов
    params = {
        'include_adult': 'true',
        'include_video': 'false',
        'language': 'en-US',
        'page': page,
        'sort_by': 'popularity.desc'
    }
    response = requests.get(base_url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        # print(data)
        movies = data['results']  # Список фильмов на текущей странице

        for movie in movies:
            movie_name.append(movie['title'])
            original_title.append(movie['original_title'])
            year.append(movie.get('release_date', 'N/A').split('-')[0])  # Извлекаем год
            language.append(movie['original_language'])
            overview.append(movie['overview'])
            popularity.append(movie['popularity'])
            vote_average.append(movie['vote_average'])
            vote_count.append(movie['vote_count'])
            adults.append(movie['adult'])
            poster_path.append(movie['poster_path'])
            genres.append(', '.join(map(str, movie['genre_ids'])))  # Преобразуем genre_ids в строку

        print(f"Page {page} processed successfully.")
    else:
        print(f"Failed to fetch data for page {page}. Status Code: {response.status_code}")
        break

# Создаем DataFrame
df = pd.DataFrame({
    'Movie Name': movie_name,
    'Adult': adults,
    'Original Title': original_title,
    'Year': year,
    'Language': language,
    'Overview': overview,
    'Popularity': popularity,
    'Vote Average': vote_average,
    'Vote Count': vote_count,
    'Genres': genres,
    'Poster path': poster_path
})

# Сохраняем в CSV
df.to_csv('tmdb_top_1000_movies.csv', index=False, encoding='utf-8')
print("Data saved to 'tmdb_top_1000_movies.csv'")