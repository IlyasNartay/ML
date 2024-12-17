import pandas as pd
import matplotlib.pyplot as plt

# Milestone 3: Exploratory Data Analysis (EDA)
# Загрузка данных
file_path = 'cleaned_tmdb_movies.csv'
movies_df = pd.read_csv(file_path)


# 1. Анализ распределения жанров
def plot_genre_distribution(df):
    genre_columns = [col for col in df.columns if col.isdigit()]
    genre_sums = df[genre_columns].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(genre_sums.index, genre_sums.values)
    plt.title("Distribution of Genres")
    plt.xlabel("Genre ID")
    plt.ylabel("Number of Movies")
    plt.xticks(rotation=45)
    plt.show()


# 2. Анализ рейтингов фильмов
def plot_rating_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['Vote Average'], bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Number of Movies")
    plt.show()


# 3. Анализ трендов популярности фильмов по годам
def plot_trend_over_years(df):
    if 'Year' in df.columns:
        year_trends = df.groupby('Year')['Popularity'].mean()
        plt.figure(figsize=(12, 6))
        plt.plot(year_trends.index, year_trends.values, marker='o')
        plt.title("Average Popularity of Movies Over Years")
        plt.xlabel("Year")
        plt.ylabel("Average Popularity")
        plt.grid(True)
        plt.show()
    else:
        print("The column 'Year' is missing in the dataset.")


# Выполнение функций
if __name__ == "__main__":
    # Распределение жанров
    plot_genre_distribution(movies_df)

    # Распределение рейтингов
    plot_rating_distribution(movies_df)

    # Тренд популярности по годам
    plot_trend_over_years(movies_df)
