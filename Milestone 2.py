import pandas as pd
import numpy as np

# Загрузка данных
movies_df = pd.read_csv('tmdb_top_1000_movies.csv')


# Milestone 2: Data Preprocessing

def preprocess_data(df):
    # 1. Удаление дубликатов
    df = df.drop_duplicates()
    print("Дубликаты удалены.")

    # 2. Обработка пропущенных значений
    # Заполняем пропущенные значения в числовых колонках медианой
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df.loc[:, col] = df[col].fillna(df[col].median())

    # Удаляем строки с пропущенными значениями в текстовых колонках
    text_cols = df.select_dtypes(include=[object]).columns
    df = df.dropna(subset=text_cols)
    print("Строки с пропущенными текстовыми значениями удалены.")

    # 3. Нормализация числовых данных (Min-Max Scaling)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].values)
    print("Числовые данные нормализованы.")

    # 4. Кодирование жанров (One-Hot Encoding)
    if 'Genres' in df.columns:
        genres_expanded = df['Genres'].str.get_dummies(sep=', ')
        df = pd.concat([df.drop(columns=['Genres']), genres_expanded], axis=1)
        print("Жанры закодированы.")

    # 5. Возвращаем чистый DataFrame
    print("Предобработка завершена.")
    return df


# Применение функции
cleaned_movies_df = preprocess_data(movies_df)

# Сохранение результатов в новый CSV
cleaned_movies_df.to_csv('cleaned_tmdb_movies.csv', index=False)
print("Чистый датасет сохранен в 'cleaned_tmdb_movies.csv'.")