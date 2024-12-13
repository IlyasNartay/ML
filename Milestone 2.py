import pandas as pd
import numpy as np

# Load the dataset
file_path = 'tmdb_top_1000_movies.csv'
data = pd.read_csv(file_path)

# Inspect the dataset
print("Initial Data Info:")
print(data.info())

# Step 1: Handle Missing Values
# Drop rows with missing 'Year' or 'Genres' (critical for recommendations)
data.dropna(subset=['Year', 'Genres'], inplace=True)
# Remove rows where 'Poster path' is null or NaN
movie_data = data[data['Poster path'].notna()]

# For 'Overview', fill missing values with a placeholder
data['Overview'] = data['Overview'].fillna('No overview available.')

# Step 2: Normalize Numerical Columns
# Normalize 'Popularity', 'Vote Average', and 'Vote Count'
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['Popularity', 'Vote Average', 'Vote Count']] = scaler.fit_transform(data[['Popularity', 'Vote Average', 'Vote Count']])

# Step 3: Encode Categorical Features
# One-hot encode 'Genres'
data['Genres'] = data['Genres'].str.split(', ')
all_genres = set(genre for genres_list in data['Genres'] for genre in genres_list)
for genre in all_genres:
    data[genre] = data['Genres'].apply(lambda x: 1 if genre in x else 0)

# Drop the original 'Genres' column
data.drop(columns=['Genres'], inplace=True)

# Encode 'Language'
language_dummies = pd.get_dummies(data['Language'], prefix='Language')
data = pd.concat([data, language_dummies], axis=1)
data.drop(columns=['Language'], inplace=True)

# Step 4: Remove Duplicates
# Remove duplicate rows based on 'Movie Name' or 'Original Title'
data.drop_duplicates(subset=['Movie Name', 'Original Title'], inplace=True)

# Step 5: Drop Unnecessary Columns
# Drop 'Poster path' if not required for analysis or recommendations
data.drop(columns=['Poster path'], inplace=True)

# Save the cleaned dataset
output_path = 'cleaned_tmdb_movies.csv'
data.to_csv(output_path, index=False)