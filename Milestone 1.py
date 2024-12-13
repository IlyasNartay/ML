import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
tmdb_movies_data = pd.read_csv('tmdb_top_1000_movies.csv')

# 1. Handle missing values
# Check for missing values
missing_values = tmdb_movies_data.isnull().sum()

# Fill missing numerical values with the median and categorical values with the mode
tmdb_movies_data['Overview'] = tmdb_movies_data['Overview'].fillna('No Overview')
tmdb_movies_data['Vote Average'] = tmdb_movies_data['Vote Average'].fillna(tmdb_movies_data['Vote Average'].median())
tmdb_movies_data['Vote Count'] = tmdb_movies_data['Vote Count'].fillna(tmdb_movies_data['Vote Count'].median())
tmdb_movies_data['Popularity'] = tmdb_movies_data['Popularity'].fillna(tmdb_movies_data['Popularity'].median())

# 2. Normalize numerical data
# Create a MinMaxScaler instance
scaler = MinMaxScaler()

# Normalize the numerical columns
tmdb_movies_data[['Popularity', 'Vote Average', 'Vote Count']] = scaler.fit_transform(
    tmdb_movies_data[['Popularity', 'Vote Average', 'Vote Count']]
)

# 3. Encode categorical features
# Encode the 'Language' column using one-hot encoding
tmdb_movies_data = pd.get_dummies(tmdb_movies_data, columns=['Language'])

# Encode the 'Genres' column (split and one-hot encode the genres)
# Split the 'Genres' column by commas and create binary columns for each genre
genres_split = tmdb_movies_data['Genres'].str.split(',', expand=True).stack().unique()
for genre in genres_split:
    tmdb_movies_data[genre] = tmdb_movies_data['Genres'].apply(lambda x: 1 if str(genre) in str(x) else 0)

# 4. Remove duplicates
tmdb_movies_data = tmdb_movies_data.drop_duplicates()

# 5. Save the cleaned data to a new CSV file
tmdb_movies_data.to_csv('tmdb_movies_cleaned.csv', index=False)
