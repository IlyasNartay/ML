import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the dataset
file_path = 'tmdb_movies_cleaned.csv'
movies_data = pd.read_csv(file_path)

# Map genre IDs to their corresponding names if provided in a dictionary
# Here we create an example mapping; replace it with the actual mapping if known
genre_id_to_name = {
    "28": "Action", "12": "Adventure", "16": "Animation", "35": "Comedy",
    "80": "Crime", "99": "Documentary", "18": "Drama", "10751": "Family",
    "14": "Fantasy", "36": "History", "27": "Horror", "10402": "Music",
    "9648": "Mystery", "10749": "Romance", "878": "Science Fiction",
    "10770": "TV Movie", "53": "Thriller", "10752": "War", "37": "Western"
}

# Replace genre IDs with names in the dataset for visualization
movies_data['Genres Named'] = movies_data['Genres'].dropna().apply(
    lambda x: ", ".join(genre_id_to_name.get(g.strip(), g) for g in x.split(', '))
)

# Split genres and count their occurrences
genres_named = movies_data['Genres Named'].dropna().str.split(', ')
flat_genres_named = [genre.strip() for sublist in genres_named for genre in sublist]

# Count occurrences of each genre
genre_named_counts = Counter(flat_genres_named)

# Convert to DataFrame for visualization
genre_named_df = pd.DataFrame(genre_named_counts.items(), columns=['Genre', 'Count']).sort_values(by='Count', ascending=False)

# Plot the updated genre distribution
plt.figure(figsize=(12, 6))
sns.barplot(data=genre_named_df, x='Count', y='Genre', color='blue')  # Use a single color instead of palette
plt.title('Genre Distribution (Text Representation)')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Ensure Year is integer and filter for valid years
movies_data['Year'] = movies_data['Year'].fillna(0).astype(int)  # Ensure Year is integer
valid_years = movies_data[movies_data['Year'] > 1900]

# Filter data for movies released from 1980 onwards
movies_from_1980 = valid_years[valid_years['Year'] >= 1980]

# Count the number of movies released each year from 1980
yearly_movie_count_1980 = movies_from_1980.groupby('Year').size().reset_index(name='Count')

# Plot the number of movies released from 1980 onwards
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_movie_count_1980, x='Year', y='Count', marker='o')
plt.title('Number of Movies Released (1980 Onwards)')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.show()
