# music_recommender.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv('songs.csv')

# Step 2: Select features for recommendation
features = ['tempo', 'danceability', 'energy', 'valence']
X = df[features]

# Step 3: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Compute cosine similarity matrix
similarity_matrix = cosine_similarity(X_scaled)

# Step 5: Recommendation function
def recommend(song_title, top_n=5):
    if song_title not in df['title'].values:
        print("Song not found in database.")
        return

    index = df[df['title'] == song_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} songs similar to '{song_title}':\n")
    for i, score in similarity_scores[1:top_n+1]:  # Skip the first one (itself)
        print(f"{df.iloc[i]['title']} by {df.iloc[i]['artist']}  (Score: {score:.2f})")

# Example usage
if __name__ == "__main__":
    recommend('Shape of You')
