import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pathlib import Path
import json

# --- Qdrant Connection ---
# Assumes you are running Qdrant in Docker
client = QdrantClient(url="http://localhost:6333")

# --- Constants ---
collection_name = "movie_recommender_tmdb"
model_dimension = 384
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Helper Functions ---
def parse_json_field(data_str):
    """Parses genres, keywords, etc."""
    try:
        data = json.loads(data_str)
        return [item['name'] for item in data]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []

def parse_cast(data_str, max_items=10):
    """Parses cast to get top actor and character names."""
    try:
        data = json.loads(data_str)
        actor_names = [item['name'] for item in data[:max_items]]
        character_names = [item['character'] for item in data[:max_items]]
        
        # Combine and filter out empty strings
        all_names = actor_names + character_names
        return [name for name in all_names if name]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []

print("Checking if collection exists...")
# Recreate the collection to ensure it's fresh and has the new data
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=model_dimension, distance=Distance.COSINE)
)
print(f"Collection '{collection_name}' created.")


# --- 1. Load and Merge Data ---
print("Loading datasets...")
SCRIPT_DIR = Path(__file__).parent 
df_movies = pd.read_csv(SCRIPT_DIR / "tmdb_5000_movies.csv")
df_credits = pd.read_csv(SCRIPT_DIR / "tmdb_5000_credits.csv")

# Merge the two dataframes
df = pd.merge(df_movies, df_credits, left_on='id', right_on='movie_id')

# --- 2. Clean and Parse ---
print("Cleaning and parsing data...")
df['overview'] = df['overview'].fillna('')
df['release_date'] = df['release_date'].fillna('')

df['genres_list'] = df['genres'].apply(parse_json_field)
df['keywords_list'] = df['keywords'].apply(parse_json_field)
df['cast_list'] = df['cast'].apply(parse_cast) # <-- Includes cast

# --- 3. Create the "Super Search Text" (with Boosting) ---
print("Creating combined search text with boosting...")

# We repeat title and cast to give them more weight
df['search_text'] = (
    df['title_x'] + ". " +  # <-- Fixed KeyError
    df['title_x'] + ". " +  # <-- Boost title
    df['cast_list'].apply(lambda x: ' '.join(x)) + ". " + # <-- Add cast
    df['cast_list'].apply(lambda x: ' '.join(x)) + ". " + # <-- Boost cast
    df['overview'] + ". " + 
    df['genres_list'].apply(lambda x: ' '.join(x)) + ". " +
    df['keywords_list'].apply(lambda x: ' '.join(x))
)

# --- 4. Encode Embeddings ---
print("Encoding embeddings (this may take a few minutes)...")
df["embedding"] = df["search_text"].apply(lambda x: model.encode(x).tolist())

# --- 5. Prepare and Upload Points ---
print("Preparing points for Qdrant...")
points = []
for i, row in df.iterrows():
    rating = float(row["vote_average"]) if pd.notnull(row["vote_average"]) else 0.0
    release_date = row["release_date"] if pd.notnull(row["release_date"]) and row["release_date"] else None
    
    points.append(
        PointStruct(
            id=int(row['id']), # <-- Use the real movie ID
            vector=row["embedding"],
            payload={
                "title": row["title_x"], # <-- Fixed KeyError
                "description": row["overview"],
                "genres": row["genres_list"],
                "rating": rating,
                "release_date": release_date
            }
        )
    )

# --- 6. Batching Fix (to prevent upload errors) ---
print(f"Uploading {len(points)} vectors to Qdrant in batches...")
BATCH_SIZE = 100

for i in range(0, len(points), BATCH_SIZE):
    end_index = min(i + BATCH_SIZE, len(points))
    batch_points = points[i:end_index]
    
    client.upsert(
        collection_name=collection_name,
        points=batch_points,
        wait=True
    )
    
    print(f"  ... uploaded batch {i // BATCH_SIZE + 1} / {len(points) // BATCH_SIZE + 1}")

print("🎉 All data successfully uploaded to Qdrant!")