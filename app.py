import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import streamlit as st
from pathlib import Path
import json

st.title("🎬 Semantic Movie Recommendation System (Qdrant)")

st.write("""
This app finds movies that match your query **based on meaning**, not just keywords.
Example: Try searching for "a sci-fi movie about space and survival"
""")

@st.cache_resource
def load_model():
    """Loads the sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_qdrant_client():
    """Initializes a Qdrant client running in memory."""
    client = QdrantClient(location=":memory:")
    return client

def parse_json_field(data_str):
    """Helper function to parse JSON fields like 'genres' and 'keywords'."""
    try:
        data = json.loads(data_str)
        return [item['name'] for item in data]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []

model = load_model()
client = load_qdrant_client()

collection_name = "movie_recommender_tmdb"
model_dimension = 384 

if 'data_uploaded' not in st.session_state:
    st.info("Setting up Qdrant collection and uploading data... Please wait.")
    
    if not client.collection_exists(collection_name):
        st.info("Creating new Qdrant collection...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=model_dimension, distance=Distance.COSINE)
        )

    SCRIPT_DIR = Path(__file__).parent 
    df = pd.read_csv(SCRIPT_DIR / "tmdb_5000_movies.csv")

    # Clean data
    df['overview'] = df['overview'].fillna('')
    df['release_date'] = df['release_date'].fillna('')
    
    # Parse the JSON fields
    df['genres_list'] = df['genres'].apply(parse_json_field)
    df['keywords_list'] = df['keywords'].apply(parse_json_field)

    # Create the "super-text" for semantic search
    df['search_text'] = (
        df['title'] + ". " + 
        df['overview'] + ". " + 
        df['genres_list'].apply(lambda x: ' '.join(x)) + ". " +
        df['keywords_list'].apply(lambda x: ' '.join(x))
    )

    df["embedding"] = df["search_text"].apply(lambda x: model.encode(x).tolist())

    points = []
    for i, row in df.iterrows():
        # Ensure rating is a standard float
        rating = float(row["vote_average"]) if pd.notnull(row["vote_average"]) else 0.0
        
        points.append(
            PointStruct(
                id=i,
                vector=row["embedding"],
                payload={
                    "title": row["title"],
                    "description": row["overview"],
                    "genres": row["genres_list"],
                    "rating": rating,
                    "release_date": row["release_date"]
                }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True
    )
    
    st.session_state['data_uploaded'] = True
    st.success("Movie data uploaded to Qdrant!")

query = st.text_input("Enter your movie idea:", "")

if st.button("Find Recommendations") and query:
    query_vector = model.encode(query).tolist()
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=10  # Let's show 5 results now
    )

    st.subheader("Top Matches")
    for match in results:
        payload = match.payload
        
        # Handle empty release date
        release_year = f"({payload['release_date'][:4]})" if payload.get('release_date') else ""
        
        st.write(
            f"**{payload['title']}** {release_year} — **Score: {match.score:.3f}**"
        )
        st.write(
            f"**Rating:** {payload['rating']}/10 | "
            f"**Genres:** {', '.join(payload['genres'])}"
        )
        st.caption(payload['description'])
        st.divider()