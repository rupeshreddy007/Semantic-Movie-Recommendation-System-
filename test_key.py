import os
from pinecone import Pinecone

# --- PASTE YOUR NEW API KEY HERE ---
MY_KEY = "YOUR_NEW_API_KEY" 
# -------------------------------------

print("--- Starting Pinecone Key Test ---")

# Check if a conflicting environment variable is set
env_key = os.environ.get("PINECONE_API_KEY")
if env_key:
    print(f"WARNING: Found environment variable 'PINECONE_API_KEY'.")
    print(f"         It starts with: {env_key[:4]}...")
else:
    print("GOOD: No 'PINECONE_API_KEY' environment variable found.")

try:
    # Initialize explicitly with your key
    pc = Pinecone(api_key=MY_KEY)

    print("\nConnecting to Pinecone...")
    index_list = pc.list_indexes()

    print("\n--- ✅ SUCCESS! ---")
    print("Connection worked. Found indexes:")
    print(index_list)

except Exception as e:
    print("\n--- ❌ FAILED! ---")
    print(f"Error: {e}")

print("\n--- Test Finished ---")