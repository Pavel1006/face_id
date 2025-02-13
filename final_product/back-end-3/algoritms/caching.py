import pickle
import os

ENCODINGS_CACHE_FILE = "encodings_cache.pkl"

def load_encodings_cache():
    """Load cached face encodings."""
    if os.path.exists(ENCODINGS_CACHE_FILE):
        with open(ENCODINGS_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_encodings_cache(encodings):
    """Save face encodings to the cache."""
    with open(ENCODINGS_CACHE_FILE, "wb") as f:
        pickle.dump(encodings, f)

def get_encoding_from_cache(name):
    """Get a specific encoding from the cache."""
    encodings = load_encodings_cache()
    return encodings.get(name)
    
def update_cache(name, encoding):
    """Update the encoding cache with a new user's encoding."""
    encodings = load_encodings_cache()
    encodings[name] = encoding
    save_encodings_cache(encodings)
