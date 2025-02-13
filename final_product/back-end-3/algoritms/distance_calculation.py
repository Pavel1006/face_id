import numpy as np

def compare_encodings(known_encoding, unknown_encoding, threshold=0.6):
    """Compare face encodings by calculating the Euclidean distance."""
    if known_encoding is None or unknown_encoding is None:
        return False
    distance = np.linalg.norm(np.array(known_encoding) - np.array(unknown_encoding))
    return distance < threshold
