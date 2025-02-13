
import face_recognition
import numpy as np
from io import BytesIO
from PIL import Image

def encode_face(image_bytes):
    """Encode the face in the image using face_recognition."""
    try:
        image_rgb = face_recognition.load_image_file(BytesIO(image_bytes))
        face_locations = face_recognition.face_locations(image_rgb)

        if not face_locations:
            return None
        
        face_encoding = face_recognition.face_encodings(image_rgb, known_face_locations=face_locations)

        if face_encoding:
            return face_encoding[0]
        return None
    except Exception as e:
        print(f"Error in face encoding: {str(e)}")
        return None
