from concurrent.futures import ThreadPoolExecutor
import face_recognition
import numpy as np

def recognize_face_parallel(image_bytes, known_encodings):
    """Recognize a face in parallel for multiple known encodings."""
    try:
        image_rgb = face_recognition.load_image_file(BytesIO(image_bytes))
        encoding = face_recognition.face_encodings(image_rgb)

        if encoding:
            encoding = encoding[0]
            for known_encoding in known_encodings:
                distance = np.linalg.norm(np.array(known_encoding) - np.array(encoding))
                if distance < 0.6:
                    return True
        return False
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        return False

def recognize_in_parallel(image_bytes, users_encodings):
    """Use parallel processing to check face recognition against multiple encodings."""
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda encoding: recognize_face_parallel(image_bytes, encoding), users_encodings)
    return any(results)
