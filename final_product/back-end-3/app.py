# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from algoritms.face_encoding import encode_face
from algoritms.distance_calculation import compare_encodings
from algoritms.user_management import load_users, save_users
from algoritms.image_preprocessing import preprocess_image
from algoritms.caching import get_encoding_from_cache, update_cache
from algoritms.parallel_processing import recognize_in_parallel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register/")
async def register_user(name: str = Form(...), file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        encoding = encode_face(image_bytes)

        if encoding is not None:

            users = load_users()
            for existing_user, existing_encoding in users.items():
                distance = compare_encodings(existing_encoding,encoding)
                if distance: 
                    return {"error": f"A user with the same face is already registered under the name {existing_user}"}

            # Check the cache before recalculating the encoding
            existing_encoding = get_encoding_from_cache(name)
            if existing_encoding is not None:
                return {"error": f"User {name} already exists in the cache."}

            # Save encoding to the cache
            update_cache(name, encoding)

            # users = load_users()
            users[name] = encoding.tolist()
            save_users(users)

            return {"message": f"User {name} registered successfully"}

        return {"error": "No face detected"}

    except Exception as e:
        return {"error": str(e)}

@app.post("/recognize/")
async def recognize_user(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        # Extract encoding from the uploaded image
        new_encoding = encode_face(image_bytes)
        if new_encoding is None:
            return {"error": "No face detected in the image."}

        # Load encodings from storage
        users = load_users()

        # Check for a match
        for existing_user, existing_encoding in users.items():
            if compare_encodings(existing_encoding, new_encoding):  
                return {"message": f"Face recognized: {existing_user}"}

        return {"error": "Face not recognized"}

    except Exception as e:
        return {"error": str(e)}
