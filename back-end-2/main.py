from fastapi import FastAPI, UploadFile, File, HTTPException
import face_recognition
import shutil
import os
from pathlib import Path
from database import engine, Base, get_db
from sqlalchemy.orm import Session
import crud, models, schemas

app = FastAPI()

# Database Initialization
Base.metadata.create_all(bind=engine)

UPLOAD_FOLDER = Path("./images")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@app.post("/register/")
async def register_user(full_name: str, image: UploadFile = File(...), db: Session = next(get_db())):
    image_path = UPLOAD_FOLDER / image.filename
    with image_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    # Load and encode the image
    face_image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(face_image)
    if not face_encoding:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    
    # Save user to database
    user = crud.create_user(db, full_name=full_name, image_path=str(image_path), encoding=face_encoding[0].tolist())
    return {"message": "User registered successfully", "user_id": user.id}

@app.post("/recognize/")
async def recognize_user(image: UploadFile = File(...), db: Session = next(get_db())):
    image_path = UPLOAD_FOLDER / image.filename
    with image_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    # Load and encode the image
    uploaded_image = face_recognition.load_image_file(image_path)
    uploaded_encoding = face_recognition.face_encodings(uploaded_image)
    if not uploaded_encoding:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")
    
    users = crud.get_all_users(db)
    for user in users:
        stored_encoding = [float(num) for num in user.encoding]
        match = face_recognition.compare_faces([stored_encoding], uploaded_encoding[0])
        if match[0]:
            return {"message": "User recognized", "user_name": user.full_name}
    
    return {"message": "User not recognized"}
