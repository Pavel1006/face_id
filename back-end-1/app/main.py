from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from app.services.face_recognition_service import FaceRecognizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = "test_images"
DATASET_DIRECTORY = "dataset"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

recognizer = FaceRecognizer(DATASET_DIRECTORY)
recognizer.train()

print(recognizer.run( os.path.join(UPLOAD_DIRECTORY, "test_image.jpg")))

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIRECTORY, "test_image.jpg")
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        label = recognizer.run(file_path)
        return JSONResponse(content={"recognized_name": label}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": "Error processing image", "error": str(e)}, status_code=500)
