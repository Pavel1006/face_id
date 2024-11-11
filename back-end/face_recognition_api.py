from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile  # Import necessary components from FastAPI
from fastapi.responses import JSONResponse  # Import JSONResponse to send JSON formatted responses
import os  # Import os module to interact with the operating system
import cv2  # Import OpenCV for image processing
import numpy as np  # Import numpy for numerical operations
from sklearn.decomposition import PCA  # Import PCA from scikit-learn for dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # Import LDA for classification
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for normalization
import uuid  # For generating unique file names

app = FastAPI()  # Create an instance of the FastAPI application

# Enable CORS for all domains (or specify your domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define a fixed directory to store uploaded images
UPLOAD_DIRECTORY = "test_images"  # Directory for saving the images

# Ensure the directory exists; if not, create it
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)  # This creates the directory if it doesn't exist yet


# Class for face recognition
class FaceRecognizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.pca = None
        self.lda = None
        self.label_map = {}
        self.images = None
        self.labels = None
        self.scaler = StandardScaler()

    def load_images(self):
        images = []
        labels = []
        label_map = {}
        current_label = 0
        
        for person_name in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_name)
            if os.path.isdir(person_path):
                label_map[current_label] = person_name
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    for (x, y, w, h) in faces:
                        face = img[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (200, 200))
                        images.append(face_resized.flatten())
                        labels.append(current_label)
                
                current_label += 1
        
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.label_map = label_map

    def train_eigenfaces(self):
        n_samples, n_features = self.images.shape
        n_components_pca = min(100, n_samples, n_features)
        pca = PCA(n_components=n_components_pca)
        
        pca_images = pca.fit_transform(self.scaler.fit_transform(self.images))
        self.pca = pca
        return pca, pca_images

    def train_fisherfaces(self, pca_images):
        lda = LDA(n_components=min(20, len(np.unique(self.labels)) - 1))
        fisherfaces = lda.fit_transform(pca_images, self.labels)
        self.lda = lda
        return lda, fisherfaces

    def recognize_face(self, image):
        faces = self.face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200, 200))
        face_flattened = face_resized.flatten()

        face_pca = self.pca.transform(self.scaler.transform([face_flattened]))

        face_fisher = self.lda.transform(face_pca)

        distances = np.linalg.norm(face_fisher - self.fisherfaces, axis=1)
        closest_idx = np.argmin(distances)
        label = self.labels[closest_idx]

        return self.label_map[label]

    def run(self, test_image_path):
        self.load_images()
        pca, pca_images = self.train_eigenfaces()
        lda, fisherfaces = self.train_fisherfaces(pca_images)
        self.fisherfaces = fisherfaces

        test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        label = self.recognize_face(test_img)
        return label if label else "No face detected"


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):  # The endpoint accepts an uploaded file
    try:
        # Generate a unique file name to avoid overwriting
        unique_file_name = "test_image.jpg"
        file_path = os.path.join(UPLOAD_DIRECTORY, unique_file_name)  # Save the file with a unique name
        
        # Open the file in write-binary mode and write the contents of the uploaded file to it
        with open(file_path, "wb") as f:
            f.write(file.file.read())  # Write the contents of the uploaded file to the server
        
        # Initialize the FaceRecognizer with the path to your dataset
        recognizer = FaceRecognizer(dataset_path="dataset")  # Adjust the path to your dataset
        
        # Use the face recognition method
        recognized_label = recognizer.run(test_image_path=file_path)
        
        # Return a response with the recognized name
        return JSONResponse(content={"recognized_name": recognized_label}, status_code=200)
    
    except Exception as e:  # Handle any errors that occur during the file upload
        # Return an error response with the error message
        return JSONResponse(content={"message": "Error uploading file", "error": str(e)}, status_code=400)
