import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
        """Load and preprocess images from the dataset directory."""
        images = []
        labels = []
        label_map = {}
        current_label = 0
        
        # Walk through the dataset directory and load images
        for person_name in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_name)
            if os.path.isdir(person_path):
                label_map[current_label] = person_name
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    # Only take the first face detected (if multiple faces, this could be improved)
                    for (x, y, w, h) in faces:
                        face = img[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (200, 200))
                        images.append(face_resized.flatten())  # Flatten face image to 1D vector
                        labels.append(current_label)
                
                current_label += 1
        
        # Convert images and labels to numpy arrays
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.label_map = label_map

    def train_eigenfaces(self):
        """Train Eigenfaces (PCA)."""
        n_samples, n_features = self.images.shape
        n_components_pca = min(100, n_samples, n_features)  # Adjust PCA components
        pca = PCA(n_components=n_components_pca)
        
        # Normalize the data before applying PCA
        pca_images = pca.fit_transform(self.scaler.fit_transform(self.images))  # Normalize before PCA
        self.pca = pca
        return pca, pca_images

    def train_fisherfaces(self, pca_images):
        """Train Fisherfaces (LDA)."""
        lda = LDA(n_components=min(20, len(np.unique(self.labels)) - 1))  # Ensure LDA components are <= classes-1
        fisherfaces = lda.fit_transform(pca_images, self.labels)
        self.lda = lda
        return lda, fisherfaces

    def recognize_face(self, image):
        """Recognize a face using PCA and LDA."""
        # Detect face
        faces = self.face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        
        # Only take the first face detected
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200, 200))
        face_flattened = face_resized.flatten()

        # Apply PCA
        face_pca = self.pca.transform(self.scaler.transform([face_flattened]))  # Normalize before PCA

        # Apply LDA (Fisherfaces)
        face_fisher = self.lda.transform(face_pca)

        # Find the closest match
        distances = np.linalg.norm(face_fisher - self.fisherfaces, axis=1)
        closest_idx = np.argmin(distances)
        label = self.labels[closest_idx]

        return self.label_map[label]

    def run(self, test_image_path):
        # Load images from dataset
        self.load_images()

        # Train Eigenfaces (PCA)
        pca, pca_images = self.train_eigenfaces()

        # Train Fisherfaces (LDA)
        lda, fisherfaces = self.train_fisherfaces(pca_images)

        self.fisherfaces = fisherfaces

        # Test the recognition on a new image
        test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        label = self.recognize_face(test_img)
        if label:
            print(f"Recognized as: {label}")
        else:
            print("No face detected")


if __name__ == "__main__":
    dataset_path = 'dataset'  # Replace with your dataset folder path
    test_image_path = 'test_images/test_image.png'  # Replace with the path of the image you want to test
    
    # Initialize recognizer and run the recognition
    recognizer = FaceRecognizer(dataset_path)
    recognizer.run(test_image_path)
