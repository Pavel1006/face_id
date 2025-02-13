import os
import cv2
import numpy as np
import json
import logging
from sklearn.preprocessing import StandardScaler
from app.services.pca_service import PCAService
from app.services.lda_service import LDAService


class FaceRecognizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.pca_service = PCAService()
        self.lda_service = LDAService()
        self.label_map = {}
        self.fisherfaces = None

    def load_images(self):
        images, labels = [], []
        label_map, current_label = {}, 0

        logging.debug(f"Loading images from dataset path: {self.dataset_path}")
        for person_name in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_name)
            if os.path.isdir(person_path):
                label_map[current_label] = person_name
                logging.debug(f"Processing images for {person_name}")
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        logging.warning(f"Image {img_file} could not be loaded.")
                        continue

                    faces = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    ).detectMultiScale(img)

                    if len(faces) == 0:
                        logging.warning(f"No faces detected in {img_file}")
                        continue

                    for (x, y, w, h) in faces:
                        face_resized = cv2.resize(img[y:y+h, x:x+w], (200, 200))
                        images.append(face_resized.flatten())
                        labels.append(current_label)
                current_label += 1

        self.label_map = label_map
        with open("app/models/label_map.json", "w") as f:
            json.dump(label_map, f)

        logging.debug(f"Images and labels loaded, total samples: {len(images)}")
        return np.array(images), np.array(labels)

    def train(self):
        logging.info("Training face recognition model...")
        images, labels = self.load_images()
        images_scaled = self.scaler.fit_transform(images)

        logging.debug("Training PCA and LDA...")
        pca_features = self.pca_service.fit_transform(images_scaled)
        fisherfaces = self.lda_service.fit_transform(pca_features, labels)

        logging.debug(f"PCA and LDA training completed. Fisherfaces shape: {fisherfaces.shape}")

        self.pca_service.save("app/models/pca_model.pkl")
        self.lda_service.save("app/models/lda_model.pkl")
        self.fisherfaces = fisherfaces
        logging.info("Models saved successfully.")

    def recognize_face(self, image):
        try:
            logging.debug("Starting face recognition...")
            with open("app/models/label_map.json", "r") as f:
                self.label_map = json.load(f)

            self.pca_service.load("app/models/pca_model.pkl")
            self.lda_service.load("app/models/lda_model.pkl")

            faces = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            ).detectMultiScale(image)
            if len(faces) == 0:
                logging.error("No face detected in the image.")
                return "No face detected"

            (x, y, w, h) = faces[0]
            face_resized = cv2.resize(image[y:y+h, x:x+w], (200, 200))
            face_flattened = self.scaler.transform([face_resized.flatten()])

            face_pca = self.pca_service.transform(face_flattened)
            face_fisher = self.lda_service.transform(face_pca)

            distances = np.linalg.norm(face_fisher - self.fisherfaces, axis=1)
            label_idx = np.argmin(distances)

            return self.label_map.get(str(label_idx), "Unknown")

        except Exception as e:
            logging.error(f"Error in face recognition: {e}")
            return f"Error processing image: {e}"

    def run(self, test_image_path):
        logging.info(f"Running face recognition for image: {test_image_path}")
        self.train()
        test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        return self.recognize_face(test_img)
