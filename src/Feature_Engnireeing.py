import cv2
import numpy as np
import pandas as pd
import dlib

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'dlib/shape_predictor_68_face_landmarks.dat'
)


def get_texture_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    return np.var(laplacian)



def get_wrinkle_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    wrinkle_percentage = np.sum(edges > 0) / (
        gray.shape[0] * gray.shape[1]
    )

    return wrinkle_percentage



def detect_eye_features(image):
    image = np.array(image, dtype=np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)

    if len(faces) == 0:
        return 0, 0

    shape = predictor(gray, faces[0])

    left_eye = shape.part(47).y - shape.part(43).y
    right_eye = shape.part(40).y - shape.part(38).y

    return left_eye, right_eye



    return df
