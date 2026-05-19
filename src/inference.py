import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    get_texture_features,
    get_wrinkle_features,
    detect_eye_features
)

from model import preprocess_model_image



def prepare_input(image_path):
    image = cv2.imread(image_path)

    processed_image = preprocess_model_image(image)

    image_input = np.expand_dims(processed_image, axis=0)


    # handcrafted features

    texture = get_texture_features(image_path)
    wrinkle = get_wrinkle_features(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    left_eye, right_eye = detect_eye_features(gray)

    tabular = np.array([
        texture,
        wrinkle,
        left_eye,
        right_eye
    return image_input, tabular
