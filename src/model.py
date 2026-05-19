import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Input,
    Concatenate
)

DESIRED_WIDTH = 128
DESIRED_HEIGHT = 128
NUM_CHANNELS = 3



def preprocess_model_image(image):
    resized = cv2.resize(
        image,
        (DESIRED_WIDTH, DESIRED_HEIGHT)
    )

    if len(resized.shape) == 2:
        resized = cv2.cvtColor(
            resized,
            cv2.COLOR_GRAY2RGB
        )

    resized = resized.astype(np.float32) / 255.0

    return resized



    return load_model(path)
