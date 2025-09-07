"""
Auto-extracted training script from notebook on 2025-09-07T22:52:35.169848
This script imports from preprocessing, model_cnn, model_vggface where applicable.
"""

from src import preprocessing as preprocessing
from src import model_cnn as model_cnn
from src import model_vggface as model_vggface

import pandas as pd
import os
import numpy as np
from tabulate import tabulate
import dlib
import cv2
from tqdm import tqdm
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Concatenate


# Constants for image preprocessing
desired_width = 128
desired_height = 128
num_channels = 3  # Assuming color images, adjust if grayscale


# Step 1: Prepare Data
# Extract 'eye openness', 'texture features', and 'wrinkle features'
tabular_features = df[['texture_features', 'wrinkle_features', 'left_eye_openness','right_eye_openness']]
# Standardize numerical features
scaler = StandardScaler()
tabular_features = scaler.fit_transform(tabular_features)


desired_channels = 3 

def preprocess_image(image):
    # Resize to a common size
    resized_image = cv2.resize(image, (desired_width, desired_height))
    # Ensure the image has the correct number of channels
    if resized_image.shape[-1] != desired_channels:
        # If the image has more than 3 channels, convert to grayscale
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # Add an extra dimension to represent the single channel
        resized_image = np.expand_dims(resized_image, axis=-1)
    # Normalize pixel values to be between 0 and 1
    resized_image = resized_image / 255.0
    # Add other preprocessing steps as needed (e.g., data augmentation)
    return resized_image

# Load and preprocess images
X_images = np.array([preprocess_image(image) for image in df['image'].values])
# Extract target
y_target = df['ageLabel'].values

# Split data into training and testing sets based on 'split' column
train_indices = df[df['split'] == 'train'].index
test_indices = df[df['split'] == 'test'].index

X_images_train = X_images[train_indices]
X_tabular_train = tabular_features[train_indices]
y_train = y_target[train_indices]

X_images_test = X_images[test_indices]
X_tabular_test = tabular_features[test_indices]
y_test = y_target[test_indices]

# The rest of your code remains the same...

# Step 2: Build Models
# Build CNN model for image feature extraction
image_input = Input(shape=(desired_width, desired_height, num_channels))
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
image_output = Dense(128, activation='relu')(x)

# Build model for tabular features
tabular_input = Input(shape=(X_tabular_train.shape[1],))
tabular_output = Dense(128, activation='relu')(tabular_input)

# Step 3: Combine Models
concatenated = Concatenate(axis=-1)([image_output, tabular_output])
x = Dense(64, activation='relu')(concatenated)
output = Dense(1, activation='linear')(x)

combined_model = Model(inputs=[image_input, tabular_input], outputs=output)

# Step 4: Train and Evaluate
combined_model.compile(optimizer='adam', loss='mean_squared_error')
combined_model.fit([X_images_train, X_tabular_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = combined_model.predict([X_images_test, X_tabular_test])

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Assuming you have binary classification labels for f1_score
y_test_binary = (y_test > 0).astype(int)
y_pred_binary = (y_pred > 0).astype(int)
f1 = f1_score(y_test_binary, y_pred_binary)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'F1 Score: {f1}')


import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Concatenate

# Assuming 'df' is your DataFrame

# Constants for image preprocessing
desired_width = 128
desired_height = 128
num_channels = 3  # Assuming color images, adjust if grayscale

# Step 1: Prepare Data
# Extract 'eye openness', 'texture features', and 'wrinkle features'
tabular_features = df_cleaned[['texture_features', 'wrinkle_features', 'left_eye_openness','right_eye_openness']]
# Standardize numerical features
scaler = StandardScaler()
tabular_features = scaler.fit_transform(tabular_features)

# Load and preprocess images
desired_channels = 3  # Assuming color images, adjust if grayscale

def preprocess_image(image):
    # Resize to a common size
    resized_image = cv2.resize(image, (desired_width, desired_height))
    # Ensure the image has the correct number of channels
    if resized_image.shape[-1] != desired_channels:
        # If the image has more than 3 channels, convert to grayscale
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # Add an extra dimension to represent the single channel
        resized_image = np.expand_dims(resized_image, axis=-1)
    # Normalize pixel values to be between 0 and 1
    resized_image = resized_image / 255.0
    # Add other preprocessing steps as needed (e.g., data augmentation)
    return resized_image

X_images = np.array([preprocess_image(image) for image in df_cleaned['image'].values])
# Extract target
y_target = df_cleaned['ageLabel'].values

# Split data into training and testing sets
X_images_train, X_images_test, X_tabular_train, X_tabular_test, y_train, y_test = train_test_split(
    X_images, tabular_features, y_target, test_size=0.2, random_state=42
)

# Step 2: Build Models
# Build CNN model for image feature extraction
image_input = Input(shape=(desired_width, desired_height, num_channels))
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
image_output = Dense(128, activation='relu')(x)

# Build model for tabular features
tabular_input = Input(shape=(X_tabular_train.shape[1],))
tabular_output = Dense(128, activation='relu')(tabular_input)

# Step 3: Combine Models
concatenated = Concatenate(axis=-1)([image_output, tabular_output])
x = Dense(64, activation='relu')(concatenated)
output = Dense(1, activation='linear')(x)

combined_model = Model(inputs=[image_input, tabular_input], outputs=output)

# Step 4: Train and Evaluate
combined_model.compile(optimizer='adam', loss='mean_squared_error')
combined_model.fit([X_images_train, X_tabular_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = combined_model.predict([X_images_test, X_tabular_test])

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
y_test_binary = (y_test > 0).astype(int)
y_pred_binary = (y_pred > 0).astype(int)
f1 = f1_score(y_test_binary, y_pred_binary)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'F1 Score: {f1}')


from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# Assuming 'df_cleaned' is your DataFrame

# Constants for image preprocessing
desired_width = 128
desired_height = 128
num_channels = 3  # Assuming color images, adjust if grayscale

# Step 1: Prepare Data
# Extract 'eye openness', 'texture features', and 'wrinkle features'
tabular_features = df_cleaned[['texture_features', 'wrinkle_features', 'left_eye_openness','right_eye_openness']]
# Standardize numerical features
scaler = StandardScaler()
tabular_features = scaler.fit_transform(tabular_features)

# Load and preprocess images
desired_channels = 3  # Assuming color images, adjust if grayscale

def preprocess_image(image):
    # Resize to a common size
    resized_image = cv2.resize(image, (desired_width, desired_height))
    # Ensure the image has the correct number of channels
    if resized_image.shape[-1] != desired_channels:
        # If the image has more than 3 channels, convert to grayscale
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # Add an extra dimension to represent the single channel
        resized_image = np.expand_dims(resized_image, axis=-1)
    # Normalize pixel values to be between 0 and 1
    resized_image = resized_image / 255.0
    # Add other preprocessing steps as needed (e.g., data augmentation)
    return resized_image

X_images = np.array([preprocess_image(image) for image in df_cleaned['image'].values])
# Extract target
y_target = df_cleaned['ageLabel'].values

# Split data into training and testing sets
X_images_train, X_images_test, X_tabular_train, X_tabular_test, y_train, y_test = train_test_split(
    X_images, tabular_features, y_target, test_size=0.2, random_state=42
)

# Step 2: Build Models
# Build VGGFace model for image feature extraction
image_input = Input(shape=(desired_width, desired_height, num_channels))
vgg_model = VGGFace(model='vgg16', include_top=False, input_tensor=image_input)

# Freeze layers to retain pretrained weights
for layer in vgg_model.layers:
    layer.trainable = False

x = Flatten()(vgg_model.output)
image_output = Dense(128, activation='relu')(x)

# Build model for tabular features
tabular_input = Input(shape=(X_tabular_train.shape[1],))
tabular_output = Dense(128, activation='relu')(tabular_input)

# Step 3: Combine Models
concatenated = Concatenate(axis=-1)([image_output, tabular_output])
x = Dense(64, activation='relu')(concatenated)
output = Dense(1, activation='linear')(x)

combined_model = Model(inputs=[image_input, tabular_input], outputs=output)

# Step 4: Train and Evaluate
combined_model.compile(optimizer='adam', loss='mean_squared_error')
combined_model.fit([X_images_train, X_tabular_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = combined_model.predict([X_images_test, X_tabular_test])

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
y_test_binary = (y_test > 0).astype(int)
y_pred_binary = (y_pred > 0).astype(int)
f1 = f1_score(y_test_binary, y_pred_binary)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'F1 Score: {f1}')


#building up the model

import os
import zipfile
import dlib
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data():
    # Load your data 
    zip_path = "dataset/age.zip"  
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    df = pd.read_csv('age_detection.csv')
    image_path = df.loc[0, 'file']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(zip_path), image_path)
    df = df[df['file'].apply(lambda x: os.path.isfile(x))]
    # Duplicate Removal
    df.drop_duplicates(subset=['file'], keep='first', inplace=True)
    df = df[df['split'].isin(['train', 'test'])]    
    # Image Quality
    # removing images that are too small
    min_width, min_height = 64, 64  # minimum acceptable dimensions
    def is_image_large_enough(file_path):
        with Image.open(file_path) as img:
            return img.width >= min_width and img.height >= min_height
    df = df[df['file'].apply(is_image_large_enough)]

    # Image Preprocessing
    # resizing and normalizing images
    def preprocess_image(file_path):
        with Image.open(file_path) as img:
            img = img.resize((min_width, min_height))  # resize
            img = img.convert('RGB')  # ensure 3 channels
            img = np.array(img) / 255.0  # normalize to [0, 1]
        return img
    df['image'] = df['file'].apply(preprocess_image)
    detector = dlib.get_frontal_face_detector()

    def contains_face(file_path):
        img = cv2.imread(file_path)
        faces = detector(img, 1)
        return len(faces) > 0

    df = df[df['file'].apply(contains_face)]
    predictor = dlib.shape_predictor('dlib/shape_predictor_68_face_landmarks.dat')  # Download this file

    def align_face(file_path):
        img = cv2.imread(file_path)
        faces = detector(img, 1)
        for rect in faces:
            shape = predictor(img, rect)
            aligned_face = dlib.get_face_chip(img, shape)
            return aligned_face

    df['aligned_face'] = df['file'].apply(align_face)

    # Background Removal

    # Lighting and Color Normalization
    def normalize_image(face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        normalized = cv2.equalizeHist(gray)
        return normalized

    df['normalized_face'] = df['aligned_face'].apply(normalize_image)

    # Data Augmentation
    def augment_image(face):
        M = cv2.getRotationMatrix2D((face.shape[1] / 2, face.shape[0] / 2), np.random.uniform(-30, 30), 1)
        rotated = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
        return rotated

    df['augmented_face'] = df['normalized_face'].apply(augment_image)
    label_encoder = LabelEncoder()
    df['ageLabel'] = label_encoder.fit_transform(df['age'])

    # Create a new column in the DataFrame to store texture features
    df['texture_features'] = None

    # Function to extract basic texture features from an image
    def get_texture_features(image_path):
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to smooth the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Compute texture features using the Laplacian operator
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian_var = np.var(laplacian)

        return laplacian_var

    # Apply the get_texture_features function to each image in the DataFrame
    texture_features_list = []
    for image_path in tqdm(df['file']):
        texture_features = get_texture_features(image_path)
        texture_features_list.append(texture_features)

    # Add the extracted texture features to the DataFrame
    df['texture_features'] = texture_features_list

    df['wrinkle_features'] = None

    # Function to extract basic wrinkle features from an image
    def get_wrinkle_features(image_path):
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection to enhance wrinkles
        edges = cv2.Canny(gray, 50, 150)

        # Compute the percentage of white pixels in the edges
        wrinkle_percentage = np.sum(edges) / (gray.shape[0] * gray.shape[1])

        return wrinkle_percentage

    # Apply the get_wrinkle_features function to each image in the DataFrame
    wrinkle_features_list = []
    for image_path in tqdm(df['file']):
        wrinkle_features = get_wrinkle_features(image_path)
        wrinkle_features_list.append(wrinkle_features)

    # Add the extracted wrinkle features to the DataFrame
    df['wrinkle_features'] = wrinkle_features_list

    # Load the pre-trained facial landmark predictor
    predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"  # Replace with the path to the shape predictor model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Function to detect facial landmarks and extract eye-related features
    def detect_eye_features(image_pixels):
        # Convert the list of pixel values to a NumPy array
        image_array = np.array(image_pixels, dtype=np.uint8)

        # Ensure the image has 3 channels (for compatibility with cv2.COLOR_BGR2GRAY)
        if image_array.ndim == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)

        # Check if a face is detected
        if len(faces) > 0:
            # Get facial landmarks for the first detected face
            shape = predictor(gray, faces[0])

            # Extract eye-related features
            left_eye_openness = shape.part(47).y - shape.part(43).y  # Example: vertical distance between eyebrow and lower eyelid
            right_eye_openness = shape.part(40).y - shape.part(38).y

            return left_eye_openness, right_eye_openness

    # Example: Apply the detect_eye_features function to each image in the DataFrame
    eye_features = df['normalized_face'].apply(detect_eye_features)

    # Example: Add the extracted eye features to the DataFrame
    df[['left_eye_openness', 'right_eye_openness']] = pd.DataFrame(eye_features.tolist(), index=df.index)
    df = df.dropna()

    return df

def load_model(df_cleaned):
    # Load the trained model
    desired_width = 128
    desired_height = 128
    num_channels = 3  # Assuming color images, adjust if grayscale

    # Step 1: Prepare Data
    # Extract 'eye openness', 'texture features', and 'wrinkle features'
    tabular_features = df_cleaned[['texture_features', 'wrinkle_features', 'left_eye_openness','right_eye_openness']]
    # Standardize numerical features
    scaler = StandardScaler()
    tabular_features = scaler.fit_transform(tabular_features)

    # Load and preprocess images
    desired_channels = 3  # Assuming color images, adjust if grayscale

    def preprocess_image(image):
        # Resize to a common size
        resized_image = cv2.resize(image, (desired_width, desired_height))
        # Ensure the image has the correct number of channels
        if resized_image.shape[-1] != desired_channels:
            # If the image has more than 3 channels, convert to grayscale
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            # Add an extra dimension to represent the single channel
            resized_image = np.expand_dims(resized_image, axis=-1)
        # Normalize pixel values to be between 0 and 1
        resized_image = resized_image / 255.0
        # Add other preprocessing steps as needed (e.g., data augmentation)
        return resized_image

    X_images = np.array([preprocess_image(image) for image in df_cleaned['image'].values])
    # Extract target
    y_target = df_cleaned['ageLabel'].values

    # Split data into training and testing sets
    X_images_train, X_images_test, X_tabular_train, X_tabular_test, y_train, y_test = train_test_split(
        X_images, tabular_features, y_target, test_size=0.2, random_state=42
    )

    # Step 2: Build Models
    # Build CNN model for image feature extraction
    image_input = Input(shape=(desired_width, desired_height, num_channels))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_output = Dense(128, activation='relu')(x)

    # Build model for tabular features
    tabular_input = Input(shape=(X_tabular_train.shape[1],))
    tabular_output = Dense(128, activation='relu')(tabular_input)

    # Step 3: Combine Models
    concatenated = Concatenate(axis=-1)([image_output, tabular_output])
    x = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='linear')(x)

    combined_model = Model(inputs=[image_input, tabular_input], outputs=output)

    # Step 4: Train and Evaluate
    combined_model.compile(optimizer='adam', loss='mean_squared_error')
    combined_model.fit([X_images_train, X_tabular_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    y_pred = combined_model.predict([X_images_test, X_tabular_test])

    return combined_model

def preprocess_input(user_input):
    processed_input = user_input.resize((desired_width, desired_height))  # Resize if needed
    processed_input = np.array(processed_input) / 255.0  # Normalize to [0, 1]
    processed_input = np.expand_dims(processed_input, axis=0)  # Add batch dimension
    return processed_input

def make_prediction(model, processed_input):
    image_input, tabular_input = processed_input

    prediction = model.predict([image_input, tabular_input])

    return prediction
