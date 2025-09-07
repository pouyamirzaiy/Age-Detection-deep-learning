"""
Auto-extracted preprocessing code from notebook on 2025-09-07T22:52:35.169848
Unnecessary environment/setup comments removed.
"""

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


# Image Quality
# removin images that are too small
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


#Face Detection
detector = dlib.get_frontal_face_detector()

def contains_face(file_path):
    img = cv2.imread(file_path)
    faces = detector(img, 1)
    return len(faces) > 0

df = df[df['file'].apply(contains_face)]


import urllib.request
import bz2

# URL of the file to be downloaded
url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# Path where the downloaded file will be stored
output_path = "shape_predictor_68_face_landmarks.dat.bz2"

# Download the file from `url` and save it locally under `output_path`:
urllib.request.urlretrieve(url, output_path)

# Open the .bz2 file for reading
with bz2.open(output_path, 'rb') as f:
    # Decompress the data
    decompressed_data = f.read()

# Write the decompressed data to a new file
with open('shape_predictor_68_face_landmarks.dat', 'wb') as f:
    f.write(decompressed_data)


# Face Alignment
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download this file

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

#Data Augmentation
def augment_image(face):
    M = cv2.getRotationMatrix2D((face.shape[1] / 2, face.shape[0] / 2), np.random.uniform(-30, 30), 1)
    rotated = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
    return rotated

df['augmented_face'] = df['normalized_face'].apply(augment_image)


def display_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    
    plt.show()

# Display the first 5 images from the training set
display_images(df['augmented_face'][:5], label_encoder.inverse_transform(df['ageLabel'])[:5])


#showing images

# Function to display images
def display_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    
    plt.show()

# Display the first 5 images from the training set
display_images(df['augmented_face'][:5], label_encoder.inverse_transform(df['ageLabel'])[:5])


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
for image_path in tqdm(train_df['filepaths']):
    texture_features = get_texture_features(image_path)
    texture_features_list.append(texture_features)

# Add the extracted texture features to the DataFrame
df['texture_features'] = texture_features_list

# Visualize Texture Features Across Age Groups
age_groups = df.groupby('label')

# Set up subplots for visualization
fig, axs = plt.subplots(len(age_groups), figsize=(8, 5 * len(age_groups)))

for i, (age_group, group_data) in enumerate(age_groups):
    axs[i].set_title(f"Age Group: {age_group}")

    # Visualize texture features for a sample of images in the age group
    axs[i].hist(group_data['texture_features'], bins=20, color='skyblue', edgecolor='black')
    axs[i].set_xlabel('Texture Features')
    axs[i].set_ylabel('Frequency')

plt.show()


# Create a new column in the DataFrame to store wrinkle features
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
for image_path in tqdm(train_df['filepaths']):
    wrinkle_features = get_wrinkle_features(image_path)
    wrinkle_features_list.append(wrinkle_features)

# Add the extracted wrinkle features to the DataFrame
df['wrinkle_features'] = wrinkle_features_list

# Visualize Wrinkle Features Across Age Groups
age_groups = df.groupby('label')

# Set up subplots for visualization
fig, axs = plt.subplots(len(age_groups), figsize=(8, 5 * len(age_groups)))

for i, (age_group, group_data) in enumerate(age_groups):
    axs[i].set_title(f"Age Group: {age_group}")

    # Visualize wrinkle features for a sample of images in the age group
    axs[i].hist(group_data['wrinkle_features'], bins=20, color='lightcoral', edgecolor='black')
    axs[i].set_xlabel('Wrinkle Features')
    axs[i].set_ylabel('Frequency')

plt.show()


#probably useless

def extract_hair_color(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to HSV color space for better color analysis
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a mask for the hair region in the HSV color space
    lower_hair_color = np.array([0, 10, 40])
    upper_hair_color = np.array([30, 200, 255])
    hair_mask = cv2.inRange(hsv_image, lower_hair_color, upper_hair_color)
    
    # Apply the hair mask to the original image
    hair_region = cv2.bitwise_and(image, image, mask=hair_mask)
    
    # Calculate the dominant hair color
    dominant_color = np.mean(hair_region, axis=(0, 1)).astype(int)
    
    return dominant_color

# Apply the extract_hair_color function to each image in the DataFrame
df['hair_color'] = df['filepaths'].apply(extract_hair_color)



# Load the pre-trained facial landmark predictor
predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"  # Replace with the path to the shape predictor model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Function to detect facial landmarks in an image
def detect_landmarks(image_pixels):
    # Convert the list of pixel values to a NumPy array
    image_array = np.array(image_pixels, dtype=np.uint8)

    # Detect faces in the image
    faces = detector(image_array)

    # Loop over each face and get facial landmarks
    landmarks_list = []
    for face in faces:
        shape = predictor(image_array, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
        landmarks_list.append(landmarks)

        # Draw landmarks on the image (comment out if you don't want to display)
        for (x, y) in landmarks:
            cv2.circle(image_array, (x, y), 2, (0, 255, 0), -1)

    # Display the image with landmarks (comment out if you don't want to display)
    cv2.imshow("Facial Landmarks", image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return landmarks_list

# Apply the detect_landmarks function to each image in the DataFrame
df['landmarks'] = df['augmented_face'].apply(detect_landmarks)

# Now df['landmarks'] contains the detected facial landmarks for each image


#eye openness detection (im not sure how accurate this is and if it works)


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

    return None

# Example: Apply the detect_eye_features function to each image in the DataFrame
eye_features = df['normalized_face'].apply(detect_eye_features)

# Example: Add the extracted eye features to the DataFrame
df[['left_eye_openness', 'right_eye_openness']] = pd.DataFrame(eye_features.tolist(), index=df.index)


# Save the trained model
age_predicton = load_model(load_and_preprocess_data())
age_predicton.save('saved_model.h5')


# streamlit_app.py

import streamlit as st

# Load the model
model = load_age_model()

def main():
    st.title("Age Detection App")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Resize the image to match the model's expected input size
        resized_image = image.resize((128, 128))  # Adjust dimensions based on your model
        st.image(resized_image, caption="Resized Image", use_column_width=True)

        # Preprocess the input
        processed_input = preprocess_input(resized_image)

        # Make a prediction
        prediction = make_prediction(model, processed_input)

        # Display the result
        # Display the result
        st.write(f"Predicted Age: {model.prediction[0]}")


if __name__ == "__main__":
    main()
