"""
Auto-extracted CNN model code from notebook on 2025-09-07T22:52:35.169848
"""

# model.py

from keras.models import load_model

def load_age_model():
    
    # Load the trained model
    model = load_model('saved_model.h5')  # Adjust the filename based on your saved model
    return model
