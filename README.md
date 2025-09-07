# Age Detection with Deep Learning

This project explores deep learning approaches for age detection from facial images.
It investigates both custom CNN architectures and pre-trained VGGFace models, combined with advanced preprocessing and feature extraction techniques.

## Project Overview
Age detection has wide applications in healthcare, surveillance, marketing, and social media.
The goal of this project is to build a robust and efficient model capable of predicting an individual’s age with high accuracy.

Key contributions:
- Implemented image preprocessing (face detection, alignment, normalization)
- Extracted facial features such as wrinkles, hair color, and facial landmarks
- Designed and trained a CNN-based model for regression
- Applied transfer learning using VGGFace for comparison
- Evaluated models using MAE, MSE, and F1-score

## Project Structure
Age-Detection-DeepLearning/
│
├── src/ # Source code (training, preprocessing, evaluation)
├── results/ # Experimental results, plots, and metrics
├── report/ # Final report (PDF)
│ └── Final_Project_Report.pdf
├── requirements.txt # Dependencies
└── README.md # Project description

bash
Copy code

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/Age-Detection-DeepLearning.git
cd Age-Detection-DeepLearning
pip install -r requirements.txt
Usage
Training
Run the training script:

bash

python src/train.py
Evaluation
Evaluate trained models:

bash

python src/evaluate.py
Results
CNN model achieved slightly better MAE/MSE than VGGFace

Both models performed comparably on F1 score

Normalized facial images improved performance significantly

Authors
Pouya Mirzaei Zadeh – p.mirzaiyzadeh@cs.sbu.ac.ir

Shadi Sefidgar – shd.wnkx@gmail.com

Department of Computer Science, Shahid Beheshti University

License
This project is licensed under the MIT License.

