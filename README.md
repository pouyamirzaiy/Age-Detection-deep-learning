# Age Detection with Deep Learning

This project explores deep learning approaches for age detection from facial images.
It uses both custom CNN architectures and pre-trained VGGFace models, combined with advanced preprocessing and feature extraction techniques.

## Project Overview

Age detection has applications in healthcare, surveillance, marketing, and social media.
The goal of this project is to build a robust and efficient model capable of predicting an individual’s age accurately.

Key contributions:
- Image preprocessing: face detection, alignment, normalization
- Feature extraction: wrinkles, hair color, facial landmarks
- CNN-based model for regression
- Transfer learning with VGGFace
- Evaluation with MAE, MSE, and F1-score

## Project Structure

Age-Detection-deep-learning/
│

├── src/ # Source code (training, preprocessing, evaluation)

├── results/ # Figures, plots, and metrics

├── report/ # Final report (PDF)

│ └── Final_Project_Report.pdf

├── requirements.txt # Dependencies

└── README.md # Project description

## Usage
Training
python src/train.py
Evaluation
python src/evaluate.py
## Results

The CNN model achieved slightly better MAE/MSE than VGGFace

Both models performed comparably on the F1 score

Normalized facial images improved performance significantly

## Authors 

Pouya Mirzaei Zadeh – p.mirzaiyzadeh@cs.sbu.ac.ir

Shadi Sefidgar – shd.wnkx@gmail.com

Department of Computer Science, Shahid Beheshti University

## License 

This project is licensed under the MIT License.
