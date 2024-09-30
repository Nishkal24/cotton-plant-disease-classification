# Cotton Plant Diseases Classification

## Overview
This project focuses on classifying diseases in cotton plants using deep learning techniques. By leveraging MobileNetV2, a lightweight convolutional neural network, the aim is to assist farmers in early disease detection, thus improving crop management and reducing yield loss.

## Table of Contents
- [Background](#background)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Implementation](#implementation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Background
Cotton is a vital crop for many economies, and diseases can severely impact yield. Early detection is crucial for effective management and treatment. This project employs machine learning to automate the identification of diseases, making it easier for farmers to take timely action.

## Technologies Used
- **Python**: Programming language used for implementation.
- **TensorFlow**: Deep learning framework for building and training the model.
- **Keras**: High-level neural networks API for simplifying model creation.
- **OpenCV**: For image processing.
- **NumPy** and **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.

## Dataset
The dataset consists of a diverse collection of cotton leaf images, categorized into various disease classes, including:
- Healthy
- Yellow Leaf Curl Virus
- Bacterial Blight
- Fungal Diseases

The images were sourced from publicly available agricultural datasets. Data augmentation techniques were applied to increase variability and robustness of the model.

## Model Architecture
The project utilizes **MobileNetV2**, known for its efficiency and accuracy in image classification tasks. The architecture includes:
- Depthwise separable convolutions
- Global average pooling
- Fully connected layers with softmax activation for multi-class classification

## Implementation
1. **Setup Environment**:
   - Install required libraries:
     ```bash
     pip install tensorflow keras opencv-python numpy pandas matplotlib
     ```

2. **Data Preprocessing**:
   - Load and preprocess images, including resizing, normalization, and augmentation.

3. **Model Training**:
   - Compile the MobileNetV2 model with an appropriate optimizer and loss function.
   - Train the model using the prepared dataset.

4. **Evaluation**:
   - Evaluate the model on a validation set and analyze performance metrics such as accuracy, precision, and recall.

5. **Prediction**:
   - Use the trained model to classify new cotton leaf images.

## Results
The model achieved an accuracy of over 90% on the validation dataset. Confusion matrices and classification reports were generated to analyze performance across different disease classes.

## Usage
To use the model for predicting diseases in new images:
1. Load the trained model.
2. Preprocess the input image in the same way as the training data.
3. Use the model to predict the disease class.

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('path_to_model.h5')

# Preprocess input image
image = cv2.imread('path_to_image.jpg')
image = cv2.resize(image, (224, 224))  # Resize to input shape
image = image / 255.0  # Normalize
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(image)
