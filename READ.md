Cotton Plant Disease Classification Using MobileNetV2
Table of Contents

    Project Overview
    Prerequisites
    Directory Structure
    Project Workflow
    Instructions
    Key Files
    Acknowledgments
    Contact

Project Overview

This project classifies cotton plant diseases into six categories using a Convolutional Neural Network (CNN) based on the MobileNetV2 architecture. The workflow includes data augmentation, model training, evaluation, and predictions on new images.

Key Features:

    Utilizes data augmentation to balance and expand the dataset.
    Implements transfer learning with MobileNetV2 for high accuracy.
    Provides tools to train, test, and predict.

Prerequisites
Required Software and Libraries

Ensure the following are installed:

    Python 3.7 or later
    TensorFlow 2.x
    NumPy
    Matplotlib
    scikit-learn

Install required packages with:

pip install tensorflow numpy matplotlib scikit-learn

Directory Structure
Input Data

The dataset should be organized into subdirectories by class:

original_dataset/
├── Class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...

Generated Files

    augmented_dataset/: Contains the augmented dataset.
    trained_model.h5: The trained MobileNetV2 model.

Project Workflow

    Data Augmentation
    Augment the dataset to ensure balanced class samples.
    Training the Model
    Train the MobileNetV2 model using the augmented dataset and save it as trained_model.h5.
    Testing and Evaluation
    Evaluate the model on the test dataset and generate metrics.
    Making Predictions
    Use the trained model to predict the class of a new image.

Instructions
Step 1: Data Augmentation

Run the script to augment your dataset:

python oversampling_script.py

Step 2: Train the Model

Train the MobileNetV2 model using:

python train_model.py

The trained model will be saved as trained_model.h5.
Step 3: Evaluate the Model

Evaluate the model on the test set:

python test_model.py

This step computes metrics such as accuracy and generates a classification report.
Step 4: Predict on a New Image

To predict the label of a single image:

    Specify the image path in predict_image.py.
    Run the script:

    python predict_image.py

Key Files

    oversampling_script.py
    Augments the dataset to balance class distributions.

    train_model.py
    Contains logic for training the MobileNetV2 model with early stopping.

    test_model.py
    Evaluates the model on the test dataset and prints metrics.

    predict_image.py
    Loads the trained model and predicts the class of a single input image.

Acknowledgments

    The project leverages MobileNetV2 for transfer learning.
    TensorFlow's ImageDataGenerator is used for data augmentation and preprocessing.

Contact

If you have any questions or need further assistance, feel free to reach out:

    Email: nishkalpokar24@gmail.com