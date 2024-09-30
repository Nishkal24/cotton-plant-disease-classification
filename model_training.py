import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Set directories (assuming your dataset is organized into subdirectories for each class)
dataset_dir = r'C:\Users\HP\OneDrive\Desktop\original_dataset' 

# Image dimensions
img_height, img_width = 224, 224  
batch_size = 32

# Step 1: Data Augmentation and Splitting

# Data augmentation setup
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% will be split as validation/test data
)

# Create train, validation, and test generators
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_test_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Split validation_test_generator into validation and test sets (50% each from 20% validation split)
X_val_test, y_val_test = validation_test_generator.__next__()
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Step 2: Load Pre-trained MobileNetV2 Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers
base_model.trainable = False

# Step 3: Build Model Architecture
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(6, activation='softmax')  # 6 classes for cotton plant disease classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Add Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 5: Train the Model
history = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=batch_size,
    callbacks=[early_stopping]
)

# Step 6: Save the Model
model.save('cotton_plant_disease_mobilenet_model.h5')

# Step 7: Load and Evaluate on Test Set
model = tf.keras.models.load_model('cotton_plant_disease_mobilenet_model.h5')
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')










# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import load_model

# # Set directories (assumes your dataset is organized into subdirectories for each class)
# dataset_dir = r'C:\Users\HP\OneDrive\Desktop\original_dataset' 

# # Image dimensions
# img_height, img_width = 224, 224  
# batch_size = 32

# # Data Preprocessing 
# datagen = ImageDataGenerator(
#     rescale=1.0/255.0, 
#     validation_split=0.1  # 10% for validation
# )

# # Load and split the dataset
# train_generator = datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training',
#     shuffle=True
# )

# validation_generator = datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation',
#     shuffle=True
# )

# # Splitting the data into train, validation, and test (20% for testing)
# X_train, y_train = train_generator.__next__()
# X_val, y_val = validation_generator.__next__()

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Model Architecture
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(6, activation='softmax')  # 6 classes for the cotton plant disease classification
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=20,
#     batch_size=batch_size
# )

# # Save the model
# model.save('cotton_plant_disease_model.h5')

# # Load and evaluate on test set
# model = load_model('cotton_plant_disease_model.h5')
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')