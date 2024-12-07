import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Step 1: Load Dataset 

dataset_dir = r'C:\Users\HP\OneDrive\Desktop\Minor Project\original_dataset' 

img_height, img_width = 224, 224  
batch_size = 32

# Step 2: Data Augmentation and Splitting

# Data Augmentation 
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.3  
)

# Create Train and Validation/Test Generators
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
    shuffle=False  
)

# Collect all Validation/Test Samples from the Generator
X_val_test, y_val_test = [], []
for i in range(len(validation_test_generator)):
    X, y = validation_test_generator[i]  
    X_val_test.extend(X)
    y_val_test.extend(y)

X_val_test = np.array(X_val_test)
y_val_test = np.array(y_val_test)

# Split Validation and Test Data 
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=2/3, random_state=42)

# Print the number of samples in each set
print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test)}")

# Step 3: Load Pre-Trained MobileNetV2 Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the Base Model Layers
base_model.trainable = False

# Step 4: Build Model Architecture
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(6, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Add Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 6: Train the Model
history = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=batch_size,
    callbacks=[early_stopping]
)

# Print the Final Training and Validation Accuracy & Loss
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

# Step 7: Save the Trained Model
model.save('trained_model.h5')

# Step 8: Print Classification Report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Step 9: Plot Training & Validation Accuracy & Loss
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()

# Step 10: Load and Test the trained Model
model = tf.keras.models.load_model('trained_model.h5')
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'\nTesting Accuracy: {test_accuracy * 100:.2f}%')











# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split

# # Set directories (assuming your dataset is organized into subdirectories for each class)
# dataset_dir = r'C:\Users\HP\OneDrive\Desktop\Minor Project\original_dataset' 

# # Image dimensions
# img_height, img_width = 224, 224  
# batch_size = 32

# # Step 1: Data Augmentation and Splitting

# # Data augmentation setup
# datagen = ImageDataGenerator(
#     rescale=1.0 / 255.0,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=0.2  # 20% will be split as validation/test data
# )

# # Create train, validation, and test generators
# train_generator = datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training',
#     shuffle=True
# )

# validation_test_generator = datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation',
#     shuffle=True
# )

# # Split validation_test_generator into validation and test sets (50% each from 20% validation split)
# X_val_test, y_val_test = validation_test_generator.__next__()
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# # Step 2: Load Pre-trained MobileNetV2 Model
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# # Freeze the base model layers
# base_model.trainable = False

# # Step 3: Build Model Architecture
# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     BatchNormalization(),
#     Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     Dropout(0.5),
#     Dense(6, activation='softmax')  # 6 classes for cotton plant disease classification
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Step 4: Add Early Stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Step 5: Train the Model
# history = model.fit(
#     train_generator,
#     validation_data=(X_val, y_val),
#     epochs=20,
#     batch_size=batch_size,
#     callbacks=[early_stopping]
# )

# # Step 6: Save the Model
# model.save('trained_model.h5')

# # Step 7: Load and Evaluate on Test Set
# model = tf.keras.models.load_model('trained_model.h5')
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')