import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Step 1: Load the Saved Model
model = tf.keras.models.load_model(r'C:\Users\nishk\Desktop\New folder (5)\trained_model.h5')

# Step 2: Load the Dataset and Extract Class Labels
dataset_dir = r'C:\Users\HP\OneDrive\Desktop\Minor Project\original_dataset'  
img_height, img_width = 224, 224  
batch_size = 32

# Load data to Access the Class Labels
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
data_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Get Class Labels from the Generator
class_labels = list(data_generator.class_indices.keys())  

# Step 3: Load and Preprocess the Input Image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

# Step 4: Predict the Label for a Single Image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  
    predicted_label = class_labels[predicted_class[0]]
    return predicted_label

# Step 5: Display Prediction for an Input Image
def display_prediction(img_path):
    predicted_label = predict_image(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.title(f'Predicted Label: {predicted_label}')
    plt.axis('off')
    plt.show()

# Step 6: Enter Image Path and Obtain Predicted Label
img_path = r"C:\Users\HP\OneDrive\Desktop\Minor Project\original_dataset\Leaf Redding\LR00101.jpg"  
display_prediction(img_path)