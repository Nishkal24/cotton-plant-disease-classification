import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Define the dataset directory path
dataset_dir = r'C:\Users\HP\OneDrive\Desktop\original_dataset'

# Get the list of class names
class_names = os.listdir(dataset_dir)
print(class_names)
# Create a dictionary to store file paths for each class
class_files = {class_name: os.listdir(os.path.join(dataset_dir, class_name)) for class_name in class_names}

# # Check the class distribution
# for class_name, files in class_files.items():
#     print(f"Class {class_name} has {len(files)} samples.")

# Define the dataset directory path
#dataset_dir = r'C:\Users\HP\OneDrive\Desktop\original_dataset'
target_samples = 500  # Target number of images for each class

# Define the data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Get the list of class names
# class_names = os.listdir(dataset_dir)

# Oversample each class to 500 images
for class_name in class_names:
    class_dir = os.path.join(dataset_dir, class_name)
    files = os.listdir(class_dir)
    num_samples = len(files)

    # If the class has fewer than 500 samples, perform oversampling
    if num_samples < target_samples:
        print(f"Class {class_name} has {num_samples} samples. Oversampling to {target_samples} samples.")

        # Number of images to generate
        num_to_generate = target_samples - num_samples

        # Randomly choose files to augment
        for i in range(num_to_generate):
            file_to_copy = random.choice(files)
            img_path = os.path.join(class_dir, file_to_copy)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Generate new augmented image
            j = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=class_dir, save_prefix='aug', save_format='jpeg'):
                j += 1
                if j >= 1:
                    break  # Only generate one new image per loop iteration

print(f"Oversampling complete. All classes now have {target_samples} samples.")
