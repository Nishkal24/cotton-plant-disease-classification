import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import shutil

# Define the dataset directory path
dataset_dir = r'C:\Users\HP\OneDrive\Desktop\original_dataset'
augmented_dir = r'C:\Users\HP\OneDrive\Desktop\augmented_dataset'

# Ensure augmented dataset folder exists
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the target number of samples per class
target_samples_per_class = 2500  # 2500 images per class

# Iterate through each class to augment the data
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    augmented_class_dir = os.path.join(augmented_dir, class_name)

    # Ensure class folder exists in the augmented dataset folder
    if not os.path.exists(augmented_class_dir):
        os.makedirs(augmented_class_dir)

    images = os.listdir(class_dir)
    num_existing_images = len(images)

    # Copy original images to the augmented dataset directory
    for img in images:
        shutil.copy(os.path.join(class_dir, img), augmented_class_dir)

    # Generate additional images if needed
    if num_existing_images < target_samples_per_class:
        print(f"Class {class_name}: Augmenting from {num_existing_images} to {target_samples_per_class} images.")
        images_to_generate = target_samples_per_class - num_existing_images

        # Generate and save new augmented images
        for i in range(images_to_generate):
            img_path = random.choice(images)  # Randomly choose an image
            img = load_img(os.path.join(class_dir, img_path))  # Load the image
            x = img_to_array(img)  # Convert to array
            x = x.reshape((1,) + x.shape)  # Reshape for the generator

            # Generate one new image and save it
            for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_class_dir, save_prefix='aug', save_format='jpeg'):
                break  # Save only one image per loop iteration

print("Data augmentation complete. 2,500 images per class have been saved.")