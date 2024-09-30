import os


# Define the dataset directory path
dataset_dir = r'C:\\Users\\nishk\\Desktop\\Original Dataset'

# Get the list of class names
class_names = os.listdir(dataset_dir)
# print(class_names)


class_files = {class_name: os.listdir(os.path.join(dataset_dir, class_name)) for class_name in class_names}
print(class_files)