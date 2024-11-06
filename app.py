from zipfile import ZipFile
import os

zip_file_path = '/content/women-fashion.zip'
extraction = '/content/women/'

if not os.path.exists(extraction):
    os.makedirs(extraction)

with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction)

extracted_files = os.listdir(extraction)
print(extracted_files[:10])

import cv2
import imgaug.augmenters as iaa
import numpy as np
import os
from glob import glob

# Set your input and output paths
input_folder = '/content/women/women fashion'
extraction_directory = '/content/women_fashion/'
os.makedirs(extraction_directory, exist_ok=True)

# Load images
image_paths = glob(os.path.join(input_folder, "*.jpg"))  # adjust for .png if needed
images = [cv2.imread(img_path) for img_path in image_paths]

# Define augmentation sequence
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),                   # Horizontal flip
    iaa.Flipud(0.2),                   # Vertical flip
    iaa.Affine(scale=(0.8, 1.2)),      # Scale images
    iaa.Affine(rotate=(-30, 30)),      # Rotate images
    iaa.Affine(translate_percent=(-0.2, 0.2)),  # Translate
    iaa.Multiply((0.8, 1.2)),          # Adjust brightness
])

# Generate augmented images
num_augmentations = 500
counter = 0

for i in range(num_augmentations):
    # Choose a random image from your list
    image = images[i % len(images)]

    # Augment the image
    augmented_image = augmenter(image=image)

    # Save the augmented image
    output_path = os.path.join(extraction_directory, f"augmented_{counter}.jpg")
    cv2.imwrite(output_path, augmented_image)
    counter += 1

print("Augmentation complete!")

# correcting the path to include the 'women fashion' directory and listing its contents
extraction_directory_updated = os.path.join('/content/', 'women_fashion')

# list the files in the updated directory
extracted_files_updated = os.listdir(extraction_directory_updated)
extracted_files_updated[:9], len(extracted_files_updated)

from PIL import Image
import matplotlib.pyplot as plt

# function to load and display an image
def display_image(file_path):
    image = Image.open(file_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# display the first image to understand its characteristics
first_image_path = os.path.join(extraction_directory_updated, extracted_files_updated[0])
display_image(first_image_path)

import glob

# directory path containing your images
image_directory = '/content/women_fashion'

image_paths_list = [file for file in glob.glob(os.path.join(image_directory, '*.*')) if file.endswith(('.jpg', '.png', '.jpeg', 'webp'))]

# print the list of image file paths
print(image_paths_list)


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

all_features = []
all_image_names = []

for img_path in image_paths_list:
    preprocessed_img = preprocess_image(img_path)
    features = extract_features(model, preprocessed_img)
    all_features.append(features)
    all_image_names.append(os.path.basename(img_path))

from scipy.spatial.distance import cosine

def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
    # pre-process the input image and extract features
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

    # calculate similarities and find the top N similar images
    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]

    # filter out the input image index from similar_indices
    similar_indices = [idx for idx in similar_indices if idx != all_image_names.index(input_image_path)]

    # display the input image
    plt.figure(figsize=(15, 10))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title("Input Image")
    plt.axis('off')

    # display similar images
    for i, idx in enumerate(similar_indices[:top_n], start=1):
        image_path = os.path.join('/content/women_fashion', all_image_names[idx])
        plt.subplot(1, top_n + 1, i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f"Recommendation {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

input_image_path = '/content/women_fashion/augmented_10.jpg'
recommend_fashion_items_cnn(input_image_path, all_features, image_paths_list, model, top_n=10)



import numpy as np
from scipy.spatial.distance import cosine
import random

# Function to simulate dynamic generation of ground truth (for example purposes)
def generate_ground_truth(image_paths, num_relevant_images=3):
    ground_truth = {}
    for image in image_paths:
        # Randomly select 'num_relevant_images' as relevant items (this could be loaded from actual data)
        relevant_images = random.sample(image_paths, num_relevant_images)
        ground_truth[image] = relevant_images
    return ground_truth

# Function to simulate dynamic recommendations (replace with actual recommendation logic)
def generate_recommendations(image_paths, top_n=5):
    recommendations = {}
    for image in image_paths:
        # Randomly select 'top_n' images as recommended (this should be replaced with actual recommendations)
        recommended_images = random.sample(image_paths, top_n)
        recommendations[image] = recommended_images
    return recommendations

# Function to calculate Precision and Recall
def calculate_accuracy(recommendations, ground_truth, top_n=5):
    precision_list = []
    recall_list = []
    
    for input_image, relevant_images in ground_truth.items():
        # Get the top-N recommended images for the current input image
        recommended_images = recommendations.get(input_image, [])
        
        # Ensure top-N (if fewer recommendations are provided, adjust accordingly)
        recommended_images = recommended_images[:top_n]
        
        # Calculate Precision: TP / (TP + FP)
        tp = len(set(recommended_images) & set(relevant_images))  # True positives
        fp = len(set(recommended_images) - set(relevant_images))  # False positives
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        
        # Calculate Recall: TP / (TP + FN)
        fn = len(set(relevant_images) - set(recommended_images))  # False negatives
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)

    # Calculate average precision and recall
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    # Calculate F1-score
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0
    
    return avg_precision, avg_recall, f1_score

# Simulate dynamic image paths (you can replace this with actual image paths from your dataset)
image_paths = [f'image_{i}.jpg' for i in range(1, 21)]  # Example: image_1.jpg, image_2.jpg, ...

# Generate ground truth and recommendations dynamically
ground_truth = generate_ground_truth(image_paths)
recommendations = generate_recommendations(image_paths)

# Calculate accuracy metrics for top-N recommendations
top_n = 5
avg_precision, avg_recall, f1_score = calculate_accuracy(recommendations, ground_truth, top_n)

# Print results
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")


