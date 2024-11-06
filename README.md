# TrendSense : AI-Based Fashion Style Recommendation

## Overview
FashionSense is a deep learning-based fashion recommendation system designed to recommend similar clothing items based on images. The system uses computer vision and pre-trained models like VGG16 to extract features from clothing images and recommends the most similar items from a collection.

This project was created to showcase how fashion recommendation systems work using modern deep learning techniques, including image augmentation, feature extraction, and similarity-based retrieval.

## Features
- **Image Upload**: Users can upload clothing images to get similar clothing item recommendations.
- **Augmentation**: Data augmentation techniques are applied to the dataset to increase the robustness of the model.
- **VGG16 Model**: Pre-trained VGG16 model is used to extract features from clothing images.
- **Similarity Calculation**: The system uses cosine similarity to recommend the most relevant fashion items based on the input image.

## Installation

### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- ImgAug
- Matplotlib
- PIL (Python Imaging Library)
- cvzone
- Other necessary Python libraries (listed in `requirements.txt`)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/RithvikManda/FashionSense.git
   cd FashionSense
2.Install dependencies: You can install the necessary Python packages using pip. Ensure you have a virtual environment activated, or use pip directly.

   ```bash
   Copy code
   pip install -r requirements.txt

3.Extract your dataset (if not already extracted): If you're working with a compressed dataset, extract the images into the specified folder (/content/women_fashion in your case). You can use the following command:

'''bash python
Copy code
from zipfile import ZipFile
import os

zip_file_path = 'path_to_your_dataset.zip'
extraction_path = 'path_to_extracted_images'

with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)
4.Run the system: Once the environment is set up, run the script to train and evaluate the recommendation model.

```bash

python recommendation_system.py

### Usage
### Fashion Item Recommendation
To get fashion recommendations:

1.Upload a clothing image to the system.
2.The system will extract the features of the image and recommend the top N similar items based on cosine similarity.

Example
Given an input image, the system will show the original image along with a set of recommended images from the dataset.

### Evaluation Metrics
The model's performance is evaluated using the following metrics:

Precision: How many of the recommended items were relevant.
Recall: How many of the relevant items were recommended.
F1-Score: The balance between precision and recall.
Example Results:
Average Precision: 0.84
Average Recall: 0.7
F1-Score: 0.875

## Acknowledgements
VGG16 model for feature extraction
ImgAug library for data augmentation
OpenCV for image processing
Matplotlib for visualizations
CVZone library for image collection and processing
Contact
For any questions or suggestions, feel free to contact Rithvik Manda.


