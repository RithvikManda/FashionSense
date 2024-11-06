# FashionSense: Fashion Recommendation System

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
