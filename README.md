# MVTec-AD Industrial Anomaly Detection
An end-to-end Machine Learning and Computer Vision project for industrial anomaly detection and defect localization using the MVTec Anomaly Detection dataset.

The project covers the complete workflow from dataset acquisition and model training in Google Colab to model evaluation, anomaly visualization, and deployment through a web-based inference application.

## Overview
Industrial quality inspection traditionally requires manual visual inspection to identify defective products. This process can be time-consuming, inconsistent, and difficult to scale.

This project applies Deep Learning and Computer Vision techniques to automatically identify anomalous patterns in industrial product images.

The system is designed to learn normal product characteristics and detect deviations that may indicate defects or anomalies.

The complete project consists of two main components:
1. Machine Learning Model Development
2. Web-Based Inference Application

```text
MVTec AD Dataset
       │
       ▼
Dataset Preparation
       │
       ▼
Data Preprocessing
       │
       ▼
Model Training
(Google Colab)
       │
       ▼
Model Evaluation
       │
       ▼
Anomaly Detection
       │
       ▼
Defect Localization
       │
       ▼
Trained Model
       │
       ▼
Model Storage
       │
       ▼
Web Application
       │
       ▼
Image Inference
       │
       ▼
Prediction Result
```

---

## Features
* Industrial anomaly detection
* Defect classification
* Anomaly score calculation
* Normal and anomaly prediction
* Defect localization
* Anomaly heatmap visualization
* Model evaluation and performance analysis
* Image upload for web-based inference
* End-to-end Machine Learning pipeline
* Reproducible training using Google Colab

---

## Dataset
This project uses the MVTec Anomaly Detection Dataset, a benchmark dataset for industrial anomaly detection and defect localization.

The dataset is downloaded directly from Kaggle during the training process.  
Dataset Source:
> https://www.kaggle.com/datasets/ipythonx/mvtec-ad/data

The dataset contains multiple industrial object and texture categories with normal and defective samples.

### Dataset Structure

```text
category/
│
├── train/
│   └── good/
│
├── test/
│   ├── good/
│   ├── defect_type_1/
│   ├── defect_type_2/
│   └── ...
│
└── ground_truth/
    ├── defect_type_1/
    └── defect_type_2/
```

The training dataset primarily contains normal images.

The testing dataset contains both normal and defective images.

Ground truth masks are available for defective samples and can be used for pixel-level anomaly detection and defect localization.

---

## Project Workflow
The complete Machine Learning pipeline is implemented inside a single Jupyter Notebook.
```text
Dataset
   │
   ▼
Dataset Download
   │
   ▼
Data Exploration
   │
   ▼
Data Preprocessing
   │
   ▼
Model Development
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Anomaly Score Calculation
   │
   ▼
Threshold Selection
   │
   ▼
Defect Localization
   │
   ▼
Result Visualization
   │
   ▼
Model Export
```

---

# Machine Learning Pipeline
## 1. Environment Setup
The training environment is configured using Google Colab.

The notebook installs and imports the required libraries and prepares the environment for dataset processing and model training.

Typical dependencies include:
* Python
* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 2. Dataset Download
The MVTec AD dataset is downloaded directly from Kaggle.
```text
Google Colab
      │
      ▼
Kaggle Authentication
      │
      ▼
Download Dataset
      │
      ▼
Extract Dataset
      │
      ▼
Load Images
```
The dataset is not included in this repository because of its size.

---

## 3. Dataset Exploration
Before training, the dataset is explored to understand its structure and characteristics.

The exploration process may include:
* Available categories
* Number of training images
* Number of testing images
* Number of normal samples
* Number of anomaly samples
* Available defect types
* Image resolution
* Ground truth masks

Example workflow:
```text
Dataset
   │
   ▼
Category Analysis
   │
   ▼
Image Statistics
   │
   ▼
Sample Visualization
   │
   ▼
Dataset Summary
```

---

## 4. Data Preprocessing
Images are processed before being passed to the Machine Learning model.

The preprocessing pipeline includes:
```text
Input Image
     │
     ▼
Image Loading
     │
     ▼
Resize
     │
     ▼
Normalization
     │
     ▼
Tensor Conversion
     │
     ▼
Model Input
```

Possible preprocessing operations include:
* Image resizing
* Pixel normalization
* Tensor conversion
* Data transformation
* Optional data augmentation

The preprocessing configuration used during inference should match the configuration used during model training.

---

## 5. Model Development
The model is developed to learn the characteristics of normal industrial products.

The general anomaly detection concept is:
```text
Normal Images
      │
      ▼
Feature Learning
      │
      ▼
Deep Learning Model
      │
      ▼
Learn Normal Patterns
      │
      ▼
Trained Model
```

After training, the model analyzes unseen images and generates an anomaly score.

Images with abnormal visual patterns are expected to produce higher anomaly scores.

---

## 6. Model Training
The model training process is performed using Google Colab.

The training notebook contains the complete training pipeline.

```text
Training Images
      │
      ▼
DataLoader
      │
      ▼
Deep Learning Model
      │
      ▼
Forward Pass
      │
      ▼
Loss Calculation
      │
      ▼
Backpropagation
      │
      ▼
Optimizer
      │
      ▼
Updated Model
```

The training process may include:
* Training configuration
* Optimizer configuration
* Learning rate scheduling
* Early stopping
* Model checkpointing
* Training history
* Validation monitoring

---

## 7. Anomaly Detection
After the model is trained, it is used to analyze input images.

The inference workflow is:
```text
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
Trained Model
      │
      ▼
Feature Analysis
      │
      ▼
Anomaly Score
      │
      ▼
Threshold Comparison
      │
      ├───────────────┐
      ▼               ▼
    NORMAL         ANOMALY
```

The anomaly score represents the degree of abnormality detected in an image.

The prediction is determined by comparing the anomaly score with a predefined threshold.

```text
Anomaly Score < Threshold

        │

        ▼

      NORMAL
```

```text
Anomaly Score >= Threshold

        │

        ▼

      ANOMALY
```

---

## 8. Threshold Selection
An anomaly detection model produces a continuous anomaly score.

A threshold is required to convert the anomaly score into a binary prediction.
```text
Anomaly Score
      │
      ▼
Threshold
      │
      ├───────────────┐
      ▼               ▼
Normal            Anomaly
```

Possible threshold selection strategies include:
* Percentile-based threshold
* Mean and standard deviation
* Validation-based threshold
* Youden's J statistic
* F1-score optimization

The selected threshold is saved and reused during web inference.

---

## 9. Defect Localization
In addition to image-level anomaly detection, the system can generate anomaly maps to identify the location of potential defects.

The localization workflow is:
```text
Input Image
      │
      ▼
Trained Model
      │
      ▼
Feature Analysis
      │
      ▼
Anomaly Map
      │
      ▼
Heatmap Generation
      │
      ▼
Defect Localization
```

The anomaly map can be visualized as a heatmap.
```text
Original Image
       +
Anomaly Heatmap
       =
Defect Visualization
```

This provides additional interpretability by showing which image regions contribute to the anomaly prediction.

---

# Model Evaluation
The trained model is evaluated using normal and defective samples from the testing dataset.
```text
Test Dataset
      │
      ▼
Model Prediction
      │
      ▼
Anomaly Score
      │
      ▼
Performance Evaluation
```

## Image-Level Metrics
The following metrics may be used:
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

## Pixel-Level Metrics
For defect localization, the following metrics may be used:
* Pixel ROC-AUC
* Intersection over Union
* Dice Score

---

# Results
The project stores training and evaluation results inside the `results` directory.
```text
results/
│
├── metrics/
│   ├── evaluation_results.json
│   └── model_metrics.csv
│
├── visualizations/
│   ├── training_history.png
│   ├── anomaly_score_distribution.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── anomaly_heatmap.png
│
└── sample_predictions/
    ├── normal/
    └── anomaly/
```

The results may include:
* Training history
* Evaluation metrics
* ROC curves
* Confusion matrices
* Anomaly score distributions
* Anomaly heatmaps
* Sample predictions

---

# Model Storage
The trained Machine Learning model is stored separately from the source code.

The recommended workflow is:
```text
Google Colab
      │
      ▼
Model Training
      │
      ▼
Model Export
      │
      ▼
Model Storage
      │
      ▼
Web Application
      │
      ▼
Model Inference
```

The model may be exported using formats such as:
```text
model.pth
```

or:
```text
model.pt
```

---

## Recommended Model Hosting
The recommended approach is to host the trained model separately from the GitHub repository.

Possible options include:
* Hugging Face
* GitHub Releases
* Google Drive

Recommended architecture:

```text
Google Colab
      │
      ▼
Train Model
      │
      ▼
Export Model
      │
      ▼
Hugging Face
      │
      ▼
Web Application
      │
      ▼
Load Model
      │
      ▼
Inference
```

This approach keeps the repository lightweight and allows the model to be updated independently.

The `models` directory contains information about the trained model and instructions for downloading or loading the model.

```text
models/
└── README.md
```

---

# Web-Based Inference
The project includes a web-based application for performing anomaly detection.

Users can upload an image and receive a prediction generated by the trained Machine Learning model.

## Web Inference Workflow
```text
User
 │
 ▼
Upload Image
 │
 ▼
Image Validation
 │
 ▼
Image Preprocessing
 │
 ▼

Load Trained Model
 │
 ▼

Model Inference
 │
 ▼
Anomaly Score
 │
 ▼

Threshold Comparison
 │
 ├───────────────┐
 ▼               ▼
NORMAL         ANOMALY
```

---

# Web Application Features
## Image Upload
Users can upload images for anomaly detection.
Supported image formats may include:
* JPG
* JPEG
* PNG

---

## Image Preview
The uploaded image is displayed before or after the inference process.

---

## Model Inference
The image is processed using the trained anomaly detection model.

```text
Input Image
      │
      ▼
Trained Model
      │
      ▼
Prediction
```

---

## Prediction Result

The application displays the prediction result.

Example:
```text
Prediction = ANOMALY
```

or:
```text
Prediction = NORMAL
```

---

## Anomaly Score
The application displays the anomaly score generated by the model.

Example:
```text
Anomaly Score = 0.87
```

---

## Anomaly Visualization
The web application can display an anomaly heatmap to visualize the potential defect location.
```text
Original Image
      │
      ▼
Anomaly Map
      │
      ▼
Heatmap
      │
      ▼
Defect Location
```

---

# Project Architecture
```text
                         ┌──────────────────────┐
                         │   MVTec AD Dataset   │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │    Google Colab      │
                         │                      │
                         │ Dataset Preparation  │
                         │ Preprocessing        │
                         │ Model Training       │
                         │ Model Evaluation     │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │    Trained Model     │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │    Model Storage     │
                         │                      │
                         │ Hugging Face         │
                         │ GitHub Releases      │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │   Web Application    │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │     User Upload      │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │   Model Inference    │
                         └───────────┬──────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │  Prediction Result   │
                         │                      │
                         │ Normal / Anomaly     │
                         └──────────────────────┘
```

---

# Project Structure
```text
MVTec-AD/
│
├── notebooks/
│   └── mvtec_ad_training.ipynb
│
├── models/
│   └── README.md
│
├── results/
│   │
│   ├── metrics/
│   │
│   ├── visualizations/
│   │
│   └── sample_predictions/
│
├── web/
│   │
│   ├── app.py
│   │
│   ├── templates/
│   │
│   │   └── index.html
│   │
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   │
│   └── services/
│       ├── model_loader.py
│       ├── preprocessing.py
│       └── inference.py
│
├── requirements.txt
│
└── README.md
```

---

# Repository Components

## `notebooks`
Contains the complete Machine Learning pipeline.
```text
notebooks/
└── mvtec_ad_training.ipynb
```

The notebook includes:
* Environment setup
* Dataset download
* Dataset exploration
* Data preprocessing
* Model development
* Model training
* Model evaluation
* Anomaly detection
* Threshold selection
* Defect localization
* Result visualization
* Model export

---

## `models`
Contains model-related information.
```text
models/
└── README.md
```

The actual trained model can be downloaded from external model storage.

---

## `results`
Contains training and evaluation outputs.
```text
results/
```
Examples include:
* Training metrics
* Evaluation metrics
* Model performance
* ROC curves
* Confusion matrices
* Anomaly heatmaps
* Sample predictions

---

## `web`
Contains the web-based inference application.
```text
web/
```

The web application handles:
* Image upload
* Image validation
* Image preprocessing
* Model loading
* Model inference
* Anomaly score calculation
* Prediction generation
* Result visualization

---

# Technology Stack
## Programming Language
* Python

## Machine Learning
* PyTorch
* Deep Learning
* Computer Vision

## Image Processing
* OpenCV
* NumPy

## Data Processing
* Pandas
* Scikit-learn

## Visualization
* Matplotlib

## Training Environment
* Google Colab

## Dataset
* MVTec AD
* Kaggle

## Model Hosting
Recommended:
* Hugging Face

Alternative:
* GitHub Releases
* Google Drive

## Web Application
Recommended:
* FastAPI
* HTML
* CSS
* JavaScript
---

# Installation
## Clone Repository
```bash
git clone https://github.com/your-username/MVTec-AD.git
```

Navigate to the project directory.
```bash
cd MVTec-AD
```

---

## Create Virtual Environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux or macOS:
```bash
python -m venv venv
source venv/bin/activate
```

---

## Install Dependencies
```bash
pip install -r requirements.txt
```

---

# Run the Web Application
Navigate to the web application directory.
```bash
cd web
```
Run the application.
```bash
python app.py
```

Open the application using a web browser.

The application will provide an interface for uploading images and performing anomaly detection.

---

# Inference Pipeline
The complete inference process is:
```text
User Upload Image
        │
        ▼
Image Validation
        │
        ▼
Image Preprocessing
        │
        ▼
Resize Image
        │
        ▼
Normalize Image
        │
        ▼
Convert to Tensor
        │
        ▼
Load Model
        │
        ▼
Model Inference
        │
        ▼
Calculate Anomaly Score
        │
        ▼
Compare with Threshold
        │
        ├───────────────┐
        ▼               ▼
      NORMAL         ANOMALY
```
---

# Future Improvements
Potential improvements for this project include:
* Support for multiple MVTec AD categories
* Multiple anomaly detection models
* Autoencoder baseline implementation
* PaDiM implementation
* PatchCore implementation
* Model comparison dashboard
* Real-time camera inference
* Advanced defect localization
* REST API
* Prediction history
* User authentication
* Cloud deployment
* Docker containerization
* Model optimization
* CI/CD integration
---

# Reproducibility
The complete Machine Learning pipeline is provided through a single Jupyter Notebook.

```text
notebooks/
└── mvtec_ad_training.ipynb
```
The notebook can be executed using Google Colab.

The workflow automatically covers:
```text
Environment Setup
      │
      ▼
Dataset Download
      │
      ▼
Dataset Preparation
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Model Export
```
This allows users to reproduce the complete model development process.

---

# Project Goal
The main objective of this project is to demonstrate a complete end-to-end Machine Learning workflow for industrial anomaly detection.

The project combines:
```text
Machine Learning
+
Computer Vision
+
Deep Learning
+
Anomaly Detection
+
Model Evaluation
+
Defect Localization
+
Web Deployment
```
The final system allows users to upload industrial product images andased application.

---

# Author
Machine Learning and Computer Vision Portfolio Project
MVTec-AD Industrial Anomaly Detection
