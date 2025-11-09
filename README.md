# Bag Synthetic Images Classification

## Project Overview  
This project focuses on **image classification** to distinguish between three types of synthetic bags: **Plastic Bag**, **Paper Bag**, and **Garbage Bag**.  
The model was developed using a **Convolutional Neural Network (CNN)** built with **TensorFlow** and **Keras**, covering the entire end-to-end workflow—from data preparation and model training to deployment conversion into **TensorFlow.js**, **SavedModel**, and **TensorFlow Lite (TFLite)** formats.  

The final model achieved a **96.73% accuracy** on the test dataset, demonstrating high reliability in classifying synthetic bag images.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Tech Stack](#tech-stack)
* [Repository Structure](#repository-structure)
* [Project Workflow](#project-workflow)
    * [1. Library Import](#1-library-import)
    * [2. Data Preparation](#2-data-preparation)
    * [3. Data Visualization](#3-data-visualization)
    * [4. Model Development](#4-model-development)
    * [5. Model Training](#5-model-training)
    * [6. Model Evaluation](#6-model-evaluation)
    * [7. Model Conversion](#7-model-conversion)
    * [8. Inference](#8-inference)
* [How to Run](#how-to-run)
    * [1. Clone Repository](#1-clone-repository)
    * [2. Install Dependencies](#2-install-dependencies)
    * [3. Run the Notebook](#3-run-the-notebook)

---
## Tech Stack  

**Programming Language**  
- Python 3.x  

**Libraries**  
- TensorFlow, Keras  
- NumPy, Pandas  
- Matplotlib  
- scikit-image, OpenCV, Pillow  
- scikit-learn  
- tqdm  

**Tools & Environment**  
- Google Colab  
- Kaggle API  
- TensorFlow.js Converter  
- TensorFlow Lite  

---

## Repository Structure  
```text
Bag-Synthetic-Images/
│
├── saved_model/ # Model saved in TensorFlow SavedModel format
├── tfjs_model/ # Model converted to TensorFlow.js
├── tflite/ # Model converted to TensorFlow Lite
│ ├── model.tflite
│ └── label.txt
│
├── model.h5 # Keras model in HDF5 format
├── notebook.ipynb # Main notebook containing the workflow
├── requirement.txt # Dependencies for the project
└── README.md # Project documentation
````

---

## Project Workflow  

### 1. Library Import  
Importing standard and third-party Python libraries for data manipulation, image processing, and CNN model development.  

### 2. Data Preparation  
- Download dataset from Kaggle: **plastic-paper-garbage-bag-synthetic-images**.  
- Extract and organize dataset folders.  
- Split the dataset into **train (80%)**, **validation (10%)**, and **test (10%)** subsets.  

### 3. Data Visualization  
- Display class distribution statistics.  
- Show sample images from each dataset split (train, validation, and test).  

### 4. Model Development  
- Apply data augmentation with transformations such as `RandomFlip`, `RandomRotation`, `RandomZoom`, and `RandomContrast`.  
- Build a CNN with multiple convolutional, pooling, and dropout layers to prevent overfitting.  
- Use **Adam optimizer** and **categorical crossentropy** as the loss function.  

### 5. Model Training  
- Train the model with augmented data.  
- Implement a custom callback, **EarlyStoppingAtAccuracy**, to automatically stop training once both training and validation accuracy reach ≥ 96%.  

### 6. Model Evaluation  
- Evaluate model performance on the test set with **96.73% accuracy**.  
- Visualize accuracy and loss curves to monitor model learning progress.  

### 7. Model Conversion  
The trained model is converted into three formats for different deployment environments:
- **SavedModel** – for TensorFlow serving.  
- **TensorFlow.js** – for web-based applications.  
- **TensorFlow Lite (TFLite)** – for mobile or embedded systems.  

### 8. Inference  
- Use the TFLite model for image prediction.  
- Uploaded images are preprocessed and displayed with their predicted label (e.g., *Prediction: Paper Bag Images*).  

---

## How to Run  

### 1. Clone Repository  
```bash
git clone https://github.com/echaan/Bag-Synthetic-Images.git
cd Bag-Synthetic-Images
````
### 2. Install Dependencies
```bash
pip install -r requirements.txt
````
### 3. Run the Notebook
Make sure your kaggle.json file is available. Then open and execute the notebook. The notebook already includes:
- Dataset download and extraction from Kaggle
- Data preprocessing and splitting
- Model training, evaluation, and conversion (SavedModel, TensorFlow.js, TFLite)
- Example inference using the trained model

