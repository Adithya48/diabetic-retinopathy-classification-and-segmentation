# 👁 Diabetic Retinopathy Classification and Segmentation

This project aims to create a web-based application for diabetic retinopathy detection and segmentation, leveraging deep learning models to classify the severity of diabetic retinopathy and segment specific features such as Microaneurysms, Hemorrhages, and Hard Exudates, facilitating early diagnosis and better treatment planning.

App.py(Frontend):
# Diabetic Retinopathy Detection and Segmentation  

## Overview  
This web application detects diabetic retinopathy (DR) severity and segments key retinal features. Users can upload fundus images and choose:  
1. **Classification:** Predict DR severity (No DR, Mild, Moderate, Severe, Proliferative).  
2. **Segmentation:** Segment specific features—Microaneurysms (MA), Hemorrhages (HE), or Hard Exudates (EX).  

## Features  
- **Classification:** Uses a Keras classification model (`diabetic_retinopathy_model.h5`) to predict DR severity.  
- **Segmentation:** Utilizes three PyTorch U-Net models (`checkpoint(MA).pth`, `checkpoint(HE).pth`, `checkpoint(EX).pth`) for feature segmentation.  

## How It Works  
1. **Frontend:** Simple HTML interface for image upload and mode selection.  
2. **Backend:** Flask handles requests, processes images, and integrates models dynamically.  

## Directory Structure  
```  
static/  
├── uploads/       # Uploaded images  
└── results/       # Segmentation outputs  
models/  
├── diabetic_retinopathy_model.h5  
├── checkpoint(MA).pth  
├── checkpoint(HE).pth  
└── checkpoint(EX).pth  
templates/  
└── index.html       # Frontend HTML  
```  

## Prerequisites  
1. **Python >= 3.8**  
2. Install required libraries:  
   ```bash  
   pip install flask torch tensorflow keras opencv-python numpy pillow  
   ```  

## How to Run  
1. **Clone the repository and navigate to the folder:**  
   ```bash  
   git clone <repository_url>  
   cd <repository_folder>  
   ```  
2. **Run the Flask server:**  
   ```bash  
   python app.py  
   ```  
3. **Open the app in a browser:**  
   ```
   http://127.0.0.1:5000  
   ```  


