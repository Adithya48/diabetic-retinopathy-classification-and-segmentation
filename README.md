# Diabetic Retinopathy Classification and Segmentation

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

## Usage  
1. Upload a fundus image.  
2. Select a mode:  
   - **Classification:** Predicts DR severity.  
   - **Segmentation:** Choose MA, HE, or EX for segmentation.  
3. Click **Submit** to view results.  

---  
**Note:** Ensure models are in the `models/` directory.

## Required Packages  
Make sure the following Python packages are installed:  
- Python (>= 3.8)  
- TensorFlow (>= 2.6)  
- NumPy  
- OpenCV  
- Matplotlib  
- Pillow  
- Scikit-learn  

Install these packages using pip:  
```bash  
pip install tensorflow numpy opencv-python matplotlib pillow scikit-learn  
```  

---

## Dataset Structure for Segmentation

This project uses the **Indian Diabetic Retinopathy Image Dataset (IDRiD)** for both classification and segmentation tasks.

🔗 [Access the IDRiD dataset here](http://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
Ensure the datasets for training and testing are organized as follows:  
```  
new_data/  
├── train/  
│   ├── image/        # Contains training images (e.g., .jpg format)  
│   └── mask/         # Contains corresponding masks (e.g., .tif format)  
└── test/  
    ├── image/        # Contains testing images  
    └── mask/         # Contains testing masks  
```  

Each feature (hemorrhages, microaneurysms, hard exudates) uses separate training and testing datasets. Ensure the correct dataset is loaded for each feature.  

