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

Segmentation uses the **Indian Diabetic Retinopathy Image Dataset (IDRiD)**.

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


---
## Dataset Structure for Classification

Classification uses the **APTOS 2019 Blindness Detection Dataset** .

🔗 [Access the APTOS dataset here](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

Organize the dataset as follows:
```python
dataset/  
├── train/  
│   ├── 0/            # Contains training images for No DR  
│   ├── 1/            # Contains training images for Mild DR  
│   ├── 2/            # Contains training images for Moderate DR  
│   ├── 3/            # Contains training images for Severe DR  
│   └── 4/            # Contains training images for Proliferative DR  
└── val/ 
```
Ensure the images are preprocessed and labeled correctly.

---

## Configure Training Parameters
1. Open the classification script train_classifier.py.
2. Update the dataset paths:

```python
train_dir = "/path_to_dataset/train"
val_dir = "/path_to_dataset/val"
```

3. Adjust hyperparameters (e.g., learning rate, batch size, epochs) as needed

```python
batch_size = 16
epochs = 50
```

## Output Visualization  
Each testing script generates visual outputs of the segmentation, including:  
- Input images.  
- Predicted segmented masks for the selected feature.  

The results are stored in the `results/` directory and can be used for analysis or presentation.  

---
