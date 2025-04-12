from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
import io
from model import build_unet  
app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load classification model
classification_model = tf.keras.models.load_model("models/diabetic_retinopathy_model.h5")  

# Segmentation model paths
model_paths = {
    'MA': "models/checkpoint(MA).pth",
    'HE': "models/checkpoint(HE).pth",
    'EX': "models/checkpoint(EX).pth"
}

# Initialize device for segmentation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load segmentation model based on user choice
def load_segmentation_model(model_choice):
    model = build_unet()
    model.load_state_dict(torch.load(model_paths[model_choice], map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to preprocess input image for segmentation
def preprocess_image_segmentation(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read image
    image = cv2.resize(image, (512, 340))  # Resize to match the model input size (H=340, W=512)
    image = image / 255.0  # Normalize to range [0, 1]
    image = np.transpose(image, (2, 0, 1))  
    image = np.expand_dims(image, axis=0).astype(np.float32)  
    return torch.from_numpy(image).to(device)

# Function to save segmentation prediction
def save_segmentation_prediction(image_path, output_path, model_choice):
    model = load_segmentation_model(model_choice)  # Load the chosen model
    image_tensor = preprocess_image_segmentation(image_path)
    
    with torch.no_grad():
        pred = model(image_tensor)
        pred = torch.sigmoid(pred)  # Apply sigmoid activation
        pred = pred[0].cpu().numpy().squeeze() > 0.5  # Threshold at 0.5
        pred = (pred * 255).astype(np.uint8)  # Convert to uint8 image
        cv2.imwrite(output_path, pred)

# Function to preprocess image for classification
def preprocess_image_classification(img_bytes):
    # Convert bytes to a PIL image with the correct target size
    img = image.load_img(io.BytesIO(img_bytes), target_size=(512, 512))  
    
    # Convert the image to an array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize if required
    
    return img_array

# Mapping for classification results
dr_stages = [
    'No Diabetic Retinopathy',
    'Mild Diabetic Retinopathy',
    'Moderate Diabetic Retinopathy', 
    'Severe Diabetic Retinopathy',
    'Proliferative Diabetic Retinopathy'
]
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    segmented_image = None
    classification_result = None

    if request.method == "POST":
       
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        # Get detection mode and model choice
        detection_mode = request.form.get("detection_mode", "classification")
        model_choice = request.form.get("model_choice", "MA")

        if file:
            # Save the uploaded image
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Prepare the relative file path for display
            uploaded_image = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            if detection_mode == "segmentation":
                # Perform segmentation
                result_path = os.path.join(app.config["RESULT_FOLDER"], f"segmented_{file.filename}")
                save_segmentation_prediction(file_path, result_path, model_choice)
                segmented_image = os.path.join(app.config['RESULT_FOLDER'], f"segmented_{file.filename}")
            
            else:  # classification mode
                file.seek(0)  
                img_bytes = file.read()
                img_array = preprocess_image_classification(img_bytes)

                # Make predictions
                predictions = classification_model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                classification_result = dr_stages[predicted_class]

            return render_template(
                "index.html", 
                uploaded_image=uploaded_image, 
                segmented_image=segmented_image, 
                classification_result=classification_result
            )

    return render_template("index.html", uploaded_image=None, segmented_image=None, classification_result=None)

if _name_ == "_main_":
    app.run(debug=True)