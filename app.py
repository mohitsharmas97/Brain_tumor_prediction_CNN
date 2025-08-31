import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
import torchvision.models as models
import torch.nn as nn
import io
import os


app = Flask(__name__)


# Global variables to hold the model and any loading errors
model = None
model_load_error = None


NUM_CLASSES = 4 #output classes are 4
MODEL_PATH = 'brain_tumor_efficientnet_model (1).pth'

try:

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please make sure it's in the same folder as app.py.")

  
    model = models.efficientnet_b0(weights=None)


    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, NUM_CLASSES)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    model.eval()
    print("Model loaded successfully using torchvision!")

except Exception as e:
    # Store the error message if loading fails
    model_load_error = str(e)
    print(f"!!!!!!!!!!!!!! ERROR LOADING MODEL !!!!!!!!!!!!!!")
    print(f"Error: {model_load_error}")
    print("Failed to load model. The model file might be corrupt or the NUM_CLASSES parameter doesn't match.")
    model = None

# Define the image transformations. These should match the ones used during training.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the class names in the correct order.
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and prediction."""
    # Check if the model failed to load at startup
    if model is None:
        error_msg = f"Model could not be loaded. Please check the server console. Error details: {model_load_error}"
        return jsonify({'error': error_msg}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            # Read the image file from the request
            image_bytes = file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Preprocess the image
            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0)
            
            # Get model prediction
            with torch.no_grad():
                output = model(batch_t)
                _, predicted_idx = torch.max(output, 1)
                prediction = class_names[predicted_idx.item()]
                
            # Return the prediction as JSON
            return jsonify({'prediction': prediction})

        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500

if __name__ == '__main__':
    # Use port 8080 to avoid potential conflicts
    app.run(debug=True, port=8080)

