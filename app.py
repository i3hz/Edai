from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
import os
skin_lesion_labels = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
    "mel": "Melanoma"
}

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
skin_model = load_model('skin_model.keras')
# Define allowed file checker
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.args.get('model')  # Get model type from query parameter
    file = request.files.get('file')
    
    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return predict_skin(file_path)
        
    
    return jsonify({'error': 'Invalid file type'})




# Skin cancer prediction function
def predict_skin(file_path):
    processed_image = preprocess_image(file_path, target_size=(28, 28))
    prediction = skin_model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    classes = {
        0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'nv', 5: 'vasc', 6: 'mel'
    }
    class_name = classes.get(predicted_class, 'Unknown')
    
    return jsonify({
        'result': skin_lesion_labels[class_name],
        'image_path': f"uploads/{os.path.basename(file_path)}"
    })



# General preprocessing function
def preprocess_image(file_path, color_mode="rgb", target_size=(28, 28)):
    img = image.load_img(file_path, color_mode=color_mode, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

if __name__ == '__main__':
    app.run(debug=True)