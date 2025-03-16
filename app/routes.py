from app.kavindi import DemandPredictionInput, PricePredictionInput, predict_demand_location, predict_market_demand, predict_price
from flask import Blueprint, request, jsonify, render_template
from app.model_manager import predict_yield
from app.preprocess import preprocess_input_data
from app.utils import handle_error
import joblib
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Import your additional blueprints
from app.watering_routes import watering_bp
from app.fertilizing_routes import fertilizing_bp
from app.protection_routes import protection_bp

api_bp = Blueprint('api', __name__)

# Register your additional blueprints with the main api_bp
api_bp.register_blueprint(watering_bp, url_prefix='/watering')
api_bp.register_blueprint(fertilizing_bp, url_prefix='/fertilizing')
api_bp.register_blueprint(protection_bp, url_prefix='/protection')

def load_model(model_name):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_name)
    return joblib.load(model_path)

def round_to_nearest_50(value):
    return round(value / 50) * 50

@api_bp.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Welcome to the API!'})


@api_bp.route('/predict/harvest', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'Invalid or missing JSON input'}), 400

        print(f"Received Input Data: {input_data}")

        new_data = pd.DataFrame(input_data, index=[0])
        processed_data = preprocess_input_data(new_data)
        print(f"Processed Data: {processed_data}")

        # ✅ Load models on demand instead of at startup
        model_p = load_model('model_P.pkl')
        model_kt = load_model('model_KT.pkl')
        model_rkt = load_model('model_RKT.pkl')

        prediction_p = predict_yield(model_p, processed_data)
        prediction_kt = predict_yield(model_kt, processed_data)
        prediction_rkt = predict_yield(model_rkt, processed_data)

        return jsonify({
            'P': round_to_nearest_50(prediction_p[0]),
            'KT': round_to_nearest_50(prediction_kt[0]),
            'RKT': round_to_nearest_50(prediction_rkt[0])
        })
        
    except Exception as e:
        print("Error:", str(e))

        return jsonify({'error': str(e)}), 500
    

#! Kavindi Routes starts here ===============>
# Endpoint: Predict price per leaf
@api_bp.post("/market/predict-price")
def predict_price_endpoint():
    data = request.get_json()  # Extract JSON data from the request

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    input_data = PricePredictionInput(**data)  # Convert JSON into an object

    print('Price prediction triggered')

    return jsonify({
        "price": predict_price(
            date=input_data.Date,
            leaf_type=input_data.Leaf_Type,
            leaf_size=input_data.Leaf_Size,
            quality_grade=input_data.Quality_Grade,
            no_of_leaves=input_data.No_of_Leaves,
            location=input_data.Location,
            season=input_data.Season
        )
    })


# Endpoint: Predict highest demand location
@api_bp.post("/market/predict-location")
def predict_location_endpoint():

    data = request.get_json()  # Extract JSON data from the request

    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    input_data = DemandPredictionInput(**data)  # Convert JSON into an object
    
    return jsonify({
        "location": predict_demand_location(
            date=input_data.Date,
            no_of_leaves=input_data.No_of_Leaves,
            leaf_type=input_data.Leaf_Type,
            leaf_size=input_data.Leaf_Size,
            quality_grade=input_data.Quality_Grade
        )
    })

@api_bp.post("/market/predict-market-demand")
def predict_market_demand_endpoint():
    try:
       return predict_market_demand()
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500


# Health check endpoint
@api_bp.get("/health")
def health_check():
    return {"status": "API is up and running!"}

# Kavindi Routes ends here ==============================================>


#! Lasiya Routes starts here ===============>

# Lasiya Routes ends here ==============================================>

#!Umesh Routes starts here ===============>

# Configuration for storing uploaded images
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the disease prediction model
DISEASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', '../models/Model_best.h5')
disease_model = None

# Define class names for disease prediction
disease_class_names = [
    'Bacterial leaf blight',
    'Brown Spots',
    'Firefly disease',
    'Healthy',
    'Red spider mite'
]

def load_disease_model():
    global disease_model
    if disease_model is None:
        try:
            disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
            print("Disease prediction model loaded successfully")
        except Exception as e:
            print(f"Error loading disease model: {e}")
    return disease_model

# Simplified inference function for disease prediction
def predict_disease_from_image(model, img):
    # Convert image to array and expand dimensions
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Perform model prediction
    predictions = model.predict(img_array)

    # Process predictions
    predicted_class = disease_class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    confidence = float(confidence)
    return predicted_class, confidence

# Flask route for disease prediction
@api_bp.route('/predict/disease', methods=['POST'])
def predict_disease_endpoint():
    try:
        # Check if an image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        print(f"Received file: {file.filename}")

        # Save the file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Load and preprocess the image
        img = Image.open(file_path).convert("RGB")  # Ensure RGB format
        img = img.resize((256, 256))  # Resize to model input size

        # Load the model and perform prediction
        model = load_disease_model()
        predicted_class, confidence = predict_disease_from_image(model, img)

        # Optionally delete the file after processing
        os.remove(file_path)

        # Return prediction as JSON
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error during disease prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Umesh Routes ends here ==============================================>