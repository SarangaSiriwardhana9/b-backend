import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Dictionary for Sinhala day names
SINHALA_DAY_NAMES = {
    "Monday": "සඳුදා",
    "Tuesday": "අඟහරුවාදා",
    "Wednesday": "බදාදා",
    "Thursday": "බ්‍රහස්පතින්දා",
    "Friday": "සිකුරාදා",
    "Saturday": "සෙනසුරාදා",
    "Sunday": "ඉරිදා"
}

# Default temperature values for Sri Lanka if not provided
DEFAULT_MIN_TEMP = 24.0
DEFAULT_MAX_TEMP = 32.0

def load_protection_model():
    """Load the trained betel protection model"""
    try:
        return joblib.load('models/betel_protection_model.pkl')
    except:
        print("Could not load protection model, using rule-based fallback")
        return None

def get_feature_names():
    """Get the feature names used in the model"""
    try:
        return joblib.load('models/protection_feature_names.pkl')
    except:
        # Default feature names if file doesn't exist
        return ['Rainfall (mm)', 'Min Temp (°C)', 'Max Temp (°C)', 
                'Location_KURUNEGALA', 'Location_PUTTALAM']

def get_protection_methods(protection_type, max_temp, min_temp, rainfall):
    """
    Determine the best protection methods based on protection type and weather conditions
    Also returns Sinhala translations for recommendations
    """
    
    if protection_type == 0:
        return {
            "english": ["No special protection needed - maintain regular care"],
            "sinhala": ["විශේෂ ආරක්ෂණ විධිවිධාන අවශ්‍ය නැත - සාමාන්‍ය රැකවරණය පවත්වා ගන්න"]
        }
    
    # Drought protection methods (Type 1)
    if protection_type == 1:
        english_methods = []
        sinhala_methods = []
        
        # High temperature, low rainfall (drought conditions)
        english_methods.append("MU - Mulching: Spread organic mulch around plants to retain soil moisture")
        sinhala_methods.append("පෙර යොදන ලද කාබනික වසුන් මේ වනවිට සම්පූර්න වශයෙන් දිරාපත් වී ඇත්නම් ,පසෙහි තෙතමනය රඳවා ගැනීම සඳහා ශාක වටා කාබනික වසුන් යොදන්න.")
        
        if max_temp >= 34:
            english_methods.append("SH - Shading: Use shade nets during peak sun hours (10am-3pm)")
            sinhala_methods.append("දැඩි හිරු එළිය ඇති වේලාවන්හිදී (පෙ.ව. 10 සිට ප.ව. 3 දක්වා) සෙවන දැල් භාවිතා කරන්න.")
        
        english_methods.append("FW - Frequent Watering: Water plants early morning or late evening")
        sinhala_methods.append(" අප ඔබට සදහන් කරන ජල පරිමාව පැළෑටි වලට උදෑසන හා සවස් යාමයේ  දෙවරකට දමන්න")
            
        return {
            "english": english_methods,
            "sinhala": sinhala_methods
        }
    
    # Excess rain protection methods (Type 2)
    if protection_type == 2:
        english_methods = []
        sinhala_methods = []
        
        if rainfall >= 30:
            english_methods.append("DS - Drainage System: Create channels to divert excess water")
            sinhala_methods.append("යම් තද වර්ශාපතනයක් අපේක්ශා කිරීම හේතුවෙන් පාත්ති තුල ජලය රැස් වීම වැලැක්වීමට පැල වලට හානි නොවන සේ තාවකාලිකව වැලිකොට්ට යොදන්න")
            
            english_methods.append("CP - Cover Plants: Use polythene sheets as temporary rain protection")
            sinhala_methods.append("යම් තද වර්ශාපතනයක් ඇති විය හැකි බැවින් කුඩා පැල සදහා තාවකාලිකව වැසි ආරක්ෂාව සඳහා තාවකාලිකව පොලිතීන් ෂීට් හෝ වැස්මක් භාවිතා කරන්න")
        else:
            english_methods.append("DS - Drainage System: Ensure proper drainage around plants")
            sinhala_methods.append("සැලකිය යුතු වර්ශාපතනයක් ඇතිවිය හැකි බැවින්පාත්ති  වටා සාදා ඇති කාණු නැවත පෑදීමක් සිදුකිරීමෙන් ක්‍රමවත්ව ජලය බැස යෑමට ඉඩ හරින්න")
        
        english_methods.append("SP - Stake Plants: Tie plants to stakes to prevent damage from wind and water")
        sinhala_methods.append(" වර්ශාව සමග යම් සුළඟ තත්වයක් ඇතිවීමට ඉඩ ඇති හෙයින් සුළඟ සහ ජලයෙන් හානි වීම වැළැක්වීමට පැළ කෝටු වලට හානි නොවන සේ සිහින්ව බැඳ තබන්න")
        
        if rainfall >= 25:
            english_methods.append("PI - Pest Inspection: Check for increased pest activity after rain")
            sinhala_methods.append(" වැස්සෙන් පසු  අනිවාර්‍යයෙන්ම පළිබෝධ ක්‍රියාකාරකම් පිලිබදව පරීක්ෂා කරන්න")
            
        return {
            "english": english_methods,
            "sinhala": sinhala_methods
        }
    
    return {
        "english": ["Standard care: Monitor plants regularly"],
        "sinhala": ["සාමාන්‍ය රැකවරණය ලබාදෙන්න ,පැල පිලිබද අවදානයෙන් සිටින්න"]
    }

def rule_based_protection(rainfall, min_temp, max_temp):
    """Simple rule-based model to determine protection type based on weather conditions"""
    
    # Define thresholds based on betel farming best practices
    HIGH_TEMP_THRESHOLD = 33.0  # High temperature threshold in Celsius
    LOW_RAIN_THRESHOLD = 2.0    # Low rainfall threshold in mm
    HIGH_RAIN_THRESHOLD = 20.0  # High rainfall threshold in mm
    
    # Excess rain conditions (highest priority)
    if rainfall >= HIGH_RAIN_THRESHOLD:
        protection_type = 2  # Excess Rain Protection
        confidence = min(100, rainfall * 2)  # Higher rainfall = higher confidence
    
    # Drought conditions
    elif max_temp >= HIGH_TEMP_THRESHOLD and rainfall <= LOW_RAIN_THRESHOLD:
        protection_type = 1  # Drought Protection
        confidence = min(100, (max_temp - HIGH_TEMP_THRESHOLD) * 20)  # Higher temp = higher confidence
    
    # Normal conditions
    else:
        protection_type = 0  # No Protection Needed
        confidence = 80  # Default confidence
    
    # Create probability distribution
    if protection_type == 0:
        probabilities = [confidence/100, (100-confidence)/200, (100-confidence)/200]
    elif protection_type == 1:
        probabilities = [(100-confidence)/200, confidence/100, (100-confidence)/200]
    else:  # protection_type == 2
        probabilities = [(100-confidence)/200, (100-confidence)/200, confidence/100]
    
    return protection_type, probabilities

def predict_7day_protection_needs(location, rainfall_forecast, min_temp_forecast=None, max_temp_forecast=None):
    """
    Predict protection needs for a 7-day forecast
    
    Args:
        location (str): Location/district (PUTTALAM or KURUNEGALA)
        rainfall_forecast (list): 7-day rainfall forecast in mm
        min_temp_forecast (list, optional): 7-day minimum temperature forecast in Celsius
        max_temp_forecast (list, optional): 7-day maximum temperature forecast in Celsius
    
    Returns:
        list: List of dictionaries with recommendations for each day
    """
    try:
        # Try to load the model, use rule-based approach if not available
        model = load_protection_model()
        feature_names = get_feature_names()
        use_model = model is not None
        
        # Create default temperature forecasts if not provided
        if min_temp_forecast is None:
            min_temp_forecast = [DEFAULT_MIN_TEMP] * len(rainfall_forecast)
        if max_temp_forecast is None:
            max_temp_forecast = [DEFAULT_MAX_TEMP] * len(rainfall_forecast)
        
        # Ensure all forecasts are the same length
        forecast_length = min(len(rainfall_forecast), len(min_temp_forecast), len(max_temp_forecast))
        rainfall_forecast = rainfall_forecast[:forecast_length]
        min_temp_forecast = min_temp_forecast[:forecast_length]
        max_temp_forecast = max_temp_forecast[:forecast_length]
        
        # Create 7-day forecast dataframe
        days = []
        
        # For each day in the forecast
        for day_idx in range(forecast_length):
            today = datetime.now() + timedelta(days=day_idx)
            rainfall = rainfall_forecast[day_idx]
            min_temp = min_temp_forecast[day_idx]
            max_temp = max_temp_forecast[day_idx]
            
            # Create feature row
            features = {
                'Rainfall (mm)': rainfall,
                'Min Temp (°C)': min_temp,
                'Max Temp (°C)': max_temp,
                'Location_KURUNEGALA': 0,
                'Location_PUTTALAM': 0
            }
            
            # Set the correct location column to 1
            location_col = f'Location_{location}'
            if location_col in features:
                features[location_col] = 1
            
            # Add day information
            day_info = {
                'date': today.strftime('%Y-%m-%d'),
                'day_name': today.strftime('%A'),
                'day_name_sinhala': SINHALA_DAY_NAMES.get(today.strftime('%A'), today.strftime('%A')),
                'rainfall': rainfall,
                'min_temp': min_temp,
                'max_temp': max_temp
            }
            
            # Combine all info
            days.append({**day_info, **features})
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(days)
        
        # Predict using model or rule-based approach
        if use_model:
            # Ensure all required columns are present
            for col in feature_names:
                if col not in forecast_df.columns:
                    forecast_df[col] = 0
            
            # Reorder columns to match training data
            input_features = forecast_df[feature_names]
            
            # Make predictions
            predictions = model.predict(input_features)
            probabilities = model.predict_proba(input_features)
            
            # Add predictions to forecast
            forecast_df['protection_type'] = predictions
            
            # Add confidence scores for each class
            for i, class_name in enumerate(['no_protection_confidence', 
                                         'drought_protection_confidence', 
                                         'excess_rain_protection_confidence']):
                forecast_df[class_name] = [prob[i] * 100 for prob in probabilities]
        else:
            # Rule-based approach
            rule_results = [rule_based_protection(r, mt, xt) 
                           for r, mt, xt in zip(rainfall_forecast, min_temp_forecast, max_temp_forecast)]
            
            forecast_df['protection_type'] = [result[0] for result in rule_results]
            
            # Create probability arrays
            probs = [result[1] for result in rule_results]
            forecast_df['no_protection_confidence'] = [p[0] * 100 for p in probs]
            forecast_df['drought_protection_confidence'] = [p[1] * 100 for p in probs]
            forecast_df['excess_rain_protection_confidence'] = [p[2] * 100 for p in probs]
        
        # Map protection types to descriptive labels
        protection_type_map = {
            0: 'No Protection Needed',
            1: 'Drought Protection',
            2: 'Excess Rain Protection'
        }
        
        protection_type_map_sinhala = {
            0: 'විශේෂ ආරක්ෂණයක් උපක්‍රමයක් අවශ්‍ය නැත',
            1: 'නියඟ ආරක්ෂණය',
            2: 'අධික වැසි ආරක්ෂණය'
        }
        
        forecast_df['protection_label'] = forecast_df['protection_type'].map(protection_type_map)
        forecast_df['protection_label_sinhala'] = forecast_df['protection_type'].map(protection_type_map_sinhala)
        
        # Add protection methods based on type and weather conditions
        forecast_df['protection_methods'] = forecast_df.apply(
            lambda row: get_protection_methods(
                row['protection_type'], row['max_temp'], row['min_temp'], row['rainfall']
            ),
            axis=1
        )
        
        # Convert to list of dictionaries
        results = []
        for _, row in forecast_df.iterrows():
            results.append({
                'date': row['date'],
                'day_name': row['day_name'],
                'day_name_sinhala': row['day_name_sinhala'],
                'rainfall': float(row['rainfall']),
                'min_temp': float(row['min_temp']),
                'max_temp': float(row['max_temp']),
                'protection_type': int(row['protection_type']),
                'protection_label': row['protection_label'],
                'protection_label_sinhala': row['protection_label_sinhala'],
                'no_protection_confidence': float(row['no_protection_confidence']),
                'drought_protection_confidence': float(row['drought_protection_confidence']),
                'excess_rain_protection_confidence': float(row['excess_rain_protection_confidence']),
                'protection_methods_english': row['protection_methods']['english'],
                'protection_methods_sinhala': row['protection_methods']['sinhala']
            })
        
        # Find the day with highest confidence for its protection type
        best_days = {}
        for ptype in [0, 1, 2]:
            type_days = [day for day in results if day['protection_type'] == ptype]
            if type_days:
                if ptype == 0:
                    confidence_key = 'no_protection_confidence'
                elif ptype == 1:
                    confidence_key = 'drought_protection_confidence'
                else:
                    confidence_key = 'excess_rain_protection_confidence'
                
                best_day = max(type_days, key=lambda x: x[confidence_key])
                best_days[ptype] = best_day
        
        # Mark best days in results
        for day in results:
            ptype = day['protection_type']
            if ptype in best_days and day['date'] == best_days[ptype]['date']:
                day['is_best_day'] = True
            else:
                day['is_best_day'] = False
        
        # Filter out any unwanted protection methods that might come from the model
        for day in results:
            if 'protection_methods_english' in day:
                day['protection_methods_english'] = [
                    method for method in day['protection_methods_english'] 
                    if not method.startswith("SE - Soil Enrichment") and not method.startswith("AF - Avoid Fertilizing")
                ]
            if 'protection_methods_sinhala' in day:
                day['protection_methods_sinhala'] = [
                    method for method in day['protection_methods_sinhala'] 
                    if not method.startswith("SE - ") and not method.startswith("AF - ")
                ]
        
        return results
    
    except Exception as e:
        print(f"Error predicting protection needs: {str(e)}")
        # Return error message
        return [{'error': str(e)}]

def predict_today_protection_needs(location, rainfall, min_temp=None, max_temp=None):
    """
    Predict protection needs for today based on weather data.
    
    Args:
        location (str): Location/district (PUTTALAM or KURUNEGALA)
        rainfall (float): Today rainfall in mm
        min_temp (float, optional): Today minimum temperature in Celsius
        max_temp (float, optional): Today maximum temperature in Celsius
    
    Returns:
        dict: Recommendation for today
    """
    today = datetime.now()
    
    try:
        # Use default temperatures if not provided
        if min_temp is None:
            min_temp = DEFAULT_MIN_TEMP
        if max_temp is None:
            max_temp = DEFAULT_MAX_TEMP
        
        # Simply call the 7-day function with just today data
        results = predict_7day_protection_needs(
            location, 
            [rainfall],
            [min_temp],
            [max_temp]
        )
        
        if results and len(results) > 0:
            if 'error' in results[0]:
                return results[0]
                
            return results[0]
        else:
            raise Exception("No prediction results returned")
        
    except Exception as e:
        print(f"Error predicting today protection needs: {str(e)}")
        return {
            'date': today.strftime('%Y-%m-%d'),
            'day_name': today.strftime('%A'),
            'day_name_sinhala': SINHALA_DAY_NAMES.get(today.strftime('%A'), today.strftime('%A')),
            'rainfall': float(rainfall),
            'min_temp': float(min_temp if min_temp is not None else DEFAULT_MIN_TEMP),
            'max_temp': float(max_temp if max_temp is not None else DEFAULT_MAX_TEMP),
            'error': str(e)
        }

# Helper function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj