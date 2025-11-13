# ml_model.py
import joblib
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class PhishingDetector:
    def __init__(self):
        self.pipeline = None
        self.feature_names = None
        self._load_trained_model()
    
    def _load_trained_model(self):
        """Load the pre-trained model from file"""
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'phishing_model.pkl')
        
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                print(f"✅ Trained model loaded successfully from {model_path}")
                
                # Extract pipeline and feature names from the saved model
                if isinstance(model_data, dict):
                    if 'pipeline' in model_data:
                        self.pipeline = model_data['pipeline']
                    elif 'model' in model_data:
                        self.pipeline = model_data['model']
                    
                    self.feature_names = model_data.get('feature_names', [])
                else:
                    # If it's a direct pipeline/model
                    self.pipeline = model_data
                    self.feature_names = []  # Will need to be set separately
                
                print(f"✅ Model type: {type(self.pipeline)}")
                if self.feature_names:
                    print(f"✅ Feature names: {self.feature_names}")
                    
            except Exception as e:
                print(f"❌ Error loading trained model: {e}")
                self._create_fallback_model()
        else:
            print(f"❌ Model file not found at {model_path}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a fallback model if trained model can't be loaded"""
        print("⚠️ Creating fallback model...")
        
        from sklearn.ensemble import RandomForestClassifier
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        self.feature_names = ['has_https', 'len_hostname', 'len_path', 'count_at', 'count_hyphen', 'count_digits']
        
        # Train on dummy data
        X_dummy = np.array([
            [1, 15, 20, 0, 2, 5],   # Legitimate
            [1, 12, 15, 0, 1, 3],   # Legitimate  
            [0, 25, 5,  1, 5, 10],  # Phishing
            [0, 30, 8,  2, 8, 15],  # Phishing
        ] * 25)
        
        y_dummy = np.array([0, 0, 1, 1] * 25)
        self.pipeline.fit(X_dummy, y_dummy)
        print("✅ Fallback model created")
    
    def predict_phishing(self, feature_vector):
        """Predict if URL is phishing"""
        if self.pipeline is None:
            raise ValueError("Model not loaded")
        
        # Convert to 2D array for prediction
        features_2d = np.array([feature_vector])
        
        try:
            prediction = self.pipeline.predict(features_2d)[0]
            probability = self.pipeline.predict_proba(features_2d)[0]
            
            return {
                'is_phishing': bool(prediction),
                'confidence': float(max(probability)),
                'probability_phishing': float(probability[1]),
                'probability_legitimate': float(probability[0])
            }
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Return a safe default prediction
            return {
                'is_phishing': False,
                'confidence': 0.5,
                'probability_phishing': 0.5,
                'probability_legitimate': 0.5
            }
