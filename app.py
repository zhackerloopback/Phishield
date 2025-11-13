# app.py
import os
from flask import Flask, render_template, request, jsonify
from ml_model import PhishingDetector
from data_preprocessing import PhishingDataProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")  # Add this if you have static files
)

# Initialize model and processor
print("🔄 Initializing Phishing Detector...")
detector = PhishingDetector()
processor = PhishingDataProcessor()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/check-url", methods=["POST"])
def check_url():
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        
        if not url:
            return jsonify(success=False, error="URL is required"), 400

        print(f"🔍 Analyzing URL: {url}")
        
        # Extract features
        features = processor.extract_features(url)
        if features is None:
            return jsonify(success=False, error="Failed to extract features from URL"), 400

        # Get feature names from processor or use default order
        if hasattr(processor, 'features') and processor.features:
            feature_keys = processor.features
        else:
            # Use the order from your trained model
            feature_keys = detector.feature_names if detector.feature_names else list(features.keys())

        # Create feature vector in correct order
        try:
            feature_vector = [features[k] for k in feature_keys]
        except KeyError as e:
            return jsonify(
                success=False, 
                error=f"Missing feature: {e}",
                available_features=list(features.keys()),
                required_features=list(feature_keys)
            ), 500

        print(f"📊 Features extracted: {features}")
        print(f"🔢 Feature vector: {feature_vector}")

        # Get prediction
        result = detector.predict_phishing(feature_vector)
        
        print(f"🎯 Prediction result: {result}")

        return jsonify({
            'success': True, 
            'result': result,
            'features': features,
            'url_analyzed': url
        }), 200

    except Exception as e:
        print(f"❌ Error in check-url: {e}")
        return jsonify(success=False, error=f"Server error: {str(e)}"), 500

@app.route("/awareness", methods=["GET"])
def awareness():
    tips = [
        "Check for HTTPS in the URL",
        "Look for spelling mistakes in the domain",
        "Be cautious of urgent action requests",
        "Verify sender email addresses",
        "Don't click on suspicious links",
    ]
    return render_template("awareness.html", tips=tips)

@app.route("/statistics", methods=["GET"])
def statistics():
    stats = {
        "total_checked": 1000,
        "phishing_detected": 230,
        "accuracy_rate": 94.5,
        "common_targets": ["Social Media", "Banking", "Email"],
    }
    return render_template("statistics.html", stats=stats)

if __name__ == "__main__":
    print("🚀 Flask app starting...")
    print("✅ Phishing Detector initialized successfully")
    print("📊 Available features:", detector.feature_names if detector.feature_names else "Using fallback features")
    app.run(host="127.0.0.1", port=5000, debug=True)
