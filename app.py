# app.py

import os
from flask import Flask, render_template, request, jsonify
from ml_model import PhishingDetector
from data_preprocessing import PhishingDataProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If you're deploying to Vercel, set template_folder as done here.
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=None
)

# Initialize model and processor (they should handle lazy init if expensive)
detector = PhishingDetector()
processor = PhishingDataProcessor()
#####
def load_model_flexibly(path: str):
    if not os.path.exists(path):
        return False, "Model file does not exist."

    try:
        payload = joblib.load(path)
    except Exception as e:
        return False, f"joblib.load failed: {e}"

    # Debug prints (will show in Flask terminal)
    print("DEBUG: loaded payload type:", type(payload))
    try:
        import sklearn.base
        print("DEBUG: is BaseEstimator?", isinstance(payload, sklearn.base.BaseEstimator))
    except Exception:
        pass
    # If it's a dict with pipeline key (our preferred format)
    if isinstance(payload, dict):
        print("DEBUG: payload keys:", list(payload.keys())[:10])
        if "pipeline" in payload:
            detector.pipeline = payload["pipeline"]
            detector.feature_names = payload.get("feature_names", detector.feature_names)
            print("DEBUG: loaded dict->pipeline, feature_names:", detector.feature_names)
            return True, "Loaded payload dict with pipeline."
        # if dict contains estimator under another key (try common ones)
        for k in ("model","estimator","clf","classifier"):
            if k in payload:
                maybe = payload[k]
                try:
                    import sklearn.base
                    if isinstance(maybe, sklearn.base.BaseEstimator):
                        from sklearn.pipeline import Pipeline
                        from sklearn.preprocessing import StandardScaler
                        detector.pipeline = Pipeline([("scaler", StandardScaler()), ("clf", maybe)])
                        detector.feature_names = detector.feature_names or ["has_https","len_hostname","len_path","count_at","count_hyphen","count_digits"]
                        print("DEBUG: wrapped dict['%s'] estimator into pipeline" % k)
                        return True, f"Loaded estimator from payload['{k}']"
                except Exception:
                    pass
    # If payload is a raw estimator (sklearn)
    try:
        import sklearn.base
        if isinstance(payload, sklearn.base.BaseEstimator):
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            detector.pipeline = Pipeline([("scaler", StandardScaler()), ("clf", payload)])
            detector.feature_names = detector.feature_names or ["has_https","len_hostname","len_path","count_at","count_hyphen","count_digits"]
            print("DEBUG: wrapped raw estimator into pipeline")
            return True, "Loaded raw estimator and wrapped into pipeline."
    except Exception as e:
        print("DEBUG: sklearn check failed:", e)

    return False, "Unknown model payload format."
######

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/check-url", methods=["POST"])
def check_url():
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify(success=False, error="URL missing"), 400

        # Extract features dict and build vector in a stable order.
        # Ensure processor.features is defined and stable.
        features = processor.extract_features(url)
        if features is None:
            return jsonify(success=False, error="Failed to extract features"), 400

        # If processor.features is not present, fallback to keys order
        if not hasattr(processor, "features") or not processor.features:
            feature_keys = list(features.keys())
        else:
            feature_keys = processor.features

        try:
            feature_vector = [features[k] for k in feature_keys]
        except KeyError:
            # If expected keys missing, return helpful error
            return jsonify(
                success=False,
                error="Feature mismatch between processor.features and extracted features",
                expected=list(feature_keys),
                got=list(features.keys())
            ), 500

        # Model prediction — detectors should accept a vector or adapt accordingly
        result = detector.predict_phishing(feature_vector)

        return jsonify(success=True, result=result, features=features), 200

    except Exception as e:
        # Return helpful error for debugging (frontend network tab)
        return jsonify(success=False, error=f"{type(e).__name__}: {e}"), 500

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
    # Optional: train or load model on startup if your detector needs it.
    # detector.train_model()  # uncomment only if you want to train at startup
    app.run(host="127.0.0.1", port=5000, debug=True)

###########################

