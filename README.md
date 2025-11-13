# Phishield – Phishing Detection System

A lightweight and effective phishing detection system built using **Flask**, **Machine Learning**, and **URL feature analysis**. This project checks URLs in real-time and predicts whether they are *legitimate* or *phishing*.

## Features

* Real-time phishing URL detection
* Machine Learning model (Random Forest / Logistic Regression)
* REST API endpoint for URL checking
* Clean and simple frontend
* Easy to deploy (Vercel / Render / Localhost)

## Project Structure

```
Phishield/
├── app.py
├── ml_model.py
├── data_preprocessing.py
├── extract_features.py
├── train_model.py (if present)
├── dataset.csv
├── requirements.txt
├── env.sh
├── api/
│   └── index.py
├── model/
│   └── phishing_model.pkl
├── templates/
│   ├── index.html
│   ├── awareness.html
│   └── statistics.html
└── utils/
    ├── extract_features.py
    └── __pycache__/
```

Phishield/
├── app.py
├── ml_model.py
├── data_preprocessing.py
├── dataset.csv
├── requirements.txt
├── api/
├── model/
├── templates/
│   ├── index.html
│   ├── awareness.html
│   └── statistics.html
├── utils/
│   └── (your utility scripts)
└── env.sh


## Installation

### 1. Clone the repository

```
git clone https://github.com/zhackerloopback/Phishield.git
cd Phishield
```

### 2. Create virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the Flask server

```
python app.py
```

### 5. Access the app

Open in browser:

```
http://127.0.0.1:5000/
```

## API Usage

### POST `/check-url`

**Sample Request:**

```
curl -X POST -H "Content-Type: application/json" \
-d '{"url":"https://example.com"}' \
http://127.0.0.1:5000/check-url
```

**Response:**

```
{
  "url": "https://example.com",
  "prediction": "legitimate",
  "score": 0.87
}
```

## Model Training

Model is trained on:

* Lexical URL features
* Suspicious patterns
* Length-based parameters

Output Model File:

```
Phishing_model.pkl
```

## Deployment Guide

### Deploy on Vercel (Frontend Only)

1. Place `index.html` and `style.css` inside a `public/` folder
2. Create a new Vercel project
3. Set backend API URL in JS (if used)

### Deploy Flask Backend on Render

1. Create a new Web Service
2. Upload code or connect GitHub repo
3. Set build command:

```
pip install -r requirements.txt
```

4. Start command:

```
python app.py
```

## Requirements

```
Flask
scikit-learn
numpy
joblib
```

## Screenshots

(Add your UI screenshots here)

## License

This project is licensed under the MIT License.

---

If you need improvements, deployment setup, or model training help, feel free to ask.
