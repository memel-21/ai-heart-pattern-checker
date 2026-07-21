# AI Heart Pattern Checker

AI Heart Pattern Checker is a Streamlit web application that analyses basic vital signs and estimates whether the entered heart pattern appears normal or unusual. The app uses a trained autoencoder model to compare user input with healthy physiological patterns learned during training.

## Project Overview

This project was developed as a final year project prototype for AI-assisted cardiovascular pattern checking. Users enter vital sign readings, and the system calculates additional cardiovascular indicators before running the AI model.

The application is intended for research and educational purposes only. It does not provide medical diagnosis and should not replace advice from a qualified healthcare professional.

## Features

- Interactive Streamlit interface for entering vital signs
- Heart rate, body temperature, systolic blood pressure, and diastolic blood pressure inputs
- Automatic calculation of Mean Arterial Pressure (MAP)
- Automatic calculation of Pulse Pressure
- Health range overview for each measurement
- AI-based anomaly detection using an autoencoder model
- Results dashboard with risk score, recommendation, and advanced analysis details
- Tooltip explanations for medical terms and output sections

## Files Included

- `app.py` - Main Streamlit application
- `final_champion_ae.keras` - Trained autoencoder model
- `unified_minmax_scaler.pkl` - Scaler used to normalize input features
- `ae_threshold.json` - Reconstruction error threshold for anomaly detection
- `requirements.txt` - Python dependencies needed to run the app

## Input Features

The model uses the following features:

- Heart Rate
- Temperature
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Mean Arterial Pressure
- Pulse Pressure

## How To Run Locally

1. Clone this repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal.

## Deployment

This repository is ready for Streamlit Community Cloud deployment.

Use these settings:

- Repository: `memel-21/ai-heart-pattern-checker`
- Branch: `main`
- Main file path: `app.py`

## Important Disclaimer

This application is a research prototype. The output should be interpreted carefully and must not be used as a medical diagnosis. If you experience symptoms such as chest pain, shortness of breath, dizziness, or other concerning signs, seek medical attention immediately.
