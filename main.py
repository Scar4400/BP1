from data_fetcher import fetch_all_data
from data_processor import process_data
from model import train_and_evaluate
import joblib
import mlflow
import mlflow.sklearn
import streamlit as st
import plotly.express as px
import pandas as pd

def train_model():
    print("Fetching data...")
    raw_data = fetch_all_data()

    print("Processing data...")
    processed_data = process_data(raw_data)

    print("Training and evaluating model...")
    with mlflow.start_run():
        model, metrics = train_and_evaluate(processed_data)

        # Log parameters, metrics, and model
        mlflow.log_params({"model_type": type(model).__name__})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("Model performance:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

    print("Saving model...")
    joblib.dump(model, "football_prediction_model.joblib")

    print("Done! Model saved as 'football_prediction_model.joblib'")

    return model, processed_data

def create_dashboard(model, data):
    st.title("Football Match Prediction Dashboard")

    st.header("Model Performance")
    metrics = model.evaluate(data.drop("result", axis=1), data["result"])
    st.write(f"Accuracy: {metrics['accuracy']:.4f}")
    st.write(f"Precision: {metrics['precision']:.4f}")
    st.write(f"Recall: {metrics['recall']:.4f}")
    st.write(f"F1 Score: {metrics['f1']:.4f}")

    st.header("Feature Importance")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame(sorted(zip(importances, data.columns)), columns=['Value','Feature'])
        fig = px.bar(feature_imp, x='Value', y='Feature', orientation='h', title='Feature Importances')
        st.plotly_chart(fig)
    else:
        st.write("Feature importance not available for this model type.")

    st.header("Prediction Tool")
    # Create input fields for key features
    home_team = st.selectbox("Home Team", data["home_team"].unique())
    away_team = st.selectbox("Away Team", data["away_team"].unique())
    # Add more input fields as needed

    if st.button("Predict"):
        # Prepare input data
        input_data = pd.DataFrame({
            "home_team": [home_team],
            "away_team": [away_team],
            # Add more features
        })

        # Make prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted outcome: {prediction[0]}")

if __name__ == "__main__":
    model, data = train_model()
    create_dashboard(model, data)

