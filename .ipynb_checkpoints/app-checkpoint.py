from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load models
freight_model = joblib.load(MODEL_DIR / "predict_freight_model.pkl")
flag_model = joblib.load(MODEL_DIR / "predict_flag_invoice.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_type = request.form.get("model_type")

        if model_type == "freight":
            df = pd.DataFrame({
                "Quantity": [float(request.form.get("quantity"))],
                "Dollars": [float(request.form.get("dollars"))]
            })

            pred = freight_model.predict(df)[0]
            result = f"₹ {round(pred,2)}"

        else:
            df = pd.DataFrame({
                "invoice_quantity": [float(request.form.get("invoice_quantity"))],
                "invoice_dollars": [float(request.form.get("invoice_dollars"))],
                "Freight": [float(request.form.get("freight"))],
                "total_item_quantity": [float(request.form.get("total_item_quantity"))],
                "total_item_dollars": [float(request.form.get("total_item_dollars"))]
            })

            df_scaled = scaler.transform(df)
            pred = flag_model.predict(df_scaled)[0]

            result = "⚠️ Fraud Detected" if pred == 1 else "✅ Safe Invoice"

        return render_template("index.html", result=result, selected=model_type)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)