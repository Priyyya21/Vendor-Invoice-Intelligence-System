import joblib 
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)

    # 🔥 FIX: ensure correct columns (Quantity + Dollars)
    input_df = input_df.reindex(columns=["Quantity", "Dollars"], fill_value=0)

    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df


if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {
        "Quantity": [10, 20, 5, 1],   
        "Dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)