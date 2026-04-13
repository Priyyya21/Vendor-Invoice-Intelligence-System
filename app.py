from flask import Flask, render_template, request

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", active="freight", form_data={})


@app.route("/predict", methods=["POST"])
def predict():
    model = request.form.get("model")

    # ================= FREIGHT =================
    if model == "freight":
        try:
            quantity = float(request.form.get("quantity"))
            dollars = float(request.form.get("dollars"))

            input_data = {
                "Quantity": [quantity],
                "Dollars": [dollars]
            }

            result_df = predict_freight_cost(input_data)
            prediction = float(result_df["Predicted_Freight"].iloc[0])

            return render_template(
                "index.html",
                freight_result=f"₹ {prediction:.2f}",
                active="freight",
                form_data=request.form
            )

        except Exception as e:
            return render_template(
                "index.html",
                error=str(e),
                active="freight",
                form_data=request.form
            )

    # ================= INVOICE =================
    else:
        try:
            input_data = {
                "invoice_quantity": [float(request.form.get("invoice_quantity"))],
                "invoice_dollars": [float(request.form.get("invoice_dollars"))],
                "Freight": [float(request.form.get("freight"))],
                "total_item_quantity": [float(request.form.get("total_item_quantity"))],
                "total_item_dollars": [float(request.form.get("total_item_dollars"))]
            }

            result_df = predict_invoice_flag(input_data)
            prediction = int(result_df["Predicted_Flag"].iloc[0])

            result = "⚠️ Manual Review Required" if prediction == 1 else "✅ Safe Invoice"

            return render_template(
                "index.html",
                flag_result=result,
                active="invoice",
                form_data=request.form
            )

        except Exception as e:
            return render_template(
                "index.html",
                error=str(e),
                active="invoice",
                form_data=request.form
            )


if __name__ == "__main__":
    app.run(debug=True)