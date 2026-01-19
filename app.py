from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        features = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]

        prediction = model.predict([features])[0]

        flower = ["Setosa", "Versicolor", "Virginica"]
        prediction = flower[prediction]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
