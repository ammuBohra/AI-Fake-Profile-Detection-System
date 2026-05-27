from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
df = pd.read_csv("../dataset/fake_instagram.csv")
print(df.columns)

# Using only 3 columns
X = df[["followers", "following", "posts"]]

y = df["fake"]

# Train model
model = RandomForestClassifier()
model.fit(X,y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    followers = int(request.form["followers"])

    following = int(request.form["following"])

    posts = int(request.form["posts"])

    input_data = [[followers, following, posts]]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "Fake Profile Detected"
    else: 
        result = "Genuine Profile"
    
    return render_template("index.html", prediction_text = result)

if __name__ == "__main__":
    app.run(debug=True)