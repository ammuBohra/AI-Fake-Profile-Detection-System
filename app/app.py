from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset
df = pd.read_csv("../dataset/fake_instagram.csv")

print(df.columns)

# Load trained model
model = pickle.load(open("../models/fake_profile_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    profile_pic = int(request.form["profile_pic"])
    nums_length_username = int(request.form["nums_length_username"])
    fullname_words = int(request.form["fullname_words"])
    nums_length_fullname = int(request.form["nums_length_fullname"])
    name_equals_username = int(request.form["name_equals_username"])
    description_length = int(request.form["description_length"])
    external_URL = int(request.form["external_URL"])
    private = int(request.form["private"])
    posts = int(request.form["posts"])
    followers = int(request.form["followers"])
    following = int(request.form["following"])


    input_data = [[
        profile_pic,
        nums_length_username,
        fullname_words,
        nums_length_fullname,
        name_equals_username,
        description_length,
        external_URL,
        private,
        posts,
        followers,
        following
    ]]

    prediction = model.predict(input_data)
    
    probability = model.predict_proba(input_data)
    
    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 1:
        result = f"Fake Profile Detected ({confidence}% Confidence)"
    else: 
        result = f"Genuine Profile ({confidence}% Confidence)"
    
    return render_template("index.html", prediction_text = result)

if __name__ == "__main__":
    app.run(debug=True)