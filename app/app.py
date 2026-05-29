from flask import Flask, render_template, request
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("../dataset/fake_instagram.csv")

print(df.columns)

# Features and target
X = df.drop("fake", axis=1)

y = df["fake"]

# Load trained model
model = pickle.load(open("../models/fake_profile_model.pkl", "rb"))

# Split data for accuracy calculation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy 
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

# Dataset statistics
total_profiles = len(df)

fake_profiles = len(df[df["fake"] == 1])

real_profiles = len(df[df["fake"] == 0])

fake_percentage = round((fake_profiles / total_profiles) *100, 2)

real_percentage = round((real_profiles / total_profiles) *100, 2)



@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy = accuracy,
        total_profiles = total_profiles,
        fake_percentage = fake_percentage,
        real_percentage = real_percentage)



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
    
    return render_template(
        "index.html",
        prediction_text = result,
        accracy = accuracy,
        total_profiles = total_profiles,
        fake_percentage = fake_percentage,
        real_percentage = real_percentage)


if __name__ == "__main__":
    app.run(debug=True)