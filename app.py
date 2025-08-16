import os  # Added to check for file existence
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)

# Load the trained model, feature names, and dataset


# Load the model without memory mapping
model = load('tour_recommendation_model.pkl', mmap_mode=None)

feature_names = joblib.load('feature_names.pkl')  # Feature names saved during training
dataset = pd.read_csv('Endgame_tour_dataset_ultimate_final_Pro_Max.csv')

@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user inputs
        user_inputs = {
            f"From_{request.form['From']}": 1,
            f"Budget_{request.form['Budget']}": 1,
            f"Type_{request.form['Type']}": 1,
            "Distance (km)": float(request.form["Distance (km)"]),
            "Duration (Days)": float(request.form["Duration (Days)"]),
            "Cost (BDT)": float(request.form["Cost (BDT)"]),
        }

        # Create a vector with all required features (initialized to 0)
        input_vector = {name: 0 for name in feature_names}
        input_vector.update(user_inputs)

        # Convert the input vector to a 2D array
        input_array = np.array([list(input_vector.values())])

        # Get probabilities for each destination
        probabilities = model.predict_proba(input_array)[0]

        # Check for no matches
        if probabilities.max() < 0.01:
            return render_template('results.html', no_match=True, destinations=[])

        # Get top 3 destinations
        destination_classes = model.classes_
        top_indices = probabilities.argsort()[-3:][::-1]
        top_destinations = destination_classes[top_indices]

        # Fetch details for each destination from the dataset
        destination_details = []
        for destination in top_destinations:
            details = dataset[dataset['Destination'] == destination].iloc[0]
            single_cost = round(details['Cost (BDT)'])
            couple_cost = round(single_cost * 1.85)  # Example calculation
            destination_details.append({
                'name': details['Destination'],
                'city': details['City'],
                'type': details['Type'],
                'distance': f"{details['Distance (km)']} km",
                'single_cost': f"{single_cost} BDT",
                'couple_cost': f"{couple_cost} BDT"
            })

        return render_template('results.html', no_match=False, destinations=destination_details)

    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/destination/<name>')
def destination_page(name):
    try:
        # Fetch details for the specific destination
        details = dataset[dataset['Destination'] == name].iloc[0]
        single_cost = round(details['Cost (BDT)'])
        couple_cost = round(single_cost * 1.85)  # Example calculation

        # Normalize the 'Type' value to match image filenames
        type_normalized = details['Type'].strip().replace(' ', '_').capitalize()
        bg_image_path = os.path.join('static', 'images', f"{type_normalized}.jpg")

        # Check if the file exists; use a fallback if missing
        if not os.path.exists(bg_image_path):
            bg_image = "/static/images/default.jpg"
        else:
            bg_image = f"/static/images/{type_normalized}.jpg"

        return render_template('destination.html', details={
            'name': details['Destination'],
            'city': details['City'],
            'type': details['Type'],
            'distance': f"{details['Distance (km)']} km",
            'single_cost': f"{single_cost} BDT",
            'couple_cost': f"{couple_cost} BDT",
            'bg_image': bg_image
        })

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
