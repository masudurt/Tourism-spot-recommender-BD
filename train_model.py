import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

# Load dataset
data = pd.read_csv('Endgame_tour_dataset_ultimate_final_Pro_Max.csv')

# Optimize numeric columns to reduce memory usage
for col in data.select_dtypes(include=['float', 'int']).columns:
    data[col] = pd.to_numeric(data[col], downcast='float')

data = data.dropna()

# Encode categorical variables
encoded_data = pd.get_dummies(data, columns=["From", "Budget", "Type"], drop_first=True)

# Ensure all required features are included
required_features = [
    "Distance (km)", "Duration (Days)", "Cost (BDT)", "From_Chittagong",
    "From_Cumilla", "From_Dhaka", "From_Khulna", "From_Mymensingh", 
    "From_Rajshahi", "From_Rangpur", "From_Sylhet", "From_Barisal", 
    "Budget_Low", "Budget_Medium", "Budget_High", "Type_Hills", 
    "Type_Forest", "Type_Historical Palace", "Type_Island", "Type_Park", 
    "Type_Picnic Spot", "Type_Sea Beach"
]

# Add missing columns with default value 0
for feature in required_features:
    if feature not in encoded_data.columns:
        encoded_data[feature] = 0

# Define features and target
X = encoded_data[required_features]  # Use only the required features
y = data["Destination"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with optimized parameters
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
dump(model, 'tour_recommendation_model.pkl')
print("Model saved as 'tour_recommendation_model.pkl'")

# Save feature names
dump(required_features, 'feature_names.pkl')
print("Feature names saved as 'feature_names.pkl'")

# Safe model loading for validation (optional)
model = load('tour_recommendation_model.pkl', mmap_mode=None)
print("Model loaded successfully for validation.")
