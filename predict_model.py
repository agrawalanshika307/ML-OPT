import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration ---
DATASET_FILE = 'consolidated_antenna_data.csv'
MODEL_FILE = 'antenna_s_model.joblib'

print("--- Starting Model Training ---")

# --- 1. Load and Prepare Data ---
print(f"Loading dataset from '{DATASET_FILE}'...")
try:
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    print(f"❌ ERROR: Dataset file not found at '{DATASET_FILE}'")
    print("Please make sure your 'consolidated_antenna_data.csv' is in the same folder.")
    exit()

print(f"Successfully loaded dataset with {len(df)} rows.")

# Define features (X) and target (y)
features = ['n', 'ws', 'frequency']
target = 's_parameter'
X = df[features]
y = df[target]

# Split data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- 2. Train the Model ---
print("\nTraining Gradient Boosting Regressor...")
# We use GradientBoostingRegressor, which is powerful for this kind of data
model = GradientBoostingRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42, 
    verbose=1  # This will print training progress
)

model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- 3. Evaluate the Model ---
print("\n--- Evaluating Model Performance on Unseen Test Data ---")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared (R²) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f} dB")
print("Interpretation: An R-squared score near 1.0 and a low MAE are excellent.")

# --- 4. Save the Model ---
print(f"\n--- Saving Model to '{MODEL_FILE}' ---")
joblib.dump(model, MODEL_FILE)
print(f"✅ Model saved successfully!")
print("You can now run 'predict_s_parameter.py' to make predictions.")
