import joblib
import pandas as pd
import os
import argparse  # Import the argument parsing library
from scipy.optimize import differential_evolution

# --- 1. CONFIGURATION ---
# This path points to the folder where this script and the model are located.
# It's set up to work within your Google Drive environment.
DRIVE_FOLDER_PATH = '/content/drive/MyDrive/Ml Project'
MODEL_FILENAME = os.path.join(DRIVE_FOLDER_PATH, 'lgbm_antenna_model_tuned.joblib')

# --- 2. PREDICTION & OBJECTIVE FUNCTIONS (The Engine) ---

def load_model_and_predict(n, x, frequency, model):
    """Uses a pre-loaded model to make a prediction."""
    input_data = pd.DataFrame([[n, x, frequency]], columns=['n', 'x', 'frequency'])
    prediction = model.predict(input_data)
    return prediction[0]

def objective_function(params, freq, model):
    """The function the optimizer will try to minimize."""
    n_candidate, x_candidate = params
    s_parameter = load_model_and_predict(n_candidate, x_candidate, freq, model)
    return s_parameter

# --- 3. MAIN EXECUTION BLOCK ---

def main():
    """
    Main function to parse arguments, run optimization, and print results.
    """
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Find the optimal antenna dimensions (n, w) for a given frequency."
    )
    parser.add_argument(
        "frequency",  # The name of our argument
        type=float,
        help="The target frequency in GHz (e.g., 5.8)"
    )
    args = parser.parse_args()
    user_target_frequency = args.frequency

    print("--- Starting Antenna Design Optimization ---")

    # --- Load the Model ---
    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ ERROR: Model file not found at '{MODEL_FILENAME}'")
        return

    print(f"✅ Loading saved model: {os.path.basename(MODEL_FILENAME)}")
    loaded_model = joblib.load(MODEL_FILENAME)

    # --- Run the Optimization ---
    bounds = [(3.0, 22.0), (13.0, 20.0)]
    print(f"🚀 Running optimizer for {user_target_frequency} GHz...")

    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(user_target_frequency, loaded_model),
        maxiter=100,
        popsize=15,
        tol=0.01,
        disp=False
    )

    print("✅ Optimization complete.")

    # --- Process and Display Results ---
    if result.success:
        optimal_n, optimal_x = result.x
        best_s_parameter = result.fun
        optimal_w = 50 - (2 * optimal_x)

        print("\n--- OPTIMAL DESIGN FOUND ---")
        print(f"Target Frequency: {user_target_frequency} GHz")
        print("----------------------------")
        print(f"Optimal 'n': {optimal_n:.4f} mm")
        print(f"Optimal 'w': {optimal_w:.4f} mm")
        print("----------------------------")
        print(f"Predicted S-parameter: {best_s_parameter:.4f} dB")
        print("============================")
    else:
        print("\n❌ WARNING: Optimization did not converge successfully.")

if __name__ == "__main__":
    main()
