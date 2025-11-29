import joblib
import pandas as pd
import numpy as np
import os

# --- Configuration ---
MODEL_FILE = 'antenna_s_model.joblib'

def load_model():
    """
    Loads the saved model file.
    """
    print(f"Loading model from '{MODEL_FILE}'...")
    if not os.path.exists(MODEL_FILE):
        print(f"❌ ERROR: Model file not found at '{MODEL_FILE}'")
        print("Please run the 'train_model.py' script first to train and save the model.")
        return None
    
    try:
        model = joblib.load(MODEL_FILE)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

def main():
    """
    Main function to load model and ask user for predictions.
    """
    model = load_model()
    if model is None:
        return # Exit if model couldn't be loaded

    print("\n--- Antenna S-Parameter Predictor ---")
    print("Enter 'q' at any time to quit.")

    while True:
        try:
            # --- Get User Input ---
            f_input = input("\nEnter Frequency (f) in GHz (e.g., 4.5): ")
            if f_input.lower() == 'q': break
            
            n_input = input("Enter Slot Length (n) in mm (e.g., 6.0): ")
            if n_input.lower() == 'q': break
            
            ws_input = input("Enter Slot Width (ws) in mm (e.g., 38.0): ")
            if ws_input.lower() == 'q': break

            # Convert inputs to numbers
            f_val = float(f_input)
            n_val = float(n_input)
            ws_val = float(ws_input)

            # --- Format Input for Model ---
            # The model expects a DataFrame with specific column names
            example_input = pd.DataFrame(
                data=[[n_val, ws_val, f_val]], 
                columns=['n', 'ws', 'frequency'] # Must match training columns
            )

            # --- Make Prediction ---
            predicted_value_array = model.predict(example_input)
            predicted_value = predicted_value_array[0] # Get the number from the array

            print("\n-------------------------")
            print(f"  Inputs: f={f_val} GHz, n={n_val} mm, ws={ws_val} mm")
            print(f"  Predicted S-parameter: {predicted_value:.4f} dB")
            print("-------------------------")
            
            if predicted_value < -10:
                print("  Interpretation: At these settings, the antenna is efficient (good match).")
            else:
                print("  Interpretation: At these settings, the antenna is inefficient (poor match).")

        except ValueError:
            print("❌ Invalid Input: Please enter numbers only (or 'q' to quit).")
        except KeyboardInterrupt:
            print("\nExiting predictor.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

