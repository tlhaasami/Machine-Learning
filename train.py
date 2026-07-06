import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train():
    # 1. Load the dataset
    data_path = "DataSet/energy_efficiency_data.csv"

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Separate features (first 8 columns) and targets (last 2 columns)
    X = df.iloc[:, :8]
    y = df.iloc[:, 8:]

    print(f"Loaded dataset shape: {df.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # 2. Train-test split
    # Using 80% of data for training and 20% for validation/testing
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")

    # 3. Define Preprocessing & Pipeline
    # All features are numeric. Scaling is generally beneficial and ensures we can swap models easily.
    numerical_features = list(X.columns)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ]
    )

    # 4. Define the Multi-Output Regressor
    # MLP Regressor natively supports multi-output regression (y can be 2D)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42, early_stopping=True))
    ])

    print("\nTraining the Multi-Layer Perceptron (MLP) Regressor...")
    model.fit(X_train, y_train)
    print("Training completed!")

    # 5. Evaluate the Model
    y_pred = model.predict(X_val)

    # Calculate metrics for each target variable
    targets_list = list(y.columns)
    print("\n--- Model Evaluation (Validation Set) ---")
    
    for i, target_name in enumerate(targets_list):
        y_true_col = y_val.iloc[:, i]
        y_pred_col = y_pred[:, i]
        
        mse = mean_squared_error(y_true_col, y_pred_col)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_col, y_pred_col)
        
        print(f"\nTarget: {target_name}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2 Score: {r2:.4f}")

    # Overall metrics
    overall_mse = mean_squared_error(y_val, y_pred)
    overall_rmse = np.sqrt(overall_mse)
    overall_r2 = r2_score(y_val, y_pred, multioutput='uniform_average')
    
    print("\nOverall Performance:")
    print(f"  Average MSE:  {overall_mse:.4f}")
    print(f"  Average RMSE: {overall_rmse:.4f}")
    print(f"  Average R2 Score: {overall_r2:.4f}")

    # 6. Save the trained model
    model_filename = "model.joblib"
    joblib.dump(model, model_filename)
    print(f"\nTrained model successfully saved to: {model_filename}")

    # 7. Demonstrate Prediction
    print("\n--- Testing Model Predictions on a few sample inputs ---")
    # Take the first 3 samples from validation set
    sample_inputs = X_val.head(3)
    sample_actual = y_val.head(3)
    sample_preds = model.predict(sample_inputs)

    for idx, (index, row) in enumerate(sample_inputs.iterrows()):
        print(f"\nSample #{idx + 1} (Row Index: {index})")
        print(f"  Inputs: {row.to_dict()}")
        print(f"  Actual:    Heating: {sample_actual.iloc[idx, 0]:.2f}, Cooling: {sample_actual.iloc[idx, 1]:.2f}")
        print(f"  Predicted: Heating: {sample_preds[idx, 0]:.2f}, Cooling: {sample_preds[idx, 1]:.2f}")

if __name__ == "__main__":
    train()
