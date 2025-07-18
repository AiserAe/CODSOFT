# Task 4: Sales Prediction Using Python
# CodSoft Data Science Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
def load_data():
    return pd.read_csv("advertising.csv")

def main():
    print("=== SALES PREDICTION USING PYTHON ===")

    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # Features and target
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

        print(f"\n{name} Results:")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R² Score: {r2:.4f}")

    # Visualizations
    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(18, 5))

    # Actual vs Predicted (dot plot)
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.7, color='dodgerblue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')

    # Residuals Histogram
    plt.subplot(1, 3, 2)
    sns.histplot(residuals, bins=20, kde=True, color='skyblue')
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')

    # Feature Importance (separate horizontal bar chart)
    plt.subplot(1, 3, 3)
    sns.barplot(data=importance, x='Importance', y='Feature')
    plt.title('Feature Importance')

    plt.tight_layout()
    plt.show()

    print("\n=== Task 4 Completed! ===")

if __name__ == "__main__":
    main()
