# Task 2: Movie Rating Prediction - CodSoft Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
def load_movie_data():
    df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
    return df

def main():
    print("🎬 Movie Rating Prediction")

    df = load_movie_data()
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Keep only useful columns
    df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Rating']]
    df.dropna(inplace=True)
    df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)

    for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
        df[col] = pd.factorize(df[col])[0]

    # Features and target
    X = df.drop('Rating', axis=1)
    y = df['Rating']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Show results
        print(f"\n{name} Results:")
        print(f"  MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        print(f"  R² Score: {r2_score(y_test, y_pred):.4f}")

    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Make plots
    plt.figure(figsize=(15, 5))

    # 1. Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.7, color='mediumseagreen')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Actual vs Predicted')

    # 2. Residuals
    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=20, kde=True, color='skyblue')
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Residuals')

    # 3. Feature Importance
    plt.subplot(1, 3, 3)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    sns.barplot(data=importance, x='Importance', y='Feature')
    plt.title('Feature Importance')

    plt.tight_layout()
    plt.show()

    print("\n Task 2 Completed")

if __name__ == "__main__":
    main()
