# Task 2: Movie Rating Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
def load_movie_data():
    df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

    # Select useful columns
    df = df[['Genre', 'Director', 'Actor 1', 'Duration', 'Year', 'Rating']]
    df.dropna(inplace=True)

    # Rename for consistency
    df.columns = ['genre', 'director', 'actor', 'runtime', 'year', 'rating']
    return df

# Main function
def main():
    print("=== MOVIE RATING PREDICTION ===")

    # Load data
    df = load_movie_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Average Rating: {df['rating'].mean():.2f}")

    # ---- VISUALIZATION ----

    plt.figure(figsize=(12, 5))

    # 1. Top 10 directors by average rating
    plt.subplot(1, 2, 1)
    top_directors = df.groupby('director')['rating'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_directors.values, y=top_directors.index)
    plt.title('Top 10 Directors by Avg Rating')
    plt.xlabel('Rating')
    plt.ylabel('Director')

    # 2. Rating distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df['rating'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

    # ---- ENCODING ----
    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df['genre'])
    df['director_encoded'] = le.fit_transform(df['director'])
    df['actor_encoded'] = le.fit_transform(df['actor'])

    # ---- FEATURES ----
    features = ['genre_encoded', 'director_encoded', 'actor_encoded', 'runtime', 'year']
    X = df[features]
    y = df['rating']

    # ---- TRAINING ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ---- EVALUATION ----
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R² Score: {r2:.2f}")

    print("\n=== Task 2 Completed! ===")

if __name__ == "__main__":
    main()
