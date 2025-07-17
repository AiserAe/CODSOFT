# Task 3: Iris Flower Classification 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the CSV dataset
def load_iris_data():
    df = pd.read_csv("IRIS.csv")
    
    # Strip any extra spaces in column names or values
    df.columns = df.columns.str.strip()
    df['species'] = df['species'].str.strip()
    
    return df

# Main function
def main():
    print("=== IRIS FLOWER CLASSIFICATION ===")

    # Load data
    df = load_iris_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Species distribution:\n{df['species'].value_counts()}")

    # Features and target
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[features]
    y = df['species']

    # Check for missing values and drop them if any
    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
        print("Missing values found. Dropping rows with NaNs.")
        df = df.dropna()
        X = df[features]
        y = df['species']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Classifiers to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }

    # Train and evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{name} Results:")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    # Visualizations for best model
    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(12, 4))

    # Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=best_model.classes_,
                yticklabels=best_model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Feature Importance
    plt.subplot(1, 2, 2)
    importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title("Feature Importance")

    plt.tight_layout()
    plt.show()

    print("\n=== Task 3 Completed ===")

if __name__ == "__main__":
    main()
