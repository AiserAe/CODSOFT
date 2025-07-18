# Task 1: Titanic Survival Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")
    return df

# Preprocess the data for modeling
def preprocess_data(df):
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    return df

def main():
    print("=== TITANIC SURVIVAL PREDICTION (USING CSV) ===")

    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Survival rate: {df['Survived'].mean():.2%}")

    # Basic survival-related visualizations
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    sns.countplot(data=df, x='Sex', hue='Survived')
    plt.title('Survival by Gender')

    plt.subplot(1, 3, 2)
    sns.countplot(data=df, x='Pclass', hue='Survived')
    plt.title('Survival by Class')

    plt.subplot(1, 3, 3)
    sns.histplot(data=df, x='Age', hue='Survived', bins=20)
    plt.title('Survival by Age')

    plt.tight_layout()
    plt.show()

    df = preprocess_data(df)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
    X = df[features]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    print("\n=== Task 1 Completed Using CSV! ===")

if __name__ == "__main__":
    main()
