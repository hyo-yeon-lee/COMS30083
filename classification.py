import pandas as pd
from PIL.ImageOps import scale
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Load the Covertype dataset
def load_data():
    dataset_path = "covtype.data"  # Update this path to the correct file path
    print("Loading dataset from local file...")
    data = pd.read_csv(dataset_path, header=None)
    X = data.iloc[:, :-1].values  # Feature values
    y = data.iloc[:, -1].values   # True class labels
    return X, y

# Task 6
def split_data(X, y):
    scaler = StandardScaler().fit(X) #standardisedata
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print('Training set size:', len(X_train))
    print('Test set size:', len(X_test))
    return X_train, X_test, y_train, y_test

# Task 7
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=3000, random_state=42, solver = 'lbfgs')
    model.fit(X_train, y_train)
    return model

# Task 8
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Task 9

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_accuracy = round(model.score(X_train, y_train), 2)
    test_accuracy = round(model.score(X_test, y_test), 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, model.predict(X_test), alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Ideal line for perfect predictions
    ax.set_xlabel('True Labels')
    ax.set_ylabel('Predicted Labels')
    ax.set_title('True vs Predicted Labels (Test Set)')

    return train_accuracy, test_accuracy


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y) #scaled splitted dataset

    # Task 7
    logistic_model = train_logistic_regression(X_train, y_train)
    log_train, log_test = evaluate_model(logistic_model, X_train, y_train, X_test, y_test)
    print("Logistic Regression Train Set: ", log_train, " Test Set Accuracy:", log_test)

    # Task 8
    tree_model = train_decision_tree(X_train, y_train)
    tree_train, tree_test = evaluate_model(tree_model, X_train, y_train, X_test, y_test)
    print("Tree Model Train Set: ", tree_train, " Test Set Accuracy:", tree_test)

    # Task 9
    ensemble_model = train_random_forest(X_train, y_train)
    ens_train, ens_test = evaluate_model(ensemble_model, X_train, y_train, X_test, y_test)
    print("Ensemble Model Train Set: ", ens_train, " Test Set Accuracy:", ens_test)


if __name__ == "__main__":
    main()
