import pandas as pd
from PIL.ImageOps import scale
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


# Load the Covertype dataset
def load_data():
    # dataset_path = " l;/.data"  # Update this path to the correct file path
    print("Loading dataset from local file...")
    # data = pd.read_csv(dataset_path, header=None)
    data = fetch_covtype()
    X = data.data # Feature values
    y = data.target   # True class labels
    return X, y

# Task 6
def split_data(X, y):
    # scaler = StandardScaler().fit(X) #standardisedata
    # X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Training set size:', len(X_train))
    print('Test set size:', len(X_test))
    return X_train, X_test, y_train, y_test

# Task 7
# def train_logistic_regression(X_train, y_train):
#     model = LogisticRegression(max_iter=3000, random_state=42, solver = 'lbfgs')
#     model.fit(X_train, y_train)
#     return model


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# Task 7 - Logistic Regression with RandomizedSearchCV
def train_logistic_regression_with_random_search(X_train, y_train):
    # Define parameter distributions
    param_distributions = {
        'C': uniform(0.01, 10),  # Regularization strength
        'solver': ['lbfgs', 'saga', 'sag', 'newton-cg'],  # Solver choices for logistic regression
        'max_iter': [100, 300, 500, 1000, 1500, 2000, 2500, 3000]  # Iteration limits
    }

    # Initialize logistic regression model
    model = LogisticRegression(random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,  # Number of different combinations to try
        cv=5,  # Cross-validation folds
        scoring='accuracy',  # Metric for evaluation
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # Fit the model with random search
    random_search.fit(X_train, y_train)

    # Retrieve the best model
    best_model = random_search.best_estimator_
    print("Best parameters found:", random_search.best_params_)
    return best_model

# Task 8
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42,
                                   criterion = 'entropy',
                                   max_depth = 16,
                                   splitter = 'best',
                                   min_samples_leaf = 5)
    model.fit(X_train, y_train)
    return model

# Task 9
def train_random_forest(X_train, y_train):
    param_distributions = {
        'n_estimators': np.arange(50, 200, 10),  # Varying from 50 to 200 in steps of 10
        'max_depth': [None, 10, 20, 30, 40, 50],  # Including None (no max depth) and values up to 50
        'max_features': ['sqrt', 'log2', None],  # Trying 'sqrt', 'log2', and None (all features)
        'min_samples_split': np.arange(2, 11),  # Minimum samples required to split a node
        'min_samples_leaf': np.arange(1, 5)  # Minimum samples required to be a leaf node
    }

    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=42)
    print("Model initialised")

    # Set up RandomizedSearchCV with 5-fold cross-validation
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_distributions,
        n_iter=10,  # Number of parameter settings sampled (adjust based on resources)
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',  # Adjust scoring metric as needed
        random_state=42,
        n_jobs=2  # Use all available cores
    )

    # Fit the RandomizedSearchCV to the data
    random_search.fit(X_train, y_train)

    # Display the best parameters and best score
    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation accuracy: ", random_search.best_score_)

    model = RandomForestClassifier(random_search.best_params_, bootstrap=True)
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
    # logistic_model = train_logistic_regression_with_random_search(X_train, y_train)
    # log_train, log_test = evaluate_model(logistic_model, X_train, y_train, X_test, y_test)
    # print("Logistic Regression Train Set: ", log_train, " Test Set Accuracy:", log_test)

    # Task 8
    # tree_model = train_decision_tree(X_train, y_train)
    # tree_train, tree_test = evaluate_model(tree_model, X_train, y_train, X_test, y_test)
    # print("Tree Model Train Set: ", tree_train, " Test Set Accuracy:", tree_test)

    # Task 9
    ensemble_model = train_random_forest(X_train, y_train)
    ens_train, ens_test = evaluate_model(ensemble_model, X_train, y_train, X_test, y_test)
    print("Ensemble Model Train Set: ", ens_train, " Test Set Accuracy:", ens_test)


if __name__ == "__main__":
    main()