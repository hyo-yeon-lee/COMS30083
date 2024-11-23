from matplotlib import pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np


# Task 6
def load_data():
    print("Loading dataset from local file...")
    data = fetch_covtype()
    X = data.data
    y = data.target
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Training set size:', len(X_train))
    print('Test set size:', len(X_test))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


# Task 7
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42,
                               C = 0.464,
                               max_iter = 25,
                               solver = 'newton-cg',
                               verbose=1)
    model.fit(X_train, y_train)
    return model


# Task 8
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42,
                                   criterion = 'entropy',
                                   max_depth = 20,
                                   splitter = 'best',
                                   min_samples_leaf = 3)
    model.fit(X_train, y_train)
    return model


# Task 9
def random_search_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42,
                                      n_estimators=150,
                                      min_samples_leaf=3,
                                      min_samples_split=5,
                                      max_features=None,
                                      max_depth=22,
                                      bootstrap=True)
    model.fit(X_train, y_train)
    return model



def plot_forest_loss_vs_trees(X_train, y_train, X_test, y_test, tree_counts):
    train_losses = []
    test_losses = []

    for n_trees in tree_counts:
        forest_model = RandomForestClassifier(
            random_state=42,
            n_estimators=n_trees,
            min_samples_leaf=3,
            min_samples_split=5,
            max_features=None,
            max_depth=22,
            bootstrap=True
        )
        forest_model.fit(X_train, y_train)
        train_loss = 1 - accuracy_score(y_train, forest_model.predict(X_train))

        train_losses.append(train_loss)

    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(tree_counts, train_losses, label='Training Loss', marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Loss (Misclassification Rate)')
    plt.title('Loss vs Number of Trees in Random Forest')
    plt.legend()
    plt.grid(True)
    plt.show()



def predict_ensemble(model, X, y):
   y_pred = model.predict(X)
   train_acc = accuracy_score(y, y_pred)

   return y_pred, train_acc



def evaluate_c_values(X_train, y_train, C_values):
    train_accuracies = []
    cv_accuracies = []

    for C in C_values:
        print(f"Calculating {C} ...")
        model = LogisticRegression(random_state=42, C=C, max_iter=500, solver='newton-cg', verbose=0)
        model.fit(X_train, y_train)
        # Training accuracy
        train_accuracy = model.score(X_train, y_train)
        train_accuracies.append(train_accuracy)
        # Cross-validation accuracy
        cv_accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))
        cv_accuracies.append(cv_accuracy)

    return train_accuracies, cv_accuracies


def plot_c_accuracies(X_train, y_train):
    # Define the range of C values
    C_values = np.logspace(-2, 1, 10)
    train_accuracies, cv_accuracies = evaluate_c_values(X_train, y_train, C_values)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(C_values, cv_accuracies, label='Cross-Validation Accuracy', marker='x')
    plt.xscale('log')  # Logarithmic scale for C values
    plt.xlabel('C Value (Regularization Strength)')
    plt.ylabel('Accuracy')
    plt.title('Training vs Cross-Validation Accuracy for Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.show()



def evaluate_train_model(model, X_train, y_train):
    train_accuracy = round(model.score(X_train,  y_train), 4)
    return train_accuracy


def evaluate_test_model(model, X_test, y_test):
    print("HERE")
    test_accuracy = round(model.score(X_test, y_test), 4)
    # plot_performance(model, X_test, y_test)

    return test_accuracy


def plot_performance(model, X, y):
    print("Entered plot performance...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, model.predict(X), alpha=0.5)
    ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')  # Ideal line for perfect predictions
    ax.set_xlabel('True Labels')
    ax.set_ylabel('Predicted Labels')
    ax.set_title('True vs Predicted Labels (Test Set)')


def plot_tree_losses(forest_model, X, y, loss_type='accuracy_loss'):
    tree_losses = []
    for i, tree in enumerate(forest_model.estimators_):
        y_pred = tree.predict(X)  # Predictions from individual tree
        if loss_type == 'accuracy_loss':
            loss = 1 - accuracy_score(y, y_pred)  # Misclassification rate
        else:
            raise ValueError("Unsupported loss type")
        tree_losses.append(loss)

    # Calculate overall Random Forest loss
    y_pred_forest = forest_model.predict(X)
    overall_loss = 1 - accuracy_score(y, y_pred_forest)  # Misclassification rate for the ensemble

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(tree_losses)), tree_losses, alpha=0.7, label='Individual Trees')
    plt.axhline(y=overall_loss, color='r', linestyle='--', label='Random Forest Overall Loss')
    plt.xlabel('Tree Index')
    plt.ylabel('Loss (Misclassification Rate)')
    plt.title('Loss of Each Decision Tree vs Random Forest (Misclassification Rate)')
    plt.legend()
    plt.show()




def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y) #scaled splitted dataset

    # Task 7
    logistic_model = train_logistic_regression(X_train, y_train)
    log_train = evaluate_train_model(logistic_model, X_train, y_train)
    print("Logistic Regression Train Set: ", log_train)

    # Task 8
    tree_model = train_decision_tree(X_train, y_train)
    tree_train = evaluate_train_model(tree_model, X_train, y_train)
    print("Tree Model Train Set: ", tree_train)

    # Task 9
    ensemble_model = random_search_forest(X_train, y_train)
    ens_train = evaluate_train_model(ensemble_model, X_train, y_train)

    #Testing set
    log_test = evaluate_test_model(logistic_model, X_test, y_test)
    print(" Log reg Test Set Accuracy:", log_test)
    tree_test = evaluate_test_model(tree_model,X_test, y_test)
    print(" Decision Tree Test Set Accuracy:", tree_test)

    ens_train = evaluate_test_model(ensemble_model, X_train, y_train)
    ens_test = evaluate_test_model(ensemble_model, X_test, y_test)
    print("Ensemble Model Test Set Accuracy:", ens_test)

    # Plot losses
    tree_counts = range(10, 201, 10)  # Number of trees to iterate over
    plot_forest_loss_vs_trees(X_train, y_train, X_test, y_test, tree_counts)

    plot_tree_losses(ensemble_model, X_train, y_train, loss_type='accuracy_loss')
    plot_tree_losses(ensemble_model, X_test, y_test, loss_type='accuracy_loss')
    ens_test = evaluate_test_model(ensemble_model, X_test, y_test)
    print("Ensemble Model Train Set: ", ens_train, " Test Set Accuracy:", ens_test)


if __name__ == "__main__":
    main()