from matplotlib import pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(random_state=42,
                               C = 0.216,
                               max_iter = 500,
                               solver = 'newton-cg',
                               verbose=1)
    model.fit(X_train, y_train)
    return model


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


def predict_ensemble(model, X, y):
   y_pred = model.predict(X)
   train_acc = accuracy_score(y, y_pred)

   return y_pred, train_acc


def evaluate_train_model(model, X_train, y_train):
    print("Received model and train data...")
    train_accuracy = round(model.score(X_train, y_train), 4)
    print("Calculated train accuracy... ", train_accuracy)
    print("Plotting into a graph...")
    plot_performance(model, X_train, y_train)

    return train_accuracy



def evaluate_test_model(model, X_test, y_test):
    print("Received model and test data...")
    test_accuracy = round(model.score(X_test, y_test), 4)
    print("Calculated terst accuracy...", test_accuracy)
    print("Plotting into a graph...")
    plot_performance(model, X_test, y_test)

    return test_accuracy


def plot_performance(model, X, y):
    print("Entered plot performance...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, model.predict(X), alpha=0.5)
    ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')  # Ideal line for perfect predictions
    ax.set_xlabel('True Labels')
    ax.set_ylabel('Predicted Labels')
    ax.set_title('True vs Predicted Labels (Test Set)')

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
    print(" Test Set Accuracy:", log_test)
    tree_test = evaluate_test_model(tree_model,X_test, y_test)
    print(" Test Set Accuracy:", tree_test)
    ens_test = evaluate_test_model(ensemble_model, X_test, y_test)
    print("Ensemble Model Train Set: ", ens_train, " Test Set Accuracy:", ens_test)


if __name__ == "__main__":
    main()