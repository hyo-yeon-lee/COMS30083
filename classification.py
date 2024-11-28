from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, cross_val_score
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
    model = LogisticRegression(random_state=42,
                               C = 0.464,
                               max_iter = 25,
                               solver = 'newton-cg')
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
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42,
                                      n_estimators=150,
                                      min_samples_leaf=3,
                                      min_samples_split=5,
                                      max_features=None,
                                      max_depth=22,
                                      bootstrap=True)
    model.fit(X_train, y_train)
    return model


# Testing functions
def evaluate_train_model(model, X_train, y_train):
    train_accuracy = round(model.score(X_train,  y_train), 4)
    return train_accuracy


def evaluate_test_model(model, X_test, y_test):
    test_accuracy = round(model.score(X_test, y_test), 4)
    return test_accuracy



def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y) #scaled splitted dataset

    # Task 7
    logistic_model = train_logistic_regression(X_train, y_train)
    log_test = evaluate_test_model(logistic_model, X_test, y_test)
    print("Logistic Regression Test Set Accuracy:", log_test)

    # Task 8
    tree_model = train_decision_tree(X_train, y_train)
    tree_test = evaluate_test_model(tree_model,X_test, y_test)
    print("Decision Tree Test Set Accuracy:", tree_test)

    # Task 9
    ensemble_model = train_random_forest(X_train, y_train)
    ens_test = evaluate_test_model(ensemble_model, X_test, y_test)
    print("Ensemble Model Test Set Accuracy:", ens_test)



if __name__ == "__main__":
    main()