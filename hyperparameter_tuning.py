from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from model import create_nn_model
import numpy as np

def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    return rf_random.best_estimator_

def tune_xgboost(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    xgb = XGBClassifier(random_state=42)
    xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    xgb_random.fit(X_train, y_train)

    return xgb_random.best_estimator_

def tune_neural_network(X_train, y_train):
    def create_model(optimizer='adam', neurons=64):
        model = create_nn_model(X_train.shape[1])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    param_dist = {
        'optimizer': ['adam', 'rmsprop', 'sgd'],
        'neurons': [32, 64, 128],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100, 150]
    }

    nn = KerasClassifier(build_fn=create_model, verbose=0)
    nn_random = RandomizedSearchCV(estimator=nn, param_distributions=param_dist, n_iter=20, cv=3, verbose=2, random_state=42)
    nn_random.fit(X_train, y_train)

    return nn_random.best_estimator_

def tune_models(X_train, y_train):
    rf_best = tune_random_forest(X_train, y_train)
    xgb_best = tune_xgboost(X_train, y_train)
    nn_best = tune_neural_network(X_train, y_train)

    return {
        "Random Forest": rf_best,
        "XGBoost": xgb_best,
        "Neural Network": nn_best
    }

if __name__ == "__main__":
    from data_fetcher import fetch_all_data
    from data_processor import process_data
    from model import prepare_data

    raw_data = fetch_all_data()
    processed_data = process_data(raw_data)
    X_train, X_test, y_train, y_test = prepare_data(processed_data)

    best_models = tune_models(X_train, y_train)

    for model_name, model in best_models.items():
        print(f"\\nBest {model_name} model:")
        print(model)

