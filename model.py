import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModel

def prepare_data(df, target_column="result"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_nn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_bert_model(input_shape):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")

    input_ids = Input(shape=(input_shape,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(input_shape,), dtype=tf.int32, name="attention_mask")

    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]
    cls_token = bert_output[:, 0, :]

    x = Dense(64, activation='relu')(cls_token)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model, tokenizer

def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    nn_model = create_nn_model(X_train.shape[1])

    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)

    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    nn_scores = nn_model.evaluate(X_train, y_train, verbose=0)[1]

    models = {
        "Random Forest": (rf_model, np.mean(rf_scores)),
        "XGBoost": (xgb_model, np.mean(xgb_scores)),
        "Neural Network": (nn_model, nn_scores)
    }

    best_model_name = max(models, key=lambda k: models[k][1])
    print(f"Using {best_model_name} model")

    best_model = models[best_model_name][0]
    if best_model_name != "Neural Network":
        best_model.fit(X_train, y_train)

    return best_model

def evaluate_model(model, X_test, y_test):
    if isinstance(model, Sequential):
        y_pred = (model.predict(X_test) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted")
    }

def plot_feature_importance(model, X):
    if isinstance(model, RandomForestClassifier) or isinstance(model, XGBClassifier):
        feature_importance = model.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.barh(pos, feature_importance[sorted_idx], align="center")
        ax.set_yticks(pos)
        ax.set_yticklabels(X.columns[sorted_idx])
        ax.set_xlabel("Relative Importance")
        ax.set_title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
    else:
        print("Feature importance plot is only available for Random Forest and XGBoost models.")

def train_and_evaluate(df):
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X_train)
    return model, metrics

if __name__ == "__main__":
    from data_fetcher import fetch_all_data
    from data_processor import process_data

    raw_data = fetch_all_data()
    processed_data = process_data(raw_data)
    model, metrics = train_and_evaluate(processed_data)
    print("Model performance:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
