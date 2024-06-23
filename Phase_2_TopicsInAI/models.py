import numpy as np
import pandas as pd
import os
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

class CustomModel:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.best_model = None

    def train(self, X_train, y_train, X_val, y_val):
        y_train = y_train.values.ravel() if len(y_train.shape) > 1 else y_train.ravel()

        grid_search = GridSearchCV(estimator=self.model, param_grid=self.params, scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        y_pred_val = self.best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        print(f"Best Parameters for {self.best_model}")
        print("Validation Accuracy:", accuracy)

    def test(self, X_test, y_test):
        if self.best_model is not None:
            y_test = y_test.values.ravel() if len(y_test.shape) > 1 else y_test.ravel()

            y_pred = self.best_model.predict(X_test)
            accuracy_test = accuracy_score(y_test, y_pred)
            print("Test Accuracy:", accuracy_test)
        else:
            print("Model not trained yet!")


if __name__ == "__main__":
    nlp = en_core_web_sm.load()

    df_final = pd.read_csv("final.csv")
    X = df_final.loc[:, ["temporal_count", "speculative_prob", "ner_count", "sequentiality", "stressful", "social_words",
         "preposition_probs", "affect_words", "perceptual_probs", "cognitive_word_probs"]]
    Y = df_final.loc[:, ["type"]]
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    models = [
        (RandomForestClassifier(), {'n_estimators': [50, 100, 200],
                                    'max_depth': [None, 10, 20, 30],
                                    'min_samples_split': [2, 5, 10],
                                    'min_samples_leaf': [1, 2, 4]}),
        (SVC(), {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'sigmoid'],
              'gamma': ['scale', 'auto', 0.1, 1, 10]}),
        (LogisticRegression(), {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                                'penalty': ['l1', 'l2'],
                                'solver': ['liblinear']}),
        (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9, 11, 13],
                                'weights': ['uniform', 'distance'],
                                'p': [1, 2]}),
        (XGBClassifier(), {'learning_rate': [0.01, 0.1, 0.2],
                            'n_estimators': [100, 200, 300],
                            'max_depth': [3, 5, 7],
                            'min_child_weight': [1, 3, 5],
                            'subsample': [0.8, 0.9, 1.0],
                            'colsample_bytree': [0.8, 0.9, 1.0],
                            'gamma': [0, 0.1, 0.2]})
    ]

    for model, param_grid in models:
        model_instance = CustomModel(model, param_grid)
        # Train the model
        model_instance.train(X_train, y_train, X_val, y_val)
        # Test the best model on test data
        model_instance.test(X_test, y_test)
        print("\n" + "=" * 30 + "\n")

