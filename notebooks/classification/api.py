"""
classification.api
~~~~~~~~~~~~~~~~~~

This module implements the high-level API.
"""


import random
from typing import Generator
import warnings

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from classification import preprocessing


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


def classic_optimization(
    dataset: pd.DataFrame, random_state: int
) -> Generator[dict, None, None]:
    algorithms = [
        (
            "Naïve Bayes",
            MultinomialNB,
            {"alpha": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5], "fit_prior": [True, False]},
        ),
        (
            "Logistic Regression",
            LogisticRegression,
            {
                "C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                "random_state": [random_state],
            },
        ),
        (
            "Support Vector Machine",
            SGDClassifier,
            {
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                "random_state": [random_state],
            },
        ),
    ]

    vectorizer = TfidfVectorizer(tokenizer=preprocessing.tokenize, max_features=10000)
    X, y = vectorizer.fit_transform(dataset["text"]), list(dataset["class"])

    for name, classifier, hyperparameters in algorithms:
        c = classifier()
        c = GridSearchCV(c, hyperparameters, cv=5)
        c.fit(X, y)
        results = pd.DataFrame(c.cv_results_)
        best_params = results.sort_values("mean_test_score", ascending=False).iloc[0][
            "params"
        ]
        yield (name, classifier, best_params)


def classic_pipeline(corpus: str, random_state: int = 23):
    dataset = preprocessing.load(corpus, split=False)
    vectorizer = TfidfVectorizer(tokenizer=preprocessing.tokenize, max_features=10000)
    X, y = vectorizer.fit_transform(dataset["text"]), list(dataset["class"])
    X, y = preprocessing.split(X, y, random_state=23)

    random.seed(random_state)
    prediction = [random.choice([0, 1]) for _ in range(len(y["test"]))]

    yield ("Random", metrics.accuracy_score(y["test"], prediction))

    algorithms = classic_optimization(dataset, random_state)
    for name, classifier, hyperparameters in algorithms:
        c = classifier(**hyperparameters)
        c.fit(X["train"], y["train"])
        prediction = c.predict(X["test"])
        yield (name, metrics.accuracy_score(y["test"], prediction))


def neural_pipeline(corpus: str, random_state: int = 23, epochs: int = 10):
    raise NotImplementedError


def evaluate():
    raise NotImplementedError
