import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from majority_vote import MajorityVoteClassifier


@pytest.fixture
def data():
    return np.array([[0], [1], [2], [3]]), np.array(["zebra", "alpha", "zebra", "alpha"])


def tied(**kwargs):
    return MajorityVoteClassifier([
        DummyClassifier(strategy="constant", constant=0),
        DummyClassifier(strategy="constant", constant=1),
    ], **kwargs)


def test_clone_parameters_and_classifier_tags():
    original = MajorityVoteClassifier([DecisionTreeClassifier(max_depth=1), LogisticRegression()])
    copied = clone(original)
    assert is_classifier(copied)
    assert copied.classifiers[0] is not original.classifiers[0]
    assert copied.get_params()["classifier0__max_depth"] == 1
    copied.set_params(classifier0__max_depth=3)
    assert copied.classifiers[0].max_depth == 3
    copied.set_params(classifier0=DecisionTreeClassifier(max_depth=2))
    assert copied.classifiers[0].max_depth == 2
    with pytest.raises(ValueError):
        copied.set_params(unknown=3)


def test_hard_ties_and_original_labels(data):
    X, y = data
    classifier = tied().fit(X, y)
    assert classifier.predict(X).tolist() == ["alpha"] * 4
    assert not hasattr(classifier.classifiers[0], "classes_")
    classifier.fit(X, np.array([10, 30, 10, 30]))
    assert classifier.predict(X).tolist() == [10] * 4


def test_weighted_hard_and_probability_votes(data):
    X, y = data
    hard = tied(weights=[1, 2]).fit(X, y)
    soft = tied(vote="probability", weights=[1, 2]).fit(X, y)
    assert hard.predict(X).tolist() == ["zebra"] * 4
    assert soft.predict(X).tolist() == hard.predict(X).tolist()
    np.testing.assert_allclose(soft.predict_proba(X), np.tile([1/3, 2/3], (4, 1)))


@pytest.mark.parametrize("weights", [[1], [-1, 2], [0, 0], [np.inf, 1], [np.nan, 1]])
def test_invalid_weights_rejected(data, weights):
    with pytest.raises(ValueError, match="weights"):
        tied(weights=weights).fit(*data)


def test_invalid_vote_and_unfitted(data):
    with pytest.raises(ValueError, match="vote"):
        tied(vote="invalid").fit(*data)
    with pytest.raises(NotFittedError):
        tied().predict(data[0])


class ReversedProbabilityClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)[::-1]
        return self

    def predict_proba(self, X):
        return np.tile([1., 0.], (len(X), 1))


def test_probability_columns_align_with_encoded_classes(data):
    model = MajorityVoteClassifier([ReversedProbabilityClassifier()], vote="probability").fit(*data)
    assert model.predict(data[0]).tolist() == ["zebra"] * 4


def test_cv_and_nested_grid_search_work():
    X, y = load_iris(return_X_y=True)
    model = MajorityVoteClassifier([
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        DecisionTreeClassifier(random_state=1),
    ])
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy", error_score="raise")
    assert np.isfinite(scores).all()
    search = GridSearchCV(model, {"classifier1__max_depth": [1, 2]}, cv=3, error_score="raise")
    search.fit(X, y)
    assert len(search.predict(X)) == len(y)
