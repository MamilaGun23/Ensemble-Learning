"""A cloneable scikit-learn classifier with explicit hard/soft voting."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class MajorityVoteClassifier(ClassifierMixin, BaseEstimator):
    """Ties choose the smallest LabelEncoder class; weights must be nonnegative."""

    def __init__(self, classifiers, vote="classlabel", weights=None):
        self.classifiers = classifiers
        self.vote = vote
        self.weights = weights

    def get_params(self, deep=True):
        params = super().get_params(deep=False)
        if deep:
            for i, classifier in enumerate(self.classifiers):
                name = f"classifier{i}"
                params[name] = classifier
                params.update({f"{name}__{key}": value for key, value in classifier.get_params(deep=True).items()})
        return params

    def set_params(self, **params):
        roots = {key: value for key, value in params.items() if key in ("classifiers", "vote", "weights")}
        super().set_params(**roots)
        remaining = {key: value for key, value in params.items() if key not in roots}
        classifiers = list(self.classifiers)
        names = {f"classifier{i}": i for i in range(len(classifiers))}
        for key, value in remaining.items():
            name, separator, nested = key.partition("__")
            if name not in names:
                raise ValueError(f"Invalid parameter {name}.")
            if not separator:
                classifiers[names[name]] = value
        for key, value in remaining.items():
            name, separator, nested = key.partition("__")
            if separator:
                classifiers[names[name]].set_params(**{nested: value})
        if remaining:
            self.classifiers = classifiers
        return self

    def fit(self, X, y):
        if self.vote not in ("classlabel", "probability"):
            raise ValueError("vote must be classlabel or probability.")
        if not self.classifiers:
            raise ValueError("At least one classifier is required.")
        weights = None if self.weights is None else np.asarray(self.weights, dtype=float)
        if weights is not None and (
            weights.shape != (len(self.classifiers),) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0
        ):
            raise ValueError("weights must be finite, nonnegative, match classifiers and have positive sum.")
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.lablenc_ = LabelEncoder().fit(y)
        self.classes_ = self.lablenc_.classes_
        encoded = self.lablenc_.transform(y)
        self.classifiers_ = [clone(classifier).fit(X, encoded) for classifier in self.classifiers]
        self.weights_ = weights
        if self.vote == "probability" and not all(hasattr(c, "predict_proba") for c in self.classifiers_):
            raise ValueError("Probability voting requires predict_proba on every classifier.")
        return self

    def predict(self, X):
        check_is_fitted(self, "classifiers_")
        X = validate_data(self, X, reset=False)
        if self.vote == "probability":
            encoded = self.predict_proba(X).argmax(axis=1)
        else:
            predictions = np.asarray([classifier.predict(X) for classifier in self.classifiers_], dtype=int).T
            encoded = np.array([
                np.bincount(row, weights=self.weights_, minlength=len(self.classes_)).argmax()
                for row in predictions
            ])
        return self.lablenc_.inverse_transform(encoded)

    def predict_proba(self, X):
        check_is_fitted(self, "classifiers_")
        X = validate_data(self, X, reset=False)
        aligned = []
        for classifier in self.classifiers_:
            if not hasattr(classifier, "predict_proba"):
                raise ValueError("Every classifier must support predict_proba.")
            probabilities = classifier.predict_proba(X)
            ordered = np.zeros((len(X), len(self.classes_)))
            ordered[:, np.asarray(classifier.classes_, dtype=int)] = probabilities
            aligned.append(ordered)
        return np.average(aligned, axis=0, weights=self.weights_)
