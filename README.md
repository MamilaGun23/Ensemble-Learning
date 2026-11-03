# Ensemble learning lab

[LAB 7.ipynb](LAB%207.ipynb) compares a custom majority-vote classifier, base
classifiers, bagging and AdaBoost. Results below are from actual execution;
they do not show that ensembles consistently outperform individual models.

## Reproduce (Python 3.12)

```sh
git clone https://github.com/MamilaGun23/Ensemble-Learning.git
cd Ensemble-Learning
python -m venv .venv
```

Activate with `.venv/Scripts/Activate.ps1` on PowerShell or
`source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check majority_vote.py tests
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "LAB 7.ipynb"
```

Interactive alternative: `python -m notebook "LAB 7.ipynb"`.
Wine and Iris use packaged scikit-learn datasets, so tests and all notebook cells
need no dataset downloads or credentials. [results.json](results.json) records
the separate experiments. Requirements are pinned to the tested environment.

## Experiments and boundaries

| Experiment | Features and classes | Stratified split (seed 1) | Training CV | Metric |
|---|---|---|---|---|
| Full Iris illustration | First two features, three species | 105 train / 45 test | No CV for this illustration | Single held-out accuracy; full-data plot is exploratory |
| Binary Iris | Sepal width and petal length; versicolor/virginica | 70 train / 30 test | Same ten shuffled folds for all four methods | ROC-AUC |
| Binary Wine | Alcohol and OD280/OD315; original classes 2/3 | 95 train / 24 test | Same five shuffled folds for fixed methods | Accuracy |
| Full Iris comparison | All four features, three species | 105 train / 45 test | Same ten shuffled folds | Accuracy |

ROC-AUC is calculated using each estimator's probability response, including
averaged probabilities from the majority-vote estimator; hard classification
predictions use classlabel voting. AUC and accuracy are different metrics and
must not be compared as interchangeable percentages.

Scaling stays inside LR/KNN pipelines in every fold, including within the custom
ensemble. Wine trees do not require scaling. UCI Wine attribution/licence is
available at [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine) (CC BY 4.0);
packaged dataset DESCR fields include source details for Wine and Iris.

Wine boosting curves use an inner validation split from training data, avoiding
the original minimum-test-error stopping choice. Full-Iris grid searches use
five training folds. Their best_score_ values are labelled **selection CV**;
a redundant post-selection CV re-run was removed. The 45 held-out Iris rows are
evaluated after selection. Original search grids and estimator counts remain,
with jobs bounded to one for portable local execution.

## Actual measured results

Binary Iris (training-only ten-fold **ROC-AUC**, 70 rows):

| Method | AUC mean ± std |
|---|---:|
| Logistic regression | 0.9167 ± 0.0913 |
| Decision tree | 0.8792 ± 0.0708 |
| KNN | 0.8875 ± 0.0987 |
| Majority vote | 0.9667 ± 0.0408 |

Full Iris (training-only ten-fold **accuracy**, 105 rows):

| Method | Accuracy mean ± std |
|---|---:|
| Logistic regression | 0.9336 ± 0.0941 |
| KNN | 0.9427 ± 0.0765 |
| Decision tree | 0.9436 ± 0.0615 |
| Random forest | 0.9436 ± 0.0842 |
| Built-in voting | 0.9436 ± 0.0842 |
| Bagging | 0.9436 ± 0.0842 |
| AdaBoost | 0.9536 ± 0.0617 |
| Custom majority vote | 0.9436 ± 0.0842 |

Five-fold selection chose AdaBoost learning_rate=0.1/n_estimators=50 (selection
CV accuracy 0.9714), and bagging max_features=0.8/max_samples=0.5/n_estimators=200
(selection CV accuracy 0.9524). On the **45 held-out full-Iris rows**, the decision
tree baseline, tuned AdaBoost and tuned bagging all scored **0.9778 accuracy**.
These selected-model scores are separate from the untuned ten-fold table above.

Binary Wine (training-only five-fold **accuracy**, 95 rows; held-out 24 rows):

| Method | CV accuracy mean ± std | Held-out accuracy |
|---|---:|---:|
| Decision tree | 0.8211 ± 0.0714 | 0.8333 |
| Bagging (500 trees) | 0.8737 ± 0.0788 | 0.9167 |
| Decision stump | 0.8632 ± 0.0421 | 0.8750 |
| AdaBoost (500, learning_rate=0.1) | 0.8842 ± 0.0774 | 0.9167 |

All tables are specific to their stated data, features, split and metric. Fold
standard deviations describe variability, not confidence intervals or statistical
significance. No universal winner, guaranteed improvement, optimal 50–200 ensemble
size, production-scale speed claim or general stopping rule follows from this lab.

## Custom estimator

[majority_vote.py](majority_vote.py) contains the same custom implementation used
by the notebook and offline tests. It provides:

- Classifier tags and clone-compatible constructor parameters.
- Dynamic nested parameters such as `classifier0__max_depth` for GridSearchCV.
- LabelEncoder mapping back to original string/nonconsecutive labels.
- Deterministic ties choosing the smallest encoded class.
- Weighted classlabel/probability votes, validated finite nonnegative weights,
  and probability-column alignment by each base estimator's classes_.
- Feature-count checks and a clear unfitted-estimator error.

Eleven offline tests exercise these behaviours, including CV and nested grid
search. See [REMEDIATION.md](REMEDIATION.md) for observed commands and limits.

Based on Chapter 7 of Raschka and Mirjalili's *Python Machine Learning*.
