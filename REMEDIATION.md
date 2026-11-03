# Remediation evidence — 3 November 2026

Changed LAB 7.ipynb and README.md. Added majority_vote.py, offline estimator tests,
pinned runtime/dev requirements, pytest.ini, .python-version, .gitignore and
results.json.

The notebook imports the tested custom estimator rather than a divergent copy.
Constructor state is cloneable; nested parameters work with GridSearchCV.
ClassifierMixin precedes BaseEstimator for classifier tags. Hard ties are
deterministic, labels are encoded/recovered, weights are validated and probability
columns are aligned. No private sklearn naming helper is used.

Evaluation separates binary Iris ROC-AUC, full Iris accuracy and binary Wine
accuracy. Each comparison shares its folds. LR/KNN scaling occurs inside
pipelines, including inside voting. Wine loads offline. Boosting curves now
choose any exploratory stopping minimum on an inner training-side validation
partition, rather than test error. The original full-Iris parameter grids remain;
post-selection CV is no longer presented as independent evidence.

## Actual checks

Environment: CPython 3.12.16, scikit-learn 1.9.1 and NumPy 2.5.4.

- `python -m pytest -q`: **11 passed in 3.71s**, including cloning, deep get/set
  parameters, CV/grid search, string/nonconsecutive labels, deterministic ties,
  weighted soft/hard voting, probability alignment and invalid weights.
- `python -m ruff check majority_vote.py tests`: passed.
- `python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "LAB 7.ipynb"`:
  all cells executed, generating real figures and results.json.
- All 12 code cells compiled; saved outputs contained no execution errors.
- `git diff --check`: passed.

README values are copied from the executed machine-readable results. On binary
Iris, majority vote's mean ROC-AUC was 0.9667 ± 0.0408. On full Iris, the baseline
tree and both tuned ensembles tied at 0.9778 held-out accuracy (45 rows).
Therefore 'ensembles consistently outperform' and universal percentage-improvement
claims were removed.

## Limits and safe claims

Only this pinned sklearn version was tested; no broad compatibility matrix or CI
run is claimed. Scores cover small coursework datasets and fixed splits, and
selection CV is optimistic after tuning. No runtime benchmark, universal winner,
optimal ensemble-size rule or deployment was verified. Windows blocked the venv
launcher; the checks used a separate newly provisioned Python interpreter rather
than fully exercising venv activation on this host.

Safe to describe: a tested sklearn-compatible custom voting classifier and
reproducible ensemble comparisons with clearly separated datasets/metrics.
Do not claim production readiness, business gains, general superiority or CI
success.
