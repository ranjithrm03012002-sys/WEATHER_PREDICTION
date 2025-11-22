"""
Logistic Regression from scratch (NumPy) + comparison with scikit-learn.

- Generates synthetic binary dataset (500 samples, 8 features, class_sep=0.85)
- Implements logistic regression with L2 regularization and batch gradient descent
- Tracks cost history for up to 10,000 iterations or until convergence
- Compares with scikit-learn's LogisticRegression on same scaled data
- Prints performance metrics and interprets feature coefficients
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression as SklearnLogistic
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Utility functions
# ---------------------------
def sigmoid(z):
    """Numerically-stable sigmoid."""
    # For large positive z, exp(-z) -> 0 fine; for large negative z, exp(-z) large but computation stays stable in numpy
    return 1.0 / (1.0 + np.exp(-z))

def safe_log(x, eps=1e-15):
    """Clip probabilities for numerical stability before log."""
    x = np.clip(x, eps, 1 - eps)
    return np.log(x)

# ---------------------------
# Logistic Regression (from scratch)
# ---------------------------
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=10000, reg_lambda=0.1, tol=1e-9, verbose=False):
        """
        lr: learning rate
        n_iter: max iterations
        reg_lambda: L2 regularization strength (lambda)
        tol: tolerance for early stopping based on cost change
        """
        self.lr = lr
        self.n_iter = n_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.verbose = verbose
        self.w = None
        self.b = None
        self.cost_history = []

    def _compute_cost(self, X, y):
        n = X.shape[0]
        z = X.dot(self.w) + self.b
        p = sigmoid(z)
        # Negative log-likelihood (binary cross-entropy)
        loss = -np.mean(y * safe_log(p) + (1 - y) * safe_log(1 - p))
        # L2 regularization on weights (not bias)
        reg = (self.reg_lambda / (2.0 * n)) * np.sum(self.w ** 2)
        return loss + reg

    def fit(self, X, y):
        """
        Train using batch gradient descent. X should be (n_samples, n_features).
        y should be {0,1}.
        """
        n, m = X.shape
        # Initialize weights
        self.w = np.zeros(m, dtype=float)
        self.b = 0.0
        self.cost_history = []

        prev_cost = None
        for it in range(1, self.n_iter + 1):
            z = X.dot(self.w) + self.b
            p = sigmoid(z)
            # Gradients
            dw = (1.0 / n) * (X.T.dot(p - y)) + (self.reg_lambda / n) * self.w
            db = (1.0 / n) * np.sum(p - y)
            # Parameter update
            self.w -= self.lr * dw
            self.b -= self.lr * db
            # Cost
            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)

            # Verbose
            if self.verbose and (it % 500 == 0 or it == 1):
                print(f"Iter {it:5d} | Cost: {cost:.6f}")

            # Early stopping condition
            if prev_cost is not None and abs(prev_cost - cost) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {it} with cost change {abs(prev_cost - cost):.2e}")
                break
            prev_cost = cost

        self.n_iter_run = it
        return self

    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------
# Generate dataset (user specified)
# 500 samples, 8 features, class separation 0.85
# ---------------------------
RANDOM_STATE = 42
n_samples = 500
n_features = 8
class_sep = 0.85  # class separation (controls difficulty)

X, y = make_classification(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=6,
                           n_redundant=2,
                           n_clusters_per_class=1,
                           class_sep=class_sep,
                           flip_y=0.01,
                           random_state=RANDOM_STATE)

# Standardize features using NumPy (important for gradient descent and fair comparison)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
X_std[X_std == 0] = 1.0  # avoid division by zero
X_scaled = (X - X_mean) / X_std

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# ---------------------------
# Train custom logistic regression
# ---------------------------
reg_lambda = 0.1      # L2 regularization strength
learning_rate = 0.5   # initial learning rate — tuned for convergence speed
max_iters = 10000
tol = 1e-9

custom_clf = LogisticRegressionScratch(lr=learning_rate, n_iter=max_iters, reg_lambda=reg_lambda, tol=tol, verbose=True)
custom_clf.fit(X_train, y_train)

# ---------------------------
# Train scikit-learn logistic regression (for comparison)
# ---------------------------
# Note: scikit-learn's C = 1 / (lambda * n_samples) if one wants exact equivalence depending on implementation.
# We'll set C = 1 / reg_lambda for practical comparison (common mapping).
sk_clf = SklearnLogistic(penalty='l2', C=1.0 / reg_lambda, solver='lbfgs', max_iter=10000)
sk_clf.fit(X_train, y_train)

# ---------------------------
# Evaluate both models
# ---------------------------
def evaluate_model(model, X_test, y_test, name="model"):
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    else:
        # our custom model's predict_proba
        y_prob = model.predict_proba(X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'y_pred': y_pred, 'y_prob': y_prob}

print("\nEvaluating custom logistic regression (NumPy implementation):")
custom_metrics = evaluate_model(custom_clf, X_test, y_test, name="Custom LogisticRegression (scratch)")

print("\nEvaluating scikit-learn logistic regression:")
# sklearn's predict_proba returns shape (n_samples, 2), second column is prob for class 1
sk_prob = sk_clf.predict_proba(X_test)[:, 1]
# adapt to same interface for printing
class SkWrapper:
    def __init__(self, prob_array):
        self.prob = prob_array
    def predict_proba(self, X):
        return self.prob  # ignores X - we pass precomputed probs below

# For evaluation function, we want a wrapper because it queries predict_proba(X_test)
# But easier: call evaluate_model-like code directly:
y_pred_sk = (sk_prob >= 0.5).astype(int)
acc_sk = accuracy_score(y_test, y_pred_sk)
prec_sk = precision_score(y_test, y_pred_sk, zero_division=0)
rec_sk = recall_score(y_test, y_pred_sk, zero_division=0)
f1_sk = f1_score(y_test, y_pred_sk, zero_division=0)
print(f"--- scikit-learn ---")
print(f"Accuracy : {acc_sk:.4f}")
print(f"Precision: {prec_sk:.4f}")
print(f"Recall   : {rec_sk:.4f}")
print(f"F1-score : {f1_sk:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred_sk, digits=4))

# ---------------------------
# Plot training cost history
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(custom_clf.cost_history) + 1), custom_clf.cost_history)
plt.xlabel("Iteration")
plt.ylabel("Regularized Cost (loss)")
plt.title("Training cost history (Custom Logistic Regression)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Coefficient interpretation
# ---------------------------
print("\nCoefficient interpretation (on standardized features):")
feature_names = [f"Feature_{i}" for i in range(n_features)]

# Custom model coefficients (w) pertain to standardized features
custom_coefs = custom_clf.w.copy()
custom_bias = custom_clf.b
sk_coefs = sk_clf.coef_.flatten()
sk_intercept = sk_clf.intercept_.item()

print(f"Custom model trained iterations: {custom_clf.n_iter_run}")
print(f"Custom model bias (intercept): {custom_bias:.6f}")
print(f"sklearn model intercept: {sk_intercept:.6f}\n")

print("Feature | Custom_coef | sk_coef | OddsRatio(custom) | Interpretation")
for i, name in enumerate(feature_names):
    w_c = custom_coefs[i]
    w_s = sk_coefs[i]
    odds = np.exp(w_c)
    direction = "increases" if w_c > 0 else ("decreases" if w_c < 0 else "no effect")
    # Magnitude guidance
    magnitude = abs(w_c)
    if magnitude > 1.0:
        strength = "strong"
    elif magnitude > 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    print(f"{name:8s} | {w_c:10.4f} | {w_s:7.4f} | {odds:16.4f} | {direction} probability ({strength})")

# Provide textual interpretation summary
print("\nSummary of coefficient interpretation:")
print(" - Coefficients are learned for STANDARDIZED features (zero mean, unit variance).")
print(" - A positive coefficient means increasing that standardized feature increases the log-odds (and hence probability) of class=1.")
print(" - Odds ratio (exp(coef)) gives multiplicative change in odds for 1 standard-deviation increase in feature.")
print(" - Compare signs and magnitudes between the custom implementation and scikit-learn to validate correctness; they should be close.")

# ---------------------------
# End of script
# ---------------------------
