# Logistic Regression - Detailed Guide

## 1. **What is Logistic Regression?**

Logistic regression is a **statistical model** used for **binary classification** (though it can be extended to multiclass). Despite its name, it's a **classification algorithm**, not regression. It predicts the **probability** that a given input belongs to a particular category.

## 2. **Core Concept: From Linear to Logistic**

### Linear Regression Problem
- Linear regression: `y = β₀ + β₁x₁ + ... + βₙxₙ`
- Outputs continuous values, unbounded
- Not suitable for probability (must be between 0 and 1)

### Solution: The Logistic Function (Sigmoid)
We apply the **sigmoid function** to linear regression output:
```
σ(z) = 1 / (1 + e^{-z})
```
Where `z = β₀ + β₁x₁ + ... + βₙxₙ`

The sigmoid function:
- Maps any real value to (0, 1) range
- S-shaped curve
- Perfect for representing probabilities

## 3. **Mathematical Formulation**

### Probability Output
```
P(y=1|x) = σ(z) = 1 / (1 + e^{-(β₀ + β₁x₁ + ... + βₙxₙ)})
P(y=0|x) = 1 - P(y=1|x)
```

### Odds and Log-Odds
- **Odds**: `P/(1-P)` (ratio of success to failure)
- **Log-Odds (Logit)**: `ln(P/(1-P)) = β₀ + β₁x₁ + ... + βₙxₙ`
- This is the **link function** connecting probability to linear combination of features

## 4. **Cost Function: Cross-Entropy Loss**

We can't use MSE (mean squared error) because:
- Would result in non-convex loss function
- Multiple local minima
- Gradient descent might not find global minimum

### Binary Cross-Entropy Loss
For a single training example:
```
Loss = -[y·log(P) + (1-y)·log(1-P)]
```
Where:
- y = actual label (0 or 1)
- P = predicted probability

For m training examples:
```
J(β) = -(1/m) Σ [yⁱ·log(Pⁱ) + (1-yⁱ)·log(1-Pⁱ)]
```

## 5. **Parameter Estimation**

### Maximum Likelihood Estimation (MLE)
We find parameters β that maximize the likelihood of observing the data:
```
L(β) = Π P(xⁱ)^{yⁱ} · (1-P(xⁱ))^{1-yⁱ}
```

Take log (log-likelihood):
```
ℓ(β) = Σ [yⁱ·log(Pⁱ) + (1-yⁱ)·log(1-Pⁱ)]
```

Maximize ℓ(β) = Minimize -ℓ(β) (which is our loss function)

### Gradient Descent Update Rules
Derivative of sigmoid: `σ'(z) = σ(z)·(1-σ(z))`

Gradient for parameter βⱼ:
```
∂J/∂βⱼ = (1/m) Σ (Pⁱ - yⁱ)·xⱼⁱ
```

Update rule:
```
βⱼ := βⱼ - α·∂J/∂βⱼ
```
Where α is learning rate

## 6. **Types of Logistic Regression**

1. **Binary**: Two classes (spam/not-spam)
2. **Multinomial**: Multiple classes (cat/dog/rabbit)
3. **Ordinal**: Ordered multiple classes (rating 1-5)

## 7. **Implementation Steps**

### Step-by-Step Algorithm
1. Initialize parameters β (usually zeros)
2. Compute predictions: P = σ(Xβ)
3. Compute loss: J(β)
4. Compute gradients: ∇J(β)
5. Update parameters: β = β - α·∇J(β)
6. Repeat until convergence

### Python Pseudocode
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    beta = np.zeros(n)
    
    for epoch in range(epochs):
        # Forward pass
        z = X @ beta
        P = sigmoid(z)
        
        # Compute loss
        loss = -(1/m) * np.sum(y*np.log(P) + (1-y)*np.log(1-P))
        
        # Backward pass
        gradient = (1/m) * X.T @ (P - y)
        
        # Update
        beta -= learning_rate * gradient
        
    return beta
```

## 8. **Decision Boundary**

The decision boundary is where:
```
β₀ + β₁x₁ + ... + βₙxₙ = 0
P(y=1|x) = 0.5
```

For two features:
```
β₀ + β₁x₁ + β₂x₂ = 0
x₂ = -(β₀/β₂) - (β₁/β₂)x₁  # Linear boundary
```

## 9. **Assumptions**

1. Binary output variable
2. No multicollinearity among features
3. Independent observations
4. Linear relationship between log-odds and features
5. Large sample size (for reliable estimates)

## 10. **Evaluation Metrics**

### Confusion Matrix Based:
- Accuracy: `(TP+TN)/Total`
- Precision: `TP/(TP+FP)`
- Recall: `TP/(TP+FN)`
- F1-Score: `2*(Precision*Recall)/(Precision+Recall)`

### Probability Based:
- Log Loss (Cross-Entropy)
- AUC-ROC (Area Under ROC Curve)

## 11. **Regularization**

To prevent overfitting:

### L2 Regularization (Ridge)
Add penalty term to loss:
```
J(β) = CrossEntropyLoss + λΣβⱼ²
```

### L1 Regularization (Lasso)
```
J(β) = CrossEntropyLoss + λΣ|βⱼ|
```

## 12. **Advantages vs. Disadvantages**

### Advantages:
- Simple, interpretable
- Outputs probabilities
- Fast to train
- Doesn't require feature scaling (though helps)
- Works well with linearly separable data

### Disadvantages:
- Assumes linear decision boundary
- Can underperform with complex relationships
- Sensitive to outliers
- Requires large sample size

## 13. **Practical Example**

Let's classify iris flowers as "virginica" or not:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X = iris.data[:, :2]  # Use only first two features
y = (iris.target == 2).astype(int)  # 1 if virginica, else 0

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities

# Interpret coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
```

## 14. **Interpretation of Coefficients**

For coefficient βⱼ:
- Odds ratio: `exp(βⱼ)`
- Interpretation: "A one-unit increase in xⱼ multiplies the odds by exp(βⱼ)"
- Example: If β₁ = 0.5, exp(0.5) ≈ 1.65
  → Each unit increase in x₁ increases odds by 65%

## 15. **Key Considerations**

1. **Feature Engineering**: Logistic regression benefits from good features
2. **Multicollinearity**: Check VIF (Variance Inflation Factor)
3. **Class Imbalance**: Use class_weight parameter or resampling
4. **Threshold Tuning**: Default is 0.5, but adjust based on business needs
5. **Feature Scaling**: Helps convergence speed

## 16. **Extensions**

- **Softmax Regression**: For multiclass classification
- **Kernel Logistic Regression**: For nonlinear boundaries
- **Regularized Variants**: L1, L2, or ElasticNet
- **Bayesian Logistic Regression**: With prior distributions on coefficients

## Practice Exercise

Try this manually with a simple dataset:

| Study Hours (x) | Pass (y) |
|-----------------|----------|
| 1               | 0        |
| 2               | 0        |
| 3               | 1        |
| 4               | 1        |

Predict: Will a student pass if they study 2.5 hours?

