# Backpropagation in Neural Networks

## 1. What is Backpropagation?

Backpropagation is an algorithm used to train Artificial Neural Networks (ANNs) and Deep Neural Networks (DNNs).

It calculates how much each weight and bias contributes to the prediction error and computes gradients to update them.

**Main purpose:**

- Calculate gradients of the loss function.
- Understand how much each weight contributes to the error.
- Update weights and biases using an optimizer such as Gradient Descent.
- Minimize the loss function.

### Real-Life Example

Imagine a neural network trained to classify cats and dogs.

- Input: An image of a cat.
- Prediction: Dog (incorrect).
- Actual output: Cat.
- Loss: High prediction error.
- Backpropagation: Calculates how the weights contributed to this error.
- Weight update: Adjusts weights to improve future predictions.

---

## 2. How Backpropagation Works

Backpropagation works through two main stages:

### A. Forward Propagation

Information moves from the input layer towards the output layer.

Steps:

1. Input data enters the neural network.
2. Each neuron calculates weighted sums.
3. Activation functions are applied.
4. The network generates a prediction.
5. Loss function calculates prediction error.

Formula:

```text
z = wx + b

a = activation(z)
```

Where:

- w = Weight
- x = Input
- b = Bias
- z = Weighted sum
- a = Neuron activation/output

### B. Backward Propagation

Information flows backward from the output layer towards the input layer.

Steps:

1. Calculate the loss.
2. Calculate gradients using the chain rule.
3. Determine gradients for weights and biases.
4. Send gradients backward through the network.
5. Update parameters using an optimizer.

---

## 3. Chain Rule

The chain rule is the mathematical foundation of backpropagation.

It helps calculate how a change in one variable affects the final loss through multiple connected operations.

For example:

```text
x → z → a → Loss
```

The derivative is:

```text
dL/dx = (dL/da) × (da/dz) × (dz/dx)
```

This allows the network to calculate gradients layer by layer.

---

## 4. Important Backpropagation Formulas

### Weight Gradient

```text
∂L/∂w
```

Measures how much the loss changes when a weight changes.

### Bias Gradient

```text
∂L/∂b
```

Measures how much the loss changes when a bias changes.

### Gradient Descent Update Rule

```text
w_new = w_old - learning_rate × (∂L/∂w)

b_new = b_old - learning_rate × (∂L/∂b)
```

Where:

- w = Weight
- b = Bias
- Learning Rate = Controls the size of parameter updates.
- Gradient = Direction and rate of loss change.

---

## 5. Numerical Example

Suppose:

```text
Weight = 2
Input = 3
Bias = 1
Target = 10
Learning Rate = 0.1
```

### Step 1: Forward Propagation

```text
Prediction = (Weight × Input) + Bias

Prediction = (2 × 3) + 1

Prediction = 7
```

### Step 2: Calculate Loss

Using Mean Squared Error for one sample:

```text
Loss = (Target - Prediction)²

Loss = (10 - 7)²

Loss = 9
```

### Step 3: Calculate Gradients

For:

```text
Loss = (Prediction - Target)²
Prediction = wx + b
```

Weight gradient:

```text
dL/dw = 2 × (Prediction - Target) × Input

dL/dw = 2 × (7 - 10) × 3

dL/dw = -18
```

Bias gradient:

```text
dL/db = 2 × (Prediction - Target)

dL/db = -6
```

### Step 4: Update Parameters

Weight:

```text
New Weight = 2 - (0.1 × -18)

New Weight = 3.8
```

Bias:

```text
New Bias = 1 - (0.1 × -6)

New Bias = 1.6
```

The weights and bias have been updated to reduce the prediction error.

---

## 6. Backpropagation in Python

```python
import numpy as np

# Initial parameters
weight = 2.0
bias = 1.0

# Training data
x = 3.0
target = 10.0

# Learning rate
lr = 0.1

# Forward propagation
prediction = weight * x + bias

# Calculate loss
loss = (prediction - target) ** 2

# Backward propagation
dw = 2 * (prediction - target) * x
db = 2 * (prediction - target)

# Update parameters
weight = weight - lr * dw
bias = bias - lr * db

print("Loss:", loss)
print("Updated Weight:", weight)
print("Updated Bias:", bias)
```

---

## 7. Backpropagation vs Gradient Descent

| Backpropagation                       | Gradient Descent          |
| ------------------------------------- | ------------------------- |
| Calculates gradients                  | Updates parameters        |
| Uses chain rule                       | Uses learning rate        |
| Determines how parameters affect loss | Minimizes the loss        |
| Computes derivatives                  | Applies parameter updates |

**Important:** Backpropagation calculates gradients, while Gradient Descent uses those gradients to update weights and biases.

---

## 8. Backpropagation in Deep Learning Frameworks

In PyTorch, backpropagation is performed using:

```python
loss.backward()
```

This calculates gradients for model parameters.

To update parameters:

```python
optimizer.step()
```

To clear previously calculated gradients:

```python
optimizer.zero_grad()
```

Typical PyTorch training sequence:

```python
optimizer.zero_grad()

prediction = model(X)

loss = loss_function(prediction, y)

loss.backward()

optimizer.step()
```

---

## 9. Common Problems in Backpropagation

### A. Vanishing Gradient

Gradients become extremely small as they move backward through deep networks.

**Effects:**

- Earlier layers learn very slowly.
- Training becomes difficult.

**Solutions:**

- ReLU activation.
- Proper weight initialization.
- Batch Normalization.
- Residual connections.

### B. Exploding Gradient

Gradients become excessively large during backpropagation.

**Effects:**

- Unstable training.
- Very large weight updates.
- Loss may become NaN.

**Solutions:**

- Gradient Clipping.
- Proper weight initialization.
- Suitable learning rate.

---

## 10. Key Takeaways

1. Backpropagation is used to train neural networks.
2. Forward propagation generates predictions.
3. Loss functions calculate prediction errors.
4. Backpropagation calculates gradients.
5. Chain Rule is the mathematical foundation.
6. Gradient Descent updates weights and biases.
7. The overall objective is to minimize the loss function.

### Next Topics to Learn

- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent
- Learning Rate
- Optimizers (Adam, RMSprop, SGD)
- Vanishing and Exploding Gradients
- Automatic Differentiation in PyTorch
