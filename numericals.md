# NPTEL Deep Learning (CS7015) — Comprehensive Solved Numericals Master-Guide

This document provides fully solved, step-by-step numerical problems designed specifically for the **NPTEL Deep Learning exam**. It covers all core mathematical patterns: forward propagation, backpropagation, weight updates, optimizer steps, CNN dimensions, parameter counts, Batch Normalization, Attention, SVD, and PCA.

---

## Table of Contents
1. [Feedforward Neural Networks (Forward Pass, Loss, Backpropagation & Weight Updates)](#1-feedforward-neural-networks)
2. [Softmax & Categorical Cross-Entropy Gradients](#2-softmax--categorical-cross-entropy)
3. [Optimization Algorithms (Momentum, NAG, AdaGrad, RMSProp, Adam)](#3-optimization-algorithms)
4. [Perceptron Learning Algorithm (PLA) & Novikoff's Bound](#4-perceptron-learning-algorithm--novikoffs-bound)
5. [Linear Algebra Foundations: PCA, Covariance & SVD](#5-linear-algebra-foundations-pca--svd)
6. [Regularization, Bias-Variance & Dropout Numericals](#6-regularization-bias-variance--dropout)
7. [Weight Initialization (Xavier vs. He) & Batch Normalization](#7-weight-initialization--batch-normalization)
8. [Convolutional Neural Networks: Spatial Dimensions, Receptive Fields & Parameters](#8-convolutional-neural-networks)
9. [Recurrent Neural Networks (RNNs, LSTMs, GRUs) & Gradient Clipping](#9-recurrent-neural-networks)
10. [Attention Mechanism & Transformer Calculations](#10-attention-mechanism--transformers)

---

# 1. Feedforward Neural Networks

### Problem 1.1: 1-Hidden-Layer Network with Sigmoid Activations (Full Forward + Backward Pass)

**Network Architecture**:
- Input: $\mathbf{x} = [x_1, x_2]^\top = [1.0, 0.5]^\top$.
- Hidden Layer (2 units):
  - Weights: $\mathbf{W}^{(1)} = \begin{bmatrix} w_{11}^{(1)} & w_{12}^{(1)} \\ w_{21}^{(1)} & w_{22}^{(1)} \end{bmatrix} = \begin{bmatrix} 0.2 & 0.4 \\ -0.3 & 0.5 \end{bmatrix}$
  - Biases: $\mathbf{b}^{(1)} = [b_1^{(1)}, b_2^{(1)}]^\top = [0.1, -0.2]^\top$
  - Activation: Logistic Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- Output Layer (1 unit):
  - Weights: $\mathbf{W}^{(2)} = [w_1^{(2)}, w_2^{(2)}] = [0.7, -0.5]$
  - Bias: $b^{(2)} = 0.3$
  - Output Activation: Linear $\hat{y} = a^{(2)}$
- Ground Truth Target: $y = 1.2$
- Loss Function: Half-Squared Error $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$
- Learning Rate: $\eta = 0.1$

---

#### Step 1: Forward Propagation

1. **Pre-activations of Hidden Layer**:
   $$a_1^{(1)} = w_{11}^{(1)} x_1 + w_{12}^{(1)} x_2 + b_1^{(1)} = 0.2(1.0) + 0.4(0.5) + 0.1 = 0.2 + 0.2 + 0.1 = 0.50$$
   $$a_2^{(1)} = w_{21}^{(1)} x_1 + w_{22}^{(1)} x_2 + b_2^{(1)} = -0.3(1.0) + 0.5(0.5) - 0.2 = -0.3 + 0.25 - 0.2 = -0.25$$

2. **Post-activations of Hidden Layer**:
   $$h_1 = \sigma(0.50) = \frac{1}{1 + e^{-0.50}} = \frac{1}{1 + 0.6065} \approx 0.6225$$
   $$h_2 = \sigma(-0.25) = \frac{1}{1 + e^{0.25}} = \frac{1}{1 + 1.2840} \approx 0.4378$$

3. **Output Layer Pre-activation and Output**:
   $$\hat{y} = a^{(2)} = w_1^{(2)} h_1 + w_2^{(2)} h_2 + b^{(2)}$$
   $$\hat{y} = 0.7(0.6225) + (-0.5)(0.4378) + 0.3 = 0.43575 - 0.2189 + 0.3 = 0.51685$$

4. **Loss Evaluation**:
   $$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0.51685 - 1.2)^2 = \frac{1}{2}(-0.68315)^2 \approx 0.2333$$

---

#### Step 2: Backward Pass (Error Gradients)

1. **Output Gradient**:
   $$\delta^{(2)} = \frac{\partial \mathcal{L}}{\partial a^{(2)}} = \hat{y} - y = 0.51685 - 1.2 = -0.68315$$

2. **Gradients w.r.t Output Layer Parameters**:
   $$\frac{\partial \mathcal{L}}{\partial w_1^{(2)}} = \delta^{(2)} \cdot h_1 = (-0.68315) \times 0.6225 \approx -0.4253$$
   $$\frac{\partial \mathcal{L}}{\partial w_2^{(2)}} = \delta^{(2)} \cdot h_2 = (-0.68315) \times 0.4378 \approx -0.2991$$
   $$\frac{\partial \mathcal{L}}{\partial b^{(2)}} = \delta^{(2)} = -0.68315$$

3. **Backpropagating Error to Hidden Units**:
   Recall $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:
   $$\sigma'(a_1^{(1)}) = h_1(1 - h_1) = 0.6225(1 - 0.6225) = 0.6225 \times 0.3775 \approx 0.2350$$
   $$\sigma'(a_2^{(1)}) = h_2(1 - h_2) = 0.4378(1 - 0.4378) = 0.4378 \times 0.5622 \approx 0.2461$$

   Error signals at hidden pre-activations:
   $$\delta_1^{(1)} = \left( \delta^{(2)} \cdot w_1^{(2)} \right) \times \sigma'(a_1^{(1)}) = [(-0.68315) \times 0.7] \times 0.2350 = (-0.4782) \times 0.2350 \approx -0.1124$$
   $$\delta_2^{(1)} = \left( \delta^{(2)} \cdot w_2^{(2)} \right) \times \sigma'(a_2^{(1)}) = [(-0.68315) \times (-0.5)] \times 0.2461 = (+0.3416) \times 0.2461 \approx +0.0841$$

4. **Gradients w.r.t Hidden Layer Weights**:
   $$\frac{\partial \mathcal{L}}{\partial w_{11}^{(1)}} = \delta_1^{(1)} \cdot x_1 = (-0.1124) \times 1.0 = -0.1124$$
   $$\frac{\partial \mathcal{L}}{\partial w_{12}^{(1)}} = \delta_1^{(1)} \cdot x_2 = (-0.1124) \times 0.5 = -0.0562$$
   $$\frac{\partial \mathcal{L}}{\partial b_1^{(1)}} = \delta_1^{(1)} = -0.1124$$

---

#### Step 3: Parameter Updates ($\eta = 0.1$)

- **Output Layer Weights**:
  $$w_1^{(2)\text{new}} = w_1^{(2)} - \eta \frac{\partial \mathcal{L}}{\partial w_1^{(2)}} = 0.7 - 0.1(-0.4253) = 0.7 + 0.04253 = 0.7425$$
  $$w_2^{(2)\text{new}} = w_2^{(2)} - \eta \frac{\partial \mathcal{L}}{\partial w_2^{(2)}} = -0.5 - 0.1(-0.2991) = -0.5 + 0.02991 = -0.4701$$
  $$b^{(2)\text{new}} = 0.3 - 0.1(-0.68315) = 0.3 + 0.0683 = 0.3683$$

- **Hidden Layer Weights**:
  $$w_{11}^{(1)\text{new}} = 0.2 - 0.1(-0.1124) = 0.2 + 0.01124 = 0.2112$$
  $$w_{12}^{(1)\text{new}} = 0.4 - 0.1(-0.0562) = 0.4 + 0.00562 = 0.4056$$
  $$b_1^{(1)\text{new}} = 0.1 - 0.1(-0.1124) = 0.1112$$

---

### Problem 1.2: 2-Hidden-Layer Network with ReLU Activations

**Problem Statement**:
A neural network has 2 hidden layers with ReLU activations:
$$h_1 = \text{ReLU}(w_1 x + b_1), \quad h_2 = \text{ReLU}(w_2 h_1 + b_2), \quad \hat{y} = w_3 h_2 + b_3$$
Given parameters:
- $w_1 = 2, b_1 = -3$
- $w_2 = -1, b_2 = 4$
- $w_3 = 3, b_3 = 1$
- Input $x = 2$, Target $y = 5$.
- Loss: $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$.

**Questions**:
1. What is the value of $h_1$?
2. What is the value of $h_2$?
3. What is the output $\hat{y}$ and loss $\mathcal{L}$?
4. What is the gradient $\frac{\partial \mathcal{L}}{\partial w_1}$?

**Solution**:
1. **Hidden layer 1**:
   $$z_1 = w_1 x + b_1 = 2(2) - 3 = 1$$
   $$h_1 = \text{ReLU}(1) = \max(0, 1) = 1$$

2. **Hidden layer 2**:
   $$z_2 = w_2 h_1 + b_2 = -1(1) + 4 = 3$$
   $$h_2 = \text{ReLU}(3) = \max(0, 3) = 3$$

3. **Output and Loss**:
   $$\hat{y} = w_3 h_2 + b_3 = 3(3) + 1 = 10$$
   $$\mathcal{L} = \frac{1}{2}(10 - 5)^2 = \frac{1}{2}(25) = 12.5$$

4. **Gradient w.r.t $w_1$ via Chain Rule**:
   $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y = 10 - 5 = 5$$
   Since $z_2 = 3 > 0$, $\text{ReLU}'(z_2) = 1$:
   $$\frac{\partial \mathcal{L}}{\partial h_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot w_3 = 5 \times 3 = 15$$
   $$\frac{\partial \mathcal{L}}{\partial z_2} = 15 \times \text{ReLU}'(z_2) = 15 \times 1 = 15$$
   $$\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial z_2} \cdot w_2 = 15 \times (-1) = -15$$
   Since $z_1 = 1 > 0$, $\text{ReLU}'(z_1) = 1$:
   $$\frac{\partial \mathcal{L}}{\partial z_1} = (-15) \times \text{ReLU}'(z_1) = -15 \times 1 = -15$$
   $$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial z_1} \cdot x = (-15) \times 2 = -30$$

---

# 2. Softmax & Categorical Cross-Entropy

### Problem 2.1: 3-Class Softmax Probabilities and Loss Calculation

**Problem Statement**:
A 3-class classification model produces raw output pre-activations (logits):
$$\mathbf{z} = [z_1, z_2, z_3]^\top = [2.0, 1.0, 0.0]^\top$$
The true class label is class 1 (one-hot vector $\mathbf{y} = [1, 0, 0]^\top$).

**Questions**:
1. Calculate the softmax output vector $\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, \hat{y}_3]^\top$.
2. Compute the Categorical Cross-Entropy loss $\mathcal{L}$.
3. Calculate the gradient vector w.r.t logits $\nabla_{\mathbf{z}} \mathcal{L}$.

**Solution**:
1. **Softmax Output**:
   $$e^{z_1} = e^{2.0} \approx 7.389$$
   $$e^{z_2} = e^{1.0} \approx 2.718$$
   $$e^{z_3} = e^{0.0} = 1.000$$
   $$\sum_{j=1}^3 e^{z_j} = 7.389 + 2.718 + 1.000 = 11.107$$

   $$\hat{y}_1 = \frac{7.389}{11.107} \approx 0.6653 \quad (66.5\%)$$
   $$\hat{y}_2 = \frac{2.718}{11.107} \approx 0.2447 \quad (24.5\%)$$
   $$\hat{y}_3 = \frac{1.000}{11.107} \approx 0.0900 \quad (9.0\%)$$
   *(Check: $0.6653 + 0.2447 + 0.0900 = 1.0000$)*.

2. **Categorical Cross-Entropy Loss**:
   $$\mathcal{L} = -\sum_{c=1}^3 y_c \log(\hat{y}_c) = -\log(\hat{y}_1) = -\ln(0.6653) \approx -(-0.4075) = 0.4075$$

3. **Gradients w.r.t. Logits**:
   By the standard Softmax + Cross-Entropy simplification: $\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$.
   $$\frac{\partial \mathcal{L}}{\partial z_1} = \hat{y}_1 - 1 = 0.6653 - 1 = -0.3347$$
   $$\frac{\partial \mathcal{L}}{\partial z_2} = \hat{y}_2 - 0 = 0.2447$$
   $$\frac{\partial \mathcal{L}}{\partial z_3} = \hat{y}_3 - 0 = 0.0900$$
   $$\nabla_{\mathbf{z}} \mathcal{L} = [-0.3347, 0.2447, 0.0900]^\top$$
   *(Exam Verification Tip: The sum of gradients w.r.t all logits is identically zero: $\sum (\hat{y}_i - y_i) = 1 - 1 = 0$!)*

---

# 3. Optimization Algorithms

### Problem 3.1: Complete Comparative Trace of GD, Momentum, AdaGrad, RMSProp, and Adam

**Setup**:
Consider a single scalar parameter $w$.
- Initial parameter: $w_0 = 2.0$.
- Learning rate: $\eta = 0.1$.
- Sequence of gradients over two successive iterations: $g_1 = 4.0$, $g_2 = 2.0$.

---

#### 1. Standard Gradient Descent
$$w_{t+1} = w_t - \eta g_t$$
- **Step 1 ($g_1 = 4.0$)**:
  $$w_1 = 2.0 - 0.1(4.0) = 2.0 - 0.4 = 1.6$$
- **Step 2 ($g_2 = 2.0$)**:
  $$w_2 = 1.6 - 0.1(2.0) = 1.6 - 0.2 = 1.4$$

---

#### 2. Momentum-Based Gradient Descent ($\beta = 0.9, v_0 = 0$)
$$v_t = \beta v_{t-1} + \eta g_t, \quad w_t = w_{t-1} - v_t$$
- **Step 1 ($g_1 = 4.0$)**:
  $$v_1 = 0.9(0) + 0.1(4.0) = 0.4$$
  $$w_1 = w_0 - v_1 = 2.0 - 0.4 = 1.6$$
- **Step 2 ($g_2 = 2.0$)**:
  $$v_2 = 0.9(v_1) + 0.1(g_2) = 0.9(0.4) + 0.1(2.0) = 0.36 + 0.20 = 0.56$$
  $$w_2 = w_1 - v_2 = 1.6 - 0.56 = 1.04$$
  *(Notice how momentum accelerated the step from $0.4$ to $0.56$ along the consistent direction!)*

---

#### 3. AdaGrad ($\epsilon = 10^{-8}, r_0 = 0$)
$$r_t = r_{t-1} + g_t^2, \quad w_t = w_{t-1} - \frac{\eta}{\sqrt{r_t + \epsilon}} g_t$$
- **Step 1 ($g_1 = 4.0$)**:
  $$r_1 = 0 + (4.0)^2 = 16.0 \implies \sqrt{r_1} = 4.0$$
  $$w_1 = 2.0 - \frac{0.1}{4.0}(4.0) = 2.0 - 0.1 = 1.9$$
- **Step 2 ($g_2 = 2.0$)**:
  $$r_2 = r_1 + (2.0)^2 = 16.0 + 4.0 = 20.0 \implies \sqrt{r_2} = \sqrt{20} \approx 4.472$$
  $$w_2 = 1.9 - \frac{0.1}{4.472}(2.0) = 1.9 - 0.0447 = 1.8553$$

---

#### 4. RMSProp ($\beta = 0.9, v_0 = 0, \epsilon = 10^{-8}$)
$$v_t = \beta v_{t-1} + (1 - \beta) g_t^2, \quad w_t = w_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$
- **Step 1 ($g_1 = 4.0$)**:
  $$v_1 = 0.9(0) + 0.1(4.0)^2 = 0.1(16.0) = 1.60 \implies \sqrt{v_1} = \sqrt{1.6} \approx 1.2649$$
  $$w_1 = 2.0 - \frac{0.1}{1.2649}(4.0) = 2.0 - 0.3162 = 1.6838$$
- **Step 2 ($g_2 = 2.0$)**:
  $$v_2 = 0.9(1.60) + 0.1(2.0)^2 = 1.44 + 0.40 = 1.84 \implies \sqrt{v_2} = \sqrt{1.84} \approx 1.3565$$
  $$w_2 = 1.6838 - \frac{0.1}{1.3565}(2.0) = 1.6838 - 0.1474 = 1.5364$$

---

#### 5. Adam with Bias Correction ($\beta_1 = 0.9, \beta_2 = 0.999, \eta = 0.1, \epsilon = 10^{-8}, m_0 = 0, v_0 = 0$)
- **Step 1 ($t = 1, g_1 = 4.0$)**:
  1. Biased 1st moment: $m_1 = 0.9(0) + (1 - 0.9)(4.0) = 0.4$
  2. Biased 2nd moment: $v_1 = 0.999(0) + (1 - 0.999)(4.0)^2 = 0.001(16.0) = 0.016$
  3. **Bias Corrections**:
     $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{0.4}{1 - 0.9} = \frac{0.4}{0.1} = 4.0$$
     $$\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{0.016}{1 - 0.999} = \frac{0.016}{0.001} = 16.0$$
  4. Parameter update:
     $$w_1 = w_0 - \frac{\eta}{\sqrt{\hat{v}_1} + \epsilon} \hat{m}_1 = 2.0 - \frac{0.1}{\sqrt{16.0}}(4.0) = 2.0 - \frac{0.1}{4.0}(4.0) = 2.0 - 0.1 = 1.9$$

---

# 4. Perceptron Learning Algorithm & Novikoff's Bound

### Problem 4.1: Full Step-by-Step Execution of PLA

**Dataset** (without bias, $b = 0$, update when $y(\mathbf{w}^\top \mathbf{x}) \le 0$):
- $P_1: \mathbf{x}_1 = [2, 1]^\top, y_1 = +1$
- $P_2: \mathbf{x}_2 = [1, 3]^\top, y_2 = +1$
- $P_3: \mathbf{x}_3 = [-1, -2]^\top, y_3 = -1$
- Initial weight vector: $\mathbf{w}^{(0)} = [0, 0]^\top$.

**Execution Trace**:
- **Epoch 1**:
  - **Sample $P_1$**: $\mathbf{w}^{(0)\top} \mathbf{x}_1 = 0(2) + 0(1) = 0$.
    Check: $y_1(\mathbf{w}^\top \mathbf{x}_1) = 1(0) = 0 \le 0 \implies \mathbf{Update!}$
    $$\mathbf{w}^{(1)} = \mathbf{w}^{(0)} + y_1 \mathbf{x}_1 = [0, 0]^\top + 1[2, 1]^\top = [2, 1]^\top$$
  - **Sample $P_2$**: $\mathbf{w}^{(1)\top} \mathbf{x}_2 = 2(1) + 1(3) = 5$.
    Check: $y_2(\mathbf{w}^\top \mathbf{x}_2) = 1(5) = 5 > 0 \implies \mathbf{No\ update}$.
  - **Sample $P_3$**: $\mathbf{w}^{(1)\top} \mathbf{x}_3 = 2(-1) + 1(-2) = -4$.
    Check: $y_3(\mathbf{w}^\top \mathbf{x}_3) = (-1)(-4) = +4 > 0 \implies \mathbf{No\ update}$.

- **Result**: Convergence in **1 update**! Final separating weight vector is $\mathbf{w}^* = [2, 1]^\top$.

---

### Problem 4.2: Novikoff's Convergence Bound Calculation

**Problem Statement**:
A linearly separable 2D dataset has a maximum sample radius $R = \max_i \|\mathbf{x}_i\|_2 = 5$.
The true separating unit hyperplane $\mathbf{w}^*$ ($\|\mathbf{w}^*\|_2 = 1$) achieves a geometric functional margin:
$$\gamma = \min_i y_i (\mathbf{w}^{*\top} \mathbf{x}_i) = 0.5$$

**Questions**:
1. What is the theoretical maximum number of weight updates before convergence?
2. If the data points are rescaled so that the margin is doubled to $\gamma = 1.0$ (while $R$ remains $5$), what is the new upper bound?

**Solution**:
1. Using Novikoff's theorem:
   $$k \le \left(\frac{R}{\gamma}\right)^2 = \left(\frac{5}{0.5}\right)^2 = 10^2 = 100 \text{ updates}$$
2. For $\gamma = 1.0$:
   $$k_{\text{new}} \le \left(\frac{5}{1.0}\right)^2 = 5^2 = 25 \text{ updates}$$
   *(Exam Insight: Doubling the margin reduces the worst-case number of updates by a factor of 4!)*

---

# 5. Linear Algebra Foundations: PCA & SVD

### Problem 5.1: Eigenvalues, Eigenvectors & Explained Variance

**Problem Statement**:
A centered dataset of 2 features has the empirical covariance matrix:
$$\Sigma = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$$

**Questions**:
1. Find the eigenvalues $\lambda_1, \lambda_2$ of $\Sigma$.
2. Find the normalized principal eigenvector corresponding to the largest eigenvalue.
3. What percentage of the total variance is explained by the first principal component?

**Solution**:
1. **Eigenvalues**:
   $$\det(\Sigma - \lambda I) = 0 \implies \det\begin{bmatrix} 5 - \lambda & 2 \\ 2 & 2 - \lambda \end{bmatrix} = 0$$
   $$(5 - \lambda)(2 - \lambda) - 4 = 0$$
   $$\lambda^2 - 7\lambda + 10 - 4 = 0 \implies \lambda^2 - 7\lambda + 6 = 0$$
   $$(\lambda - 6)(\lambda - 1) = 0 \implies \mathbf{\lambda_1 = 6, \quad \lambda_2 = 1}$$
   *(Check: $\lambda_1 + \lambda_2 = 7 = \text{Tr}(\Sigma) = 5 + 2$)*.

2. **Principal Eigenvector for $\lambda_1 = 6$**:
   $$(\Sigma - 6I)\mathbf{u}_1 = \mathbf{0} \implies \begin{bmatrix} 5 - 6 & 2 \\ 2 & 2 - 6 \end{bmatrix} \begin{bmatrix} u_x \\ u_y \end{bmatrix} = \begin{bmatrix} -1 & 2 \\ 2 & -4 \end{bmatrix} \begin{bmatrix} u_x \\ u_y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
   $$-u_x + 2u_y = 0 \implies u_x = 2 u_y$$
   Let $u_y = 1 \implies \mathbf{u} = [2, 1]^\top$.
   Normalize to unit length:
   $$\|\mathbf{u}\|_2 = \sqrt{2^2 + 1^2} = \sqrt{5}$$
   $$\mathbf{u}_1 = \begin{bmatrix} \frac{2}{\sqrt{5}} \\ \frac{1}{\sqrt{5}} \end{bmatrix}$$

3. **Percentage of Variance Explained**:
   $$\text{Explained Ratio} = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{6}{6 + 1} = \frac{6}{7} \approx 85.71\%$$

---

### Problem 5.2: Singular Value Decomposition (SVD) Low-Rank Approximation Error

**Problem Statement**:
A matrix $M$ has singular values $\sigma_1 = 8.0, \sigma_2 = 4.0, \sigma_3 = 2.0, \sigma_4 = 1.0$.

**Questions**:
1. What is the rank of matrix $M$?
2. What is the squared Frobenius norm $\|M\|_F^2$?
3. If $M$ is approximated by a rank-2 truncated matrix $M_2$, what is the reconstruction error $\|M - M_2\|_F^2$?
4. What fraction of energy (variance) is retained in $M_2$?

**Solution**:
1. The rank equals the number of non-zero singular values: $\text{rank}(M) = 4$.
2. By the properties of SVD:
   $$\|M\|_F^2 = \sum_{i=1}^4 \sigma_i^2 = 8^2 + 4^2 + 2^2 + 1^2 = 64 + 16 + 4 + 1 = 85$$
3. By the Eckart-Young-Mirsky Theorem, the reconstruction error equals the sum of squares of the discarded singular values:
   $$\|M - M_2\|_F^2 = \sigma_3^2 + \sigma_4^2 = 2^2 + 1^2 = 4 + 1 = 5$$
4. Fraction of retained energy:
   $$\text{Retained} = \frac{\sigma_1^2 + \sigma_2^2}{\|M\|_F^2} = \frac{64 + 16}{85} = \frac{80}{85} \approx 94.12\%$$

---

# 6. Regularization, Bias-Variance & Dropout

### Problem 6.1: Bias-Variance Decomposition Calculation

**Problem Statement**:
At an input query point $x_0$, the true function value is $f(x_0) = 10.0$.
The observational noise has variance $\sigma^2 = 2.0$.
A deep learning model is trained on 5 different random training datasets $\mathcal{D}_1, \dots, \mathcal{D}_5$, producing the following predictions at $x_0$:
$$\hat{f}_1 = 8.0, \quad \hat{f}_2 = 12.0, \quad \hat{f}_3 = 9.0, \quad \hat{f}_4 = 11.0, \quad \hat{f}_5 = 10.0$$

**Questions**:
1. What is the expected prediction $\mathbb{E}[\hat{f}(x_0)]$?
2. What is the model Bias and $\text{Bias}^2$?
3. What is the model Variance?
4. What is the total Expected Test Error $\mathbb{E}[(y - \hat{f}(x_0))^2]$?

**Solution**:
1. **Expected Prediction**:
   $$\mathbb{E}[\hat{f}(x_0)] = \frac{8.0 + 12.0 + 9.0 + 11.0 + 10.0}{5} = \frac{50.0}{5} = 10.0$$
2. **Bias**:
   $$\text{Bias} = \mathbb{E}[\hat{f}(x_0)] - f(x_0) = 10.0 - 10.0 = 0 \implies \mathbf{\text{Bias}^2 = 0}$$
3. **Variance**:
   $$\text{Var} = \frac{1}{5} \sum_{i=1}^5 (\hat{f}_i - 10.0)^2 = \frac{(8-10)^2 + (12-10)^2 + (9-10)^2 + (11-10)^2 + (10-10)^2}{5}$$
   $$\text{Var} = \frac{(-2)^2 + (2)^2 + (-1)^2 + (1)^2 + 0^2}{5} = \frac{4 + 4 + 1 + 1 + 0}{5} = \frac{10}{5} = 2.0$$
4. **Total Expected Test Error**:
   $$\mathbb{E}[(y - \hat{f}(x_0))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2 = 0 + 2.0 + 2.0 = 4.0$$

---

### Problem 6.2: $L_2$ Weight Decay Step

**Problem Statement**:
A neural network parameter has current value $w = 5.0$.
The data loss gradient is $\frac{\partial \mathcal{J}_{\text{data}}}{\partial w} = 2.0$.
Regularization strength is $\lambda = 0.5$, and learning rate is $\eta = 0.1$.
The total objective is $\mathcal{J}_{\text{total}} = \mathcal{J}_{\text{data}} + \frac{\lambda}{2} w^2$.

**Question**:
Compute the updated weight $w_{\text{new}}$ using gradient descent.

**Solution**:
$$\frac{\partial \mathcal{J}_{\text{total}}}{\partial w} = \frac{\partial \mathcal{J}_{\text{data}}}{\partial w} + \lambda w = 2.0 + 0.5(5.0) = 2.0 + 2.5 = 4.5$$
$$w_{\text{new}} = w - \eta \frac{\partial \mathcal{J}_{\text{total}}}{\partial w} = 5.0 - 0.1(4.5) = 5.0 - 0.45 = 4.55$$

*Alternative Weight-Decay Formula Check*:
$$w_{\text{new}} = (1 - \eta \lambda)w - \eta \nabla \mathcal{J}_{\text{data}} = (1 - 0.1 \times 0.5)(5.0) - 0.1(2.0) = (0.95)(5.0) - 0.2 = 4.75 - 0.2 = 4.55$$

---

# 7. Weight Initialization & Batch Normalization

### Problem 7.1: Xavier vs. He Initialization Variance

**Problem Statement**:
Consider a hidden layer with $n_{\text{in}} = 256$ incoming connections and $n_{\text{out}} = 128$ outgoing connections.

**Questions**:
1. If the layer uses **tanh** activations, what is the required weight variance under standard Xavier (Glorot) initialization?
2. If the layer uses **ReLU** activations, what is the required weight variance under He (Kaiming) initialization?
3. If weights are sampled uniformly from $[-a, a]$, what is the value of $a$ for He initialization?

**Solution**:
1. **Xavier Variance**:
   $$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}} = \frac{2}{256 + 128} = \frac{2}{384} \approx 0.00521$$
   *(Or input-only version: $\frac{1}{n_{\text{in}}} = \frac{1}{256} \approx 0.00391$)*.
2. **He Variance**:
   $$\text{Var}(W) = \frac{2}{n_{\text{in}}} = \frac{2}{256} = 0.0078125$$
3. **Uniform Bound $a$ for He Initialization**:
   For a continuous uniform distribution $\mathcal{U}(-a, a)$, variance is $\frac{(2a)^2}{12} = \frac{4a^2}{12} = \frac{a^2}{3}$.
   $$\frac{a^2}{3} = \frac{2}{n_{\text{in}}} \implies a = \sqrt{\frac{6}{n_{\text{in}}}} = \sqrt{\frac{6}{256}} \approx \sqrt{0.0234375} \approx 0.1531$$

---

### Problem 7.2: Full Batch Normalization Calculation

**Problem Statement**:
A mini-batch of 4 activations for a single neuron is:
$$\mathcal{B} = \{2.0, 4.0, 6.0, 8.0\}$$
Learnable parameters: scale $\gamma = 1.5$, shift $\beta = 0.5$. Stability constant $\epsilon = 0$.

**Step-by-Step Calculation**:
1. **Mini-batch Mean ($\mu_{\mathcal{B}}$)**:
   $$\mu_{\mathcal{B}} = \frac{2.0 + 4.0 + 6.0 + 8.0}{4} = \frac{20.0}{4} = 5.0$$
2. **Mini-batch Variance ($\sigma_{\mathcal{B}}^2$)**:
   $$\sigma_{\mathcal{B}}^2 = \frac{1}{4} \sum_{i=1}^4 (x_i - 5.0)^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4}$$
   $$\sigma_{\mathcal{B}}^2 = \frac{(-3)^2 + (-1)^2 + (1)^2 + (3)^2}{4} = \frac{9 + 1 + 1 + 9}{4} = \frac{20}{4} = 5.0$$
   $$\sigma_{\mathcal{B}} = \sqrt{5.0} \approx 2.236$$
3. **Normalized Activations ($\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sigma_{\mathcal{B}}}$)**:
   $$\hat{x}_1 = \frac{2.0 - 5.0}{2.236} = \frac{-3.0}{2.236} \approx -1.3416$$
   $$\hat{x}_2 = \frac{4.0 - 5.0}{2.236} = \frac{-1.0}{2.236} \approx -0.4472$$
   $$\hat{x}_3 = \frac{6.0 - 5.0}{2.236} = \frac{1.0}{2.236} \approx +0.4472$$
   $$\hat{x}_4 = \frac{8.0 - 5.0}{2.236} = \frac{3.0}{2.236} \approx +1.3416$$
4. **Scale and Shift ($y_i = \gamma \hat{x}_i + \beta$)**:
   $$y_1 = 1.5(-1.3416) + 0.5 = -2.0124 + 0.5 = -1.5124$$
   $$y_2 = 1.5(-0.4472) + 0.5 = -0.6708 + 0.5 = -0.1708$$
   $$y_3 = 1.5(+0.4472) + 0.5 = +0.6708 + 0.5 = +1.1708$$
   $$y_4 = 1.5(+1.3416) + 0.5 = +2.0124 + 0.5 = +2.5124$$

---

# 8. Convolutional Neural Networks

### Problem 8.1: Multi-Layer Spatial Dimensions and Parameter Count

**Architecture**:
- **Input**: RGB Image of size $224 \times 224 \times 3$.
- **Layer 1 (Conv1)**: 64 filters of size $7 \times 7$, Stride $S = 2$, Padding $P = 3$.
- **Layer 2 (Pool1)**: $3 \times 3$ Max Pooling, Stride $S = 2$, Padding $P = 0$.
- **Layer 3 (Conv2)**: 128 filters of size $3 \times 3$, Stride $S = 1$, Padding $P = 1$.
- **Layer 4 (GAP)**: Global Average Pooling.
- **Layer 5 (FC)**: Fully connected output layer to 10 classes.

---

#### Step 1: Spatial Dimension Tracking
$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

1. **After Conv1**:
   $$O_{\text{conv1}} = \left\lfloor \frac{224 - 7 + 2(3)}{2} \right\rfloor + 1 = \left\lfloor \frac{224 - 7 + 6}{2} \right\rfloor + 1 = \left\lfloor \frac{223}{2} \right\rfloor + 1 = 111 + 1 = 112$$
   Feature Tensor: $\mathbf{112 \times 112 \times 64}$.
2. **After Pool1**:
   $$O_{\text{pool1}} = \left\lfloor \frac{112 - 3 + 2(0)}{2} \right\rfloor + 1 = \left\lfloor \frac{109}{2} \right\rfloor + 1 = 54 + 1 = 55$$
   Feature Tensor: $\mathbf{55 \times 55 \times 64}$.
3. **After Conv2**:
   $$O_{\text{conv2}} = \left\lfloor \frac{55 - 3 + 2(1)}{1} \right\rfloor + 1 = \frac{54}{1} + 1 = 55$$
   Feature Tensor: $\mathbf{55 \times 55 \times 128}$.
4. **After GAP**:
   Collapses each $55 \times 55$ feature slice into 1 value $\implies \mathbf{1 \times 1 \times 128}$ (a 128-dim vector).

---

#### Step 2: Parameter Count Calculations

$$\text{Params} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$$

1. **Conv1**:
   $$\text{Weights} = 7 \times 7 \times 3 \times 64 = 9,408$$
   $$\text{Biases} = 64$$
   $$\text{Total Conv1 Params} = 9,408 + 64 = \mathbf{9,472}$$
2. **Pool1**:
   $$\text{Params} = \mathbf{0}$$
3. **Conv2**:
   $$\text{Weights} = 3 \times 3 \times 64 \times 128 = 73,728$$
   $$\text{Biases} = 128$$
   $$\text{Total Conv2 Params} = 73,728 + 128 = \mathbf{73,856}$$
4. **GAP**:
   $$\text{Params} = \mathbf{0}$$
5. **FC Output Layer (128 inputs $\to$ 10 outputs)**:
   $$\text{Weights} = 128 \times 10 = 1,280$$
   $$\text{Biases} = 10$$
   $$\text{Total FC Params} = 1,280 + 10 = \mathbf{1,290}$$

**Total Trainable Network Parameters**:
$$9,472 + 0 + 73,856 + 0 + 1,290 = \mathbf{84,618}$$

---

### Problem 8.2: Effective Receptive Field of Stacked Convolutions

**Problem Statement**:
Calculate the effective receptive field (in pixels) on the input image for:
1. Two stacked $3 \times 3$ convolutional layers (both with stride 1).
2. Three stacked $3 \times 3$ convolutional layers (all with stride 1).

**Formula**:
$$RF_{l} = RF_{l-1} + (K_l - 1) \times S_{\text{cum}}$$
where $S_{\text{cum}}$ is the cumulative stride of all preceding layers.

**Calculations**:
1. **Layer 1 ($3 \times 3$)**:
   $$RF_1 = 3 \times 3$$
2. **Layer 2 ($3 \times 3$, stride 1)**:
   $$RF_2 = 3 + (3 - 1) \times 1 = 3 + 2 = 5 \implies \mathbf{5 \times 5}$$
3. **Layer 3 ($3 \times 3$, stride 1)**:
   $$RF_3 = 5 + (3 - 1) \times 1 = 5 + 2 = 7 \implies \mathbf{7 \times 7}$$

---

# 9. Recurrent Neural Networks

### Problem 9.1: Vanilla RNN Parameter Count vs. LSTM vs. GRU

**Problem Statement**:
Given:
- Input vector dimension: $d_x = 100$.
- Hidden state dimension: $d_h = 200$.
- Output vocabulary size: $d_y = 5000$.

**Calculate total parameters for each recurrent architecture**:

#### 1. Vanilla RNN:
- Input-to-hidden: $\mathbf{U} \in \mathbb{R}^{200 \times 100} \implies 20,000$
- Hidden-to-hidden: $\mathbf{W} \in \mathbb{R}^{200 \times 200} \implies 40,000$
- Hidden bias: $\mathbf{b} \in \mathbb{R}^{200} \implies 200$
- Hidden-to-output: $\mathbf{V} \in \mathbb{R}^{5000 \times 200} \implies 1,000,000$
- Output bias: $\mathbf{c} \in \mathbb{R}^{5000} \implies 5,000$
- **Total Recurrent Cell Params**: $20,000 + 40,000 + 200 = \mathbf{60,200}$
- **Total Model Params**: $60,200 + 1,005,000 = \mathbf{1,065,200}$

#### 2. LSTM (4 internal gates: $f, i, \tilde{c}, o$):
Each gate has identical matrix sizes: $(d_h \times d_x) + (d_h \times d_h) + d_h = 60,200$.
- **Total LSTM Cell Params**:
  $$4 \times 60,200 = \mathbf{240,800}$$
- **Total Model Params** (with output layer):
  $$240,800 + 1,005,000 = \mathbf{1,245,800}$$

#### 3. GRU (3 internal gates: $r, z, \tilde{h}$):
- **Total GRU Cell Params**:
  $$3 \times 60,200 = \mathbf{180,600}$$
- **Total Model Params** (with output layer):
  $$180,600 + 1,005,000 = \mathbf{1,185,600}$$

---

### Problem 9.2: Gradient Clipping Calculation

**Problem Statement**:
During BPTT, the computed gradient vector across all parameters is:
$$\mathbf{g} = [6.0, -8.0, 0.0]^\top$$
The maximum gradient norm threshold is set to $\theta = 5.0$.

**Question**:
Compute the clipped gradient vector $\mathbf{g}_{\text{clipped}}$.

**Solution**:
1. Compute Euclidean norm of $\mathbf{g}$:
   $$\|\mathbf{g}\|_2 = \sqrt{6.0^2 + (-8.0)^2 + 0^2} = \sqrt{36 + 64 + 0} = \sqrt{100} = 10.0$$
2. Since $\|\mathbf{g}\|_2 = 10.0 > \theta = 5.0$, clipping is triggered:
   $$\mathbf{g}_{\text{clipped}} = \frac{\theta}{\|\mathbf{g}\|_2} \mathbf{g} = \frac{5.0}{10.0} \begin{bmatrix} 6.0 \\ -8.0 \\ 0.0 \end{bmatrix} = 0.5 \begin{bmatrix} 6.0 \\ -8.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 3.0 \\ -4.0 \\ 0.0 \end{bmatrix}$$
3. Verification:
   $$\|\mathbf{g}_{\text{clipped}}\|_2 = \sqrt{3^2 + (-4)^2} = \sqrt{25} = 5.0 = \theta$$

---

# 10. Attention Mechanism & Transformers

### Problem 10.1: Context Vector Computation in Seq2Seq Attention

**Problem Statement**:
In an English-to-Hindi translation model, the source sentence has 3 words with encoder hidden states:
$$\mathbf{h}_1 = [1, 0]^\top, \quad \mathbf{h}_2 = [0, 2]^\top, \quad \mathbf{h}_3 = [1, 1]^\top$$
At the current decoding step, the decoder hidden state is $\mathbf{s}_{t-1} = [2, 1]^\top$.
The alignment score is the dot product $e_i = \mathbf{s}_{t-1}^\top \mathbf{h}_i$.

**Questions**:
1. Calculate the raw alignment scores $e_1, e_2, e_3$.
2. Compute the attention weights $\alpha_1, \alpha_2, \alpha_3$ using softmax.
3. Compute the synthesized context vector $\mathbf{c}_t$.

**Solution**:
1. **Raw Dot-Product Scores**:
   $$e_1 = [2, 1] \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 2(1) + 1(0) = 2$$
   $$e_2 = [2, 1] \begin{bmatrix} 0 \\ 2 \end{bmatrix} = 2(0) + 1(2) = 2$$
   $$e_3 = [2, 1] \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 2(1) + 1(1) = 3$$

2. **Softmax Attention Weights**:
   $$e^{e_1} = e^2 \approx 7.389$$
   $$e^{e_2} = e^2 \approx 7.389$$
   $$e^{e_3} = e^3 \approx 20.086$$
   $$\sum_{i=1}^3 e^{e_i} = 7.389 + 7.389 + 20.086 = 34.864$$

   $$\alpha_1 = \frac{7.389}{34.864} \approx 0.2119 \quad (21.2\%)$$
   $$\alpha_2 = \frac{7.389}{34.864} \approx 0.2119 \quad (21.2\%)$$
   $$\alpha_3 = \frac{20.086}{34.864} \approx 0.5761 \quad (57.6\%)$$
   *(Check: $0.2119 + 0.2119 + 0.5761 = 1.0000$)*.

3. **Context Vector**:
   $$\mathbf{c}_t = \alpha_1 \mathbf{h}_1 + \alpha_2 \mathbf{h}_2 + \alpha_3 \mathbf{h}_3$$
   $$\mathbf{c}_t = 0.2119 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.2119 \begin{bmatrix} 0 \\ 2 \end{bmatrix} + 0.5761 \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
   $$\mathbf{c}_t = \begin{bmatrix} 0.2119 + 0 + 0.5761 \\ 0 + 0.4238 + 0.5761 \end{bmatrix} = \begin{bmatrix} 0.7880 \\ 0.9999 \end{bmatrix} \approx \begin{bmatrix} 0.79 \\ 1.00 \end{bmatrix}$$

---

### Problem 10.2: Scaled Dot-Product Self-Attention Calculation

**Problem Statement**:
Given Query vector $\mathbf{q}$ and two Key vectors $\mathbf{k}_1, \mathbf{k}_2$ with key dimension $d_k = 4$:
$$\mathbf{q} = [2, 0, 2, 0]^\top$$
$$\mathbf{k}_1 = [1, 1, 1, 1]^\top, \quad \mathbf{k}_2 = [2, -1, 0, 1]^\top$$
Value vectors:
$$\mathbf{v}_1 = [10, 0]^\top, \quad \mathbf{v}_2 = [0, 20]^\top$$

**Calculate the self-attention output vector**:

1. **Dot Products**:
   $$\mathbf{q}^\top \mathbf{k}_1 = 2(1) + 0(1) + 2(1) + 0(1) = 2 + 2 = 4$$
   $$\mathbf{q}^\top \mathbf{k}_2 = 2(2) + 0(-1) + 2(0) + 0(1) = 4 + 0 = 4$$

2. **Scaling by $\sqrt{d_k} = \sqrt{4} = 2$**:
   $$s_1 = \frac{\mathbf{q}^\top \mathbf{k}_1}{\sqrt{d_k}} = \frac{4}{2} = 2.0$$
   $$s_2 = \frac{\mathbf{q}^\top \mathbf{k}_2}{\sqrt{d_k}} = \frac{4}{2} = 2.0$$

3. **Softmax Weights**:
   Since $s_1 = s_2 = 2.0$, by symmetry:
   $$\alpha_1 = 0.5, \quad \alpha_2 = 0.5$$

4. **Attention Output**:
   $$\text{Output} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 = 0.5 \begin{bmatrix} 10 \\ 0 \end{bmatrix} + 0.5 \begin{bmatrix} 0 \\ 20 \end{bmatrix} = \begin{bmatrix} 5.0 \\ 10.0 \end{bmatrix}$$
