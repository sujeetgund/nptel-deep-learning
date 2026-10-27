# Week 3: Feedforward Neural Networks and the Backpropagation Algorithm

---

## 1. Executive Summary & Exam Roadmap

Week 3 explores the operational engine of deep learning:
1. **Multilayer Feedforward Architecture**: Layer-by-layer forward propagation notation ($\mathbf{x} = \mathbf{h}_0$, pre-activations $\mathbf{a}_k$, post-activations $\mathbf{h}_k$).
2. **Output Functions and Loss Functions**:
   - Regression: Linear output activation with Squared Error loss.
   - Binary Classification: Sigmoid activation with Binary Cross-Entropy loss.
   - Multi-class Classification: **Softmax activation** with **Categorical Cross-Entropy loss**.
3. **The Softmax + Cross-Entropy Gradient**: Full mathematical derivation of $\frac{\partial \mathcal{L}}{\partial a_i} = \hat{y}_i - y_i$.
4. **Computational Graphs & The Chain Rule**: Forward pass evaluation followed by backward pass local gradient accumulation.
5. **The Vectorized Backpropagation Algorithm**: Complete derivation of output error vectors, parameter gradients ($\nabla_{\mathbf{W}_k}$, $\nabla_{\mathbf{b}_k}$), and backward message passing ($\nabla_{\mathbf{h}_{k-1}}$, $\nabla_{\mathbf{a}_{k-1}}$).

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 Multilayer Network Architecture and Notation

Consider a deep neural network with $L$ layers ($L-1$ hidden layers and $1$ output layer):
- $\mathbf{x} \in \mathbb{R}^{d}$: Input vector, defined as $\mathbf{h}_0 = \mathbf{x}$.
- For each layer $k \in \{1, 2, \dots, L\}$:
  - $\mathbf{W}_k \in \mathbb{R}^{n_k \times n_{k-1}}$: Weight matrix connecting layer $k-1$ to layer $k$.
  - $\mathbf{b}_k \in \mathbb{R}^{n_k}$: Bias vector for layer $k$.
  - $\mathbf{a}_k \in \mathbb{R}^{n_k}$: **Pre-activation** vector:
    $$\mathbf{a}_k = \mathbf{b}_k + \mathbf{W}_k \mathbf{h}_{k-1}$$
  - $\mathbf{h}_k \in \mathbb{R}^{n_k}$: **Post-activation** vector:
    $$\mathbf{h}_k = g(\mathbf{a}_k)$$
    where $g(\cdot)$ is an element-wise activation function (e.g. sigmoid, tanh, ReLU).
- At the output layer ($k = L$):
  $$\hat{\mathbf{y}} = O(\mathbf{a}_L)$$
  where $O(\cdot)$ is the output activation function.

```
x = h_0 ---> [ W_1, b_1 ] ---> a_1 ---> g(.) ---> h_1 ... ---> a_L ---> O(.) ---> y_hat ---> Loss L(y, y_hat)
```

---

### 2.2 Output Functions and Loss Functions

The choice of output activation and loss function depends directly on the task:

| Task | Output Activation $O(\mathbf{a}_L)$ | Target $y$ | Loss Function $\mathcal{L}(\boldsymbol{\theta})$ | Loss Formula |
|:---|:---|:---|:---|:---|
| **Regression** | Linear: $\hat{y}_i = a_{L,i}$ | Real values $\mathbb{R}^K$ | **Mean Squared Error** | $\frac{1}{2} \sum_{i} (\hat{y}_i - y_i)^2$ |
| **Binary Classification** | Sigmoid: $\hat{y} = \sigma(a_L)$ | Binary $y \in \{0, 1\}$ | **Binary Cross-Entropy** | $- [y \log \hat{y} + (1-y)\log(1-\hat{y})]$ |
| **Multi-class Classification** | **Softmax**: $\hat{y}_i = \frac{e^{a_{L,i}}}{\sum_j e^{a_{L,j}}}$ | One-hot vector $\mathbf{y} \in \{0, 1\}^K$ | **Categorical Cross-Entropy** | $-\sum_{c=1}^K y_c \log \hat{y}_c = -\log \hat{y}_\ell$ |

#### The Softmax Function
For a $K$-dimensional pre-activation vector $\mathbf{a}_L$:
$$\hat{y}_i = \frac{e^{a_{L,i}}}{\sum_{j=1}^K e^{a_{L,j}}}, \quad \forall i \in \{1, \dots, K\}$$
Properties:
1. $0 < \hat{y}_i < 1$ for all $i$.
2. $\sum_{i=1}^K \hat{y}_i = 1$, representing a valid discrete probability distribution.
3. Exponentiation accentuates differences, causing the largest input to dominate the output probabilities.

---

### 2.3 Elegant Derivation: Softmax with Cross-Entropy Loss

Let $\mathbf{y}$ be a one-hot vector with $y_\ell = 1$ for the true class $\ell$ and $y_c = 0$ for all $c \neq \ell$.
The loss is:
$$\mathcal{L} = -\sum_{c=1}^K y_c \log \hat{y}_c = -\log \hat{y}_\ell$$
We want to find the gradient with respect to any output pre-activation $a_{L,i}$:
$$\frac{\partial \mathcal{L}}{\partial a_{L,i}} = -\frac{1}{\hat{y}_\ell} \frac{\partial \hat{y}_\ell}{\partial a_{L,i}}$$

#### Step 1: Derivative of Softmax $\frac{\partial \hat{y}_\ell}{\partial a_{L,i}}$
Recall $\hat{y}_\ell = \frac{e^{a_{L,\ell}}}{\sum_{j=1}^K e^{a_{L,j}}}$. Let $S = \sum_{j=1}^K e^{a_{L,j}}$.
- **Case 1: $i = \ell$** (derivative w.r.t the true class pre-activation):
  $$\frac{\partial \hat{y}_\ell}{\partial a_{L,\ell}} = \frac{e^{a_{L,\ell}} \cdot S - e^{a_{L,\ell}} \cdot e^{a_{L,\ell}}}{S^2} = \frac{e^{a_{L,\ell}}}{S} - \left(\frac{e^{a_{L,\ell}}}{S}\right)^2 = \hat{y}_\ell (1 - \hat{y}_\ell)$$
- **Case 2: $i \neq \ell$** (derivative w.r.t an incorrect class pre-activation):
  $$\frac{\partial \hat{y}_\ell}{\partial a_{L,i}} = \frac{0 \cdot S - e^{a_{L,\ell}} \cdot e^{a_{L,i}}}{S^2} = -\frac{e^{a_{L,\ell}}}{S} \cdot \frac{e^{a_{L,i}}}{S} = -\hat{y}_\ell \hat{y}_i$$

#### Step 2: Substituting into the Loss Derivative
- For $i = \ell$:
  $$\frac{\partial \mathcal{L}}{\partial a_{L,\ell}} = -\frac{1}{\hat{y}_\ell} \cdot \hat{y}_\ell(1 - \hat{y}_\ell) = -(1 - \hat{y}_\ell) = \hat{y}_\ell - 1 = \hat{y}_\ell - y_\ell$$
- For $i \neq \ell$:
  $$\frac{\partial \mathcal{L}}{\partial a_{L,i}} = -\frac{1}{\hat{y}_\ell} \cdot (-\hat{y}_\ell \hat{y}_i) = \hat{y}_i = \hat{y}_i - 0 = \hat{y}_i - y_i$$

In vector notation:
$$\nabla_{\mathbf{a}_L} \mathcal{L} = \hat{\mathbf{y}} - \mathbf{y}$$

> [!NOTE]
> The derivative of Categorical Cross-Entropy with Softmax simplifies exactly to $(\hat{\mathbf{y}} - \mathbf{y})$, the raw difference between the predicted probability and the true one-hot target.

---

### 2.4 Computational Graphs & Backpropagation Intuition

In a computational graph:
- **Forward Pass**: Nodes compute output values from inputs along directed edges.
- **Backward Pass**: Reverse traversal. Every node receives an incoming gradient from downstream:
  $$\frac{\partial \mathcal{L}}{\partial \text{output}}$$
  and computes its local gradient w.r.t each input using the chain rule:
  $$\frac{\partial \mathcal{L}}{\partial \text{input}} = \frac{\partial \mathcal{L}}{\partial \text{output}} \times \frac{\partial \text{output}}{\partial \text{input}}$$
- **Branching Nodes**: If an intermediate quantity $u$ feeds into multiple subsequent nodes $v_1, v_2, \dots, v_m$, its gradient is the sum of gradients from all outgoing paths:
  $$\frac{\partial \mathcal{L}}{\partial u} = \sum_{j=1}^m \frac{\partial \mathcal{L}}{\partial v_j} \frac{\partial v_j}{\partial u}$$

---

### 2.5 The Full Matrix Backpropagation Algorithm

Given pre-activations $\mathbf{a}_k$ and post-activations $\mathbf{h}_k$ from the forward pass:

#### 1. Output Layer Gradient
$$\nabla_{\mathbf{a}_L} \mathcal{L} = \hat{\mathbf{y}} - \mathbf{y}$$

#### 2. Backward Recursion (for $k = L$ down to $1$):
- **Gradient w.r.t. Weights Matrix $\mathbf{W}_k$**:
  $$\nabla_{\mathbf{W}_k} \mathcal{L} = (\nabla_{\mathbf{a}_k} \mathcal{L}) \, (\mathbf{h}_{k-1})^\top$$
  *(Outer product of layer $k$'s error vector and layer $k-1$'s activation vector)*
- **Gradient w.r.t. Bias Vector $\mathbf{b}_k$**:
  $$\nabla_{\mathbf{b}_k} \mathcal{L} = \nabla_{\mathbf{a}_k} \mathcal{L}$$
- **Backpropagate Error to Previous Post-Activations $\mathbf{h}_{k-1}$**:
  $$\nabla_{\mathbf{h}_{k-1}} \mathcal{L} = \mathbf{W}_k^\top (\nabla_{\mathbf{a}_k} \mathcal{L})$$
- **Pass Through Previous Activation Derivative to Pre-Activations $\mathbf{a}_{k-1}$**:
  $$\nabla_{\mathbf{a}_{k-1}} \mathcal{L} = (\nabla_{\mathbf{h}_{k-1}} \mathcal{L}) \odot g'(\mathbf{a}_{k-1})$$
  *(where $\odot$ denotes element-wise Hadamard product)*

#### 3. Parameter Update:
$$\mathbf{W}_k \leftarrow \mathbf{W}_k - \eta \nabla_{\mathbf{W}_k} \mathcal{L}$$
$$\mathbf{b}_k \leftarrow \mathbf{b}_k - \eta \nabla_{\mathbf{b}_k} \mathcal{L}$$

---

## 3. Core Formulas & Cheat Sheet

| Step | Vector / Matrix Equation | Dimensions |
|:---|:---|:---|
| **Forward Pre-activation** | $\mathbf{a}_k = \mathbf{W}_k \mathbf{h}_{k-1} + \mathbf{b}_k$ | $(n_k \times 1) = (n_k \times n_{k-1})(n_{k-1} \times 1) + (n_k \times 1)$ |
| **Forward Post-activation** | $\mathbf{h}_k = g(\mathbf{a}_k)$ | $(n_k \times 1)$ |
| **Output Error** | $\nabla_{\mathbf{a}_L} \mathcal{L} = \hat{\mathbf{y}} - \mathbf{y}$ | $(n_L \times 1)$ |
| **Weight Gradient** | $\nabla_{\mathbf{W}_k} \mathcal{L} = (\nabla_{\mathbf{a}_k} \mathcal{L}) \mathbf{h}_{k-1}^\top$ | $(n_k \times n_{k-1}) = (n_k \times 1)(1 \times n_{k-1})$ |
| **Bias Gradient** | $\nabla_{\mathbf{b}_k} \mathcal{L} = \nabla_{\mathbf{a}_k} \mathcal{L}$ | $(n_k \times 1)$ |
| **Hidden Activation Grad** | $\nabla_{\mathbf{h}_{k-1}} \mathcal{L} = \mathbf{W}_k^\top (\nabla_{\mathbf{a}_k} \mathcal{L})$ | $(n_{k-1} \times 1) = (n_{k-1} \times n_k)(n_k \times 1)$ |
| **Previous Pre-act Grad** | $\nabla_{\mathbf{a}_{k-1}} \mathcal{L} = (\nabla_{\mathbf{h}_{k-1}} \mathcal{L}) \odot g'(\mathbf{a}_{k-1})$ | $(n_{k-1} \times 1)$ |
| **Computational Chain Rule** | $\frac{\partial \mathcal{L}}{\partial x} = \sum_m \frac{\partial \mathcal{L}}{\partial q_m} \frac{\partial q_m}{\partial x}$ | Sum over all forward paths branching from $x$ |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Tutoring Session Readiness Score (Questions 1 to 5)

#### Context & Data
A tutoring platform uses a feedforward network:
Inputs: $x_1 = 2$ (student preparation score), $x_2 = 1$ (topic difficulty score).
Equations:
$$z_1 = 2x_1 + x_2$$
$$z_2 = x_1 - x_2 + 1$$
$$\hat{y} = z_1 + 2z_2$$

---

#### Question 1
**What is the value of $z_1$ for this session?**
- A) 6
- B) 5
- C) 4
- D) 3

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$z_1 = 2x_1 + x_2 = 2(2) + 1 = 4 + 1 = 5$$

---

#### Question 2
**What is the value of $z_2$?**
- A) 0
- B) 1
- C) 3
- D) 2

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$z_2 = x_1 - x_2 + 1 = 2 - 1 + 1 = 2$$

---

#### Question 3
**What is the final value of $\hat{y}$?**
- A) 9
- B) 7
- C) 8
- D) 6

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$\hat{y} = z_1 + 2z_2 = 5 + 2(2) = 5 + 4 = 9$$

---

#### Question 4
**Which computation must be completed before $\hat{y}$ is evaluated?**
- A) The loss must be minimized over all stored tutoring sessions.
- B) The target value must be changed by the model before scoring.
- C) The hidden quantities $z_1$ and $z_2$ must be computed.
- D) The learning rate must be set to zero.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  In a feedforward network, functional evaluation follows topological ordering. Since $\hat{y} = z_1 + 2z_2$, its inputs are $z_1$ and $z_2$. Therefore, $z_1$ and $z_2$ must be computed before $\hat{y}$ can be evaluated.

---

#### Question 5
**What makes this a feedforward computation?**
- A) The output is repeatedly fed back as a new input.
- B) Information flows from inputs to hidden quantities to output.
- C) The target label is used before the input features.
- D) The network stores a separate state for each previous session.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  A feedforward neural network computes activations strictly in a forward acyclic direction from input nodes through intermediate hidden layers to output nodes, without cycles or loops.

---

### Case Study 2: Inventory Forecast Calculation Review (Questions 6 to 10)

#### Context & Data
A computational graph:
$$u = 2x$$
$$v = u + 3$$
$$\mathcal{L} = v^2$$
For the current product, $x = 1$.

---

#### Question 6
**What is the value of $u$?**
- A) 2
- B) 1
- C) 3
- D) 4

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$u = 2x = 2(1) = 2$$

---

#### Question 7
**What is the value of $v$?**
- A) 3
- B) 4
- C) 6
- D) 5

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$v = u + 3 = 2 + 3 = 5$$

---

#### Question 8
**What is the loss $\mathcal{L}$?**
- A) 20
- B) 15
- C) 25
- D) 10

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$\mathcal{L} = v^2 = 5^2 = 25$$

---

#### Question 9
**What is $\frac{\partial \mathcal{L}}{\partial v}$ at $v = 5$?**
- A) 5
- B) 10
- C) 25
- D) 2

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$\frac{\partial \mathcal{L}}{\partial v} = \frac{d}{dv}(v^2) = 2v$$
  At $v = 5$:
  $$\frac{\partial \mathcal{L}}{\partial v} = 2(5) = 10$$

---

#### Question 10
**What is $\frac{\partial \mathcal{L}}{\partial x}$?**
- A) 10
- B) 5
- C) 40
- D) 20

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Applying the chain rule through the computational graph:
  $$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial v} \cdot \frac{\partial v}{\partial u} \cdot \frac{\partial u}{\partial x}$$
  - $\frac{\partial \mathcal{L}}{\partial v} = 2v = 10$
  - $\frac{\partial v}{\partial u} = \frac{d}{du}(u + 3) = 1$
  - $\frac{\partial u}{\partial x} = \frac{d}{dx}(2x) = 2$
  $$\frac{\partial \mathcal{L}}{\partial x} = 10 \cdot 1 \cdot 2 = 20$$

---

### Case Study 3: Sensor Calibration Error Analysis (Questions 11 to 15)

#### Context & Data
A model with one hidden unit:
$$h = w_1 x + b$$
$$\hat{y} = w_2 h$$
Parameters: $w_1 = 1, b = 0, w_2 = 2$.
Training example: $x = 2, y = 5$.
Loss function:
$$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$$

---

#### Question 11
**What is the hidden value $h$?**
- A) 1
- B) 4
- C) 2
- D) 5

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$h = w_1 x + b = 1(2) + 0 = 2$$

---

#### Question 12
**What is the prediction $\hat{y}$?**
- A) 3
- B) 5
- C) 2
- D) 4

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$\hat{y} = w_2 h = 2(2) = 4$$

---

#### Question 13
**What is $\frac{\partial \mathcal{L}}{\partial \hat{y}}$?**
- A) -1
- B) 1
- C) -2
- D) 2

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{d}{d\hat{y}}\left[\frac{1}{2}(\hat{y} - y)^2\right] = \hat{y} - y = 4 - 5 = -1$$

---

#### Question 14
**What is $\frac{\partial \mathcal{L}}{\partial w_2}$?**
- A) -1
- B) -2
- C) 2
- D) 1

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Since $\hat{y} = w_2 h$, by chain rule:
  $$\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_2} = (\hat{y} - y) \cdot h = (-1) \cdot 2 = -2$$

---

#### Question 15
**What is $\frac{\partial \mathcal{L}}{\partial w_1}$?**
- A) -1
- B) 1
- C) -4
- D) 4

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  By backpropagation through $h$:
  $$\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} = (-1) \cdot w_2 = (-1)(2) = -2$$
  Since $h = w_1 x + b$, $\frac{\partial h}{\partial w_1} = x = 2$.
  $$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial w_1} = (-2) \cdot 2 = -4$$

---

### Case Study 4: Course Recommendation Score Update (Questions 16 to 20)

#### Context & Data
Linear prediction model:
$$\hat{y} = w_1 x_1 + w_2 x_2 + b$$
Training example: $x_1 = 1, x_2 = 2, y = 6$.
Initial parameters: $w_1 = 1, w_2 = 1, b = 0$.
Loss: $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$.
Learning rate: $\eta = 0.1$.
Update rule: $\theta \leftarrow \theta - \eta \frac{\partial \mathcal{L}}{\partial \theta}$.

---

#### Question 16
**What is the initial prediction $\hat{y}$?**
- A) 4
- B) 3
- C) 5
- D) 2

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$\hat{y} = w_1 x_1 + w_2 x_2 + b = 1(1) + 1(2) + 0 = 1 + 2 + 0 = 3$$

---

#### Question 17
**What is $\frac{\partial \mathcal{L}}{\partial \hat{y}}$?**
- A) -3
- B) 3
- C) -6
- D) 6

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y = 3 - 6 = -3$$

---

#### Question 18
**What is $\frac{\partial \mathcal{L}}{\partial w_2}$?**
- A) 3
- B) -3
- C) 6
- D) -6

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_2} = (\hat{y} - y) \cdot x_2 = (-3) \cdot 2 = -6$$

---

#### Question 19
**What is the updated value of $w_1$?**
- A) 0.7
- B) 1.6
- C) 1.3
- D) 0.4

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  First compute gradient w.r.t $w_1$:
  $$\frac{\partial \mathcal{L}}{\partial w_1} = (\hat{y} - y) \cdot x_1 = (-3) \cdot 1 = -3$$
  Apply gradient update:
  $$w_1^{(1)} = w_1^{(0)} - \eta \frac{\partial \mathcal{L}}{\partial w_1} = 1 - 0.1 \times (-3) = 1 + 0.3 = 1.3$$

---

#### Question 20
**What is the updated value of $b$?**
- A) 0.3
- B) -0.3
- C) 0.6
- D) -0.6

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Compute gradient w.r.t $b$:
  $$\frac{\partial \mathcal{L}}{\partial b} = (\hat{y} - y) \cdot \frac{\partial \hat{y}}{\partial b} = (-3) \cdot 1 = -3$$
  Apply gradient update:
  $$b^{(1)} = b^{(0)} - \eta \frac{\partial \mathcal{L}}{\partial b} = 0 - 0.1 \times (-3) = 0 + 0.3 = 0.3$$

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Matrix Dimensions in Backpropagation**:
   - Remember: if $\mathbf{h}_{k-1}$ is $n_{k-1} \times 1$ and $\mathbf{a}_k$ is $n_k \times 1$, the weight matrix $\mathbf{W}_k$ has dimensions $n_k \times n_{k-1}$.
   - The gradient $\nabla_{\mathbf{W}_k} \mathcal{L}$ must have the same dimension $n_k \times n_{k-1}$. It is computed as $(\nabla_{\mathbf{a}_k} \mathcal{L}) (\mathbf{h}_{k-1})^\top$, **NOT** $(\mathbf{h}_{k-1})(\nabla_{\mathbf{a}_k} \mathcal{L})^\top$.
2. **Transpose in Backpropagating Hidden Errors**:
   - When pulling the error back through weights to layer $k-1$: $\nabla_{\mathbf{h}_{k-1}} \mathcal{L} = \mathbf{W}_k^\top (\nabla_{\mathbf{a}_k} \mathcal{L})$. The weight matrix **must be transposed**!
3. **Cross-Entropy Derivative Simplification**:
   - For Softmax + Cross-Entropy, the derivative w.r.t logits $a_{L,i}$ simplifies completely to $\hat{y}_i - y_i$. You do not need to evaluate messy quotients during exams!
