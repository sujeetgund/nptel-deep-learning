# Week 2: Sigmoid Neurons, Gradient Descent, Feedforward Neural Networks, and Universal Approximation

---

## 1. Executive Summary & Exam Roadmap

Week 2 transitions the course from discrete, non-differentiable threshold logic (Perceptrons) to continuous, smooth, differentiable optimization:
1. **The Need for Smoothness**: The limitations of step functions; small weight changes cause catastrophic, discontinuous output flips.
2. **The Sigmoid (Logistic) Neuron**: Mathematical properties, output interpretation as posterior probability, derivative derivation $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, geometric behavior.
3. **Formal Derivation of Gradient Descent**: Why move opposite to the gradient? Taylor Series expansion, directional derivative, Cauchy-Schwarz condition ($\cos \beta = -1$).
4. **Learning Parameters for a Sigmoid Neuron**: Deriving gradients $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ under squared error loss.
5. **Feedforward Neural Networks & Universal Approximation Theorem**: How networks of sigmoids compose "tower functions" (bump functions) to approximate any continuous mathematical function to arbitrary precision (Cybenko, 1989).

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 Why Not Perceptrons? The Discontinuity Bottleneck
In a standard Perceptron with step activation:
$$y = f(z) = \begin{cases} 1 & \text{if } \mathbf{w}^\top \mathbf{x} + b \ge 0 \\ 0 & \text{otherwise} \end{cases}$$
The activation function has two fatal flaws for continuous optimization:
1. **Zero gradient almost everywhere**: $\frac{df}{dz} = 0$ for all $z \neq 0$.
2. **Undefined / Infinite gradient at origin**: At $z = 0$, $f(z)$ is non-differentiable.

Because the gradient is zero, gradient-based optimization algorithms cannot receive any informative signal indicating whether a small weight adjustment $\Delta \mathbf{w}$ moves the model closer to or further from the desired threshold.

---

### 2.2 The Sigmoid Neuron (Logistic Activation)

#### Mathematical Definition
The sigmoid neuron replaces the step threshold with the **logistic function**:
$$\hat{y} = f(\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^n w_i x_i + b$.

#### Key Algebraic and Geometric Properties
1. **Range**: $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$.
2. **Symmetry**:
   $$\sigma(-z) = \frac{1}{1 + e^z} = \frac{e^{-z}}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)$$
3. **Asymptotes**:
   $$\lim_{z \to +\infty} \sigma(z) = 1, \quad \lim_{z \to -\infty} \sigma(z) = 0, \quad \sigma(0) = \frac{1}{2} = 0.5$$
4. **Controllable Steepness**:
   - For $z = w x + b$, if $w \to \infty$, $\sigma(w x + b)$ approaches a sharp Heaviside step function centered at $x = -b/w$.
   - If $w$ is small, the transition is very gradual and linear-like near the origin.

#### Complete Derivation of the Sigmoid Derivative
The derivative of $\sigma(z)$ with respect to $z$ is:
$$\frac{d\sigma(z)}{dz} = \frac{d}{dz} (1 + e^{-z})^{-1} = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$
Factor the expression:
$$\frac{d\sigma(z)}{dz} = \left(\frac{1}{1 + e^{-z}}\right) \left(\frac{e^{-z}}{1 + e^{-z}}\right) = \sigma(z) \cdot (1 - \sigma(z))$$

> [!IMPORTANT]
> **Maximum Gradient Value**:
> Since $\sigma(z) \in (0, 1)$, the product $\sigma(z)(1 - \sigma(z))$ achieves its maximum at $\sigma(z) = 0.5$ (which occurs at $z = 0$):
> $$\max \sigma'(z) = 0.5 \cdot (1 - 0.5) = 0.25$$
> For large $|z|$, $\sigma'(z) \to 0$. This gives rise to the **vanishing gradient problem** in deep networks.

---

### 2.3 Formal Derivation of Gradient Descent via Taylor Series

Consider a loss function $\mathcal{L}(\boldsymbol{\theta})$ where $\boldsymbol{\theta} = [w_1, \dots, w_n, b]^\top$.
We wish to find a small perturbation $\Delta \boldsymbol{\theta} = \eta \mathbf{u}$ (where $\|\mathbf{u}\| = 1$ and step size $\eta > 0$) such that the loss decreases:
$$\mathcal{L}(\boldsymbol{\theta} + \eta \mathbf{u}) - \mathcal{L}(\boldsymbol{\theta}) < 0$$

#### Taylor Series Expansion
Expanding $\mathcal{L}(\boldsymbol{\theta} + \eta \mathbf{u})$ around $\boldsymbol{\theta}$:
$$\mathcal{L}(\boldsymbol{\theta} + \eta \mathbf{u}) = \mathcal{L}(\boldsymbol{\theta}) + \eta \mathbf{u}^\top \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) + \frac{\eta^2}{2!} \mathbf{u}^\top \nabla^2 \mathcal{L}(\boldsymbol{\theta}) \mathbf{u} + \mathcal{O}(\eta^3)$$
For sufficiently small learning rate $\eta \ll 1$, higher-order terms vanish:
$$\Delta \mathcal{L} \approx \eta \, \mathbf{u}^\top \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$$

#### Maximizing the Rate of Decrease
To achieve the maximum decrease in $\mathcal{L}$, we must minimize the inner product $\mathbf{u}^\top \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$.
By the definition of the dot product:
$$\mathbf{u}^\top \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) = \|\mathbf{u}\| \, \|\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})\| \cos \beta = \|\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})\| \cos \beta$$
where $\beta$ is the angle between the update direction $\mathbf{u}$ and the gradient vector $\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$.
- The cosine function $\cos \beta$ is minimized when $\cos \beta = -1 \implies \beta = 180^\circ$ ($\pi$ radians).
- Therefore, the optimal direction $\mathbf{u}$ is **directly opposite** to the gradient:
  $$\mathbf{u}^* = -\frac{\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})}{\|\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})\|}$$

#### Gradient Descent Update Rule
Absorbing constants into the learning rate $\eta$:
$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta \, \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}^{(t)})$$
Specifically for weight $w$ and bias $b$:
$$w^{(t+1)} = w^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial w}, \quad b^{(t+1)} = b^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial b}$$

---

### 2.4 Gradient Derivation for a Single Sigmoid Neuron

Let the squared error loss for a sample $(\mathbf{x}, y)$ be:
$$\mathcal{L}(w, b) = \frac{1}{2} (\hat{y} - y)^2 = \frac{1}{2} (\sigma(z) - y)^2, \quad z = \mathbf{w}^\top \mathbf{x} + b$$
Applying the chain rule:
$$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$
- $\frac{\partial \mathcal{L}}{\partial \hat{y}} = (\hat{y} - y)$
- $\frac{\partial \hat{y}}{\partial z} = \sigma'(z) = \hat{y}(1 - \hat{y})$
- $\frac{\partial z}{\partial w_j} = x_j$, and $\frac{\partial z}{\partial b} = 1$

Therefore:
$$\frac{\partial \mathcal{L}}{\partial w_j} = (\hat{y} - y) \, \hat{y}(1 - \hat{y}) \, x_j$$
$$\frac{\partial \mathcal{L}}{\partial b} = (\hat{y} - y) \, \hat{y}(1 - \hat{y})$$
For $N$ training points, the gradient is the sum over all points:
$$\nabla_{\mathbf{w}} \mathcal{L} = \sum_{i=1}^N (\hat{y}_i - y_i) \, \hat{y}_i(1 - \hat{y}_i) \, \mathbf{x}_i$$

---

### 2.5 Representation Power & Universal Approximation Theorem

#### Building "Tower" (Bump) Functions
1. **Single Sigmoid Step**:
   $f_1(x) = \sigma(w(x - s_1))$. With a large $w$, this approximates a step jumping from $0$ to $1$ at $x = s_1$.
2. **Second Shifted Step**:
   $f_2(x) = \sigma(w(x - s_2))$ where $s_2 > s_1$. This jumps from $0$ to $1$ at $x = s_2$.
3. **Subtracting Two Steps (Tower / Pulse Function)**:
   $$h(x) = f_1(x) - f_2(x) = \sigma(w(x - s_1)) - \sigma(w(x - s_2))$$
   $h(x)$ is approximately $0$ for $x < s_1$, jumps to $1$ for $s_1 \le x \le s_2$, and drops back to $0$ for $x > s_2$. This forms a rectangular "tower" of width $(s_2 - s_1)$ and height $1$.
4. **Weighted Combination of Towers**:
   Any arbitrary 1D continuous curve can be sliced into narrow rectangles of height $A_k$ and width $\Delta s_k$:
   $$f(x) \approx \sum_{k} A_k \left[ \sigma(w(x - s_k)) - \sigma(w(x - s_{k+1})) \right]$$

#### Generalization to Multiple Dimensions (2D / $n$D)
- In 2D, two perpendicular 1D step combinations along $x_1$ and $x_2$ are fed into an AND-like sigmoid neuron, producing a localized 3D cylinder/box tower.
- By summing thousands of tiny towers, any continuous 3D surface can be reconstructed.

#### Universal Approximation Theorem (Cybenko 1989, Hornik 1991)
> A feedforward neural network with a **single hidden layer** containing a finite number of non-linear (e.g. sigmoid) activation functions can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary precision $\epsilon > 0$.

*Catch*: The theorem guarantees *existence*, not *learnability*. The required number of hidden neurons may grow exponentially with the input dimension, and finding those weights via gradient descent is non-trivial.

---

## 3. Core Formulas & Cheat Sheet

| Concept | Mathematical Formula | Key Properties / Usage |
|:---|:---|:---|
| **Sigmoid Function** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Maps $\mathbb{R} \to (0, 1)$; $\sigma(0) = 0.5$ |
| **Sigmoid Derivative** | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ | Maximum value is $0.25$ at $z = 0$ |
| **Squared Error Loss** | $\mathcal{L} = (y - \hat{y})^2$ | Derivative w.r.t prediction: $\frac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y} - y)$ |
| **Half-Squared Loss** | $\mathcal{L} = \frac{1}{2}(y - \hat{y})^2$ | Derivative w.r.t prediction: $\frac{\partial \mathcal{L}}{\partial \hat{y}} = (\hat{y} - y)$ |
| **Mean Squared Error (MSE)** | $\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$ | Average squared penalty over $n$ samples |
| **Gradient Descent Step** | $\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}$ | Moves in direction of steepest descent ($\beta = 180^\circ$) |
| **Sigmoid Weight Gradient** | $\frac{\partial \mathcal{L}}{\partial w_j} = (\hat{y} - y)\hat{y}(1-\hat{y})x_j$ | Product of error, activation derivative, and input |
| **Tower Function** | $h(x) = \sigma(w(x-s_1)) - \sigma(w(x-s_2))$ | Requires 2 hidden neurons to produce a 1D pulse |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Clinic Follow-up Priority Scoring (Questions 1 to 5)

#### Context & Data
A clinic produces a probability-like score $\hat{p}$ using normalized inputs:
- $x_1$: Symptom persistence score
- $x_2$: Abnormal reading score
Linear score:
$$z = 3x_1 + 2x_2 - 2$$
Priority score:
$$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
Decision rule: Send for priority follow-up when $\hat{p} \ge 0.5$.

---

#### Question 1
**Why is the sigmoid function suitable for this priority score?**
- A) It maps a real-valued input to a value between 0 and 1.
- B) It removes the need to compute any weighted input value.
- C) It always gives a hard class label before thresholding.
- D) It forces both input features to have identical weights.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The logistic sigmoid function maps the entire real line $(-\infty, \infty)$ into the open interval $(0, 1)$. This makes its output smooth, bounded, and naturally interpretable as a probability or priority score.

---

#### Question 2
**For $x_1 = 0.4$ and $x_2 = 0.6$, what is the value of $z$?**
- A) 0.2
- B) 0.6
- C) 0.4
- D) 0.8

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Substitute the values into $z = 3x_1 + 2x_2 - 2$:
  $$z = 3(0.4) + 2(0.6) - 2 = 1.2 + 1.2 - 2 = 2.4 - 2 = 0.4$$

---

#### Question 3
**The rule $\hat{p} \ge 0.5$ is equivalent to which condition?**
- A) $z < 0$
- B) $z \ge 0$
- C) $z > 1$
- D) $z \le -1$

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Recall that $\sigma(0) = \frac{1}{1 + e^0} = \frac{1}{2} = 0.5$.
  Because $\sigma(z)$ is a strictly monotonically increasing function of $z$:
  $$\sigma(z) \ge 0.5 \iff z \ge 0$$

---

#### Question 4
**Which input is not sent for priority follow-up?**
- A) $x_1 = 0.8, x_2 = 0.1$
- B) $x_1 = 0.2, x_2 = 0.8$
- C) $x_1 = 0.5, x_2 = 0.4$
- D) $x_1 = 0.1, x_2 = 0.4$

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  A record is sent for follow-up if $z \ge 0$. Let's compute $z = 3x_1 + 2x_2 - 2$ for each option:
  - Option A: $z = 3(0.8) + 2(0.1) - 2 = 2.4 + 0.2 - 2 = +0.6 \ge 0$ (Sent)
  - Option B: $z = 3(0.2) + 2(0.8) - 2 = 0.6 + 1.6 - 2 = +0.2 \ge 0$ (Sent)
  - Option C: $z = 3(0.5) + 2(0.4) - 2 = 1.5 + 0.8 - 2 = +0.3 \ge 0$ (Sent)
  - Option D: $z = 3(0.1) + 2(0.4) - 2 = 0.3 + 0.8 - 2 = -0.9 < 0 \implies \hat{p} < 0.5$ (**Not sent**)

---

#### Question 5
**If the clinic wants fewer records to satisfy $\hat{p} \ge 0.5$, what should happen to the bias term $-2$?**
- A) It should become more negative.
- B) It should be removed from the model entirely.
- C) It should be made equal to $x_1 + x_2$.
- D) It should be increased to make every score larger.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Decreasing the bias term (making it more negative, e.g. from $-2$ to $-3$) shifts the linear score $z$ downward for all patient records. As a result, fewer inputs will satisfy $z \ge 0$, reducing the number of priority follow-up flags.

---

### Case Study 2: Delivery Time Prediction Review (Questions 6 to 10)

#### Context & Data
A regression model predicts delivery delay in minutes:
| Order | Actual Delay ($y$) | Predicted Delay ($\hat{y}$) |
|:---:|:---:|:---:|
| 1 | 12 | 10 |
| 2 | 20 | 24 |
| 3 | 15 | 13 |
| 4 | 18 | 21 |

Formulas:
$$\mathcal{L} = (y - \hat{y})^2, \quad \text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2, \quad \frac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y} - y)$$

---

#### Question 6
**What is the squared error for Order 2?**
- A) 4
- B) 16
- C) 9
- D) 25

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  For Order 2: $y = 20, \hat{y} = 24$.
  Error $e = y - \hat{y} = 20 - 24 = -4$.
  Squared error $\mathcal{L} = (-4)^2 = 16$.

---

#### Question 7
**What is the mean squared error over the four orders?**
- A) 7.25
- B) 8.00
- C) 9.50
- D) 8.25

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Calculate squared error for each order:
  - Order 1: $(12 - 10)^2 = 2^2 = 4$
  - Order 2: $(20 - 24)^2 = (-4)^2 = 16$
  - Order 3: $(15 - 13)^2 = 2^2 = 4$
  - Order 4: $(18 - 21)^2 = (-3)^2 = 9$
  Sum of squared errors $= 4 + 16 + 4 + 9 = 33$.
  $$\text{MSE} = \frac{33}{4} = 8.25$$

---

#### Question 8
**Which order contributes the largest squared error?**
- A) Order 1
- B) Order 3
- C) Order 2
- D) Order 4

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Errors are: Order 1 (4), Order 2 (16), Order 3 (4), Order 4 (9).
  The maximum squared error is 16, coming from Order 2.

---

#### Question 9
**If $y = 20$ and $\hat{y} = 24$, what is $\frac{\partial \mathcal{L}}{\partial \hat{y}}$?**
- A) 8
- B) -8
- C) 4
- D) -4

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Using the given formula $\frac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y} - y)$:
  $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = 2(24 - 20) = 2(4) = 8$$

---

#### Question 10
**A second model predicts $11, 21, 14, 19$. Which model has lower MSE?**
- A) The original model, because its MSE is 8.25.
- B) The second model, because its MSE is 1.00.
- C) Both models, because their errors are identical.
- D) Neither model, because MSE cannot be computed here.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Calculate squared errors for Model 2 against actual values $(12, 20, 15, 18)$:
  - Order 1: $(12 - 11)^2 = 1^2 = 1$
  - Order 2: $(20 - 21)^2 = (-1)^2 = 1$
  - Order 3: $(15 - 14)^2 = 1^2 = 1$
  - Order 4: $(18 - 19)^2 = (-1)^2 = 1$
  Sum $= 1 + 1 + 1 + 1 = 4 \implies \text{MSE} = \frac{4}{4} = 1.00$.
  Since $1.00 < 8.25$, the second model has a substantially lower MSE.

---

### Case Study 3: Machine Energy Use Prediction (Questions 11 to 15)

#### Context & Data
A feedforward neural network computes:
$$h_1 = \sigma(x_1 + x_2 - 1)$$
$$h_2 = \sigma(2x_1 - x_2)$$
$$\hat{y} = \sigma(3h_1 + 2h_2 - 2)$$
Inputs $x_1$ (duration score) and $x_2$ (load score).

---

#### Question 11
**Which quantities are computed first in the feedforward direction?**
- A) The final loss value
- B) The parameter update
- C) The hidden units $h_1$ and $h_2$
- D) The target label chosen by the model

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  In forward propagation, computations proceed strictly layer by layer from input to output:
  $\text{Inputs } (x_1, x_2) \longrightarrow \text{Hidden activations } (h_1, h_2) \longrightarrow \text{Output } \hat{y} \longrightarrow \text{Loss } \mathcal{L}$.
  Therefore, the hidden unit activations $h_1$ and $h_2$ must be computed before the output or loss.

---

#### Question 12
**For $x_1 = 1$ and $x_2 = 1$, what is the input to the sigmoid in $h_1$?**
- A) 1
- B) 0
- C) 2
- D) 3

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The net input (pre-activation) to $h_1$ is:
  $$z_{h1} = x_1 + x_2 - 1 = 1 + 1 - 1 = 1$$

---

#### Question 13
**For $x_1 = 1$ and $x_2 = 0$, what is the input to the sigmoid in $h_2$?**
- A) -2
- B) -1
- C) 1
- D) 2

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  The pre-activation to $h_2$ is:
  $$z_{h2} = 2x_1 - x_2 = 2(1) - 0 = 2$$

---

#### Question 14
**Why can sigmoid hidden units introduce nonlinearity?**
- A) They copy inputs without changing any value.
- B) They apply a nonlinear function to weighted input combinations.
- C) They remove all weights from the network.
- D) They force the output to be exactly linear for all possible inputs.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  If hidden units were linear ($f(z) = c z$), any sequence of layers would simply collapse into a single linear matrix multiplication: $\mathbf{W}_2(\mathbf{W}_1 \mathbf{x}) = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} = \mathbf{W}_{\text{comb}} \mathbf{x}$. Applying a nonlinear function like the sigmoid to the affine combination creates a nonlinear feature space, allowing the network to model complex boundaries.

---

#### Question 15
**What is the final output range of $\hat{y}$?**
- A) Any real number from $-\infty$ to $\infty$
- B) Only the two values -1 and 1
- C) A value between 0 and 1
- D) A value greater than 1 for all inputs

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Since $\hat{y} = \sigma(3h_1 + 2h_2 - 2)$ uses a logistic sigmoid as its output activation, its range is strictly the open interval $(0, 1)$.

---

### Case Study 4: Warehouse Routing Time Analysis (Questions 16 to 20)

#### Context & Data
One-parameter regression model:
$$\hat{y} = w x$$
For a single training example: $x = 3, y = 15$.
Squared error loss:
$$\mathcal{L}(w) = (y - \hat{y})^2 = (15 - 3w)^2$$

---

#### Question 16
**What is $\mathcal{L}(w)$ when $w = 4$?**
- A) 6
- B) 4
- C) 12
- D) 9

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Predicted value: $\hat{y} = w x = 4 \times 3 = 12$.
  Loss: $\mathcal{L}(4) = (15 - 12)^2 = 3^2 = 9$.

---

#### Question 17
**Which value of $w$ gives zero loss?**
- A) $w = 4$
- B) $w = 5$
- C) $w = 6$
- D) $w = 3$

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Zero loss requires $\mathcal{L}(w) = 0 \implies (15 - 3w)^2 = 0 \implies 15 - 3w = 0 \implies 3w = 15 \implies w = 5$.

---

#### Question 18
**What is $\frac{d\mathcal{L}}{dw}$ at $w = 4$?**
- A) -18
- B) -6
- C) 18.0
- D) 6

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Using the chain rule:
  $$\frac{d\mathcal{L}}{dw} = \frac{d}{dw} (15 - 3w)^2 = 2(15 - 3w) \cdot \frac{d}{dw}(15 - 3w) = 2(15 - 3w)(-3) = -6(15 - 3w)$$
  Evaluating at $w = 4$:
  $$\left. \frac{d\mathcal{L}}{dw} \right|_{w=4} = -6(15 - 3(4)) = -6(15 - 12) = -6(3) = -18$$

---

#### Question 19
**If gradient descent uses $w \leftarrow w - \eta \frac{d\mathcal{L}}{dw}$ with $\eta = 0.1$, what is the next value from $w = 4$?**
- A) 3.2
- B) 4.6
- C) 5.8
- D) 2.2

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$w^{(1)} = w^{(0)} - \eta \frac{d\mathcal{L}}{dw} = 4 - 0.1 \times (-18) = 4 + 1.8 = 5.8$$

---

#### Question 20
**What does the gradient indicate in this example?**
- A) It gives the class label directly without a model.
- B) It removes the input feature from the prediction.
- C) It makes the squared error impossible to compute.
- D) It shows how the loss changes as $w$ changes.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  The gradient $\frac{d\mathcal{L}}{dw}$ represents the derivative of the loss function with respect to the parameter $w$. It quantifies the instantaneous rate of change and the direction of greatest increase of the loss function.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Sign of Gradient Update**:
   - The update is $w \leftarrow w - \eta \nabla \mathcal{L}$.
   - If the gradient is negative (e.g. $-18$), subtracting a negative quantity **increases** the parameter ($4 - 0.1(-18) = 5.8$). Do not forget the minus sign!
2. **Derivative Factor of 2**:
   - Pay close attention to whether the loss is written as $\mathcal{L} = (y - \hat{y})^2$ or $\mathcal{L} = \frac{1}{2}(y - \hat{y})^2$.
   - For $(y - \hat{y})^2$, the derivative w.r.t $\hat{y}$ is $2(\hat{y} - y)$.
   - For $\frac{1}{2}(y - \hat{y})^2$, the factor of 2 cancels out, leaving $(\hat{y} - y)$.
3. **Sigmoid Slope Maximum**:
   - A common multiple choice question asks for the maximum derivative of the sigmoid function: it is **0.25** (occurring at $z = 0$, where $\sigma(0) = 0.5$).
4. **Tower Function Construction**:
   - To make a tower function in 1D, you need **at least 2 hidden neurons** (one for the rising edge, one for the falling edge).
