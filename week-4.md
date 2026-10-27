# Week 4: Optimization Algorithms — Gradient Descent, Momentum, NAG, AdaGrad, RMSProp, and Adam

---

## 1. Executive Summary & Exam Roadmap

Week 4 focuses on training modern deep architectures efficiently:
1. **Gradient Descent Variants**:
   - **Batch GD**: Computes exact gradients over the entire dataset $N$.
   - **Stochastic GD (SGD)**: Computes noisy gradients on 1 sample at a time.
   - **Mini-Batch GD**: Evaluates gradients over a batch of size $B$ (e.g. 32, 64, 128), balancing vectorization, memory, and variance.
2. **Pathologies of Standard Gradient Descent**:
   - Ravines/valleys (steep in one dimension, flat in another) cause wild cross-valley oscillations and slow progress along the valley floor.
   - Saddle points and plateaus with near-zero gradients.
3. **Momentum-Based Gradient Descent**: Uses exponential averaging of previous gradients to accelerate along persistent directions and dampen oscillations.
4. **Nesterov Accelerated Gradient (NAG)**: Looks ahead along the momentum trajectory before evaluating the gradient, preventing overshoot.
5. **Adaptive Learning Rate Optimizers**:
   - **AdaGrad**: Accumulates sum of squared gradients to scale learning rates inversely per parameter.
   - **RMSProp**: Resolves AdaGrad's premature stopping using an exponentially decaying moving average of squared gradients.
   - **Adam**: Combines first-moment momentum with second-moment RMSProp, including critical initial bias correction steps.

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 The Spectrum of Gradient Descent: Batch vs. Stochastic vs. Mini-Batch

Let the objective function be the average loss over $N$ training examples:
$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_i(\boldsymbol{\theta})$$

| Attribute | Batch Gradient Descent (BGD) | Stochastic Gradient Descent (SGD) | Mini-Batch Gradient Descent (MBGD) |
|:---|:---|:---|:---|
| **Batch Size ($B$)** | $B = N$ (all samples) | $B = 1$ (single sample) | $1 < B < N$ (typically 32 to 512) |
| **Updates per Epoch** | Exactly 1 update per epoch | $N$ updates per epoch | $N / B$ updates per epoch |
| **Gradient Nature** | True, exact gradient $\nabla \mathcal{L}$ | High-variance, noisy estimate | Moderately smooth, low variance |
| **Computational Cost** | High memory; cannot process massive datasets in RAM | Very cheap per step, but misses GPU matrix parallelization | Highly efficient with GPU vectorization |
| **Convergence Behavior** | Smooth, monotonic loss decrease | Erratic/noisy trajectory; can escape shallow local minima | Smooth convergence with slight stochastic noise |

---

### 2.2 Pathologies of Standard Gradient Descent

1. **Ill-Conditioned Curvature (Ravines)**:
   - When the Hessian matrix has a high condition number ($\lambda_{\max} \gg \lambda_{\min}$), the error surface resembles an elongated ravine.
   - Standard GD oscillates wildly back and forth across the steep walls while making agonizingly slow progress along the flat base.
2. **Saddle Points and Local Plateaus**:
   - At a saddle point or zero-gradient plateau, $\|\nabla_{\boldsymbol{\theta}} \mathcal{L}\| \approx 0$.
   - Because standard GD updates are strictly proportional to the gradient magnitude ($\Delta \boldsymbol{\theta} = -\eta \nabla \mathcal{L}$), the optimizer can become trapped or take thousands of iterations to crawl out.

---

### 2.3 Momentum-Based Gradient Descent

#### Physical Intuition
Think of a heavy ball rolling down a hilly terrain. As it rolls down the slope, it accumulates velocity. When it encounters slight bumps or flat plateaus, its accumulated inertia carries it forward. Across narrow ravines, opposing vertical forces cancel out while forward forces reinforce each other.

#### Mathematical Formulation
At step $t$, maintain a velocity vector $\mathbf{v}_t$:
$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_{t-1})$$
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \mathbf{v}_t$$
*(where $\beta \in [0, 1)$ is the momentum decay factor, typically $0.9$, and $\eta$ is the learning rate).*

#### Expanding the Velocity History
Expanding $\mathbf{v}_t$ recursively:
$$\mathbf{v}_t = \eta \sum_{\tau=1}^t \beta^{t-\tau} \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_{\tau-1})$$
- Along oscillating directions where gradients alternate signs ($+g, -g$), the terms cancel: $\sum \beta^k g_k \approx 0$.
- Along consistent downhill directions where gradients have the same sign ($+g, +g$), velocity builds up to an effective terminal velocity:
  $$\mathbf{v}_{\text{terminal}} = \frac{\eta}{1 - \beta} \mathbf{g}$$
  For $\beta = 0.9$, the step size along consistent directions is amplified by a factor of $\frac{1}{1 - 0.9} = 10\times$!

---

### 2.4 Nesterov Accelerated Gradient (NAG)

#### The Problem with Momentum: Overshooting
Because momentum accumulates velocity, a ball rolling at high speed down a slope might overshoot the minimum and roll up the opposite slope before turning around.

#### The Nesterov Solution: "Look-Ahead" Gradient
Instead of computing the gradient at the current position $\boldsymbol{\theta}_{t-1}$, compute the gradient at the predicted "look-ahead" point $\boldsymbol{\theta}_{t-1} - \beta \mathbf{v}_{t-1}$:
$$\boldsymbol{\theta}_{\text{look-ahead}} = \boldsymbol{\theta}_{t-1} - \beta \mathbf{v}_{t-1}$$
$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_{t-1} - \beta \mathbf{v}_{t-1})$$
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \mathbf{v}_t$$

```
Standard Momentum:
Current Position ----(Momentum Step)----> Intermediate ----(Gradient Step)----> Final Position
                         \                                 /
                          \------- Net Update Vector -----/

Nesterov Momentum:
Current Position ----(Momentum Step)----> Look-ahead Point ----(Corrective Gradient)----> Final Position
```
If the look-ahead point starts climbing up the opposite valley wall, the gradient at that point immediately points backwards, acting as an automatic brake to prevent overshoot!

---

### 2.5 AdaGrad (Adaptive Gradient Algorithm)

#### Motivation: Parameter-Specific Learning Rates
In many tasks (e.g. NLP, sparse features), some weights receive frequent, large gradients, while others receive infrequent, tiny gradients. A single global learning rate $\eta$ either causes frequent features to explode or sparse features to never train.

#### Update Equations
Accumulate the sum of historical squared gradients for each parameter:
$$v_t = v_{t-1} + g_t^2$$
Update parameter:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$
*(where $\epsilon > 0$, typically $10^{-8}$, prevents division by zero).*

#### Drawback of AdaGrad
Since $g_t^2 \ge 0$, the accumulator $v_t$ **monotonically increases** with every iteration:
$$v_t \ge v_{t-1} \implies \frac{\eta}{\sqrt{v_t + \epsilon}} \to 0 \quad \text{as } t \to \infty$$
The effective learning rate continuously decays and eventually becomes infinitesimal, causing the algorithm to freeze and stop learning before reaching the optimum.

---

### 2.6 RMSProp (Root Mean Square Propagation)

#### Fixing AdaGrad's Shrinking Learning Rate
Geoff Hinton proposed modifying AdaGrad by replacing the indefinite sum with an **exponentially decaying moving average** of squared gradients:
$$v_t = \beta v_{t-1} + (1 - \beta) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$
*(typically $\beta = 0.9$ or $0.99$).*

- The denominator $\sqrt{v_t}$ now tracks the *recent* magnitude of gradients rather than the entire infinite history.
- If recent gradients are large, the step is downscaled; if recent gradients are small, the step is scaled up. Learning never prematurely grinds to a halt.

---

### 2.7 Adam (Adaptive Moment Estimation)

Adam combines the advantages of **Momentum** (first moment: running average of gradients) and **RMSProp** (second moment: running average of squared gradients).

#### Complete Algorithm Formulation
1. **Compute Current Gradient**:
   $$\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_{t-1})$$
2. **Update Biased First Moment (Mean)**:
   $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$
3. **Update Biased Second Raw Moment (Uncentered Variance)**:
   $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$
4. **Bias Correction**:
   Since $\mathbf{m}_0 = \mathbf{0}$ and $\mathbf{v}_0 = \mathbf{0}$, both estimates are heavily biased towards zero during early iterations (especially when $\beta_1, \beta_2 \approx 1$).
   $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$
   $$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
   *(Note: $\beta^t$ is $\beta$ raised to the power of step count $t$. As $t \to \infty$, $1 - \beta^t \to 1$ and correction vanishes).*
5. **Update Parameters**:
   $$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

Standard default hyperparameters:
$$\eta = 0.001, \quad \beta_1 = 0.9, \quad \beta_2 = 0.999, \quad \epsilon = 10^{-8}$$

---

## 3. Core Formulas & Cheat Sheet

| Optimizer | Accumulator / Velocity Update | Parameter Update Rule | Key Advantage / Drawback |
|:---|:---|:---|:---|
| **Standard GD** | — | $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \mathbf{g}_t$ | Simple; oscillates in ravines |
| **Momentum** | $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \eta \mathbf{g}_t$ | $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \mathbf{v}_t$ | Accelerates in flat ravines; may overshoot |
| **NAG** | $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \eta \nabla \mathcal{L}(\boldsymbol{\theta} - \beta \mathbf{v}_{t-1})$ | $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \mathbf{v}_t$ | Look-ahead brakes prevent overshoot |
| **AdaGrad** | $\mathbf{v}_t = \mathbf{v}_{t-1} + \mathbf{g}_t^2$ | $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \mathbf{g}_t$ | Parameter-specific steps; learning rate dies |
| **RMSProp** | $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\mathbf{g}_t^2$ | $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \mathbf{g}_t$ | Solves dying LR using exponential moving average |
| **Adam** | $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$<br>$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$ | $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$ | Combines momentum + RMSProp with bias corrections |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Delivery Delay Parameter Tuning (Questions 1 to 5)

#### Context & Data
One-parameter loss function:
$$\mathcal{L}(w) = (w - 4)^2$$
Gradient descent rule:
$$w_{t+1} = w_t - \eta \frac{d\mathcal{L}}{dw}$$
Current state: $w_t = 1$, learning rate $\eta = 0.1$.

---

#### Question 1
**What is $\frac{d\mathcal{L}}{dw}$ at $w = 1$?**
- A) 6
- B) -3
- C) 3
- D) -6

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$\frac{d\mathcal{L}}{dw} = \frac{d}{dw}(w - 4)^2 = 2(w - 4)$$
  At $w = 1$:
  $$\left. \frac{d\mathcal{L}}{dw} \right|_{w=1} = 2(1 - 4) = 2(-3) = -6$$

---

#### Question 2
**What is the next value of $w$?**
- A) 1.3
- B) 1.6
- C) 0.4
- D) 2.0

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$w_{t+1} = w_t - \eta \frac{d\mathcal{L}}{dw} = 1 - 0.1(-6) = 1 + 0.6 = 1.6$$

---

#### Question 3
**Why does $w$ increase in this update?**
- A) The learning rate is negative.
- B) The loss is already zero.
- C) The gradient at $w = 1$ is negative.
- D) The update ignores the derivative.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Gradient descent subtracts the scaled gradient: $-\eta \frac{d\mathcal{L}}{dw}$. Because the gradient is negative ($-6$), subtracting a negative quantity adds a positive offset ($+0.6$), increasing $w$ toward the minimum at $w = 4$.

---

#### Question 4
**Which value of $w$ minimizes this loss?**
- A) 1
- B) 2
- C) 3
- D) 4

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Since $\mathcal{L}(w) = (w - 4)^2 \ge 0$, its global minimum occurs where $(w - 4)^2 = 0 \implies w = 4$.

---

#### Question 5
**What is the role of the learning rate $\eta$?**
- A) It controls the update step size.
- B) It stores the final target value.
- C) It removes the need for a gradient.
- D) It makes every loss value equal.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The learning rate $\eta$ is a positive scalar hyperparameter that determines how large of a step the parameters take in the direction of the negative gradient.

---

### Case Study 2: Energy Forecast Training Schedule (Questions 6 to 10)

#### Context & Data
A dataset contains $N = 12$ training records.
In one experiment, the mini-batch size is $B = 3$.
One epoch means that all 12 records have been processed exactly once.

---

#### Question 6
**How many mini-batches are processed in one epoch?**
- A) 3
- B) 4
- C) 6
- D) 12

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$\text{Number of mini-batches per epoch} = \frac{\text{Total records}}{\text{Mini-batch size}} = \frac{12}{3} = 4$$

---

#### Question 7
**Which method uses all 12 records for each parameter update?**
- A) Stochastic gradient descent
- B) Mini-batch gradient descent
- C) Batch gradient descent
- D) Momentum-only update

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Batch Gradient Descent (BGD) computes the cost function and its gradients over the complete training set (all 12 records) before performing a single parameter update.

---

#### Question 8
**Which method uses one record per parameter update?**
- A) Batch gradient descent
- B) Adam without gradients or single-example updates
- C) Full-batch averaging
- D) Stochastic gradient descent

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Stochastic Gradient Descent (SGD) updates parameters sequentially after inspecting each individual training record ($B = 1$).

---

#### Question 9
**If the mini-batch gradients for four batches are 2, 4, 6, 8, what is their average?**
- A) 5
- B) 4
- C) 6
- D) 8

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$\text{Average gradient} = \frac{2 + 4 + 6 + 8}{4} = \frac{20}{4} = 5$$

---

#### Question 10
**Why might mini-batch gradient descent be preferred in practice?**
- A) It never uses gradients.
- B) It balances noisy updates and computational efficiency.
- C) It requires the loss to be exactly zero before every mini-batch update.
- D) It updates only after training is complete.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Mini-batch GD combines the computational advantages of vectorized hardware (GPUs) with the variance reduction of batching, while retaining enough stochasticity to escape shallow saddles.

---

### Case Study 3: Queue Prediction Parameter Update (Questions 11 to 15)

#### Context & Data
A momentum-based update rule:
$$v_t = \beta v_{t-1} + \eta g_t$$
$$w_t = w_{t-1} - v_t$$
Given:
- Previous velocity: $v_{t-1} = 0.5$
- Momentum parameter: $\beta = 0.8$
- Learning rate: $\eta = 0.1$
- Current gradient: $g_t = 3$
- Previous parameter: $w_{t-1} = 5$

---

#### Question 11
**What is the new velocity $v_t$?**
- A) 0.5
- B) 0.6
- C) 0.7
- D) 0.8

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$v_t = \beta v_{t-1} + \eta g_t = 0.8(0.5) + 0.1(3) = 0.4 + 0.3 = 0.7$$

---

#### Question 12
**What is the updated parameter $w_t$?**
- A) 4.3
- B) 5.7
- C) 4.7
- D) 5.3

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$w_t = w_{t-1} - v_t = 5 - 0.7 = 4.3$$

---

#### Question 13
**What does the momentum term $\beta v_{t-1}$ represent?**
- A) It removes all history from the update and uses only the current gradient.
- B) It sets the learning rate to zero for every future update.
- C) It replaces the loss function.
- D) It carries part of the previous update direction.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  The term $\beta v_{t-1}$ retains a decayed fraction of the past velocity vector, preserving directional inertia.

---

#### Question 14
**If $\beta = 0$, what would $v_t$ be for the same gradient and learning rate?**
- A) 0.3
- B) 0.1
- C) 0.2
- D) 0.4

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  With $\beta = 0$:
  $$v_t = 0(0.5) + 0.1(3) = 0.3$$

---

#### Question 15
**Why can momentum help optimization?**
- A) It makes gradients unnecessary.
- B) It always increases the loss.
- C) It can smooth updates across steps.
- D) It forces all parameters to be equal.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Momentum averages out high-frequency orthogonal oscillations in narrow valleys while reinforcing consistent descent directions, yielding smoother trajectories.

---

### Case Study 4: Demand Prediction Step-Size Analysis (Questions 16 to 20)

#### Context & Data
AdaGrad-style adaptive step size for a single parameter:
$$r_t = r_{t-1} + g^2$$
Update magnitude:
$$\Delta = \frac{\eta g}{\sqrt{r_t}}$$
Given:
- Current gradient: $g = 4$
- Previous accumulated squared gradient: $r_{t-1} = 9$
- Learning rate: $\eta = 0.5$

---

#### Question 16
**What is $r_t$?**
- A) 16
- B) 25
- C) 13
- D) 9

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$r_t = r_{t-1} + g^2 = 9 + 4^2 = 9 + 16 = 25$$

---

#### Question 17
**What is $\sqrt{r_t}$?**
- A) 3
- B) 4
- C) 5
- D) 6

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$\sqrt{r_t} = \sqrt{25} = 5$$

---

#### Question 18
**What is the update magnitude?**
- A) 0.2
- B) 0.4
- C) 0.8
- D) 0.6

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$\text{Update magnitude} = \frac{\eta g}{\sqrt{r_t}} = \frac{0.5 \times 4}{5} = \frac{2}{5} = 0.4$$

---

#### Question 19
**What is the main idea behind AdaGrad-style scaling?**
- A) It ignores all previous gradient information while claiming to adapt steps.
- B) It requires the same update for every parameter.
- C) It removes the need for a loss function.
- D) It adapts the step size using accumulated squared gradients.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  AdaGrad individually divides each parameter's update by the square root of its cumulative historical squared gradients, automatically performing smaller steps for frequently updated weights and larger steps for infrequent weights.

---

#### Question 20
**Which optimizer combines momentum-like averaging with adaptive scaling?**
- A) Adam
- B) Plain batch gradient descent
- C) Stochastic gradient descent without memory
- D) A zero-learning-rate update

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Adam (Adaptive Moment Estimation) explicitly integrates first-moment exponential averaging (like momentum) with second-moment exponential averaging of squared gradients (like RMSProp).

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Epoch vs Iteration**:
   - $\text{Iterations per epoch} = \lceil N / B \rceil$.
   - If $N = 1000$ and $B = 50$, there are $20$ iterations per epoch.
2. **Velocity Formula Variations**:
   - In some textbooks, the learning rate $\eta$ is placed inside velocity: $v_t = \beta v_{t-1} + \eta g_t, w_t = w_{t-1} - v_t$.
   - In others, $\eta$ is placed in the parameter step: $v_t = \beta v_{t-1} + g_t, w_t = w_{t-1} - \eta v_t$.
   - Always check the given formula in the exam question prompt!
3. **Adam Bias Correction Formula**:
   - The correction is $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$, where $t$ is the exponent (iteration number), not a product $\beta_1 \cdot t$!
   - Since $0 < \beta_1 < 1$, $\beta_1^t \to 0$ as $t$ increases, so the denominator $1 - \beta_1^t \to 1$.
