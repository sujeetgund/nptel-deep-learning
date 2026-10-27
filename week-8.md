# Week 8: Training Deep Networks — Greedy Pre-training, Modern Activations, Initialization, and Batch Normalization

---

## 1. Executive Summary & Exam Roadmap

Week 8 explores the breakthroughs that enabled the training of deep architectures without falling victim to vanishing or exploding gradients:
1. **Vanishing and Exploding Gradients Pathology**: The mathematical cause of gradient decay through deep chains of saturated activations ($\sigma'(z) \le 0.25$).
2. **Greedy Layer-wise Unsupervised Pre-training (Hinton, Bengio 2006)**: Layer-by-layer unsupervised feature discovery, freezing weights, and end-to-end supervised fine-tuning.
3. **Modern Activation Functions**:
   - **ReLU**: Non-saturating positive regime ($f'(x) = 1$), computational speed, and the "Dying ReLU" pathology.
   - **Leaky ReLU, PReLU, ELU, and tanh**: Preserving negative gradients and zero-centered dynamics.
4. **Principled Weight Initialization**:
   - Preserving variance across layers ($\text{Var}(y) \approx \text{Var}(x)$).
   - **Xavier (Glorot) Initialization**: $\text{Var}(W) = \frac{1}{n_{\text{in}}}$ or $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ (optimal for tanh/sigmoid).
   - **He (Kaiming) Initialization**: $\text{Var}(W) = \frac{2}{n_{\text{in}}}$ (compensating for ReLU's half-space clipping).
5. **Batch Normalization (Ioffe & Szegedy, 2015)**: Reducing internal covariate shift, batch normalization transform equations, learnable scale ($\gamma$) and shift ($\beta$), and test-time running statistics.

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 The Vanishing and Exploding Gradient Problem

Consider an $L$-layer feedforward network with scalar activations:
$$\frac{\partial \mathcal{L}}{\partial a_1} = \frac{\partial \mathcal{L}}{\partial a_L} \cdot \prod_{l=1}^{L-1} \frac{\partial a_{l+1}}{\partial a_l} = \frac{\partial \mathcal{L}}{\partial a_L} \cdot \prod_{l=1}^{L-1} \left( W_{l+1} \cdot g'(a_l) \right)$$
In vectorized form:
$$\nabla_{\mathbf{a}_1} \mathcal{L} = \left[ \prod_{l=2}^L \mathbf{W}_l^\top \text{diag}(g'(\mathbf{a}_l)) \right] \nabla_{\mathbf{a}_L} \mathcal{L}$$

#### The Two Failure Modes
1. **Vanishing Gradients**:
   - If the singular values or eigenvalues of $\mathbf{W}_l^\top \text{diag}(g'(\mathbf{a}_l))$ are strictly less than 1, multiplying $L$ such matrices causes the gradient to decay **exponentially with depth**:
     $$\|\nabla_{\mathbf{a}_1} \mathcal{L}\| \propto \lambda^L \to 0 \quad (\text{for } \lambda < 1)$$
   - For a sigmoid neuron, $\max g'(z) = 0.25$. Even if weights are of order 1, the product includes $(0.25)^L$. For $L = 10$, $(0.25)^{10} \approx 9.5 \times 10^{-7}$, rendering the earliest layers completely frozen and incapable of learning.
2. **Exploding Gradients**:
   - If the eigenvalues are significantly greater than 1, the gradient norm grows exponentially ($\lambda^L \to \infty$), causing numerical overflow (`NaN`), wild parameter oscillations, and optimization divergence.

---

### 2.2 Greedy Layer-wise Pre-training (2006 Breakthrough)

Before 2006, training networks deeper than 2–3 layers from random initializations consistently failed. Geoff Hinton and Yoshua Bengio solved this using **Greedy Layer-wise Pre-training**:

```
Step 1: Train Layer 1 Autoencoder (Input x ---> Latent h_1 ---> Output x_hat)
Step 2: Freeze Layer 1 weights W_1. Compute h_1 = f(W_1 * x).
Step 3: Train Layer 2 Autoencoder (Input h_1 ---> Latent h_2 ---> Output h_1_hat)
Step 4: Repeat greedily up to Layer L.
Step 5: Attach classification/regression head. Fine-tune ALL layers via Backprop.
```

#### Why It Works: Regularization & Optimization Perspective
1. **Unsupervised Discovery**: Uses vast unlabeled data to discover the underlying geometric manifold of the data distribution.
2. **Better Optimization Basin**: The error surface of a deep network is non-convex with numerous poor local plateaus. Pre-training initializes the weights in an attractor basin close to a good local optimum, avoiding bad saddle regions.

---

### 2.3 Modern Activation Functions

```
 Sigmoid                 tanh                   ReLU                   Leaky ReLU
   |   /                  |   /                  |  /                    |  /
   |  /                   |  /                   | /                     | /
---+---                ---+---                ---+---                 ---+---
  /|                     /|                      |                      /|
 / |                    / |                      |                    /  |
```

#### 1. Rectified Linear Unit (ReLU) (Nair & Hinton, 2010)
$$f(x) = \max(0, x)$$
$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$
- **Advantages**:
  - **No saturation in the positive domain**: For any $x > 0$, the derivative is strictly $1$. This enables gradients to flow through dozens of layers without decay!
  - **Extreme computational efficiency**: Computed via simple hardware thresholding (no costly transcendentals like $\exp$).
  - **Sparse representations**: Approximately $50\%$ of neurons are zero for random inputs.
- **The "Dying ReLU" Problem**:
  - If a large gradient updates a neuron such that its pre-activation is negative for *all* samples in the dataset, its activation is always $0$ and its gradient is always $0$. The neuron becomes permanently dead and never recovers.

#### 2. Leaky ReLU & PReLU
$$f(x) = \max(\alpha x, x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \le 0 \end{cases}$$
- In Leaky ReLU, $\alpha$ is a small fixed constant (e.g., $\alpha = 0.01$).
- In Parametric ReLU (**PReLU**), $\alpha$ is treated as a learnable parameter updated via backprop.
- Ensures a small, non-zero gradient ($\alpha$) flows even when $x < 0$, preventing dead neurons.

#### 3. Exponential Linear Unit (ELU)
$$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \le 0 \end{cases}$$
- Smooth everywhere; brings mean activations closer to zero, accelerating learning.

---

### 2.4 Principled Weight Initialization

To prevent signals from either vanishing to zero or exploding to infinity as they traverse forward and backward through $L$ layers, we require the **variance of activations to remain constant across layers**:
$$\text{Var}(h_l) = \text{Var}(h_{l-1})$$

Let $y = \sum_{i=1}^{n_{\text{in}}} w_i x_i$, where $w_i$ and $x_i$ are independent, zero-mean random variables:
$$\text{Var}(y) = \sum_{i=1}^{n_{\text{in}}} \text{Var}(w_i x_i) = \sum_{i=1}^{n_{\text{in}}} \text{Var}(w_i)\text{Var}(x_i) = n_{\text{in}} \, \text{Var}(w) \, \text{Var}(x)$$
To ensure $\text{Var}(y) = \text{Var}(x)$, we must set:
$$n_{\text{in}} \, \text{Var}(w) = 1 \implies \mathbf{\text{Var}(w) = \frac{1}{n_{\text{in}}}}$$

#### 1. Xavier (Glorot) Initialization (for Symmetric Activations: tanh, Sigmoid)
Taking both forward flow ($n_{\text{in}}$) and backward gradient flow ($n_{\text{out}}$) into account:
$$\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$
- **Gaussian**: $W \sim \mathcal{N}\left(0, \, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$
- **Uniform**: $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \, +\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$

#### 2. He (Kaiming) Initialization (for ReLU Activations)
Because ReLU sets all negative inputs to zero, it discards half of the variance:
$$\text{Var}(f(x)) = \frac{1}{2} \text{Var}(x)$$
To counteract this $50\%$ energy loss per layer, the weight variance must be **doubled**:
$$\mathbf{\text{Var}(w) = \frac{2}{n_{\text{in}}}}$$
- **Gaussian**: $W \sim \mathcal{N}\left(0, \, \frac{2}{n_{\text{in}}}\right)$
- **Uniform**: $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \, +\sqrt{\frac{6}{n_{\text{in}}}}\right)$

---

### 2.5 Batch Normalization (Ioffe & Szegedy, 2015)

#### Internal Covariate Shift
During training, as parameters in earlier layers update, the distribution of inputs to later layers changes continuously. Later layers must constantly adapt to a non-stationary input distribution, forcing the use of very small learning rates.

#### The Batch Normalizing Transform
For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ of a specific neuron's pre-activation:
1. **Mini-batch Mean**:
   $$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i$$
2. **Mini-batch Variance**:
   $$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$$
3. **Normalize to Zero Mean, Unit Variance**:
   $$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$
   *(where $\epsilon > 0$ is a stability constant, typically $10^{-5}$).*
4. **Scale and Shift (Learnable Affine Transformation)**:
   $$y_i = \text{BN}_{\gamma, \beta}(x_i) = \gamma \hat{x}_i + \beta$$
   where $\gamma$ (scale) and $\beta$ (shift) are learnable parameters optimized via backpropagation!
   - *Why include $\gamma$ and $\beta$?* If normalizing restricts the representation capacity (e.g. forcing a sigmoid into its linear regime), the network can adaptively learn $\gamma = \sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}$ and $\beta = \mu_{\mathcal{B}}$ to restore the original identity mapping.

#### Placement in Architecture
Batch Normalization is standardly placed **immediately after the linear transformation and right before the nonlinear activation function**:
$$\mathbf{x} \longrightarrow [\mathbf{W}\mathbf{x}] \longrightarrow [\text{Batch Norm}] \longrightarrow [\text{Activation } g(\cdot)] \longrightarrow \mathbf{h}$$

#### Test Time / Inference
During inference, mini-batches may not exist (e.g. evaluating 1 sample). The network uses fixed running moving averages computed during training:
$$\mu_{\text{test}} = \mathbb{E}[\mu_{\mathcal{B}}], \quad \sigma^2_{\text{test}} = \frac{m}{m-1}\mathbb{E}[\sigma^2_{\mathcal{B}}]$$
$$\hat{x} = \frac{x - \mu_{\text{test}}}{\sqrt{\sigma^2_{\text{test}} + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

---

## 3. Core Formulas & Cheat Sheet

| Technique | Mathematical Equation | Key Purpose / Rule of Thumb |
|:---|:---|:---|
| **ReLU** | $f(x) = \max(0, x)$ | Gradient is 1 for $x > 0$; solves vanishing gradients |
| **Leaky ReLU** | $f(x) = \max(\alpha x, x)$ | Prevents dead neurons; $\alpha \approx 0.01$ |
| **Xavier Variance** | $\text{Var}(W) = \frac{1}{n_{\text{in}}}$ or $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | Used with tanh / Sigmoid activations |
| **He Variance** | $\text{Var}(W) = \frac{2}{n_{\text{in}}}$ | Used with ReLU / Leaky ReLU activations |
| **BN Mean** | $\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i$ | Centering mini-batch |
| **BN Variance** | $\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$ | Normalizing mini-batch dispersion |
| **BN Normalization**| $\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$ | Produces zero mean and unit variance |
| **BN Scale & Shift**| $y_i = \gamma \hat{x}_i + \beta$ | Restores network representation power |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Insurance Claim Approval Prediction (Questions 1 to 5)

#### Context & Data
A deep feedforward network with 8 hidden layers classifies insurance claims.
Output layer: Sigmoid (binary probability).
Hidden layers originally: Sigmoid.
Hidden layers replaced with:
$$\text{ReLU}(x) = \max(0, x)$$
Given values: $\sigma(2) \approx 0.88$, $\sigma(-4) \approx 0.018$.

---

#### Question 1
**Why can sigmoid hidden layers slow down training in a deep network?**
- A) They can saturate and produce very small gradients.
- B) They always output negative values under the given conditions.
- C) They remove all nonlinear behavior.
- D) They force every weight to become zero.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  For inputs with large positive or negative magnitudes, the sigmoid function saturates near 1 or 0, where its derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \to 0$. Multiplying these tiny fractions through 8 hidden layers drives the gradients to near zero (vanishing gradients), halting learning.

---

#### Question 2
**What is $\text{ReLU}(-4)$?**
- A) -4
- B) -1
- C) 0
- D) 4

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$\text{ReLU}(-4) = \max(0, -4) = 0$$

---

#### Question 3
**What is $\text{ReLU}(2)$?**
- A) 0
- B) -2
- C) 1
- D) 2

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$\text{ReLU}(2) = \max(0, 2) = 2$$

---

#### Question 4
**Which property makes ReLU helpful for positive pre-activation values?**
- A) It bounds hidden outputs between 0 and 1, as sigmoid would.
- B) It avoids saturation and keeps a useful gradient.
- C) It converts classification into clustering.
- D) It makes the loss function unnecessary.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  For all $x > 0$, the derivative of ReLU is constantly $1$. Gradients flow backward without attenuation, effectively eliminating saturation and the vanishing gradient problem.

---

#### Question 5
**Why is the output layer still kept as sigmoid in this binary classifier?**
- A) It maps the final score to a probability-like value between 0 and 1.
- B) It is required for every hidden layer for the described model.
- C) It guarantees no validation error after training on every future test case.
- D) It prevents all negative input features for the data shown here.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The task is binary classification. A sigmoid output squashes the real-valued logit into $(0, 1)$, allowing it to be directly interpreted as the posterior probability $P(y = 1|\mathbf{x})$.

---

### Case Study 2: Solar Power Forecasting for a Rooftop Plant (Questions 6 to 10)

#### Context & Data
A 7-hidden-layer network with ReLU activations predicts solar generation.
Formulas:
- Xavier Initialization: $\text{Var}(w) = \frac{1}{n}$
- He Initialization: $\text{Var}(w) = \frac{2}{n}$
Hidden layer with $n = 200$ inputs.

---

#### Question 6
**What problem can poor initialization create in a deep network?**
- A) It automatically removes all noise from data.
- B) It can cause activations or gradients to explode or vanish.
- C) It guarantees underfitting only under the given conditions.
- D) It changes regression targets into class labels.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  If initial weights are too small, forward activations and backward gradients shrink exponentially with layer depth (vanishing). If initial weights are too large, activations and gradients explode, causing unstable training.

---

#### Question 7
**What is the Xavier variance for $n = 200$?**
- A) 0.020
- B) 0.200
- C) 0.100
- D) 0.005

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$\text{Var}(w) = \frac{1}{n} = \frac{1}{200} = 0.005$$

---

#### Question 8
**What is the He variance for $n = 200$?**
- A) 0.010
- B) 0.005
- C) 0.020
- D) 0.100

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$\text{Var}(w) = \frac{2}{n} = \frac{2}{200} = 0.010$$

---

#### Question 9
**Why is He initialization more suitable for these hidden layers?**
- A) The output layer uses no activation.
- B) The network has only one layer.
- C) The hidden layers use ReLU activations.
- D) The target variable is binary in the stated case setting.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  He initialization is specifically derived for ReLU activations, where the extra factor of 2 compensates for the fact that half of the inputs are clipped to zero.

---

#### Question 10
**Which observation suggests the new initialization improved training stability?**
- A) Hidden activations are all forced to zero.
- B) Loss decreases more smoothly across epochs.
- C) No validation set is needed in the stated case setting.
- D) All weights become identical.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Stable gradient and activation variances prevent numerical oscillations, resulting in steady, monotonic decreases in training loss across epochs.

---

### Case Study 3: Warehouse Sensor Fault Detection (Questions 11 to 15)

#### Context & Data
A deep neural network with 10 hidden layers detects sensor faults. Direct end-to-end supervised training is slow due to vanishing gradients in early layers. The team uses **greedy layer-wise pre-training**.

---

#### Question 11
**What is the main goal of greedy layer-wise pretraining here?**
- A) To delete the hidden layers before training.
- B) To avoid using input features during both unsupervised pretraining and fine-tuning.
- C) To give each layer a useful initialization before supervised fine-tuning.
- D) To guarantee zero test error in this specific scenario.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Greedy layer-wise pre-training provides a structured initialization for each layer's weights based on input data geometry, placing the network in a favorable region of parameter space before full backpropagation.

---

#### Question 12
**During pretraining of the second hidden layer, what happens to the already trained first layer?**
- A) It is randomly reinitialized in the stated case setting.
- B) It is removed from the model.
- C) It is trained with the output labels only.
- D) It is kept fixed while the next layer is trained.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  In greedy pre-training, previously trained layers are frozen, and their output representations serve as fixed inputs for training the subsequent layer.

---

#### Question 13
**Why is the first pretraining objective described as unsupervised?**
- A) It uses the input data without using the abnormal-condition labels.
- B) It uses no numerical features under the given conditions.
- C) It uses only the final classifier output for the data shown here.
- D) It requires the validation labels to be hidden in the stated case setting.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The pre-training phase relies solely on input reconstruction (e.g. autoencoders) without utilizing target class labels.

---

#### Question 14
**What happens after all hidden layers have been pretrained?**
- A) The network is discarded under the given conditions.
- B) The full model is fine-tuned with the supervised task objective.
- C) The input features are replaced by random values.
- D) The loss function is no longer required in the stated case setting.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Once all layers are initialized, the complete network is un-frozen and fine-tuned end-to-end using standard supervised backpropagation on the labeled task.

---

#### Question 15
**Which issue is layer-wise pretraining intended to reduce?**
- A) The need to store data files in this specific scenario.
- B) The number of output classes under the case assumptions.
- C) The difficulty of training early layers in a very deep network.
- D) The existence of nonlinear activations under the given conditions.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Because supervised gradients vanish before reaching early layers in very deep networks, unsupervised pre-training ensures that early layers already compute meaningful hierarchical features.

---

### Case Study 4: Laboratory Test Result Prediction (Questions 16 to 20)

#### Context & Data
A 6-hidden-layer network predicts test repeats.
Batch Normalization is inserted between linear transformations and ReLU.
For one mini-batch, a neuron's pre-activations are:
$$1, 3, 5, 7$$

---

#### Question 16
**What problem does batch normalization primarily address?**
- A) It changes labels from binary to continuous.
- B) It removes the need for backpropagation.
- C) It guarantees a perfect classifier in the stated case setting.
- D) It reduces shifting layer-input distributions during training.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Batch Normalization reduces internal covariate shift by stabilizing the mean and variance of layer inputs across mini-batches during training.

---

#### Question 17
**What is the mean of the mini-batch values $1, 3, 5, 7$?**
- A) 3
- B) 4
- C) 5
- D) 16

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$\mu_{\mathcal{B}} = \frac{1 + 3 + 5 + 7}{4} = \frac{16}{4} = 4$$

---

#### Question 18
**What is the batch variance using the average squared deviation?**
- A) 4
- B) 6
- C) 3
- D) 5

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Compute squared deviations from mean $\mu_{\mathcal{B}} = 4$:
  - $(1 - 4)^2 = (-3)^2 = 9$
  - $(3 - 4)^2 = (-1)^2 = 1$
  - $(5 - 4)^2 = 1^2 = 1$
  - $(7 - 4)^2 = 3^2 = 9$
  $$\sigma_{\mathcal{B}}^2 = \frac{9 + 1 + 1 + 9}{4} = \frac{20}{4} = 5$$

---

#### Question 19
**Where is batch normalization placed in this case?**
- A) Only after the output probability for the described model.
- B) Before every input feature is collected.
- C) Between the linear transformation and ReLU activation.
- D) After the final prediction is thresholded.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The standard placement introduced by Ioffe & Szegedy is directly after the linear transformation $\mathbf{W}\mathbf{x} + \mathbf{b}$ and before the nonlinear activation function (ReLU).

---

#### Question 20
**Why are learnable scale and shift parameters useful in batch normalization?**
- A) They allow the network to choose an appropriate normalized representation.
- B) They delete the mini-batch statistics in this specific scenario.
- C) They prevent gradients from being computed under the case assumptions.
- D) They force every activation to remain exactly zero for the described model.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The parameters $\gamma$ and $\beta$ allow the network to learn any scale and shift it requires, ensuring that normalization does not restrict the model's expressive capacity.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **He vs Xavier Factor**:
   - For **tanh / Sigmoid**: Use Xavier: $\text{Var}(W) = \frac{1}{n_{\text{in}}}$.
   - For **ReLU**: Use He: $\text{Var}(W) = \frac{2}{n_{\text{in}}}$.
   - Forgetting the factor of $2$ for ReLU is one of the most frequent mistakes in numerical MCQs!
2. **ReLU Derivative at Zero**:
   - Technically $f'(0)$ is undefined, but by convention in deep learning frameworks, it is set to either $0$ or $1$ (a subgradient).
3. **Batch Norm Parameters Count**:
   - For a layer with $D$ channels/neurons, Batch Normalization introduces **$2D$ learnable parameters**: $D$ scale parameters ($\gamma$) and $D$ shift parameters ($\beta$).
