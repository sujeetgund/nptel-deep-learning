# Week 7: Regularization Techniques — Bias-Variance Tradeoff, L2/L1, Early Stopping, Data Augmentation, Ensembles, and Dropout

---

## 1. Executive Summary & Exam Roadmap

Week 7 tackles the central challenge of statistical machine learning: **Generalization**:
1. **Bias-Variance Decomposition**: Mathematical derivation of expected test error: $\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$.
2. **$L_2$ Regularization (Weight Decay)**: Mathematical formulation, gradient step modification $(1 - \eta \lambda)w$, and geometric shrinkage along Hessian eigenvectors.
3. **$L_1$ Regularization (Lasso)**: Subgradient derivation, diamond-shaped constraint geometry, and sparse feature selection.
4. **Early Stopping**: Using a validation set to halt gradient descent at minimal validation loss; equivalence to implicit $L_2$ regularization.
5. **Data Augmentation & Noise Injection**: Invariance engineering and injecting noise into inputs, weights, or labels.
6. **Ensemble Methods & Bagging**: Why averaging $M$ independent estimators cuts variance by up to $\frac{1}{M}$.
7. **Dropout**: Training-time stochastic masking, inverted dropout scaling ($1/p$), preventing feature co-adaptation, and interpreting dropout as an implicit ensemble of $2^H$ sub-networks.

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 The Bias-Variance Decomposition

Assume the true data-generating process is:
$$y = f(\mathbf{x}) + \epsilon, \quad \text{where } \mathbb{E}[\epsilon] = 0, \text{Var}(\epsilon) = \sigma^2$$
Given a training set $\mathcal{D}$, we fit an estimator $\hat{f}(\mathbf{x}; \mathcal{D})$. We evaluate the expected squared prediction error at a fixed query point $\mathbf{x}$ across all possible training datasets $\mathcal{D}$:
$$\mathbb{E}_{\mathcal{D}, \epsilon}\left[ (y - \hat{f}(\mathbf{x}))^2 \right]$$

#### Mathematical Derivation:
Add and subtract $\mathbb{E}[\hat{f}(\mathbf{x})]$ inside the expectation:
$$y - \hat{f}(\mathbf{x}) = \left( f(\mathbf{x}) + \epsilon \right) - \hat{f}(\mathbf{x}) = \left( f(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})] \right) + \left( \mathbb{E}[\hat{f}(\mathbf{x})] - \hat{f}(\mathbf{x}) \right) + \epsilon$$
Squaring and expanding the three terms:
1. $\left( f(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})] \right)^2$: Deterministic w.r.t $\mathcal{D}$ and $\epsilon$. This is the **$\text{Bias}^2$**.
2. $\mathbb{E}_{\mathcal{D}}\left[ \left( \hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})] \right)^2 \right]$: This is the **$\text{Variance}$**.
3. $\mathbb{E}_{\epsilon}[\epsilon^2] = \sigma^2$: The **Irreducible Error**.
4. All cross-product terms evaluate to zero because $\mathbb{E}[\epsilon] = 0$ and $\mathbb{E}_{\mathcal{D}}[\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})]] = 0$.

Therefore:
$$\mathbb{E}\left[ (y - \hat{f}(\mathbf{x}))^2 \right] = \underbrace{\left( f(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})] \right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[ (\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2 \right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}$$

- **High Bias (Underfitting)**: Model is overly simplistic (e.g. linear model for nonlinear curve); fails to capture true trends. Train error is high, validation error is high.
- **High Variance (Overfitting)**: Model is overly complex (e.g. unregularized deep neural network); fits random noise in the training set. Train error is low, validation error is high.

---

### 2.2 $L_2$ Regularization (Weight Decay / Ridge)

#### Objective Function
Adds a penalty on the squared Euclidean norm of the weight parameters:
$$\mathcal{J}_{\text{total}}(\mathbf{w}) = \mathcal{J}_{\text{data}}(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2 = \mathcal{J}_{\text{data}}(\mathbf{w}) + \frac{\lambda}{2} \sum_{i} w_i^2$$
*(Note: Biases $b$ are typically left unregularized because regularizing bias introduces unnecessary underfitting without controlling capacity).*

#### Gradient & Parameter Update
$$\nabla_{\mathbf{w}} \mathcal{J}_{\text{total}} = \nabla_{\mathbf{w}} \mathcal{J}_{\text{data}} + \lambda \mathbf{w}$$
Performing a gradient descent update with step size $\eta$:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{J}_{\text{total}} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{J}_{\text{data}} - \eta \lambda \mathbf{w}_t$$
Rearranging:
$$\mathbf{w}_{t+1} = (1 - \eta \lambda) \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{J}_{\text{data}}$$

> [!IMPORTANT]
> Because $0 < 1 - \eta \lambda < 1$, the parameter vector is multiplied by a shrinkage factor strictly less than 1 before taking the data gradient step. This is why $L_2$ regularization is called **Weight Decay**.

#### Effect on Hessian Eigenvectors
Using a quadratic approximation of $\mathcal{J}_{\text{data}}$ near optimal weights $\mathbf{w}^*$:
$$w_i = \frac{\gamma_i}{\gamma_i + \lambda} w_i^*$$
where $\gamma_i$ are eigenvalues of the Hessian $H$:
- If $\gamma_i \gg \lambda$ (direction of high curvature/steep descent): $\frac{\gamma_i}{\gamma_i + \lambda} \approx 1$, weights are preserved.
- If $\gamma_i \ll \lambda$ (flat, uninformative direction): $\frac{\gamma_i}{\gamma_i + \lambda} \to 0$, weights are shunk toward zero!

---

### 2.3 $L_1$ Regularization (Lasso)

#### Objective Function
$$\mathcal{J}_{\text{total}}(\mathbf{w}) = \mathcal{J}_{\text{data}}(\mathbf{w}) + \alpha \|\mathbf{w}\|_1 = \mathcal{J}_{\text{data}}(\mathbf{w}) + \alpha \sum_{i} |w_i|$$
Gradient (subgradient for $w_i \neq 0$):
$$\nabla_{\mathbf{w}} \mathcal{J}_{\text{total}} = \nabla_{\mathbf{w}} \mathcal{J}_{\text{data}} + \alpha \, \text{sign}(\mathbf{w})$$
Update rule:
$$w_{t+1} = w_t - \eta \nabla \mathcal{J}_{\text{data}} - \eta \alpha \, \text{sign}(w_t)$$
- While $L_2$ shrinks weights proportionally to their magnitude ($-\lambda w$), $L_1$ subtracts a constant force ($-\alpha \, \text{sign}(w)$).
- **Sparsity**: $L_1$ drives small, unimportant weights all the way to **strictly zero**, acting as an embedded feature selector. Geometrically, the diamond-shaped $L_1$ ball has sharp corners on the coordinate axes where the elliptical loss contours intersect.

---

### 2.4 Early Stopping

#### Mechanism
1. Split available training data into a Training Set and a Validation Set.
2. At regular intervals (every epoch), evaluate the loss on the validation set.
3. Save the model checkpoint that achieves the minimal validation loss.
4. If validation loss fails to improve for a set number of epochs (the *patience* hyperparameter), terminate training early.

#### Theoretical Equivalence to $L_2$ Regularization
Starting from initial weights $\mathbf{w}_0 = \mathbf{0}$, limiting training to $T$ gradient descent steps confines the weights to a ball of radius $\approx \eta T \|\nabla \mathcal{L}\|$.
Mathematical analysis proves that for quadratic error surfaces, **Early Stopping after $T$ iterations is mathematically equivalent to $L_2$ weight decay with parameter $\lambda \approx \frac{1}{\eta T}$**.

---

### 2.5 Dataset Augmentation & Invariance Injection

- Neural networks generalize best when trained on massive, varied data distributions.
- **Data Augmentation**: Generating synthetic examples by applying realistic, label-preserving transformations:
  - Vision: Horizontal flips, random rotations ($\pm 15^\circ$), random crops, scaling, color jitter, cutout/mixup.
  - Audio: Time shifting, pitch modification, background noise addition.
  - NLP: Back-translation, synonym replacement, masked language token insertion.
- Effectively injects domain-specific prior knowledge into the model, enforcing invariance without needing architectural changes.

---

### 2.6 Ensemble Methods & Bagging

Let $M$ separate, diverse models produce predictions $\hat{y}_1, \dots, \hat{y}_M$. The ensemble prediction is:
$$\hat{y}_{\text{ensemble}} = \frac{1}{M} \sum_{m=1}^M \hat{y}_m$$
Assume each model's prediction error $\epsilon_m = \hat{y}_m - y$ has zero mean $\mathbb{E}[\epsilon_m] = 0$, individual variance $\mathbb{E}[\epsilon_m^2] = v$, and average pairwise covariance $c = \mathbb{E}[\epsilon_i \epsilon_j]$ ($i \neq j$).
The expected squared error of the ensemble is:
$$\mathbb{E}\left[ \left( \frac{1}{M} \sum_{m=1}^M \epsilon_m \right)^2 \right] = \frac{1}{M} v + \frac{M - 1}{M} c$$
- If the models are **completely uncorrelated** ($c = 0$), the ensemble variance drops to $\frac{v}{M}$, which approaches $0$ as $M \to \infty$!
- In practice, even partially uncorrelated errors significantly reduce prediction variance.

---

### 2.7 Dropout (Srivastava, Hinton, et al., 2014)

#### Training Time
During each forward pass on a mini-batch, each hidden unit $j$ in a dropout layer is randomly retained with keep probability $p \in (0, 1]$ (or dropped with probability $1 - p$):
$$r_j \sim \text{Bernoulli}(p)$$
$$\tilde{h}_j = r_j \cdot h_j$$

#### Test Time (Standard Dropout vs. Inverted Dropout)
- **Standard Dropout**: At test time, all neurons remain active, but the incoming weight matrix is scaled down by $p$:
  $$\mathbf{W}_{\text{test}} = p \mathbf{W}_{\text{train}}$$
  This ensures that the expected input to each neuron at test time matches its expectation during training: $\mathbb{E}[\tilde{h}_j] = p h_j$.
- **Inverted Dropout (Modern standard)**: Scale activations during training by $\frac{1}{p}$:
  $$\tilde{h}_j = \frac{r_j}{p} h_j$$
  This preserves the expected activation magnitude during training ($\mathbb{E}[\tilde{h}_j] = h_j$), meaning **no modifications or scaling are needed during inference/testing**!

#### Why Dropout Works
1. **Prevents Feature Co-adaptation**: Neurons cannot rely on specific other neurons being present. Every neuron must learn features that are independently robust and useful in random combinations.
2. **Implicit Ensemble**: A network with $H$ dropout units can form $2^H$ possible thinned subnetworks. Dropout implicitly trains an ensemble of $2^H$ sub-networks that share parameters!

---

## 3. Core Formulas & Cheat Sheet

| Regularizer | Mathematical Formula | Key Operational Property |
|:---|:---|:---|
| **Bias-Variance** | $\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2 + \text{Var} + \sigma^2$ | Fundamental trade-off governing model capacity |
| **$L_2$ Regularization** | $\mathcal{J} = \mathcal{J}_{\text{data}} + \frac{\lambda}{2}\|\mathbf{w}\|^2$ | Gradient update: $(1 - \eta \lambda)\mathbf{w} - \eta \nabla \mathcal{J}_{\text{data}}$ |
| **$L_1$ Regularization** | $\mathcal{J} = \mathcal{J}_{\text{data}} + \alpha \|\mathbf{w}\|_1$ | Drives weights to exact zero; produces sparse models |
| **Early Stopping** | Halt when $\mathcal{L}_{\text{val}}$ reaches minimum | Equivalent to $L_2$ regularization with $\lambda \propto \frac{1}{\eta T}$ |
| **Augmentation Size** | $N_{\text{new}} = N_{\text{orig}} \times (1 + k_{\text{aug}})$ | Increases effective training size by exposing transformations |
| **Dropout Keep Prob** | $\mathbb{E}[\text{active units}] = p \times N_{\text{units}}$ | $p = 0.8$ with 50 units $\implies 40$ active units |
| **Inverted Dropout** | $\tilde{\mathbf{h}} = \frac{\mathbf{r} \odot \mathbf{h}}{p}$ | Requires no test-time scaling |
| **Ensemble Averaging** | $\text{Var}_{\text{ens}} = \frac{v}{M} + \frac{M-1}{M}c$ | Reduces variance toward $\frac{v}{M}$ as correlation $c \to 0$ |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Water Demand Forecasting for a Residential Zone (Questions 1 to 5)

#### Context & Data
A municipal analytics team predicts next-day water demand:
- Simple linear model: Similar predictions across different training samples, but far from true demand.
- Neural network: Fits training data closely, but predictions change substantially across different training subsets.
- True demand: $y = 48$ thousand liters.
- Predictions of the neural network from 4 training subsets:
  $$41, 49, 45, 53$$
- Error decomposition:
  $$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$$

---

#### Question 1
**Which description best matches the simple linear model?**
- A) High bias and low variance
- B) Low bias and high variance
- C) Low bias and low variance
- D) High variance and zero bias

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  - Stable predictions across different training subsets indicate low variability/sensitivity, meaning **low variance**.
  - Being consistently far from the true target value indicates systematic under-fitting, meaning **high bias**.

---

#### Question 2
**What is the average neural-network prediction for the given input?**
- A) 46
- B) 48
- C) 47
- D) 50

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$\mathbb{E}[\hat{f}(x)] = \frac{41 + 49 + 45 + 53}{4} = \frac{188}{4} = 47$$

---

#### Question 3
**What is the squared bias of the neural-network prediction?**
- A) 0
- B) 1
- C) 4
- D) 9

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$\text{Bias} = \mathbb{E}[\hat{f}(x)] - f(x) = 47 - 48 = -1$$
  $$\text{Squared Bias} = (\text{Bias})^2 = (-1)^2 = 1$$

---

#### Question 4
**What does the large variation among the neural-network predictions indicate?**
- A) The model has no irreducible error.
- B) The model has high variance under the case assumptions.
- C) The output labels are unavailable for the described model.
- D) The model is too simple for the data.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Large differences in predictions (ranging from 41 to 53) when trained on different random samples of training data is the hallmark definition of **high variance** (overfitting).

---

#### Question 5
**Which action is most directly aimed at reducing the neural network's variance?**
- A) Add a suitable regularization method.
- B) Remove the validation set entirely.
- C) Increase model depth without constraints.
- D) Train until training loss becomes zero.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Regularization techniques (such as $L_2$ weight decay, dropout, data augmentation, and early stopping) constrain model complexity, directly curbing overfitting and reducing model variance.

---

### Case Study 2: Small-Business Loan Risk Scoring (Questions 6 to 10)

#### Context & Data
Logistic regression with $L_2$ regularization:
$$\mathcal{J}_{\text{total}}(\mathbf{w}) = \mathcal{J}_{\text{data}}(\mathbf{w}) + \alpha \sum_{i} w_i^2$$
Given:
- Current parameter: $w = 3$
- Data-loss gradient: $\frac{\partial \mathcal{J}_{\text{data}}}{\partial w} = 4$
- Regularization strength: $\alpha = 2$
- Learning rate: $\eta = 0.1$
- Regularized gradient formula:
  $$\frac{\partial \mathcal{J}_{\text{total}}}{\partial w} = 4 + 2\alpha w$$

---

#### Question 6
**What is the main purpose of adding $L_2$ regularization here?**
- A) To remove the need for training data for the described model
- B) To discourage large weights for the described model
- C) To guarantee perfect validation accuracy
- D) To convert logistic regression into clustering

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The quadratic penalty $\alpha \sum w_i^2$ penalizes large weight magnitudes, keeping the decision boundary smooth and preventing individual features from exerting disproportionate leverage.

---

#### Question 7
**What is the regularization contribution $2\alpha w$ to the gradient?**
- A) 10
- B) 6
- C) 8
- D) 12

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$\text{Regularization gradient} = 2 \alpha w = 2(2)(3) = 12$$

---

#### Question 8
**What is the full regularized gradient for this parameter?**
- A) 16
- B) 12
- C) 10
- D) 14

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$\frac{\partial \mathcal{J}_{\text{total}}}{\partial w} = \frac{\partial \mathcal{J}_{\text{data}}}{\partial w} + 2\alpha w = 4 + 12 = 16$$

---

#### Question 9
**What is the updated value of $w$ after one gradient-descent step?**
- A) 2.0
- B) 2.6
- C) 1.4
- D) 3.6

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$w_{\text{new}} = w - \eta \frac{\partial \mathcal{J}_{\text{total}}}{\partial w} = 3 - 0.1(16) = 3 - 1.6 = 1.4$$

---

#### Question 10
**Which outcome is expected after applying an appropriate $L_2$ penalty?**
- A) Training accuracy must become zero for the data shown here.
- B) Validation performance may improve due to reduced overfitting.
- C) All feature weights must become exactly zero.
- D) The model stops using gradient descent under the given conditions.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  By restricting over-parameterization, $L_2$ regularization reduces variance on unseen validation samples, often improving test generalization even if training accuracy drops slightly.

---

### Case Study 3: Energy Consumption Prediction for an Office Campus (Questions 11 to 15)

#### Context & Data
Training and validation loss progression:
| Epoch | Training Loss | Validation Loss |
|:---:|:---:|:---:|
| 5 | 0.62 | 0.70 |
| 10 | 0.44 | 0.52 |
| 15 | 0.31 | 0.43 |
| 20 | 0.22 | 0.46 |
| 25 | 0.16 | 0.51 |

---

#### Question 11
**At which epoch is the validation loss minimum?**
- A) Epoch 10
- B) Epoch 20
- C) Epoch 15
- D) Epoch 25

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Inspecting the validation loss column: $\{0.70, 0.52, 0.43, 0.46, 0.51\}$. The lowest value is $0.43$, which occurs at **Epoch 15**.

---

#### Question 12
**What pattern after epoch 15 suggests overfitting?**
- A) Both losses increase together.
- B) Training loss increases while validation loss decreases.
- C) Training loss becomes exactly zero.
- D) Training loss decreases while validation loss increases.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  From Epoch 15 to 25, the training loss steadily declines ($0.31 \to 0.22 \to 0.16$) while the validation loss rises ($0.43 \to 0.46 \to 0.51$). Divergence where training error drops while validation error deteriorates is the classic indicator of **overfitting**.

---

#### Question 13
**Why should early stopping monitor validation loss rather than only training loss?**
- A) Validation loss reflects performance on unseen data.
- B) Training loss is not computed during neural-network training.
- C) Validation loss is always exactly equal to test loss.
- D) Training loss cannot decrease across epochs.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Training loss solely reflects memorization of seen samples. Validation loss serves as an unbiased proxy for out-of-sample generalization.

---

#### Question 14
**Which statement about early stopping is correct?**
- A) It guarantees the globally optimal weights.
- B) It acts as a regularization method by limiting over-training.
- C) It prevents all forms of noise in input data.
- D) It removes the need for a validation set under the case assumptions.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Early stopping halts gradient descent before weights grow excessively large to fit idiosyncrasies of the training set, functioning as an effective regularizer.

---

#### Question 15
**If the team keeps the model from epoch 25 instead of epoch 15, what is the main risk?**
- A) The training set becomes larger automatically.
- B) The model cannot produce numerical predictions.
- C) The validation loss is ignored by definition under the case assumptions.
- D) The model may generalize worse despite lower training loss.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  At epoch 25, the model has overfit the training data (validation loss is $0.51$ vs $0.43$ at epoch 15), meaning it will generalize worse on future unseen days.

---

### Case Study 4: Product-Surface Image Quality Inspection (Questions 16 to 20)

#### Context & Data
A dataset contains $N = 900$ labeled images.
- Train split: $70\%$, Validation split: $30\%$.
- Data augmentation: Each original training image generates **two additional** augmented images.
- Dropout: Hidden layer keep probability $p = 0.8$.
- Ensemble: Averaging 3 independently trained CNNs.

---

#### Question 16
**How many original images are in the training split before augmentation?**
- A) 270
- B) 900
- C) 630
- D) 450

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$\text{Training images} = 0.70 \times 900 = 630$$

---

#### Question 17
**What is the effective training-set size after adding two augmented images per original training image?**
- A) 1890
- B) 1260
- C) 630
- D) 1530

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Each original image now yields $1 \text{ (original)} + 2 \text{ (augmented)} = 3$ training instances.
  $$\text{Effective size} = 630 \times 3 = 1890$$

---

#### Question 18
**What is the main reason augmentation can improve validation accuracy?**
- A) It changes all class labels randomly in the stated case setting.
- B) It removes the need to learn features.
- C) It guarantees that training loss is zero.
- D) It exposes the model to realistic input variations.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Augmentation simulates realistic variations (rotations, illumination shifts, noise) that occur in real-world deployment, preventing the CNN from latching onto spurious pixel-level artifacts.

---

#### Question 19
**During dropout training with keep probability $p = 0.8$, approximately how many units remain active in a layer of 50 hidden units?**
- A) 25
- B) 10
- C) 40
- D) 50

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The number of active units follows a Binomial distribution $B(n, p)$ with expected value:
  $$\mathbb{E}[\text{active units}] = n \times p = 50 \times 0.8 = 40$$

---

#### Question 20
**Why can averaging three independently trained CNNs improve reliability?**
- A) It increases the learning rate of each CNN.
- B) It can reduce variance by combining diverse predictions.
- C) It removes all bias from every CNN for the data shown here.
- D) It makes the dataset labels unnecessary.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Independent models make errors on different subsets of the data space. Averaging their posterior probabilities cancels out idiosyncratic mistakes, significantly reducing prediction variance.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Augmentation Multiplier**:
   - Pay close attention to wording: "adding two augmented images *per* original image" means the total is $1 + 2 = 3\times$ the original size, **not** $2\times$!
2. **Dropout Probability ($p$ vs $1-p$)**:
   - Check whether the problem defines $p$ as the **retention/keep probability** or the **drop probability**.
   - If keep probability is $0.8$, then $80\%$ of units are active.
   - If drop probability is $0.2$, then $1 - 0.2 = 0.8$ are active.
3. **Weight Decay Factor**:
   - In $L_2$ regularization, the shrinkage factor is $(1 - \eta \lambda)$. If $\eta = 0.01$ and $\lambda = 0.1$, weights are scaled by $1 - 0.001 = 0.999$ each step.
