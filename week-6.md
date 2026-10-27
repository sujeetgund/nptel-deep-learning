# Week 6: Autoencoders, Relation to PCA, and Regularized Architectures

---

## 1. Executive Summary & Exam Roadmap

Week 6 shifts the focus from supervised learning to unsupervised feature learning via **Autoencoders**:
1. **Autoencoder Fundamentals**: Encoder function $f(\mathbf{x})$, bottleneck latent representation $\mathbf{h}$, decoder function $g(\mathbf{h})$, reconstruction loss $\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}})$, and tied weights ($\mathbf{W}' = \mathbf{W}^\top$).
2. **Autoencoders vs. PCA**: The Bourlard & Kamp (1988) theorem proving that a linear autoencoder with squared error loss spans the exact same $k$-dimensional subspace as the top $k$ principal components of PCA.
3. **Undercomplete vs. Overcomplete Autoencoders**:
   - Undercomplete ($k < d$): Imposes an architectural bottleneck.
   - Overcomplete ($k > d$): Can memorize the identity mapping without regularization.
4. **Regularized Autoencoders**:
   - **Sparse Autoencoders**: KL divergence and $L_1$ penalties enforcing low average neuronal activation.
   - **Denoising Autoencoders (DAE)**: Reconstructing clean inputs $\mathbf{x}$ from stochastically corrupted inputs $\tilde{\mathbf{x}}$ (manifold learning).
   - **Contractive Autoencoders (CAE)**: Penalizing the Frobenius norm of the encoder's Jacobian matrix $\|J_f(\mathbf{x})\|_F^2$ to guarantee local invariance.

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 Basic Autoencoder Architecture

An autoencoder is a feedforward neural network trained to reconstruct its own input in an unsupervised manner:

```
Input x (d-dim) ---> [ Encoder: W, b ] ---> Latent Code h (k-dim) ---> [ Decoder: W', b' ] ---> Output x_hat (d-dim)
```

#### The Encoder
Maps an input vector $\mathbf{x} \in \mathbb{R}^d$ to a latent code $\mathbf{h} \in \mathbb{R}^k$:
$$\mathbf{h} = f(\mathbf{x}) = s_f(\mathbf{W}\mathbf{x} + \mathbf{b})$$
where $\mathbf{W} \in \mathbb{R}^{k \times d}$, $\mathbf{b} \in \mathbb{R}^k$, and $s_f$ is an activation function (e.g., sigmoid, tanh, or identity).

#### The Decoder
Maps the latent code $\mathbf{h}$ back to the input space as a reconstruction $\hat{\mathbf{x}} \in \mathbb{R}^d$:
$$\hat{\mathbf{x}} = g(\mathbf{h}) = s_g(\mathbf{W}'\mathbf{h} + \mathbf{b}')$$
where $\mathbf{W}' \in \mathbb{R}^{d \times k}$, $\mathbf{b}' \in \mathbb{R}^d$, and $s_g$ is the decoder activation function.

#### Tied Weights
To halve the number of parameters and regularize the model, the decoder weights are often "tied" to the encoder weights:
$$\mathbf{W}' = \mathbf{W}^\top$$

#### Reconstruction Loss Functions
1. **Mean Squared Error (for real-valued continuous inputs)**:
   $$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{2} \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \frac{1}{2} \sum_{j=1}^d (x_j - \hat{x}_j)^2$$
2. **Binary Cross-Entropy (for normalized inputs $x_j \in [0, 1]$)**:
   $$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = -\sum_{j=1}^d \left[ x_j \log \hat{x}_j + (1 - x_j) \log(1 - \hat{x}_j) \right]$$

---

### 2.2 Deep Link: Linear Autoencoders and PCA

#### The Bourlard & Kamp Theorem (1988)
Consider an undercomplete autoencoder where:
1. Both encoder and decoder activation functions are strictly **linear (identity)**:
   $$\mathbf{h} = \mathbf{W} \mathbf{x}, \quad \hat{\mathbf{x}} = \mathbf{W}' \mathbf{h} = \mathbf{W}' \mathbf{W} \mathbf{x}$$
2. The loss function is the mean squared reconstruction error:
   $$\min_{\mathbf{W}, \mathbf{W}'} \frac{1}{N} \sum_{i=1}^N \|\mathbf{x}_i - \mathbf{W}' \mathbf{W} \mathbf{x}_i\|^2$$

**Theorem Result**:
- The global minimum of this objective function is achieved when the $k$-dimensional latent space spanned by the columns of $\mathbf{W}'$ (or rows of $\mathbf{W}$) is **identical to the subspace spanned by the top $k$ principal eigenvectors of the data covariance matrix $\Sigma$**.

#### Key Differences Between Linear Autoencoders and PCA:
| Property | PCA | Linear Autoencoder |
|:---|:---|:---|
| **Orthogonality** | Projection vectors $\mathbf{u}_i$ are strictly orthogonal ($\mathbf{u}_i^\top \mathbf{u}_j = 0$) | Weights are not inherently orthogonal unless explicitly constrained |
| **Ordering** | Components are strictly sorted in descending order of explained variance | Latent dimensions have no inherent ordering |
| **Uniqueness** | Unique solution (up to sign flips) | Infinite equivalent solutions (any rotation within the $k$-dimensional subspace) |
| **Optimization** | Closed-form solution via SVD / Eigendecomposition | Iterative numerical optimization via Gradient Descent |
| **Nonlinear Extensions**| Requires kernel tricks (Kernel PCA) | Trivial: simply add nonlinear activation functions ($\sigma$, ReLU) and deeper layers! |

---

### 2.3 Undercomplete vs. Overcomplete Autoencoders

- **Undercomplete Autoencoder ($k < d$)**:
  The hidden layer dimension is strictly less than the input dimension. The network cannot trivially copy inputs and is forced to extract the most informative, low-dimensional manifold.
- **Overcomplete Autoencoder ($k > d$)**:
  The hidden dimension is larger than the input dimension. Without constraints, the network can simply learn the identity function $\hat{\mathbf{x}} = \mathbf{x}$ (e.g., storing the input in the first $d$ neurons and ignoring the rest), failing to learn any meaningful representation.
  To prevent this trivial memorization, we add **regularization penalties**.

---

### 2.4 Regularized Autoencoders

#### 1. Sparse Autoencoders
A sparse autoencoder allows $k \ge d$, but penalizes the network if too many hidden neurons fire simultaneously.
Let $a_j^{(i)} = h_j(\mathbf{x}^{(i)})$ be the activation of hidden unit $j$ on sample $\mathbf{x}^{(i)}$. The average activation of neuron $j$ over $m$ training samples is:
$$\hat{\rho}_j = \frac{1}{m} \sum_{i=1}^m a_j^{(i)}$$
We enforce $\hat{\rho}_j \approx \rho$, where $\rho$ is a tiny target sparsity parameter (e.g., $\rho = 0.05$ or $0.01$).

**KL Divergence Sparsity Penalty**:
The difference between the Bernoulli distribution $\text{Bernoulli}(\rho)$ and $\text{Bernoulli}(\hat{\rho}_j)$ is penalized:
$$\Omega_{\text{sparse}} = \sum_{j=1}^k \text{KL}(\rho \| \hat{\rho}_j) = \sum_{j=1}^k \left[ \rho \log \frac{\rho}{\hat{\rho}_j} + (1 - \rho)\log \frac{1 - \rho}{1 - \hat{\rho}_j} \right]$$
Total objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstruction}} + \beta \, \Omega_{\text{sparse}} + \lambda \, \Omega_{\text{weights}}$$

#### 2. Denoising Autoencoders (DAE) (Vincent et al., 2008)
Instead of feeding the clean input $\mathbf{x}$, the DAE is fed a corrupted version $\tilde{\mathbf{x}} \sim q_D(\tilde{\mathbf{x}}|\mathbf{x})$.
Common corruption processes:
- Additive Gaussian noise: $\tilde{\mathbf{x}} = \mathbf{x} + \mathcal{N}(\mathbf{0}, \sigma^2 I)$.
- Masking noise: Randomly setting a fraction $\nu$ of coordinates to 0.
- Salt-and-pepper noise.

**Objective**:
$$\mathcal{L}_{\text{DAE}} = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} \sim q_D(\tilde{\mathbf{x}}|\mathbf{x})} \left[ \mathcal{L}(\mathbf{x}, g(f(\tilde{\mathbf{x}}))) \right]$$
Notice that the reconstruction loss is calculated against the **original, clean vector $\mathbf{x}$**, NOT the corrupted vector $\tilde{\mathbf{x}}$!

**Geometric / Manifold Learning Intuition**:
Data points in high-dimensional space concentrate near a lower-dimensional manifold. Adding noise pushes the point orthogonal to the manifold. By learning to reconstruct the clean point, the DAE learns a vector field that projects off-manifold corrupted points directly back onto the true data manifold!

#### 3. Contractive Autoencoders (CAE) (Rifai et al., 2011)
The CAE encourages the learned representation $\mathbf{h} = f(\mathbf{x})$ to be robust to infinitesimal input perturbations by penalizing the squared Frobenius norm of the encoder's Jacobian matrix $J_f(\mathbf{x})$:
$$\Omega_{\text{contractive}} = \|J_f(\mathbf{x})\|_F^2 = \sum_{i=1}^k \sum_{j=1}^d \left( \frac{\partial h_i}{\partial x_j} \right)^2$$
For a sigmoid encoder $h_i = \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)$:
$$\frac{\partial h_i}{\partial x_j} = h_i (1 - h_i) W_{ij}$$
$$\|J_f(\mathbf{x})\|_F^2 = \sum_{i=1}^k \left( h_i (1 - h_i) \right)^2 \sum_{j=1}^d W_{ij}^2$$
Total loss:
$$\mathcal{L}_{\text{CAE}} = \mathcal{L}_{\text{reconstruction}} + \lambda \, \|J_f(\mathbf{x})\|_F^2$$
- The contractive penalty flattens the mapping, pulling variations toward zero.
- The reconstruction loss resists this flattening to preserve signal information.
- Result: The representation is sensitive only along the directions of the data manifold (where variation is necessary for reconstruction) and flat/invariant orthogonal to the manifold.

---

## 3. Core Formulas & Cheat Sheet

| Architecture | Forward Equations | Objective Function | Key Mechanism |
|:---|:---|:---|:---|
| **Undercomplete AE** | $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x}+\mathbf{b})$<br>$\hat{\mathbf{x}} = \sigma(\mathbf{W}'\mathbf{h}+\mathbf{b}')$ | $\min \|\mathbf{x} - \hat{\mathbf{x}}\|^2$ | Latent dim $k < d$ forces dimensionality compression |
| **Linear AE** | $\mathbf{h} = \mathbf{W}\mathbf{x}, \hat{\mathbf{x}} = \mathbf{W}'\mathbf{h}$ | $\min \|\mathbf{x} - \mathbf{W}'\mathbf{W}\mathbf{x}\|^2$ | Spans same subspace as PCA |
| **Sparse AE** | $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x}+\mathbf{b})$ | $\mathcal{L}_{\text{rec}} + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$ | Prevents memorization in overcomplete ($k > d$) networks |
| **Denoising AE** | $\tilde{\mathbf{x}} \sim q(\tilde{\mathbf{x}}\|\mathbf{x})$<br>$\hat{\mathbf{x}} = g(f(\tilde{\mathbf{x}}))$ | $\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \|\mathbf{x} - g(f(\tilde{\mathbf{x}}))\|^2$ | Input is corrupted $\tilde{\mathbf{x}}$; target is clean $\mathbf{x}$ |
| **Contractive AE** | $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x}+\mathbf{b})$ | $\mathcal{L}_{\text{rec}} + \lambda \sum_{i,j} \left(\frac{\partial h_i}{\partial x_j}\right)^2$ | Penalizes sensitivity via Frobenius norm of Jacobian |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Machine Reading Compression (Questions 1 to 5)

#### Context & Data
A two-feature reading $\mathbf{x} = (4, 2)^\top$.
Encoder:
$$z = 0.5x_1 + 0.5x_2$$
Decoder:
$$\hat{x}_1 = z, \quad \hat{x}_2 = z$$
Reconstruction loss:
$$\mathcal{L} = (x_1 - \hat{x}_1)^2 + (x_2 - \hat{x}_2)^2$$

---

#### Question 1
**What is the latent value $z$?**
- A) 3
- B) 2
- C) 4
- D) 1

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$z = 0.5(4) + 0.5(2) = 2.0 + 1.0 = 3$$

---

#### Question 2
**What is the reconstructed vector $\hat{\mathbf{x}}$?**
- A) $(2, 2)^\top$
- B) $(3, 3)^\top$
- C) $(4, 4)^\top$
- D) $(3, 2)^\top$

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The decoder sets both output components equal to $z$:
  $$\hat{\mathbf{x}} = (\hat{x}_1, \hat{x}_2)^\top = (z, z)^\top = (3, 3)^\top$$

---

#### Question 3
**What is the reconstruction loss?**
- A) 4
- B) 1
- C) 2
- D) 0

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$\mathcal{L} = (x_1 - \hat{x}_1)^2 + (x_2 - \hat{x}_2)^2 = (4 - 3)^2 + (2 - 3)^2 = 1^2 + (-1)^2 = 1 + 1 = 2$$

---

#### Question 4
**Which part of the autoencoder creates the latent representation?**
- A) The decoder output layer
- B) The reconstruction loss
- C) The target label table
- D) The encoder layer

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  By architectural definition, the **encoder** maps the high-dimensional input vector into the compressed bottleneck/latent representation.

---

#### Question 5
**What is the training objective of this autoencoder?**
- A) To minimize reconstruction error.
- B) To assign external class labels before any reconstruction is attempted.
- C) To ignore the input features.
- D) To make every latent value zero.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Autoencoders are self-supervised models whose primary loss function optimizes the fidelity of the reconstructed output relative to the input.

---

### Case Study 2: Report Feature Compression (Questions 6 to 10)

#### Context & Data
A document system uses $d = 5$ input features, compresses them into $k = 2$ latent hidden units, and reconstructs the 5 features. The model is undercomplete.

---

#### Question 6
**What is the input dimension?**
- A) 2
- B) 5
- C) 7
- D) 10

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  There are 5 input features, so the input dimension is $d = 5$.

---

#### Question 7
**What is the latent dimension?**
- A) 1
- B) 5
- C) 2
- D) 7

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The latent bottleneck contains 2 hidden units, so the latent dimension is $k = 2$.

---

#### Question 8
**Why is the model undercomplete?**
- A) The decoder has no reconstruction target.
- B) The output layer ignores all reconstructed features during training.
- C) The input has fewer features than the latent layer.
- D) The latent dimension is smaller than the input dimension.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  An autoencoder is formally classified as **undercomplete** when its latent bottleneck dimension is strictly less than its input dimension ($k < d$, here $2 < 5$).

---

#### Question 9
**Which output dimension is required to reconstruct the original input?**
- A) 5
- B) 1
- C) 2
- D) 3

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The decoder's output must reconstruct the input $\mathbf{x} \in \mathbb{R}^5$, so the output dimension must identically equal the input dimension, which is 5.

---

#### Question 10
**How is this related to PCA in the linear case?**
- A) It must always produce a one-hot class label.
- B) It can learn a low-dimensional representation similar to PCA.
- C) It removes the need for reconstruction in every training example.
- D) It makes every component have zero variance.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  When an undercomplete autoencoder employs linear activations with squared reconstruction error, the hidden layer spans the exact same optimal low-dimensional subspace as Principal Component Analysis (PCA).

---

### Case Study 3: Meter Reading Recovery (Questions 11 to 15)

#### Context & Data
Denoising autoencoder:
- Clean reading: $\mathbf{x} = (5, 1)^\top$
- Corrupted reading: $\tilde{\mathbf{x}} = (5, 0)^\top$
- Network output: $\hat{\mathbf{x}} = (4, 1)^\top$
- Loss: $\mathcal{L} = (x_1 - \hat{x}_1)^2 + (x_2 - \hat{x}_2)^2$

---

#### Question 11
**Which vector is given to the encoder during training?**
- A) $\mathbf{x} = (5, 1)^\top$
- B) $\hat{\mathbf{x}} = (4, 1)^\top$
- C) $(0, 0)^\top$
- D) $\tilde{\mathbf{x}} = (5, 0)^\top$

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  In a Denoising Autoencoder (DAE), the encoder receives the stochastically corrupted vector $\tilde{\mathbf{x}} = (5, 0)^\top$ as input.

---

#### Question 12
**Which vector is the clean reconstruction target?**
- A) $(4, 1)^\top$
- B) $(5, 0)^\top$
- C) $(5, 1)^\top$
- D) $(0, 1)^\top$

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The loss in a DAE is measured by comparing the reconstructed output $\hat{\mathbf{x}}$ against the original **clean** target $\mathbf{x} = (5, 1)^\top$.

---

#### Question 13
**What is the reconstruction loss?**
- A) 0
- B) 2
- C) 4
- D) 1

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$\mathcal{L} = (x_1 - \hat{x}_1)^2 + (x_2 - \hat{x}_2)^2 = (5 - 4)^2 + (1 - 1)^2 = 1^2 + 0^2 = 1$$

---

#### Question 14
**What is the purpose of using corrupted inputs?**
- A) To train the model to recover clean structure.
- B) To remove the decoder from the network before training begins.
- C) To force every output to equal zero.
- D) To avoid computing any reconstruction loss.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Feeding corrupted data while targeting clean data forces the model to learn the true underlying data distribution and project noisy off-manifold points back onto the manifold.

---

#### Question 15
**Which statement best describes this training setup?**
- A) The model ignores all input values.
- B) The model maps noisy input toward a clean target.
- C) The target is the corrupted vector only.
- D) The encoder output is used as the final label for every noisy example.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The training setup maps noisy, corrupted patterns $\tilde{\mathbf{x}}$ toward their clean, ground-truth targets $\mathbf{x}$.

---

### Case Study 4: Compact Representation Control (Questions 16 to 20)

#### Context & Data
Base reconstruction loss: $\mathcal{L}_{\text{rec}} = 0.8$.
Sparsity penalty: $0.3$ with weighting hyperparameter $\lambda = 2$.
Total loss:
$$\mathcal{L}_{\text{sparse}} = \mathcal{L}_{\text{rec}} + \lambda(0.3)$$

---

#### Question 16
**What is the sparsity penalty contribution?**
- A) 0.3
- B) 0.8
- C) 0.6
- D) 1.1

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$\text{Penalty contribution} = \lambda \times (\text{raw penalty}) = 2 \times 0.3 = 0.6$$

---

#### Question 17
**What is $\mathcal{L}_{\text{sparse}}$?**
- A) 1.4
- B) 1.1
- C) 0.8
- D) 0.6

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$\mathcal{L}_{\text{sparse}} = \mathcal{L}_{\text{rec}} + \text{Penalty contribution} = 0.8 + 0.6 = 1.4$$

---

#### Question 18
**What does sparse regularization encourage?**
- A) Every hidden unit must always be active during the full reconstruction process.
- B) The decoder is removed from the model.
- C) Only a small number of hidden units should be active.
- D) The input dimension must equal zero.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Sparsity regularization forces the vast majority of hidden neurons to remain inactive or near zero, ensuring that only a small, specialized subset of features fires for any given input.

---

#### Question 19
**What is the goal of a contractive penalty?**
- A) To force reconstruction loss to be ignored throughout all training.
- B) To make all weights increase without bound after every input change.
- C) To replace the encoder with class labels.
- D) To make the representation less sensitive to small input changes.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  A contractive penalty penalizes the Frobenius norm of the Jacobian $\sum_{i,j} (\partial h_i / \partial x_j)^2$. This minimizes the derivative of the encoder outputs with respect to the inputs, ensuring that small perturbations in $\mathbf{x}$ do not cause large fluctuations in the hidden code $\mathbf{h}$.

---

#### Question 20
**Why are such penalties added to the reconstruction loss?**
- A) To remove all training data from the objective.
- B) To control the learned representation beyond plain reconstruction.
- C) To stop the model from producing reconstructions.
- D) To make every feature identical to every other feature before encoding.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Plain reconstruction alone can lead to trivial identity mappings (especially in overcomplete networks). Regularization terms (sparsity, contractive, denoising) impose inductive biases that shape the geometry and semantics of the latent manifold.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Target in Denoising Autoencoder**:
   - The target for computing loss is **ALWAYS the clean $\mathbf{x}$**, NOT the corrupted $\tilde{\mathbf{x}}$. The input is $\tilde{\mathbf{x}}$, and the target is $\mathbf{x}$.
2. **Contractive vs Denoising**:
   - DAE achieves robustness globally through stochastic sampling of corrupted points.
   - CAE achieves robustness analytically and locally through the Jacobian Frobenius norm penalty.
3. **Linear Autoencoder vs PCA Subspace**:
   - A linear AE learns the *same subspace* as PCA, but **not necessarily the same orthogonal eigenvectors**. Any rotated basis of the PCA subspace yields the exact same minimum reconstruction error.
