# Week 5: Linear Algebra Foundations — Eigenvalues, Eigenvectors, PCA, and SVD

---

## 1. Executive Summary & Exam Roadmap

Week 5 covers the linear algebra principles underpinning dimensionality reduction and representation learning in Deep Learning:
1. **Eigenvalues & Eigenvectors**: Algebraic and geometric definitions ($A\mathbf{u} = \lambda \mathbf{u}$), characteristic equation $\det(A - \lambda I) = 0$, trace and determinant relations.
2. **Eigenvalue Decomposition (Spectral Theorem)**: Diagonalization of symmetric matrices $A = U \Lambda U^\top$ with orthogonal eigenvectors ($U^\top U = I$).
3. **Principal Component Analysis (PCA)**:
   - Data centering and the empirical covariance matrix $\Sigma = \frac{1}{m} X^\top X$.
   - **Dual perspectives**: Maximizing projected variance vs. minimizing reconstruction error.
   - Eigenvectors of $\Sigma$ as principal directions; fraction of total variance explained.
4. **Singular Value Decomposition (SVD)**:
   - Factorization $A = U \Sigma V^\top$ for arbitrary rectangular matrices.
   - Left singular vectors ($AA^\top$), right singular vectors ($A^\top A$), and singular values $\sigma_i = \sqrt{\lambda_i}$.
   - **Eckart-Young-Mirsky Theorem**: Optimal low-rank matrix approximation.

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 Eigenvalues and Eigenvectors

#### Definition
Let $A \in \mathbb{R}^{n \times n}$ be a square matrix. A non-zero vector $\mathbf{u} \in \mathbb{R}^n \setminus \{\mathbf{0}\}$ is an **eigenvector** of $A$ corresponding to **eigenvalue** $\lambda \in \mathbb{C}$ if:
$$A \mathbf{u} = \lambda \mathbf{u}$$

#### Geometric Interpretation
Multiplying a vector $\mathbf{u}$ by matrix $A$ generally changes both its length and its direction. An eigenvector represents a special invariant direction: $A$ merely stretches, shrinks, or reverses $\mathbf{u}$ along its original line of action by a scaling factor of $\lambda$.
- If $\lambda > 1$: vector is stretched.
- If $0 < \lambda < 1$: vector is shrunk.
- If $\lambda < 0$: vector direction is reversed.
- If $\lambda = 0$: vector is mapped to the null space ($A\mathbf{u} = \mathbf{0}$).

#### Characteristic Equation
Rewriting $A\mathbf{u} = \lambda \mathbf{u}$:
$$(A - \lambda I)\mathbf{u} = \mathbf{0}$$
For a non-trivial solution $\mathbf{u} \neq \mathbf{0}$ to exist, the matrix $(A - \lambda I)$ must be singular:
$$\det(A - \lambda I) = 0$$

#### Fundamental Properties
1. **Sum of Eigenvalues equals Trace**:
   $$\sum_{i=1}^n \lambda_i = \text{Tr}(A) = \sum_{i=1}^n A_{ii}$$
2. **Product of Eigenvalues equals Determinant**:
   $$\prod_{i=1}^n \lambda_i = \det(A)$$
3. **Diagonal Matrices**:
   For a diagonal matrix $D = \text{diag}(d_1, d_2, \dots, d_n)$, the eigenvalues are simply the diagonal entries $d_i$, and the eigenvectors are the standard Cartesian basis vectors $\mathbf{e}_i$.

---

### 2.2 Eigenvalue Decomposition & Symmetric Matrices

If $A$ has $n$ linearly independent eigenvectors $\mathbf{u}_1, \dots, \mathbf{u}_n$ with eigenvalues $\lambda_1, \dots, \lambda_n$:
$$A U = U \Lambda \implies A = U \Lambda U^{-1}$$
where $U = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_n]$ and $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$.

#### The Spectral Theorem for Real Symmetric Matrices
If $A = A^\top \in \mathbb{R}^{n \times n}$:
1. All eigenvalues $\lambda_i$ are **guaranteed to be real** ($\lambda_i \in \mathbb{R}$).
2. Eigenvectors corresponding to distinct eigenvalues are **mutually orthogonal**:
   $$\mathbf{u}_i^\top \mathbf{u}_j = 0 \quad (\forall i \neq j)$$
3. The eigenvector matrix $U$ can be chosen to be orthonormal ($U^\top = U^{-1}$):
   $$A = U \Lambda U^\top = \sum_{i=1}^n \lambda_i \mathbf{u}_i \mathbf{u}_i^\top$$

---

### 2.3 Principal Component Analysis (PCA)

#### Data Preprocessing: Centering
Given a dataset of $m$ vectors $\{\mathbf{x}_1, \dots, \mathbf{x}_m\} \subset \mathbb{R}^d$, compute the sample mean $\boldsymbol{\mu} = \frac{1}{m}\sum_{i=1}^m \mathbf{x}_i$.
Center the data:
$$\tilde{\mathbf{x}}_i = \mathbf{x}_i - \boldsymbol{\mu}$$
The empirical covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$ is:
$$\Sigma = \frac{1}{m} \sum_{i=1}^m \tilde{\mathbf{x}}_i \tilde{\mathbf{x}}_i^\top = \frac{1}{m} X^\top X$$
where $X \in \mathbb{R}^{m \times d}$ is the mean-centered design matrix. Note that $\Sigma$ is symmetric and positive semi-definite ($\Sigma = \Sigma^\top, \lambda_i \ge 0$).

#### Perspective 1: Maximizing Variance of Projection
We want to project data onto a unit direction $\mathbf{u}$ ($\|\mathbf{u}\| = 1$) such that the variance of the projected points is maximized:
$$\text{Var}(\mathbf{u}^\top \tilde{\mathbf{x}}) = \frac{1}{m}\sum_{i=1}^m (\mathbf{u}^\top \tilde{\mathbf{x}}_i)^2 = \mathbf{u}^\top \left( \frac{1}{m}\sum_{i=1}^m \tilde{\mathbf{x}}_i \tilde{\mathbf{x}}_i^\top \right) \mathbf{u} = \mathbf{u}^\top \Sigma \mathbf{u}$$
Formulate the Lagrangian with constraint $\mathbf{u}^\top \mathbf{u} = 1$:
$$\mathcal{L}(\mathbf{u}, \lambda) = \mathbf{u}^\top \Sigma \mathbf{u} - \lambda (\mathbf{u}^\top \mathbf{u} - 1)$$
Setting the gradient w.r.t $\mathbf{u}$ to zero:
$$\nabla_{\mathbf{u}} \mathcal{L} = 2\Sigma \mathbf{u} - 2\lambda \mathbf{u} = \mathbf{0} \implies \Sigma \mathbf{u} = \lambda \mathbf{u}$$
The projected variance is:
$$\mathbf{u}^\top \Sigma \mathbf{u} = \mathbf{u}^\top (\lambda \mathbf{u}) = \lambda \|\mathbf{u}\|^2 = \lambda$$
To maximize variance, $\mathbf{u}$ must be the eigenvector of $\Sigma$ corresponding to the **largest eigenvalue** $\lambda_{\max}$.

#### Perspective 2: Minimizing Reconstruction Error
Approximating $\tilde{\mathbf{x}}_i$ using $k$ orthonormal basis vectors $\{\mathbf{u}_1, \dots, \mathbf{u}_k\}$:
$$\hat{\mathbf{x}}_i = \sum_{j=1}^k (\tilde{\mathbf{x}}_i^\top \mathbf{u}_j)\mathbf{u}_j$$
Minimizing the mean squared reconstruction error:
$$\min \frac{1}{m}\sum_{i=1}^m \|\tilde{\mathbf{x}}_i - \hat{\mathbf{x}}_i\|^2$$
leads mathematically to the exact same solution: pick the eigenvectors corresponding to the $k$ largest eigenvalues of $\Sigma$!

#### Variance Explained Ratio
$$\text{Fraction of Variance Explained by top } k \text{ components} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j} = \frac{\sum_{i=1}^k \lambda_i}{\text{Tr}(\Sigma)}$$

---

### 2.4 Singular Value Decomposition (SVD)

#### Theorem
Any real matrix $M \in \mathbb{R}^{m \times n}$ of rank $r \le \min(m, n)$ can be factorized as:
$$M = U \Sigma V^\top$$
where:
- $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix ($U^\top U = I_m$). Its columns are the **left singular vectors**, which are eigenvectors of $M M^\top$.
- $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix ($V^\top V = I_n$). Its columns are the **right singular vectors**, which are eigenvectors of $M^\top M$.
- $\Sigma \in \mathbb{R}^{m \times n}$ is a rectangular diagonal matrix containing non-negative **singular values** sorted in descending order:
  $$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0 = \dots = 0$$
  where $\sigma_i = \sqrt{\lambda_i(M^\top M)} = \sqrt{\lambda_i(MM^\top)}$.

#### Outer Product Expansion
$$M = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

#### Eckart-Young-Mirsky Theorem (Low-Rank Approximation)
The optimal rank-$k$ ($k < r$) approximation $M_k$ that minimizes the Frobenius error $\|M - M_k\|_F$ is obtained by truncating the SVD to the top $k$ singular values:
$$M_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$
The approximation error is directly given by the discarded singular values:
$$\|M - M_k\|_F^2 = \sum_{j=k+1}^r \sigma_j^2$$

---

## 3. Core Formulas & Cheat Sheet

| Mathematical Concept | Formula | Key Properties |
|:---|:---|:---|
| **Eigenvalue Equation** | $A\mathbf{u} = \lambda \mathbf{u}$ | $\det(A - \lambda I) = 0$ |
| **Matrix Trace** | $\text{Tr}(A) = \sum A_{ii} = \sum \lambda_i$ | Sum of diagonal entries = sum of eigenvalues |
| **Matrix Determinant** | $\det(A) = \prod \lambda_i$ | Product of eigenvalues |
| **Spectral Decomposition** | $A = U \Lambda U^\top$ | Valid for symmetric $A = A^\top$; $U^\top U = I$ |
| **Sample Covariance** | $\Sigma = \frac{1}{m} X^\top X$ | Data must be zero-centered first |
| **Projected Variance** | $\text{Var} = \mathbf{u}^\top \Sigma \mathbf{u} = \lambda$ | Maximized along dominant eigenvector |
| **SVD Factorization** | $M = U \Sigma V^\top$ | $U$: eigenvectors of $MM^\top$; $V$: eigenvectors of $M^\top M$ |
| **Singular Value Relation** | $\sigma_i = \sqrt{\lambda_i(M^\top M)}$ | Always non-negative ($\sigma_i \ge 0$) |
| **Truncated SVD** | $M_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ | Optimal rank-$k$ approximation under Frobenius norm |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Sensor Direction Scaling (Questions 1 to 5)

#### Context & Data
A diagonal linear transformation:
$$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}$$
Independent sensor directions: $\mathbf{u}_1 = (1, 0)^\top$, $\mathbf{u}_2 = (0, 1)^\top$.

---

#### Question 1
**What is $A\mathbf{u}_1$?**
- A) $(3, 0)^\top$
- B) $(0, 3)^\top$
- C) $(2, 0)^\top$
- D) $(1, 3)^\top$

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  $$A \mathbf{u}_1 = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3(1) + 0(0) \\ 0(1) + 2(0) \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix} = (3, 0)^\top$$

---

#### Question 2
**What eigenvalue corresponds to $\mathbf{u}_2$?**
- A) 1
- B) 3
- C) 2
- D) 0

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$A \mathbf{u}_2 = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 2 \end{bmatrix} = 2 \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 2 \mathbf{u}_2$$
  Since $A \mathbf{u}_2 = \lambda_2 \mathbf{u}_2$, the eigenvalue is $\lambda_2 = 2$.

---

#### Question 3
**Which statement is true about $\mathbf{u}_1$?**
- A) It is not changed by $A$ for any input scale.
- B) It is rotated by 90 degrees after multiplication by $A$.
- C) It becomes the zero vector.
- D) It is scaled by a factor of 3.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $A\mathbf{u}_1 = (3, 0)^\top = 3 \mathbf{u}_1$. Its direction is completely preserved, and its length is scaled by 3.

---

#### Question 4
**What is the trace of $A$?**
- A) 6
- B) 5
- C) 1
- D) 3

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The trace of a matrix is the sum of its diagonal entries:
  $$\text{Tr}(A) = 3 + 2 = 5$$
  (Also equals $\lambda_1 + \lambda_2 = 3 + 2 = 5$).

---

#### Question 5
**How are the eigenvalues related to the diagonal entries in this matrix?**
- A) They are equal to the diagonal entries for this diagonal matrix.
- B) They must both be zero for every diagonal matrix in this setting.
- C) They are found by adding all entries in each row.
- D) They depend only on the off-diagonal entries here.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  For any diagonal matrix $D = \text{diag}(d_1, \dots, d_n)$, the characteristic equation is $\det(D - \lambda I) = \prod_{i=1}^n (d_i - \lambda) = 0$, so the eigenvalues are identically equal to the diagonal entries.

---

### Case Study 2: Customer Feature Coordinate Review (Questions 6 to 10)

#### Context & Data
Basis vectors in $\mathbb{R}^2$:
$$\mathbf{b}_1 = (1, 1)^\top, \quad \mathbf{b}_2 = (1, -1)^\top$$
Customer feature vector:
$$\mathbf{x} = (4, 2)^\top$$
Coordinate representation:
$$\mathbf{x} = c_1 \mathbf{b}_1 + c_2 \mathbf{b}_2$$

---

#### Question 6
**Which equation is obtained from the first coordinate?**
- A) $c_1 - c_2 = 4$
- B) $c_1 + c_2 = 4$
- C) $2c_1 = 4$
- D) $c_2 = 4$

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  $$c_1 \begin{bmatrix} 1 \\ 1 \end{bmatrix} + c_2 \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} c_1 + c_2 \\ c_1 - c_2 \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$$
  Equating the first coordinate yields: $c_1 + c_2 = 4$.

---

#### Question 7
**Which equation is obtained from the second coordinate?**
- A) $c_1 + c_2 = 2$
- B) $c_1 = 2$
- C) $c_1 - c_2 = 2$
- D) $2c_2 = 2$

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Equating the second coordinate yields: $c_1 - c_2 = 2$.

---

#### Question 8
**What is $c_1$?**
- A) 3
- B) 1
- C) 4
- D) 2

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Add the two linear equations:
  $$(c_1 + c_2) + (c_1 - c_2) = 4 + 2 \implies 2c_1 = 6 \implies c_1 = 3$$

---

#### Question 9
**What is $c_2$?**
- A) 0
- B) 1
- C) 3
- D) 2

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Substitute $c_1 = 3$ into $c_1 + c_2 = 4$:
  $$3 + c_2 = 4 \implies c_2 = 1$$

---

#### Question 10
**Why do $\mathbf{b}_1$ and $\mathbf{b}_2$ form a basis for $\mathbb{R}^2$?**
- A) They point in the same direction.
- B) They both have zero length.
- C) They always produce only positive coordinates.
- D) They are linearly independent and span $\mathbb{R}^2$.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Checking dot product: $\mathbf{b}_1^\top \mathbf{b}_2 = (1)(1) + (1)(-1) = 0$. Since they are non-zero and orthogonal, they are linearly independent. Any set of 2 linearly independent vectors in $\mathbb{R}^2$ spans $\mathbb{R}^2$ and forms a basis.

---

### Case Study 3: Store Performance Feature Reduction (Questions 11 to 15)

#### Context & Data
Centered covariance matrix:
$$\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 1 \end{bmatrix}$$

---

#### Question 11
**What is the variance along the first coordinate direction?**
- A) 1
- B) 2
- C) 4
- D) 0

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The diagonal entries of a covariance matrix represent the individual feature variances. For feature 1, the variance is $\Sigma_{11} = 4$.

---

#### Question 12
**Which direction is the first principal component?**
- A) The second coordinate direction
- B) The direction $(1, 1)^\top$
- C) The zero vector direction
- D) The first coordinate direction

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  The first principal component corresponds to the eigenvector with the largest eigenvalue. Here $\lambda_1 = 4 > \lambda_2 = 1$. The eigenvector corresponding to $\lambda_1 = 4$ is $\mathbf{e}_1 = (1, 0)^\top$, which is the first coordinate direction.

---

#### Question 13
**What fraction of total variance is explained by the first principal component?**
- A) 4/5
- B) 1/5
- C) 1/4
- D) 5/4

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Total variance $= \text{Tr}(\Sigma) = 4 + 1 = 5$.
  Variance explained by component 1 $= 4$.
  $$\text{Fraction} = \frac{4}{4 + 1} = \frac{4}{5} = 80\%$$

---

#### Question 14
**If the data is projected onto the first principal component only, how many dimensions remain?**
- A) 0
- B) 1
- C) 2
- D) 4

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Projecting $d$-dimensional data onto a single 1D line defined by the first principal component yields a 1-dimensional scalar representation.

---

#### Question 15
**What is the purpose of centering data before PCA?**
- A) To make every covariance value negative after centering.
- B) To remove all variance from the dataset before PCA.
- C) To force all eigenvalues to be equal.
- D) To express variation relative to the mean.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Centering shifts the coordinate origin to the centroid (sample mean) of the data cloud. This ensures that the covariance matrix measures true spread and dispersion about the mean, rather than being distorted by the offset from the origin.

---

### Case Study 4: Product-User Interaction Compression (Questions 16 to 20)

#### Context & Data
Interaction matrix:
$$M = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$$
SVD form: $M = U \Sigma V^\top$ with singular values $\sigma_1 = 3, \sigma_2 = 1$.

---

#### Question 16
**What is the largest singular value?**
- A) 1
- B) 2
- C) 3
- D) 0

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The singular values are $\sigma_1 = 3$ and $\sigma_2 = 1$. The largest singular value is 3.

---

#### Question 17
**What does $V^\top$ represent in the SVD expression?**
- A) The transpose of the right singular-vector matrix.
- B) The loss value of the recommendation model after a full training step.
- C) The number of users in every batch.
- D) The covariance matrix after centering.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  In $M = U \Sigma V^\top$, $V$ is the orthogonal matrix of right singular vectors (eigenvectors of $M^\top M$). Thus, $V^\top$ is the transpose of the right singular-vector matrix.

---

#### Question 18
**What rank-one approximation keeps only the largest singular value?**
- A) $\begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}$
- B) $\begin{bmatrix} 3 & 0 \\ 0 & 0 \end{bmatrix}$
- C) $\begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$
- D) $\begin{bmatrix} 0 & 3 \\ 1 & 0 \end{bmatrix}$

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  By the Eckart-Young theorem, the best rank-1 approximation sets all singular values except the largest to zero:
  $$M_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top = 3 \begin{bmatrix} 1 \\ 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 0 & 0 \end{bmatrix}$$

---

#### Question 19
**What is the rank of $M$?**
- A) 0
- B) 1
- C) 2
- D) 3

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The rank of a matrix equals the number of strictly non-zero singular values. Since both $\sigma_1 = 3$ and $\sigma_2 = 1$ are non-zero, $\text{rank}(M) = 2$.

---

#### Question 20
**Why is SVD useful for dimensionality reduction?**
- A) It requires every matrix entry to be exactly zero before the factorization.
- B) It removes the need for any matrix factorization.
- C) It changes all singular values to negative values.
- D) It can keep dominant singular directions and discard smaller ones.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  SVD decomposes a matrix into a sum of rank-1 matrices weighted by their singular values. Discarding smaller singular values filters out high-frequency noise and compresses the data while preserving the dominant geometric variation.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Covariance Matrix Symmetry**:
   - Covariance matrix $\Sigma$ is **always symmetric** ($\Sigma = \Sigma^\top$) and **positive semi-definite** ($\mathbf{x}^\top \Sigma \mathbf{x} \ge 0$). All eigenvalues of $\Sigma$ are $\ge 0$.
2. **Singular Values vs Eigenvalues**:
   - Singular values $\sigma_i$ are **always real and non-negative** ($\sigma_i \ge 0$).
   - Eigenvalues $\lambda_i$ of a general matrix can be negative or complex.
   - For any matrix $M$, $\sigma_i(M) = \sqrt{\lambda_i(M^\top M)}$.
3. **Fraction of Variance Formula**:
   - $\frac{\lambda_1}{\lambda_1 + \lambda_2 + \dots + \lambda_d}$. Be careful not to square the eigenvalues when calculating variance explained in PCA (the eigenvalues of $\Sigma$ *already* represent variances).
