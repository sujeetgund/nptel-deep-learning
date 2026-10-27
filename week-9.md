# Week 9: Learning Vectorial Representations of Words — Word Embeddings, Word2Vec, and SVD

---

## 1. Executive Summary & Exam Roadmap

Week 9 explores how discrete symbolic text is converted into dense, continuous mathematical vectors for deep neural networks:
1. **One-Hot Representation & Bag-of-Words (BoW)**: High dimensionality ($V$), severe sparsity, orthogonality, and the inability to capture semantic relatedness.
2. **The Distributional Hypothesis (Firth, 1957)**: "A word is characterized by the company it keeps."
3. **Count-based Approaches (Co-occurrence Matrix + Truncated SVD)**:
   - Constructing word-context matrices $C \in \mathbb{R}^{V \times V}$.
   - Truncated SVD dimensionality reduction: $C \approx U_k \Sigma_k V_k^\top$.
   - Strengths and computational limitations ($O(V^3)$).
4. **Prediction-based Approaches (Word2Vec - Mikolov et al., 2013)**:
   - **Continuous Bag of Words (CBOW)**: Predicting center target word from context words.
   - **Skip-Gram**: Predicting context words from the center target word.
5. **Computational Acceleration**:
   - The softmax denominator bottleneck over vocabulary $V$.
   - **Negative Sampling (SGNS)**: Converting multi-class classification into $k$ binary classification tasks with $P_n(w) \propto U(w)^{3/4}$.
   - **Hierarchical Softmax**: Huffman tree routing ($O(\log V)$ complexity).
6. **Vector Arithmetic & Evaluation**: Cosine similarity, word analogies ($\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$), and correlation with human judgments.

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 One-Hot Word Representations and Their Fatal Flaws

Let $\mathcal{V} = \{w_1, w_2, \dots, w_V\}$ be a fixed vocabulary of $V$ unique words.
In a one-hot representation, each word $w_i$ is mapped to a standard basis vector $\mathbf{e}_i \in \{0, 1\}^V$:
$$\mathbf{e}_{\text{cat}} = [1, 0, 0, \dots, 0]^\top, \quad \mathbf{e}_{\text{dog}} = [0, 1, 0, \dots, 0]^\top$$

#### Severe Limitations:
1. **Orthogonality (Zero Semantic Similarity)**:
   For any two distinct words $i \neq j$:
   $$\mathbf{e}_i^\top \mathbf{e}_j = 0, \quad \text{cosine\_sim}(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i^\top \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|} = 0$$
   The distance between "cat" and "kitten" is identically equal to the distance between "cat" and "airplane".
2. **Sparsity & Memory Explosion**:
   For modern corpora, $V \approx 10^5 \text{ to } 10^6$. Every word requires a megabyte-scale sparse vector containing exactly one $1$ and $999,999$ zeros.

---

### 2.2 The Distributional Hypothesis & Count-Based SVD

#### The Distributional Hypothesis
Proposed by linguist J.R. Firth (1957): Words that appear in similar linguistic contexts tend to share similar semantic and syntactic meanings.

#### Co-occurrence Matrix Construction
Define a symmetric context window of size $c$ (e.g. $c = 1$ considers 1 word to the left and 1 word to the right).
A square matrix $C \in \mathbb{R}^{V \times V}$ is constructed where entry $C_{ij}$ records the number of times word $w_j$ appears within a window of $c$ tokens around word $w_i$.
```
Corpus: "deep learning is fun" (window c = 1)
Target "learning" has context words: "deep" (left) and "is" (right).
```

#### Dimensionality Reduction via Truncated SVD
The raw matrix $C \in \mathbb{R}^{V \times V}$ is sparse and noisy. Applying SVD:
$$C = U \Sigma V^\top$$
Retaining only the top $k$ singular values ($k \ll V$, typically $k \in [100, 300]$):
$$C \approx U_k \Sigma_k V_k^\top$$
- $U_k \in \mathbb{R}^{V \times k}$: Each row $i$ of $U_k$ (or $U_k \Sigma_k$) provides a compact, dense, $k$-dimensional continuous vector embedding for word $w_i$.
- **Major Drawbacks**:
  - Computational complexity is $O(V^3)$, making it intractable for $V > 50,000$.
  - High-frequency functional stopwords ("the", "is", "a") dominate co-occurrence counts without carrying semantic content.

---

### 2.3 Prediction-Based Models: Word2Vec (Mikolov et al., 2013)

Instead of counting global frequencies and factoring a huge matrix, Word2Vec learns word vectors iteratively using small sliding windows and backpropagation.

```
       CBOW Architecture                          Skip-Gram Architecture
Context Words: w(t-2), w(t-1), ...                  Center Target Word: w(t)
               |                                              |
      [ Average Projection ]                           [ Projection ]
               |                                              |
     Target Center Word: w(t)                  Context Words: w(t-1), w(t+1), ...
```

#### 1. Continuous Bag of Words (CBOW)
- **Objective**: Given a sequence of $2c$ context words $\{w_{t-c}, \dots, w_{t+c}\} \setminus \{w_t\}$, predict the center target word $w_t$.
- **Hidden Layer Representation**:
  Let $\mathbf{v}_w \in \mathbb{R}^d$ be the input embedding of word $w$. The hidden layer computes the average of the context embeddings:
  $$\mathbf{h} = \frac{1}{2c} \sum_{-c \le j \le c, j \neq 0} \mathbf{v}_{w_{t+j}}$$
- **Output Probability (Softmax)**:
  Let $\mathbf{u}_w \in \mathbb{R}^d$ denote the output representation of word $w$:
  $$P(w_t | \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{h})}$$
- **Loss**: Categorical Cross-Entropy: $\mathcal{L} = -\log P(w_t | \text{context})$.

#### 2. Skip-Gram Model
- **Objective**: Given the center target word $w_t$, predict each of the surrounding context words $w_{t+j}$ for $-c \le j \le c, j \neq 0$.
- **Probability of context word $w_c$ given center word $w_t$**:
  $$P(w_c | w_t) = \frac{\exp(\mathbf{u}_{w_c}^\top \mathbf{v}_{w_t})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_t})}$$
- **Total Corpus Objective**: Maximize the average log probability:
  $$\mathcal{L} = \sum_{t=1}^T \sum_{-c \le j \le c, j \neq 0} \log P(w_{t+j} | w_t)$$

---

### 2.4 Accelerating Training: Negative Sampling (SGNS)

#### The Computational Bottleneck
In the standard softmax:
$$P(w_c | w_t) = \frac{\exp(\mathbf{u}_{w_c}^\top \mathbf{v}_{w_t})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_t})}$$
The denominator requires summing over all $V$ words in the vocabulary for every training step! For $V = 10^6$, this is impossible to train.

#### Negative Sampling Formulation
Mikolov et al. replaced the multinomial softmax with multiple independent binary logistic regressions:
1. Maximize the probability that the true target-context pair $(w_t, w_c)$ came from the data ($\mathcal{D} = 1$).
2. Minimize the probability that $K$ randomly sampled noise pairs $(w_t, w_{\text{neg}})$ came from the data ($\mathcal{D} = 0$).

**Loss Function for a Target Word $w_t$ and Context Word $w_c$**:
$$\mathcal{L}_{\text{SGNS}} = \log \sigma(\mathbf{u}_{w_c}^\top \mathbf{v}_{w_t}) + \sum_{k=1}^K \mathbb{E}_{w_k \sim P_n(w)} \left[ \log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_{w_t}) \right]$$
where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid.
- $K$ is the number of negative samples (typically $5 \le K \le 20$ for small datasets; $2 \le K \le 5$ for massive datasets).
- **The Noise Distribution $P_n(w)$**:
  Sampled from the unigram frequency distribution raised to the $3/4$ power:
  $$P_n(w) = \frac{U(w)^{3/4}}{\sum_{w'} U(w')^{3/4}}$$
  *Why $3/4$?* Raising to $0.75$ slightly suppresses the overwhelming frequency of common words ("the", "of") while boosting the sampling probability of rare words.

---

### 2.5 Vector Space Geometry & Word Analogies

Dense word embeddings capture linear relational composition in the semantic vector space:
- **Cosine Similarity**:
  $$\text{cosine\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
- **Linear Analogies ($a : b :: c : d$)**:
  "Man is to King as Woman is to ?"
  $$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} \approx \mathbf{v}_{\text{queen}} - \mathbf{v}_{\text{woman}}$$
  To solve for the unknown word $d$, search for the vocabulary vector $\mathbf{v}_w$ ($w \neq a, b, c$) that maximizes cosine similarity to:
  $$\mathbf{v}^* = \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$$

---

## 3. Core Formulas & Cheat Sheet

| Concept | Mathematical Equation | Key Notes |
|:---|:---|:---|
| **One-Hot Vector** | $\mathbf{e}_i = [0, \dots, 1, \dots, 0]^\top \in \{0, 1\}^V$ | Length $V$, exactly one 1; all pairs orthogonal |
| **Co-occurrence SVD** | $C_{V \times V} \approx U_{V \times k} \Sigma_{k \times k} V_{k \times V}^\top$ | Embeddings are rows of $U_k$; cost $O(V^3)$ |
| **CBOW Target Prob** | $P(w_t\|\mathbf{h}) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_w \exp(\mathbf{u}_w^\top \mathbf{h})}$ | $\mathbf{h} = \frac{1}{2c}\sum \mathbf{v}_c$ (context average) |
| **Skip-Gram Prob** | $P(w_c\|w_t) = \frac{\exp(\mathbf{u}_{w_c}^\top \mathbf{v}_{w_t})}{\sum_w \exp(\mathbf{u}_w^\top \mathbf{v}_{w_t})}$ | Predicts context given center word |
| **Negative Sampling**| $\log \sigma(\mathbf{u}_c^\top \mathbf{v}_t) + \sum_{k=1}^K \log \sigma(-\mathbf{u}_{k}^\top \mathbf{v}_t)$ | Reduces $O(V)$ softmax to $O(K)$ binary steps |
| **Noise Distribution**| $P_n(w) \propto U(w)^{0.75}$ | Dampens stopwords; enhances rare words |
| **Vector Analogy** | $\mathbf{d} = \arg\max_{w} \text{sim}(\mathbf{v}_w, \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c)$ | Linear semantic vector arithmetic |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Customer Support Ticket Classification (Questions 1 to 5)

#### Context & Data
A telecom company processes 50,000 tickets with a vocabulary of $V = 12,000$ unique words.
Each word is represented as a one-hot vector of length $V$.
Bag-of-words ticket vectors are formed by summing word vectors:
- Ticket A: `recharge failed refund`
- Ticket B: `recharge plan refund`

---

#### Question 1
**What is the length of each one-hot word vector?**
- A) 12,000
- B) 50,000
- C) 3
- D) 4

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  A one-hot vector has a dedicated dimension for every unique token in the vocabulary. Since the vocabulary size is $V = 12,000$, each one-hot vector has length $12,000$.

---

#### Question 2
**How many non-zero entries does a single one-hot word vector contain?**
- A) 12,000
- B) 0
- C) 1
- D) 3

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  By mathematical definition, a one-hot vector has exactly one coordinate set to $1$ (corresponding to the word's index) and all other $V-1$ coordinates set to $0$.

---

#### Question 3
**How many shared words appear in Ticket A and Ticket B?**
- A) 1
- B) 3
- C) 0
- D) 2

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Ticket A words: `{"recharge", "failed", "refund"}`.
  Ticket B words: `{"recharge", "plan", "refund"}`.
  Intersection: `{"recharge", "refund"}` $\implies 2$ shared words.

---

#### Question 4
**What is a major limitation of one-hot word vectors?**
- A) They are always dense for the data shown here.
- B) They do not directly capture semantic similarity between words.
- C) They require labels for every context window for the described model.
- D) They have variable length for the same vocabulary.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  All pairs of distinct one-hot vectors are mutually orthogonal ($\mathbf{e}_i^\top \mathbf{e}_j = 0$), meaning they cannot encode linguistic nuances, synonyms, or conceptual similarity.

---

#### Question 5
**Why can cosine similarity still be useful for bag-of-words ticket vectors?**
- A) It measures overlap in word usage while reducing sensitivity to length.
- B) It learns neural hidden layers automatically from the sparse ticket vectors.
- C) It converts sparse vectors into images in this specific scenario.
- D) It guarantees semantic understanding under the given conditions.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Cosine similarity evaluates the normalized dot product $\frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$, which reflects relative word co-occurrence while normalizing for document length.

---

### Case Study 2: Product Review Term Analysis (Questions 6 to 10)

#### Context & Data
Corpus:
1. `battery lasts long`
2. `screen looks bright`
3. `battery charges fast`
4. `bright screen response`
Vocabulary: `[battery, screen, bright, long, fast]` ($V = 5$).
Context window: $c = 1$ (immediate left and right neighbors).

---

#### Question 6
**What is the size of the co-occurrence matrix for this selected vocabulary?**
- A) $4 \times 5$
- B) $5 \times 5$
- C) $5 \times 4$
- D) $25 \times 25$

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The word-word co-occurrence matrix has rows corresponding to target words ($V = 5$) and columns corresponding to context words ($V = 5$), resulting in a $5 \times 5$ matrix.

---

#### Question 7
**What does a context window of 1 mean?**
- A) Only the target word is counted.
- B) All words in the corpus are counted as context.
- C) The vocabulary is reduced to one word.
- D) Only immediate neighboring words are counted.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  A symmetric window of size 1 encompasses only the immediate predecessor (left neighbor) and successor (right neighbor) of the target token.

---

#### Question 8
**In the fragment "battery lasts long", how many context contributions does "lasts" receive with window size 1?**
- A) 2
- B) 1
- C) 0
- D) 3

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  With $c = 1$, the word "lasts" has one word to its left ("battery") and one word to its right ("long"), giving 2 context contributions.

---

#### Question 9
**How many times are "screen" and "bright" adjacent in the toy corpus?**
- A) 0
- B) 2
- C) 1
- D) 3

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Let's check each fragment:
  1. `battery lasts long`: Neither word appears.
  2. `screen looks bright`: "screen" and "bright" are separated by "looks" (distance is 2, not adjacent).
  3. `battery charges fast`: Neither appears.
  4. `bright screen response`: "bright" is immediately adjacent to "screen" (distance 1).
  Thus, they are immediately adjacent exactly **1** time.

---

#### Question 10
**What does a row of the co-occurrence matrix represent?**
- A) The number of labels in the classifier under the case assumptions.
- B) The vectorial representation of a target word based on its context counts.
- C) The one-hot position of a document category in the downstream product-review classifier.
- D) The learning rate selected for truncated SVD.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Each row of the co-occurrence matrix corresponds to a target word, and the entries represent its co-occurrence distribution across all context words.

---

### Case Study 3: Legal Search Term Representation (Questions 11 to 15)

#### Context & Data
A legal search co-occurrence matrix $X$ has dimension $6000 \times 6000$.
Truncated SVD:
$$X \approx U_k \Sigma_k V_k^\top, \quad k = 200$$

---

#### Question 11
**Why is truncated SVD applied here?**
- A) To increase the vocabulary size in the stated case setting.
- B) To make every context vector one-hot.
- C) To reduce dimensionality while preserving important structure.
- D) To remove the need for context words in every downstream search task.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Truncated SVD compresses the high-dimensional ($6000$) sparse matrix into a compact ($200$) dense semantic embedding space while filtering out random noise.

---

#### Question 12
**What is the dimension of each reduced word vector when $k = 200$?**
- A) 6000
- B) 36,000,000
- C) 400
- D) 200

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Selecting $k = 200$ singular components projects each word onto a $200$-dimensional latent subspace.

---

#### Question 13
**What is the size of $U_k$ in this case?**
- A) $6000 \times 200$
- B) $200 \times 200$
- C) $200 \times 6000$
- D) $6000 \times 6000$

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  For an input matrix of size $V \times V$ ($6000 \times 6000$), the left singular matrix truncated to $k$ columns has dimension $V \times k = 6000 \times 200$.

---

#### Question 14
**Which advantage does the reduced representation provide?**
- A) It makes all unrelated words identical under the given conditions.
- B) It can make similarity computation more efficient.
- C) It prevents any word from appearing in context.
- D) It removes all latent semantic structure.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Computing dot products and cosine similarities in $200$ dimensions requires drastically fewer FLOPs and less memory than computing over $6000$ dimensions.

---

#### Question 15
**What kind of information can SVD help uncover?**
- A) The number of documents in chronological order.
- B) The exact spelling of every rare word.
- C) The chronological order of every document.
- D) Latent patterns in word-context usage.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  SVD discovers latent semantic concepts by grouping words that share similar context distributions, even if the words never co-occur directly.

---

### Case Study 4: Search Term Similarity Evaluation (Questions 16 to 20)

#### Context & Data
Candidate cosine similarities for a query $q$:
- A: $0.18$
- B: $0.74$
- C: $0.41$
- D: $0.62$
Analogy formulation: $a : b :: c : ?$ uses vector search near $\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$.

---

#### Question 16
**Which candidate is selected by the synonym rule?**
- A) A
- B) B
- C) C
- D) D

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The synonym rule selects the candidate with the maximum cosine similarity. Among $\{0.18, 0.74, 0.41, 0.62\}$, the maximum is $0.74$ (Candidate B).

---

#### Question 17
**What does a high correlation with human relatedness scores indicate?**
- A) The model's similarity judgments agree well with human judgments.
- B) The vectors remain one-hot after training.
- C) The vocabulary has only one word under the given conditions.
- D) The model ignores cosine similarity during all human-score evaluations.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  A high Spearman/Pearson correlation between vector cosine similarities and human-annotated similarity scores demonstrates that the model's geometric representation accurately reflects human linguistic intuition.

---

#### Question 18
**In an analogy task $a : b :: c : ?$, what vector operation is commonly used?**
- A) $\mathbf{v}_a + \mathbf{v}_b + \mathbf{v}_c$ only
- B) $\mathbf{v}_a - \mathbf{v}_b - \mathbf{v}_c$
- C) $\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$
- D) $\mathbf{v}_a / \mathbf{v}_b$

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The relationship vector is $\mathbf{d}_{\text{rel}} = \mathbf{v}_b - \mathbf{v}_a$. Adding this displacement to $\mathbf{v}_c$ yields $\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$.

---

#### Question 19
**Why are dense word vectors useful for query expansion?**
- A) They make every vector a one-hot vector under the given conditions.
- B) They require no training corpus under the given conditions.
- C) They make every pair of different words orthogonal.
- D) They can place related words near each other in vector space.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Dense word embeddings encode distributional semantics, clustering synonyms and related concepts in close proximity in vector space.

---

#### Question 20
**What metric is used in the synonym rule described in the case?**
- A) Training loss under the given conditions
- B) Matrix rank only
- C) Cosine similarity
- D) Vocabulary size

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The decision rule explicitly ranks candidates based on their **cosine similarity** to the query vector.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **CBOW vs Skip-Gram Direction**:
   - **CBOW**: Input = Context words $\longrightarrow$ Output = Target center word.
   - **Skip-Gram**: Input = Target center word $\longrightarrow$ Output = Context words.
2. **Negative Sampling Exponent**:
   - The unigram distribution is raised to the power of **$0.75$ (or $3/4$)**, not $0.5$ or $1.0$.
3. **Analogy Vector Sign**:
   - For $a : b :: c : d$, the formula is $\mathbf{v}_d \approx \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$. Be careful not to swap $\mathbf{v}_a$ and $\mathbf{v}_b$!
