# Week 12: Encoder-Decoder Models, Attention Mechanisms, and Transformers

---

## 1. Executive Summary & Exam Roadmap

Week 12 culminates the course with modern sequence transduction and the Attention revolution:
1. **The Sequence-to-Sequence (Seq2Seq) Architecture**:
   - The Encoder compresses an arbitrary-length input sequence into a fixed-length vector.
   - The Decoder unrolls autoregressively, predicting target tokens one step at a time conditioning on previous outputs.
2. **The Information Bottleneck**: Performance degrades precipitously as sequence length increases because a single vector cannot retain all syntactic and factual nuances.
3. **The Attention Mechanism (Bahdanau et al., 2014; Luong et al., 2015)**:
   - Calculating alignment scores $e_{ti}$ between decoder states and all encoder hidden states.
   - Softmax normalization yielding non-negative attention weights $\alpha_{ti}$ ($\sum \alpha_{ti} = 1$).
   - Dynamic context vector computation as a weighted sum $\mathbf{c}_t = \sum_i \alpha_{ti} \mathbf{h}_i$.
4. **Self-Attention & Scaled Dot-Product Attention (Vaswani et al., 2017)**:
   - Query ($Q$), Key ($K$), and Value ($V$) projections.
   - The scaled dot-product formula: $\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.
   - Why scale by $\sqrt{d_k}$? Preventing softmax saturation for large vector dimensions.
5. **Positional Encodings**: Supplying essential word order and distance signals to permutation-invariant attention matrices.
6. **Hierarchical Attention Networks**: Word-level attention composing sentences, followed by sentence-level attention composing documents.

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 The Classic Encoder-Decoder (Seq2Seq) Architecture

Used for asynchronous sequence-to-sequence mappings where input and output lengths differ ($T_x \neq T_y$), such as Machine Translation, Summarization, and Question Answering.

```
Source: x_1, x_2, ..., x_Tx
          |    |          |
Encoder: (h_1)-(h_2)-...-(h_Tx) ===[ Fixed Context Vector c = h_Tx ]===> Decoder: (s_1) ---> (s_2) ---> ...
                                                                                    |          |
                                                                          Target:  y_1        y_2
```

#### Encoder
Processes source sequence $\mathbf{x} = (x_1, x_2, \dots, x_{T_x})$:
$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1})$$
The final hidden state $\mathbf{h}_{T_x}$ is taken as the single fixed-length summary context vector $\mathbf{c}$:
$$\mathbf{c} = \mathbf{h}_{T_x}$$

#### Decoder
Autoregressively generates output tokens $\mathbf{y} = (y_1, y_2, \dots, y_{T_y})$:
$$\mathbf{s}_t = g(y_{t-1}, \mathbf{s}_{t-1}, \mathbf{c})$$
$$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{V}\mathbf{s}_t)$$

#### The Information Bottleneck
Forcing all semantic details of a 50-word sentence or multi-page medical record into a single fixed-size vector (e.g., 512 dimensions) causes severe information loss. Early critical clauses (e.g., "avoid aspirin", "recharge failed") are washed away by subsequent tokens.

---

### 2.2 The Attention Mechanism (Bahdanau et al., 2014)

Instead of forcing the encoder to compress everything into one fixed vector $\mathbf{c}$, the Attention mechanism retains **all intermediate encoder states** $\{\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_{T_x}\}$.
At every decoding time step $t$, the decoder generates a tailored, **time-dependent context vector $\mathbf{c}_t$** by dynamically attending to the most relevant source positions.

```
Decoder State s_(t-1)
       |
       +-----> [ Alignment Score e_(t, i) ] <----- All Encoder States h_i
                        |
                 [ Softmax Normalization ]
                        |
                 Attention Weights alpha_(t, i)  (sum to 1)
                        |
                 Context Vector c_t = Sum(alpha_(t, i) * h_i)
```

#### The Four Mathematical Steps of Attention:

1. **Alignment / Scoring Function ($e_{ti}$)**:
   Measures the affinity or relevance between the previous decoder state $\mathbf{s}_{t-1}$ and each source encoder state $\mathbf{h}_i$:
   - **Dot product (Luong)**:
     $$e_{ti} = \mathbf{s}_{t-1}^\top \mathbf{h}_i$$
   - **General / Multiplicative (Luong)**:
     $$e_{ti} = \mathbf{s}_{t-1}^\top \mathbf{W}_a \mathbf{h}_i$$
   - **Additive / Concatenation (Bahdanau)**:
     $$e_{ti} = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i)$$
2. **Softmax Normalization (Attention Weights $\alpha_{ti}$)**:
   Converts raw unbounded alignment scores into a valid probability distribution over source positions:
   $$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k=1}^{T_x} \exp(e_{tk})}, \quad \text{where } \alpha_{ti} \ge 0 \text{ and } \sum_{i=1}^{T_x} \alpha_{ti} = 1$$
3. **Dynamic Context Vector Computation ($\mathbf{c}_t$)**:
   Takes the expectation (weighted average) of the encoder hidden states:
   $$\mathbf{c}_t = \sum_{i=1}^{T_x} \alpha_{ti} \mathbf{h}_i$$
4. **Conditional Prediction**:
   The current decoder state $\mathbf{s}_t$ and output probability are conditioned directly on $\mathbf{c}_t$:
   $$\mathbf{s}_t = g(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$$
   $$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o [\mathbf{s}_t; \mathbf{c}_t])$$

---

### 2.3 Self-Attention & The Transformer (Vaswani et al., 2017)

While Bahdanau attention connects a decoder to an encoder, **Self-Attention** connects tokens within the *same* sequence to one another, removing recurrent loops and enabling massive GPU parallelization.

#### Query, Key, and Value ($Q, K, V$) Abstraction
For each input token embedding $\mathbf{x}_i \in \mathbb{R}^d$, linear projection matrices project it into three distinct roles:
1. **Query ($\mathbf{q}_i = \mathbf{W}_Q \mathbf{x}_i \in \mathbb{R}^{d_k}$)**: What this token is looking for.
2. **Key ($\mathbf{k}_i = \mathbf{W}_K \mathbf{x}_i \in \mathbb{R}^{d_k}$)**: What this token contains (used for matching).
3. **Value ($\mathbf{v}_i = \mathbf{W}_V \mathbf{x}_i \in \mathbb{R}^{d_v}$)**: The actual informational content passed forward.

#### Scaled Dot-Product Attention
Pack vectors into matrices $Q, K \in \mathbb{R}^{N \times d_k}$ and $V \in \mathbb{R}^{N \times d_v}$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

#### Why Scale by $\frac{1}{\sqrt{d_k}}$?
Assume components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with zero mean and variance $1$.
The dot product $\mathbf{q}^\top \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ has mean $0$ and variance $d_k$:
$$\text{Var}(\mathbf{q}^\top \mathbf{k}) = d_k$$
- For large dimensions (e.g. $d_k = 64$ or $128$), the dot products grow very large in magnitude.
- Feeding large values into the softmax function pushes it into extreme saturation regions where gradients $\frac{\partial \text{softmax}}{\partial z}$ are virtually zero.
- Dividing by $\sqrt{d_k}$ scales the variance back down to $1$, preserving healthy, robust gradient flow during backpropagation!

---

### 2.4 Positional Encodings

#### The Permutation Invariance Problem
In self-attention, the operation $\text{softmax}(Q K^\top / \sqrt{d_k}) V$ is strictly permutation-equivariant: shuffling the input tokens shuffles the output representations identically without changing the computed features. Unlike RNNs or CNNs, attention has **no inherent notion of word order**.

#### Sinusoidal Positional Encoding
To inject order, fixed positional vectors $PE_{(pos)}$ of dimension $d$ are directly added to the input token embeddings:
$$\mathbf{x}_{\text{input}} = \text{Embedding}(\text{token}) + PE_{(pos)}$$
Using sine and cosine functions of varying frequencies:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
- Allows the model to attend easily by relative position, because for any fixed offset $k$, $PE_{(pos + k)}$ can be represented as a linear function of $PE_{(pos)}$.

---

### 2.5 Multi-Head Attention & Hierarchical Structures

#### Multi-Head Attention
Instead of performing a single attention function, Multi-Head Attention linearly projects queries, keys, and values $h$ times with different, learned linear projections to $d_k, d_k$, and $d_v$ dimensions:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O$$
$$\text{where } \text{head}_j = \text{Attention}(Q \mathbf{W}_j^Q, K \mathbf{W}_j^K, V \mathbf{W}_j^V)$$
- Enables the network to jointly attend to information from different representation subspaces (e.g., one head tracks syntactic subject-verb dependencies, another tracks coreference, another tracks topic semantics).

#### Hierarchical Attention Networks (HAN)
For long hierarchical documents:
1. **Word-Level Attention**: Encodes word vectors into a unified sentence vector, weighting key phrases.
2. **Sentence-Level Attention**: Encodes sentence vectors into a final document vector, weighting key paragraphs.

---

## 3. Core Formulas & Cheat Sheet

| Mechanism | Mathematical Formula | Function / Role |
|:---|:---|:---|
| **Alignment Score** | $e_{ti} = \mathbf{s}_{t-1}^\top \mathbf{h}_i$ | Affinity between decoder state $\mathbf{s}$ and encoder state $\mathbf{h}$ |
| **Attention Weights** | $\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_k \exp(e_{tk})}$ | Softmax probability distribution ($\sum_i \alpha_{ti} = 1$) |
| **Context Vector** | $\mathbf{c}_t = \sum_i \alpha_{ti} \mathbf{h}_i$ | Dynamic, time-dependent weighted representation |
| **Scaled Dot-Product**| $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | Fundamental engine of Transformer architectures |
| **Scaling Factor** | $\frac{1}{\sqrt{d_k}}$ | Normalizes variance of dot products to 1; prevents softmax saturation |
| **Positional Encoding**| $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$ | Injects sequence order into attention representations |
| **Multi-Head Attention**| $\text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O$ | Captures diverse subspace relationships simultaneously |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Clinic Discharge Note Conversion (Questions 1 to 5)

#### Context & Data
A hospital converts doctors' discharge notes into patient-friendly language using an encoder-decoder architecture. In an early prototype, a single fixed-length summary vector is passed from the encoder to the decoder.

---

#### Question 1
**In the early prototype, what is the main risk of using only one final summary vector?**
- A) Important source details may be compressed away.
- B) The decoder must output every source word unchanged.
- C) The encoder is unable to read numerical tokens.
- D) The vocabulary becomes empty during generation.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Forcing a long, detailed document through a single fixed-length vector creates an information bottleneck, causing critical early details (e.g., drug contraindications or fever warnings) to be lost.

---

#### Question 2
**What does the encoder mainly provide to the decoder in this sequence model?**
- A) A sorted list of output tokens for the note
- B) The final patient label after thresholding
- C) Representations derived from the source note
- D) A replacement for the training loss function

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The encoder's role is to ingest the source text tokens and construct contextual hidden representations that capture the semantic content of the document.

---

#### Question 3
**Why is the decoder described as producing the converted note token by token?**
- A) It ignores all tokens generated earlier.
- B) Each new token can depend on earlier output tokens.
- C) It computes the encoder states after every output only.
- D) Every output token must have the same probability.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Autoregressive decoding generates language sequentially, allowing each newly predicted word to condition on the full history of previously generated tokens: $P(y_t | y_1, \dots, y_{t-1}, \mathbf{c})$.

---

#### Question 4
**Which change most directly helps the decoder recover a warning sentence from early in a long note?**
- A) Removing all source-token representations
- B) Replacing the decoder with a lookup table
- C) Using only the last generated token as input
- D) Allowing access to relevant encoder states

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Implementing an attention mechanism grants the decoder direct, dynamic access to all intermediate encoder hidden states, allowing it to retrieve early warning tokens regardless of document length.

---

#### Question 5
**If a long converted note omits an early allergy warning, what does that suggest about the fixed-summary design?**
- A) The final summary did not preserve all needed information.
- B) The output language has no word order requirement at any step.
- C) The source note was processed without any token representations.
- D) The decoder cannot produce medical vocabulary for short notes.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Omitting early source information is the classic symptom of information degradation caused by fixed-vector bottleneck compression.

---

### Case Study 2: Maintenance Ticket Digest (Questions 6 to 10)

#### Context & Data
A support desk generates digests from maintenance tickets with 4 sections:
1. Machine ID
2. Fault symptoms
3. Technician notes
4. Replacement parts
For a specific output token, the attention weights over the four sections are:
$$0.10, \quad 0.25, \quad 0.15, \quad 0.50$$

---

#### Question 6
**What is the role of the attention weights in this decoder step?**
- A) They replace all encoder states with fixed zero vectors.
- B) They set the number of output classes for the digest.
- C) They specify how much each section contributes.
- D) They sort the ticket sections into alphabetical order.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Attention weights act as a convex combination weighting: they define the exact percentage contribution of each source section to the synthesized context vector.

---

#### Question 7
**Why is softmax applied to the alignment scores?**
- A) To convert scores into normalized nonnegative weights.
- B) To remove the decoder hidden state from every step.
- C) To force the largest raw score to become exactly one.
- D) To make all ticket sections contribute with equal weights.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The softmax function exponentiates and normalizes raw, unbounded alignment scores into non-negative values that strictly sum to $1.0$ ($0.10 + 0.25 + 0.15 + 0.50 = 1.00$).

---

#### Question 8
**Which ticket section contributes most to this output token?**
- A) Section 1
- B) Section 2
- C) Section 3
- D) Section 4

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Comparing the weights $\{0.10, 0.25, 0.15, 0.50\}$, Section 4 has the largest weight ($0.50$ or $50\%$).

---

#### Question 9
**How is the context vector computed after these weights are available?**
- A) By choosing the first encoder state only
- B) By forming a weighted sum of encoder states
- C) By averaging decoder outputs from all steps
- D) By discarding the highest-weight section

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The context vector is the linear weighted sum: $\mathbf{c}_t = \sum_{i=1}^4 \alpha_i \mathbf{h}_i = 0.10 \mathbf{h}_1 + 0.25 \mathbf{h}_2 + 0.15 \mathbf{h}_3 + 0.50 \mathbf{h}_4$.

---

#### Question 10
**If the support team expects the replacement-part section to drive this token, what attention pattern would support that interpretation?**
- A) All four sections receiving negative attention weights
- B) The decoder state being removed from the scoring step
- C) The replacement-part section receiving the largest weight
- D) Every ticket section receiving exactly the same weight

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  If the replacement-part section receives the highest attention weight (as Section 4 does with $0.50$), it mathematically dominates the context vector, confirming its primary influence.

---

### Case Study 3: Procurement Rule Checking (Questions 11 to 15)

#### Context & Data
A model reviews tender documents using **self-attention** with Query ($Q$), Key ($K$), and Value ($V$) vectors, scaled by $\sqrt{d_k}$, followed by softmax and positional encodings.

---

#### Question 11
**What advantage does self-attention give for checking references across distant clauses?**
- A) It forces every token to use only its previous neighbor.
- B) It removes all long-distance comparisons.
- C) It replaces the document with a single class label.
- D) It lets tokens compare directly with other tokens.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  In self-attention, every token computes an inner product directly with every other token in $O(1)$ path length, allowing instantaneous reference checking across arbitrary document distances without recurrent decay.

---

#### Question 12
**In the query-key comparison, what do the resulting scores represent?**
- A) The final number of tender documents reviewed
- B) Relevance between one token and another token
- C) The optimizer’s learning-rate value during training
- D) The length of the output vocabulary for labels

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The query-key dot product $\mathbf{q}_i^\top \mathbf{k}_j$ measures the alignment and semantic relevance between the query of token $i$ and the key of token $j$.

---

#### Question 13
**What is the purpose of scaling the dot product by $\sqrt{d_k}$?**
- A) To keep score magnitudes controlled before softmax
- B) To erase the effect of all value vectors in the layer
- C) To make positional encodings unnecessary for ordering
- D) To force all attention scores to become identical

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Dividing by $\sqrt{d_k}$ counteracts the variance growth of high-dimensional dot products, preventing the softmax function from saturating into regions with vanishingly small gradients.

---

#### Question 14
**What role do Value vectors play after attention weights are computed?**
- A) They choose the batch size for training runs.
- B) They remove token-order information from clauses.
- C) They provide the content that gets combined.
- D) They store only the document identifier value.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Once attention weights $\alpha_{ij}$ are derived from query-key matching, they are multiplied by the Value vectors $\mathbf{v}_j$ to compute the weighted output representation: $\sum_j \alpha_{ij} \mathbf{v}_j$.

---

#### Question 15
**Why are positional encodings needed in this tender-review model?**
- A) They delete repeated words from the document.
- B) They replace softmax in attention scoring.
- C) They make every token attend to itself only.
- D) They supply information about token order.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Self-attention is fundamentally order-invariant. Positional encodings provide distinct mathematical coordinates for word order, allowing the model to distinguish antecedent clauses from subsequent clauses.

---

### Case Study 4: Campus Feedback Review (Questions 16 to 20)

#### Context & Data
A hierarchical model processes student feedback:
- Words form sentences (Word-level attention).
- Sentences form reports (Sentence-level attention).
- Multi-head attention tracks diverse feedback categories.

---

#### Question 16
**Why is a two-level representation natural for these reports?**
- A) Reports contain only a single unordered word list.
- B) Words form sentences, and sentences form reports.
- C) The model must ignore sentence boundaries.
- D) Every paragraph has exactly one token.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Text documents naturally follow a hierarchical compositional structure: characters compose words, words compose sentences, and sentences compose documents.

---

#### Question 17
**What does word-level attention help identify?**
- A) The number of reports in the dataset split
- B) The final optimizer update rule for training
- C) The batch size used for report classification
- D) Important words or phrases inside a sentence

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Word-level attention highlights salient words (e.g. "hazardous", "defective") within a sentence when constructing the sentence embedding.

---

#### Question 18
**What does sentence-level attention help identify?**
- A) Sentences that are most useful for the report decision
- B) Characters that should be removed from each word
- C) Output tokens generated by a decoder layer
- D) The number of attention heads in the layer

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Sentence-level attention assigns weights to individual sentences, prioritizing the most informative sentences when constructing the overall document summary.

---

#### Question 19
**Why can multiple attention heads be useful in this feedback model?**
- A) They force all heads to learn identical attention weights.
- B) They delete word-level and sentence-level evidence.
- C) They allow different evidence patterns to be tracked.
- D) They make positional information impossible to use.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Multi-head attention enables distinct heads to independently focus on different semantic concerns simultaneously (e.g. one head attends to safety keywords, another to equipment complaints).

---

#### Question 20
**If sentence attention is almost uniform across a long report, what concern should the team consider?**
- A) The report contains no sentence representations.
- B) The model may not be selecting the most informative sentences.
- C) Multi-head attention cannot operate on textual inputs.
- D) Word-level attention must always be zero for all words.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Uniform attention weights ($1/N$ for all sentences) imply an inability to discriminate key sentences from routine sentences, acting as a simple unweighted average.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Context Vector Computation**:
   - The context vector is a **linear combination of encoder states** $\mathbf{c}_t = \sum_i \alpha_{ti} \mathbf{h}_i$, NOT decoder states!
2. **Softmax Normalization Dimension**:
   - In attention, softmax is computed across the **source sequence length $T_x$**, ensuring the weights over source tokens sum to 1.
3. **Scaling Factor in Self-Attention**:
   - Remember the formula $\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.
   - The scaling denominator is the **square root of key dimension $\sqrt{d_k}$**, not $d_k$ and not sequence length $N$.
