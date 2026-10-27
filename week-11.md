# Week 11: Recurrent Neural Networks (RNNs), Backpropagation Through Time, LSTMs, and GRUs

---

## 1. Executive Summary & Exam Roadmap

Week 11 covers temporal and sequential deep learning architectures:
1. **Sequence Modeling Paradigms**: One-to-one, one-to-many, many-to-one, and many-to-many sequence problems.
2. **Vanilla Recurrent Neural Networks (RNNs)**: Hidden state recurrence $\mathbf{s}_t = \tanh(\mathbf{W}\mathbf{s}_{t-1} + \mathbf{U}\mathbf{x}_t + \mathbf{b})$, parameter sharing across time, and forward unfolding.
3. **Backpropagation Through Time (BPTT)**: Unrolling the computational graph across $T$ steps and calculating temporal parameter gradients.
4. **Vanishing and Exploding Gradients in Sequences**: The exponential decay/growth of Jacobian products $\prod \frac{\partial \mathbf{s}_j}{\partial \mathbf{s}_{j-1}}$; **Gradient Clipping** and **Truncated BPTT**.
5. **Long Short-Term Memory Networks (LSTMs)**: The Constant Error Carousel (CEC), the additive cell state $\mathbf{c}_t$, and the three regulating gates: Forget ($\mathbf{f}_t$), Input ($\mathbf{i}_t$), and Output ($\mathbf{o}_t$).
6. **Gated Recurrent Units (GRUs)**: Compact architectural design merging cell and hidden states, utilizing only two gates: Reset ($\mathbf{r}_t$) and Update ($\mathbf{z}_t$).

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 The Need for Recurrent Networks

Standard feedforward networks suffer from three fundamental limitations when modeling sequences:
1. **Fixed Input / Output Size**: Cannot process variable-length inputs (e.g. sentences with 5 vs. 50 words).
2. **No Position/Order Awareness**: Permuting input coordinates destroys meaning in temporal series.
3. **No Feature Sharing Across Time**: Features learned at time $t_1$ are not reused if they appear at time $t_2$.

#### Taxonomy of Sequence Learning Tasks
- **One-to-One**: Standard classification (Image $\to$ Category).
- **One-to-Many**: Image Captioning (Image $\to$ Sequence of words).
- **Many-to-One**: Sentiment Analysis / Transcript Intent (Sequence of words $\to$ Class label).
- **Many-to-Many (Synchronized)**: Video frame tagging / POS Tagging (Sequence of frames $\to$ Sequence of labels of equal length).
- **Many-to-Many (Asynchronous / Seq2Seq)**: Machine Translation (Sequence in source language $\to$ Sequence in target language of differing length).

---

### 2.2 Vanilla Recurrent Neural Networks (RNNs)

```
       y_1                y_2                y_t
        ^                  ^                  ^
        | V                | V                | V
s_0 -> (s_1) --- W ---> (s_2) --- W ---> ... (s_t)
        ^                  ^                  ^
        | U                | U                | U
       x_1                x_2                x_t
```

#### Forward Equations
At each discrete time step $t \in \{1, 2, \dots, T\}$:
1. **Hidden State Pre-activation**:
   $$\mathbf{a}_t = \mathbf{W}\mathbf{s}_{t-1} + \mathbf{U}\mathbf{x}_t + \mathbf{b}$$
2. **Hidden State Activation**:
   $$\mathbf{s}_t = \tanh(\mathbf{a}_t) = \tanh(\mathbf{W}\mathbf{s}_{t-1} + \mathbf{U}\mathbf{x}_t + \mathbf{b})$$
3. **Output Layer**:
   $$\hat{\mathbf{y}}_t = \text{softmax}(\mathbf{V}\mathbf{s}_t + \mathbf{c})$$
where:
- $\mathbf{U} \in \mathbb{R}^{d_h \times d_x}$: Input-to-hidden weight matrix.
- $\mathbf{W} \in \mathbb{R}^{d_h \times d_h}$: Recurrent hidden-to-hidden weight matrix.
- $\mathbf{V} \in \mathbb{R}^{d_y \times d_h}$: Hidden-to-output weight matrix.
- **Shared Parameters**: $\mathbf{U}, \mathbf{W}, \mathbf{V}$ are identically shared across all time steps.

---

### 2.3 Backpropagation Through Time (BPTT) & Gradient Pathologies

The total loss across a sequence of length $T$ is:
$$\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t$$
Evaluating the gradient w.r.t the recurrent matrix $\mathbf{W}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}}$$
Using the chain rule across recurrent states:
$$\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}} = \sum_{k=1}^t \frac{\partial \mathcal{L}_t}{\partial \mathbf{s}_t} \frac{\partial \mathbf{s}_t}{\partial \mathbf{s}_k} \frac{\partial \mathbf{s}_k}{\partial \mathbf{W}}$$
The critical term is the temporal Jacobian product:
$$\frac{\partial \mathbf{s}_t}{\partial \mathbf{s}_k} = \prod_{j=k+1}^t \frac{\partial \mathbf{s}_j}{\partial \mathbf{s}_{j-1}} = \prod_{j=k+1}^t \mathbf{W}^\top \text{diag}(1 - \mathbf{s}_j^2)$$

#### The Dilemma:
Let $\lambda$ be the dominant eigenvalue of $\mathbf{W}$:
1. **If $\lambda > 1$ (Exploding Gradients)**:
   The product grows as $\lambda^{t-k}$. For large $T$, gradients become astronomical, causing NaN overflows and destabilizing parameters.
   - **Remedy: Gradient Clipping**:
     $$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 \le \theta \\ \frac{\theta}{\|\mathbf{g}\|_2} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 > \theta \end{cases}$$
2. **If $\lambda < 1$ (Vanishing Gradients)**:
   The factor $(1 - \mathbf{s}_j^2) \le 1$ combined with $\lambda < 1$ causes $\frac{\partial \mathbf{s}_t}{\partial \mathbf{s}_k} \to 0$ exponentially fast as $t - k$ grows. The network cannot bridge dependencies beyond 5–10 steps (forgetting long-term history).
   - **Remedy: Gated Architectures (LSTM / GRU)** and **Truncated BPTT**.

---

### 2.4 Long Short-Term Memory (LSTM) (Hochreiter & Schmidhuber, 1997)

The LSTM resolves vanishing gradients by replacing the plain hidden state with an internal memory cell governed by a **Constant Error Carousel (CEC)** with additive linear updates.

```
                    Cell State: c_(t-1) --------[*]---------------------[+]---------> c_t
                                                 ^                       ^             |
                                                 | (f_t: Forget)         |             |
                                                 |               (i_t * c~_t: Input)  [tanh]
                                                 |                       |             |
                                                 |                       |             v
Hidden State: h_(t-1) ---[Split]---> [Gate Ops: f_t, i_t, c~_t, o_t] ----+----------> [*] ---> h_t
Input: x_t --------------/                                                             ^
                                                                                       | (o_t: Output)
```

#### The Six Core LSTM Equations:
1. **Forget Gate ($\mathbf{f}_t$)**:
   Decides what fraction of information to erase from the previous cell state $\mathbf{c}_{t-1}$:
   $$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{h}_{t-1} + \mathbf{U}_f \mathbf{x}_t + \mathbf{b}_f)$$
2. **Input Gate ($\mathbf{i}_t$)**:
   Decides which new components to write into the cell state:
   $$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{h}_{t-1} + \mathbf{U}_i \mathbf{x}_t + \mathbf{b}_i)$$
3. **Candidate Cell State ($\tilde{\mathbf{c}}_t$)**:
   New candidate information synthesized from current input:
   $$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{h}_{t-1} + \mathbf{U}_c \mathbf{x}_t + \mathbf{b}_c)$$
4. **Cell State Update ($\mathbf{c}_t$)**:
   Linear combination of past memory and new candidate:
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
5. **Output Gate ($\mathbf{o}_t$)**:
   Controls what parts of the internal cell state are exposed to the external network:
   $$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{h}_{t-1} + \mathbf{U}_o \mathbf{x}_t + \mathbf{b}_o)$$
6. **Hidden State ($\mathbf{h}_t$)**:
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

> [!NOTE]
> **Why LSTM Doesn't Suffer Vanishing Gradients**:
> In the cell update equation $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \dots$, the derivative w.r.t past memory is $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t$. If the forget gate is learned to be near $1$ ($\mathbf{f}_t \approx 1$), the gradient flows through the cell state across hundreds of time steps with **zero attenuation**!

---

### 2.5 Gated Recurrent Unit (GRU) (Cho et al., 2014)

The GRU simplifies the LSTM by eliminating the separate cell state $\mathbf{c}_t$ and merging the forget and input gates into a coupled mechanism:

#### The Four Core GRU Equations:
1. **Reset Gate ($\mathbf{r}_t$)**:
   Determines how much past hidden state to forget when computing the candidate state:
   $$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{h}_{t-1} + \mathbf{U}_r \mathbf{x}_t + \mathbf{b}_r)$$
2. **Update Gate ($\mathbf{z}_t$)**:
   Controls how much of the old state to carry forward vs. how much of the new candidate to accept:
   $$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{h}_{t-1} + \mathbf{U}_z \mathbf{x}_t + \mathbf{b}_z)$$
3. **Candidate Hidden State ($\tilde{\mathbf{h}}_t$)**:
   $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{U} \mathbf{x}_t + \mathbf{b})$$
4. **Hidden State Update ($\mathbf{h}_t$)**:
   Linear interpolation between previous state $\mathbf{h}_{t-1}$ and candidate $\tilde{\mathbf{h}}_t$:
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

---

## 3. Core Formulas & Cheat Sheet

| Model | Gate Formulation | State Update Rule | Parameter Multiplier |
|:---|:---|:---|:---|
| **Vanilla RNN** | None | $\mathbf{s}_t = \tanh(\mathbf{W}\mathbf{s}_{t-1} + \mathbf{U}\mathbf{x}_t + \mathbf{b})$ | $1 \times (d_h^2 + d_h d_x)$ |
| **LSTM** | $\mathbf{f}_t, \mathbf{i}_t, \mathbf{o}_t = \sigma(\dots)$<br>$\tilde{\mathbf{c}}_t = \tanh(\dots)$ | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$<br>$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$ | $4 \times (d_h^2 + d_h d_x + d_h)$ |
| **GRU** | $\mathbf{r}_t, \mathbf{z}_t = \sigma(\dots)$<br>$\tilde{\mathbf{h}}_t = \tanh(\dots)$ | $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$ | $3 \times (d_h^2 + d_h d_x + d_h)$ |
| **Gradient Clipping** | $\|\mathbf{g}\|_2 > \theta \implies \mathbf{g} \leftarrow \frac{\theta}{\|\mathbf{g}\|_2} \mathbf{g}$ | Preserves gradient direction while capping magnitude | Prevents catastrophic parameter divergence |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Call Transcript Intent Prediction (Questions 1 to 5)

#### Context & Data
A customer-care system predicts intent from phone transcripts. The model processes variable-length sequences using a many-to-one RNN architecture. In vanilla RNNs, early transcript details are forgotten by the time the final word is read.

---

#### Question 1
**Why is an RNN suitable for transcript modeling?**
- A) It maintains a hidden state across sequence steps.
- B) It requires fixed-size images under the given conditions.
- C) It ignores word order completely.
- D) It removes the need for training.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Recurrent neural networks maintain an evolving hidden state vector $\mathbf{s}_t$ that carries contextual memory from prior words across arbitrary sequence lengths.

---

#### Question 2
**What does forgetting early transcript details mainly indicate?**
- A) The vocabulary is empty under the given conditions.
- B) The hidden state has no update rule.
- C) Earlier information is overwritten or weakened over time.
- D) The output layer is convolutional in the stated case setting.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  In a vanilla RNN, the hidden state is repeatedly overwritten via nonlinear matrix multiplications at each step, causing early information to be attenuated or overwritten (vanishing gradient problem).

---

#### Question 3
**Which task structure describes predicting one transcript intent from a sequence of words?**
- A) Many-to-one
- B) One-to-many
- C) One-to-one
- D) Many-to-many without recurrence

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Processing a variable-length sequence of input tokens to produce a single final category prediction is the canonical definition of a **Many-to-One** sequence learning problem.

---

#### Question 4
**Why can parameter sharing across time steps help in this task?**
- A) It forces every sentence to contain exactly the same words.
- B) It allows the same pattern to be recognized anywhere in the transcript.
- C) It removes the need for backpropagation through time at all.
- D) It replaces word vectors with scalar numbers only.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Sharing transition matrices $\mathbf{W}, \mathbf{U}$ across all time steps enforces temporal translation invariance: a complaint phrase carries the same semantic meaning whether it appears at the beginning, middle, or end of the transcript.

---

#### Question 5
**Which architecture is designed to reduce the forgetting of early information using gates?**
- A) Gated architectures such as LSTM and GRU
- B) Unregularized linear regression
- C) Plain bag-of-words without state
- D) Single convolution filters only

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Gated architectures (LSTMs and GRUs) use learnable multiplicative gates to explicitly regulate what to remember, write, and erase, protecting early signals from being overwritten.

---

### Case Study 2: Machine Log Monitoring for Factory Alerts (Questions 6 to 10)

#### Context & Data
An RNN monitors factory event logs using Backpropagation Through Time (BPTT).
- Truncated BPTT uses a window of 12 steps.
- Gradient clipping is added to prevent unstable parameter updates.

---

#### Question 6
**What does BPTT do?**
- A) It trains only the output layer under the given conditions.
- B) It backpropagates errors through the unrolled sequence.
- C) It removes recurrence under the given conditions.
- D) It converts logs into images for the data shown here.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Backpropagation Through Time unfolds the recurrent computational graph across the time steps of the sequence and backpropagates error gradients from later steps back to earlier steps.

---

#### Question 7
**What limitation is introduced by truncated BPTT with window size 12?**
- A) The network cannot make predictions for the described model.
- B) The input sequence must have exactly 12 symbols.
- C) Gradients are propagated only across the chosen recent window.
- D) The hidden state must be zero forever under the case assumptions.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Truncated BPTT cuts the backward gradient flow after a fixed window (here, 12 steps), preventing the network from learning dependencies that span beyond that temporal horizon.

---

#### Question 8
**What is the symptom of exploding gradients?**
- A) Very large parameter updates and unstable loss
- B) No hidden state in this specific scenario
- C) Fixed output vocabulary under the case assumptions
- D) Smaller-than-zero sequence length

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  When gradients explode, parameter updates become excessively large, causing weights to oscillate wildly, resulting in loss spikes and numerical overflow (`NaN`).

---

#### Question 9
**What does gradient clipping do?**
- A) It removes all gradients in this specific scenario.
- B) It limits the magnitude of gradients above a threshold.
- C) It increases every gradient without bound.
- D) It replaces recurrence with convolution for the described model.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Gradient clipping rescales the gradient vector whenever its $L_2$ norm exceeds a threshold $\theta$, ensuring the update step remains bounded while preserving the gradient's directional vector.

---

#### Question 10
**Why can long recurrent chains create gradient problems?**
- A) The input has no meaningful order.
- B) The sequence has no time steps during backpropagation through time at all.
- C) The model has no trainable weights under the given conditions.
- D) Repeated multiplication through time can shrink or grow gradients.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Backpropagating through time involves repeated matrix multiplications by $\mathbf{W}^\top \text{diag}(1 - \mathbf{s}^2)$, which exponentially contracts gradients to zero (if eigenvalues $< 1$) or blows them up to infinity (if eigenvalues $> 1$).

---

### Case Study 3: Weather Alert Prediction from Hourly Observations (Questions 11 to 15)

#### Context & Data
An LSTM predicts storm alerts from hourly sensor logs using:
- Forget gate ($\mathbf{f}_t$)
- Input gate ($\mathbf{i}_t$)
- Output gate ($\mathbf{o}_t$)
- Cell state ($\mathbf{c}_t$)

---

#### Question 11
**Which LSTM component directly stores long-term information?**
- A) The softmax label only
- B) The one-hot input only
- C) The cell state
- D) The batch size

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The internal **cell state** ($\mathbf{c}_t$) serves as the primary linear memory highway where information is preserved across extended temporal horizons.

---

#### Question 12
**Which gate decides how much old cell-state information is retained?**
- A) Output gate
- B) ReLU gate
- C) Reset gate
- D) Forget gate

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  The **forget gate** $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{h}_{t-1} + \mathbf{U}_f \mathbf{x}_t + \mathbf{b}_f)$ outputs values in $[0, 1]$ that modulate how much of $\mathbf{c}_{t-1}$ is retained.

---

#### Question 13
**Which gate controls how much new candidate information is written?**
- A) Input gate for the described model
- B) Output gate in this specific scenario
- C) Pooling gate
- D) Softmax gate

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The **input gate** $\mathbf{i}_t$ scales the candidate vector $\tilde{\mathbf{c}}_t$, deciding which new features are accumulated into the cell state.

---

#### Question 14
**Why are sigmoid activations commonly used in gates?**
- A) They always output integers in this specific scenario.
- B) They output values between 0 and 1 for soft control.
- C) They remove all memory for the data shown here.
- D) They make gradients infinite.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The sigmoid function bounds outputs strictly between $0$ (completely closed / block signal) and $1$ (completely open / pass signal), providing a smooth, differentiable gating mechanism.

---

#### Question 15
**What is the main advantage of LSTM gates in this case?**
- A) They remove all sequence dependence from the model.
- B) They require fixed-length inputs only under the case assumptions.
- C) They regulate remembering, writing, and exposing information over time.
- D) They replace all observations with images before recurrent processing begins for alerts.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The gates allow the network to selectively retain persistent trends (e.g. barometric pressure drops) while filtering out transient sensor noise.

---

### Case Study 4: Delivery Status Prediction from Event Logs (Questions 16 to 20)

#### Context & Data
A courier company uses a Gated Recurrent Unit (GRU) to predict delivery status:
- Update gate ($\mathbf{z}_t$)
- Reset gate ($\mathbf{r}_t$)

---

#### Question 16
**What is a key difference between a GRU and an LSTM?**
- A) A GRU uses convolution filters only.
- B) A GRU has no recurrent state under the case assumptions.
- C) A GRU cannot process sequences.
- D) A GRU is a more compact gated recurrent architecture.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  The GRU combines cell state and hidden state into one representation and uses 2 gates instead of 3, resulting in a more compact model with fewer parameters and faster execution.

---

#### Question 17
**Which GRU gate controls how much previous hidden state is retained?**
- A) Update gate
- B) Output gate
- C) Pooling gate
- D) Vocabulary gate

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  In a GRU, the **update gate** $\mathbf{z}_t$ acts as a linear blender between the old hidden state and the candidate state: $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$.

---

#### Question 18
**What does the reset gate control?**
- A) The number of output classes for the described model
- B) The amount of past hidden state used in forming the candidate state
- C) The image filter size used by the model
- D) The batch-normalization variance computed before each recurrent candidate update

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  The **reset gate** $\mathbf{r}_t$ scales $\mathbf{h}_{t-1}$ when computing the candidate activation: $\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}(\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{U}\mathbf{x}_t + \mathbf{b})$.

---

#### Question 19
**Why might the courier team choose a GRU?**
- A) It removes all trainable parameters in this specific scenario.
- B) It requires no event ordering in this specific scenario.
- C) It is compact while still handling sequence dependencies.
- D) It works only on images for the data shown here.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  GRUs offer comparable modeling capacity to LSTMs on many real-world tasks while utilizing fewer parameters, training faster, and requiring less memory.

---

#### Question 20
**Which problem do GRU and LSTM architectures primarily address compared with vanilla RNNs?**
- A) Increasing vocabulary sparsity in this specific scenario
- B) Removing all sequence order from the data
- C) Removing the need for hidden states in recurrent sequence models entirely
- D) Capturing longer-range dependencies using controlled information flow

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Both gated architectures prevent gradient vanishing through additive highways, allowing the network to capture dependencies across long temporal intervals.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Parameter Count Comparison**:
   - An LSTM layer with $d_h$ hidden units and $d_x$ input units has **$4 \times$** the parameters of a vanilla RNN:
     $$\text{Params}_{\text{LSTM}} = 4 \times [d_h(d_h + d_x) + d_h]$$
   - A GRU layer has **$3 \times$** the parameters of a vanilla RNN:
     $$\text{Params}_{\text{GRU}} = 3 \times [d_h(d_h + d_x) + d_h]$$
2. **Gradient Clipping Function**:
   - Clipping scales down the *norm* of the gradient vector: $\mathbf{g} \leftarrow \frac{\theta}{\|\mathbf{g}\|} \mathbf{g}$. It **does not** change the direction of the gradient, only its magnitude.
3. **Activation Functions in Gates**:
   - Gates always use **Sigmoid ($\sigma$)** because outputs must be strictly in $[0, 1]$ to act as percentage filters.
   - States and candidates always use **$\tanh$** to allow activations to be both positive and negative around zero.
