# Week 1: History of Deep Learning, McCulloch-Pitts Neuron, Perceptrons, and Representation Power

---

## 1. Executive Summary & Exam Roadmap

Week 1 lays the mathematical and historical foundations of modern Deep Learning. In NPTEL CS7015, Week 1 covers:
1. **History of Artificial Neural Networks**: Milestones from biological foundations to modern revival.
2. **McCulloch-Pitts (MP) Neuron (1943)**: Threshold-based binary logic unit, excitatory vs. inhibitory inputs, geometric interpretation, representation capacity.
3. **Rosenblatt's Perceptron (1958)**: Real-valued inputs, real-valued weights, explicit bias / threshold, linear hyperplanes.
4. **Perceptron Learning Algorithm (PLA)**: Iterative weight update rule, geometric intuition, Novikoff's convergence theorem and formal proof.
5. **Linear Separability & Multilayer Perceptrons (MLPs)**: Limitations of single-layer models (the XOR problem), Minsky & Papert's critique, representation power of 2-layer and 3-layer networks.

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 Historical Timeline & Key Milestones (Lecture 1)
- **1943 - McCulloch & Pitts**: Introduced the first computational abstraction of a biological neuron (threshold logic unit).
- **1949 - Donald Hebb**: Hebbian Learning rule ("neurons that fire together, wire together").
- **1958 - Frank Rosenblatt**: Developed the Perceptron with learnable continuous weights.
- **1969 - Marvin Minsky & Seymour Papert**: Published *Perceptrons*, proving that single-layer perceptrons cannot compute non-linearly separable functions like XOR. This triggered the first **"AI Winter"**.
- **1986 - Rumelhart, Hinton, & Williams**: Popularized the Backpropagation algorithm for training multi-layer neural networks, sparking the connectionist revival.
- **1989 - Cybenko / 1991 - Hornik**: Proved the Universal Approximation Theorem.
- **1990s**: Rise of Support Vector Machines (SVMs) and kernel methods; neural networks fell out of favor due to vanishing gradients, slow training, and overfitting.
- **2006 - Hinton, Salakhutdinov, Bengio**: Deep Belief Networks and Greedy Layer-wise Pre-training revitalized deep architectures (birth of "Deep Learning").
- **2012 - Krizhevsky, Sutskever, & Hinton (AlexNet)**: Outperformed classical computer vision in ImageNet 2012 by an unprecedented margin using GPUs, ReLU activations, and Dropout.

---

### 2.2 Biological Neuron vs. Artificial Neuron
A biological neuron consists of:
- **Dendrites**: Receive incoming electrochemical signals (inputs $x_i$).
- **Soma (Cell Body)**: Accumulates and processes the electrical potentials.
- **Axon**: Transmits the action potential (output signal $y$) to subsequent neurons.
- **Synapses**: Junction points where signal strength is modulated (weights $w_i$).

```
Biological:  Dendrites (Inputs) ---> Soma (Summation & Threshold) ---> Axon (Output)
Artificial:  x_1, x_2, ..., x_n ---> Sum(w_i * x_i) + b ---> Activation f(z) ---> y
```

---

### 2.3 McCulloch-Pitts (MP) Neuron (1943)

#### Definition and Mathematical Model
- **Inputs**: Binary vector $\mathbf{x} = [x_1, x_2, \dots, x_n]^\top$, where $x_i \in \{0, 1\}$.
- **Weights**: All inputs have equal, unit weight ($w_i = 1$).
- **Input Types**:
  - **Excitatory inputs**: Add to the net potential.
  - **Inhibitory inputs**: If any inhibitory input is active ($1$), the neuron is completely suppressed and output is forcefully $0$.
- **Aggregation & Firing Rule**:
  Assuming no active inhibitory inputs:
  $$g(x_1, x_2, \dots, x_n) = \sum_{i=1}^n x_i$$
  $$y = f(g(\mathbf{x})) = \begin{cases} 1 & \text{if } \sum_{i=1}^n x_i \ge \theta \\ 0 & \text{if } \sum_{i=1}^n x_i < \theta \end{cases}$$
  where $\theta \in \mathbb{Z}^+$ is a predefined integer threshold.

#### Boolean Functions Implementable by MP Neuron
1. **AND Function ($n$ inputs)**:
   - Output is $1$ if and only if all $n$ inputs are $1$.
   - Aggregation: $\sum_{i=1}^n x_i \ge n \implies \mathbf{\theta = n}$.
2. **OR Function ($n$ inputs)**:
   - Output is $1$ if at least one input is $1$.
   - Aggregation: $\sum_{i=1}^n x_i \ge 1 \implies \mathbf{\theta = 1}$.
3. **NOT Function (1 input)**:
   - Requires an inhibitory input. Let $x_1$ be inhibitory: if $x_1 = 1$, output is $0$; if $x_1 = 0$, output fires with $\theta = 0$.
4. **NAND / NOR Functions**:
   - Implemented via inhibitory inputs or threshold logic combinations.

#### Geometric Interpretation & Limitations of MP Neuron
- The equation $\sum_{i=1}^n x_i = \theta$ represents a hyperplane in $n$-dimensional space with normal vector $\mathbf{w} = [1, 1, \dots, 1]^\top$.
- **Severe limitation**: The orientation of the separating hyperplane is strictly fixed at $45^\circ$ (all weights must be $1$). It can only be translated by changing the threshold $\theta$. It cannot rotate to fit general linear boundaries, nor can it assign different importances to different inputs.

---

### 2.4 Frank Rosenblatt's Perceptron (1958)

#### Mathematical Formulation
Rosenblatt generalized the MP neuron by allowing continuous real-valued inputs and learnable real-valued weights:
- **Inputs**: $\mathbf{x} = [x_1, x_2, \dots, x_n]^\top \in \mathbb{R}^n$.
- **Weights**: $\mathbf{w} = [w_1, w_2, \dots, w_n]^\top \in \mathbb{R}^n$.
- **Bias**: $b \in \mathbb{R}$ (often written as $w_0$ with dummy input $x_0 = 1$, or as $-\theta$).
- **Net Input (Logit / Score)**:
  $$z = \mathbf{w}^\top \mathbf{x} + b = \sum_{i=1}^n w_i x_i + b$$
- **Activation Function (Signum / Step Function)**:
  - **0/1 formulation**:
    $$y = \begin{cases} 1 & \text{if } \mathbf{w}^\top \mathbf{x} + b \ge 0 \\ 0 & \text{otherwise} \end{cases}$$
  - **-1/+1 formulation**:
    $$y = \begin{cases} +1 & \text{if } \mathbf{w}^\top \mathbf{x} + b \ge 0 \\ -1 & \text{otherwise} \end{cases}$$

#### Geometric Interpretation
- The decision boundary is the hyperplane given by:
  $$\mathbf{w}^\top \mathbf{x} + b = 0$$
- The vector $\mathbf{w}$ is **orthogonal (perpendicular)** to the decision hyperplane and points in the direction of the positive half-space ($\mathbf{w}^\top \mathbf{x} + b > 0$).
- The perpendicular distance of any point $\mathbf{x}$ to the hyperplane is:
  $$d = \frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|_2}$$

---

### 2.5 Perceptron Learning Algorithm (PLA)

#### The Update Rule
For training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with labels $y_i \in \{-1, +1\}$ and no explicit bias (or absorbed into augmented $\mathbf{x} = [1, x_1, \dots, x_n]^\top$):
- Condition for misclassification:
  $$y_i (\mathbf{w}^\top \mathbf{x}_i) \le 0$$
- When misclassified, update:
  $$\mathbf{w}^{(t+1)} \leftarrow \mathbf{w}^{(t)} + \eta \, y_i \mathbf{x}_i$$
  where $\eta > 0$ is the learning rate (typically $\eta = 1$).

#### Intuition Behind the Update
- If $y_i = +1$ and $\mathbf{w}^\top \mathbf{x}_i \le 0$ (false negative):
  $$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + \mathbf{x}_i \implies \mathbf{w}_{\text{new}}^\top \mathbf{x}_i = \mathbf{w}_{\text{old}}^\top \mathbf{x}_i + \|\mathbf{x}_i\|^2 > \mathbf{w}_{\text{old}}^\top \mathbf{x}_i$$
  The angle between $\mathbf{w}$ and $\mathbf{x}_i$ decreases, pulling the boundary toward the correct side.
- If $y_i = -1$ and $\mathbf{w}^\top \mathbf{x}_i \ge 0$ (false positive):
  $$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \mathbf{x}_i \implies \mathbf{w}_{\text{new}}^\top \mathbf{x}_i = \mathbf{w}_{\text{old}}^\top \mathbf{x}_i - \|\mathbf{x}_i\|^2 < \mathbf{w}_{\text{old}}^\top \mathbf{x}_i$$
  The angle increases, pushing the hyperplane away.

---

### 2.6 Novikoff's Perceptron Convergence Theorem & Formal Proof

#### Theorem Statement
Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ be a linearly separable dataset with $y_i \in \{-1, +1\}$.
Assume:
1. All inputs are bounded: $\|\mathbf{x}_i\| \le R$ for all $i$.
2. There exists an optimal unit weight vector $\mathbf{w}^*$ ($\|\mathbf{w}^*\| = 1$) and margin $\gamma > 0$ such that:
   $$y_i (\mathbf{w}^{*\top} \mathbf{x}_i) \ge \gamma, \quad \forall i$$
Then, starting from $\mathbf{w}^{(0)} = \mathbf{0}$, the perceptron learning algorithm makes at most $k$ mistake updates, where:
$$k \le \left(\frac{R}{\gamma}\right)^2$$

#### Formal Mathematical Proof (Step-by-Step)
Suppose an update occurs at step $t$ on misclassified point $(\mathbf{x}_i, y_i)$ where $y_i (\mathbf{w}^{(t)\top} \mathbf{x}_i) \le 0$:
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + y_i \mathbf{x}_i$$

**Step 1: Lower bounding the inner product with $\mathbf{w}^*$**
Take the dot product with $\mathbf{w}^*$:
$$\mathbf{w}^{*\top} \mathbf{w}^{(t+1)} = \mathbf{w}^{*\top} \mathbf{w}^{(t)} + y_i (\mathbf{w}^{*\top} \mathbf{x}_i) \ge \mathbf{w}^{*\top} \mathbf{w}^{(t)} + \gamma$$
By induction over $k$ updates starting from $\mathbf{w}^{(0)} = \mathbf{0}$:
$$\mathbf{w}^{*\top} \mathbf{w}^{(k)} \ge k \gamma$$

**Step 2: Upper bounding the squared norm $\|\mathbf{w}^{(k)}\|^2$**
$$\|\mathbf{w}^{(t+1)}\|^2 = \|\mathbf{w}^{(t)} + y_i \mathbf{x}_i\|^2 = \|\mathbf{w}^{(t)}\|^2 + 2 y_i (\mathbf{w}^{(t)\top} \mathbf{x}_i) + \|\mathbf{x}_i\|^2$$
Since an update occurs only when $y_i (\mathbf{w}^{(t)\top} \mathbf{x}_i) \le 0$ and $\|\mathbf{x}_i\| \le R$:
$$\|\mathbf{w}^{(t+1)}\|^2 \le \|\mathbf{w}^{(t)}\|^2 + R^2$$
By induction over $k$ updates starting from $\mathbf{w}^{(0)} = \mathbf{0}$:
$$\|\mathbf{w}^{(k)}\|^2 \le k R^2$$

**Step 3: Combining using Cauchy-Schwarz inequality**
By Cauchy-Schwarz:
$$\mathbf{w}^{*\top} \mathbf{w}^{(k)} \le \|\mathbf{w}^*\| \, \|\mathbf{w}^{(k)}\| = 1 \cdot \|\mathbf{w}^{(k)}\| = \|\mathbf{w}^{(k)}\|$$
Squaring both sides:
$$(k \gamma)^2 \le (\mathbf{w}^{*\top} \mathbf{w}^{(k)})^2 \le \|\mathbf{w}^{(k)}\|^2 \le k R^2$$
$$k^2 \gamma^2 \le k R^2 \implies k \le \frac{R^2}{\gamma^2} = \left(\frac{R}{\gamma}\right)^2$$
$\blacksquare$

---

### 2.7 Linear Separability & Representation Power of Networks

#### Boolean Functions of $n$ Variables
- For $n$ binary variables, there are $2^n$ possible input combinations.
- A Boolean function assigns an output $0$ or $1$ to each input combination.
- Total number of distinct Boolean functions of $n$ inputs:
  $$\text{Total functions} = 2^{2^n}$$
  - For $n = 2$: $2^{2^2} = 2^4 = 16$ functions.
  - Of these 16 functions, **14 are linearly separable**; only **2 (XOR and XNOR)** are non-linearly separable.
  - For $n \ge 3$, the fraction of linearly separable functions approaches zero exponentially!

#### The XOR Problem
| $x_1$ | $x_2$ | $y$ (XOR) |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Geometrically, the positive instances $(0, 1)$ and $(1, 0)$ lie on one diagonal of the unit square, while the negative instances $(0, 0)$ and $(1, 1)$ lie on the opposite diagonal. No single straight line $w_1 x_1 + w_2 x_2 + b = 0$ can separate them.

#### Representation Power Theorems
1. **Any Boolean function can be represented by a 2-layer network of perceptrons**:
   - Hidden layer: Up to $2^n$ perceptrons (each perceptron activates for exactly one positive input minterm).
   - Output layer: 1 perceptron computing an OR over the active hidden units.
2. **Geometric Regions**:
   - **Single Perceptron**: Can separate half-spaces (open linear decision boundary).
   - **1 Hidden Layer (2 layers total)**: Can form any open or closed **convex polygon** by taking the intersection of half-spaces.
   - **2 Hidden Layers (3 layers total)**: Can form **arbitrary, disjoint, non-convex regions** by taking unions of convex polygons.

---

## 3. Core Formulas & Cheat Sheet

| Concept | Mathematical Equation | Notes / Key Constraints |
|:---|:---|:---|
| **MP Neuron Firing** | $y = 1 \iff \sum_{i=1}^n x_i \ge \theta$ | Inputs $x_i \in \{0, 1\}$, all weights $w_i = 1$, threshold $\theta \in \mathbb{Z}^+$ |
| **Perceptron Net Input** | $z = \mathbf{w}^\top \mathbf{x} + b$ | Real inputs $x_i \in \mathbb{R}$, real weights $w_i \in \mathbb{R}$, real bias $b \in \mathbb{R}$ |
| **Perceptron Decision Rule** | $y = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$ | $1$ if $\ge 0$, else $0$ (or $-1$) |
| **Perceptron Update Rule** | $\mathbf{w} \leftarrow \mathbf{w} + y_i \mathbf{x}_i$ | Applied only when $y_i (\mathbf{w}^\top \mathbf{x}_i) \le 0$ |
| **PLA Max Updates Bound** | $k \le \left(\frac{R}{\gamma}\right)^2$ | $R = \max \|\mathbf{x}_i\|$, $\gamma = \min y_i(\mathbf{w}^{*\top} \mathbf{x}_i)$ |
| **Decision Boundary** | $\mathbf{w}^\top \mathbf{x} + b = 0$ | Normal vector is $\mathbf{w}$; distance to origin is $\frac{\|b\|}{\|\mathbf{w}\|}$ |
| **Total Boolean Functions** | $N = 2^{2^n}$ | Grows double-exponentially with number of inputs $n$ |
| **Sufficient Hidden Units** | $2^n$ perceptrons | Sufficient to represent any arbitrary $n$-input Boolean function |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Smart Classroom Ventilation Control (Questions 1 to 5)

#### Context & Data
A smart classroom controller uses four binary indicators:
- $x_1$: Carbon dioxide level is high
- $x_2$: Room occupancy is high
- $x_3$: Indoor temperature is above comfort limit
- $x_4$: Outside air quality is acceptable
Each indicator is $1$ if true, $0$ otherwise.
The ventilation fan is governed by an MP-style threshold unit:
$$y = \begin{cases} 1 & \text{if } x_1 + x_2 + x_3 + x_4 \ge \theta \\ 0 & \text{otherwise} \end{cases}$$
Requirement: The fan switches on ($y = 1$) only when at least 3 of the 4 indicators are active ($1$).

---

#### Question 1
**Which threshold value implements the desired fan-control rule?**
- A) $\theta = 1$
- B) $\theta = 2$
- C) $\theta = 3$
- D) $\theta = 4$

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The condition states: "fan switches on only when at least three of the four indicators are active."
  Thus, $\sum_{i=1}^4 x_i \ge 3$.
  Comparing with the decision rule $\sum_{i=1}^4 x_i \ge \theta$, we get $\theta = 3$.
  - If $\theta = 1$, fan turns on if $\ge 1$ indicator is active (OR logic).
  - If $\theta = 2$, fan turns on if $\ge 2$ indicators are active (majority of 4).
  - If $\theta = 4$, fan turns on only if all 4 are active (AND logic).

---

#### Question 2
**For input $(1, 0, 1, 1)$, what output does the controller produce?**
- A) $y = 0$, because only one condition is absent
- B) $y = 1$, because three indicators are active
- C) $y = 0$, because all four indicators must be active
- D) $y = 1$, because the first indicator is active

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Input vector: $\mathbf{x} = (1, 0, 1, 1)$.
  Sum of inputs: $\sum_{i=1}^4 x_i = 1 + 0 + 1 + 1 = 3$.
  Since $\theta = 3$, we check: $3 \ge 3$, which is True.
  Therefore, $y = 1$. The fan switches on because three indicators are active.

---

#### Question 3
**How many binary input patterns switch on the fan?**
- A) 5
- B) 4
- C) 6
- D) 8

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The total number of binary inputs of length 4 is $2^4 = 16$.
  The fan turns on when the number of active indicators is either 3 or 4:
  - Number of patterns with exactly 3 ones: $\binom{4}{3} = \frac{4!}{3!1!} = 4$.
    Specifically: $(1,1,1,0), (1,1,0,1), (1,0,1,1), (0,1,1,1)$.
  - Number of patterns with all 4 ones: $\binom{4}{4} = 1$.
    Specifically: $(1,1,1,1)$.
  - Total firing patterns: $\binom{4}{3} + \binom{4}{4} = 4 + 1 = 5$.

---

#### Question 4
**If the threshold is accidentally changed to $\theta = 2$, what happens?**
- A) The fan switches on less often than intended.
- B) The fan switches on when two or more indicators are active.
- C) The fan switches on only when all inputs are 1, making activation too strict.
- D) The fan never switches on.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  With threshold $\theta = 2$, the condition for activation becomes $\sum_{i=1}^4 x_i \ge 2$.
  This triggers whenever 2, 3, or 4 indicators are active (total patterns = $\binom{4}{2} + \binom{4}{3} + \binom{4}{4} = 6 + 4 + 1 = 11$ patterns). Hence, the fan turns on when two or more indicators are active, which is more frequent than intended.

---

#### Question 5
**Which property makes this controller a threshold-logic system?**
- A) It predicts future classroom occupancy states.
- B) It updates weights after every incorrect output during a training process.
- C) It requires a hidden layer to make a decision.
- D) It compares a summed input value with a fixed threshold.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  A threshold-logic unit (such as the McCulloch-Pitts neuron or threshold gate) is fundamentally defined by taking an unweighted or weighted sum of inputs and comparing it against a predetermined numerical threshold.

---

### Case Study 2: Scholarship Application Screening (Questions 6 to 10)

#### Context & Data
A scholarship office screening aid flags applications for immediate review based on two binary features:
- $x_1$: Academic score above cutoff
- $x_2$: Financial-need certificate verified
The office wants the system to flag an application when **at least one** of these two conditions is true.
A perceptron with decision rule:
$$\hat{y} = \begin{cases} 1 & \text{if } w_1 x_1 + w_2 x_2 + b \ge 0 \\ 0 & \text{otherwise} \end{cases}$$
is tested.

---

#### Question 6
**Which logical behavior is described by the desired screening rule?**
- A) AND
- B) XOR
- C) OR
- D) XNOR

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The phrase "at least one of these two conditions is true" corresponds directly to the logical **OR** operator: $y = x_1 \lor x_2$.
  The output is 1 if $(x_1, x_2) \in \{(0,1), (1,0), (1,1)\}$ and 0 only when $(0,0)$.

---

#### Question 7
**Which parameter setting correctly implements this rule?**
- A) $w_1 = 1, w_2 = 1, b = -1.5$
- B) $w_1 = 1, w_2 = 1, b = -0.5$
- C) $w_1 = -1, w_2 = -1, b = 0.5$
- D) $w_1 = 1, w_2 = -1, b = 0$

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  We evaluate $z = w_1 x_1 + w_2 x_2 + b$ for all input combinations under option B ($w_1 = 1, w_2 = 1, b = -0.5$):
  - $(0,0): 1(0) + 1(0) - 0.5 = -0.5 < 0 \implies \hat{y} = 0$ (Correct)
  - $(0,1): 1(0) + 1(1) - 0.5 = +0.5 \ge 0 \implies \hat{y} = 1$ (Correct)
  - $(1,0): 1(1) + 1(0) - 0.5 = +0.5 \ge 0 \implies \hat{y} = 1$ (Correct)
  - $(1,1): 1(1) + 1(1) - 0.5 = +1.5 \ge 0 \implies \hat{y} = 1$ (Correct)
  Option A gives $b = -1.5$, which yields AND logic since $(0,1) \implies -0.5 < 0$.

---

#### Question 8
**For $w_1 = 1, w_2 = 1, b = -0.5$, what is the score for input $(0, 1)$?**
- A) 0.5
- B) -0.5
- C) 0
- D) 1.5

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Score $z = w_1 x_1 + w_2 x_2 + b = 1(0) + 1(1) - 0.5 = 1 - 0.5 = 0.5$.

---

#### Question 9
**What is the decision boundary for the parameter setting $w_1 = 1, w_2 = 1, b = -0.5$?**
- A) $x_1 - x_2 - 0.5 = 0$
- B) $x_1 + x_2 + 0.5 = 0$
- C) $2x_1 + x_2 = 0$
- D) $x_1 + x_2 - 0.5 = 0$

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  The decision boundary of a perceptron is the set of points where the linear score equals zero:
  $$w_1 x_1 + w_2 x_2 + b = 0 \implies 1\cdot x_1 + 1\cdot x_2 - 0.5 = 0 \implies x_1 + x_2 - 0.5 = 0$$

---

#### Question 10
**Why can a single perceptron represent this screening rule?**
- A) The positive and negative cases are linearly separable.
- B) The rule requires remembering earlier applications before deciding the next output.
- C) The rule has no decision boundary.
- D) The output changes randomly for the same input.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  A single perceptron can represent any function if and only if the positive class ($y = 1$) and negative class ($y = 0$) are linearly separable by a hyperplane. For OR, $(0,0)$ is separated from $(0,1), (1,0), (1,1)$ by the line $x_1 + x_2 - 0.5 = 0$.

---

### Case Study 3: Equipment Handover Record Review (Questions 11 to 15)

#### Context & Data
An engineering lab tracks:
- $x_1$: Battery pack returned
- $x_2$: Sensor module returned
Rule: Flag ($y = 1$) when **exactly one** of the two items has been returned. If neither or both are returned, $y = 0$.
Truth table:
- $(0,0) \to 0$
- $(0,1) \to 1$
- $(1,0) \to 1$
- $(1,1) \to 0$

---

#### Question 11
**Which logical behavior does the handover rule represent?**
- A) AND
- B) OR
- C) XOR
- D) NAND

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The output is 1 when inputs differ, and 0 when inputs are the same. This is the exclusive-OR (**XOR**) logic function.

---

#### Question 12
**What output is assigned to the input pattern $(1, 1)$?**
- A) $y = 1$, because one item has been returned
- B) $y = 0$, because exactly one item has not been returned
- C) $y = 1$, because both checks are complete
- D) $y = 0$, because both values are the same

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  For $(1, 1)$, both items have been returned. The rule explicitly states that if both items have been returned, the record is not flagged ($y = 0$), because both values are identical.

---

#### Question 13
**Which geometric property prevents a single perceptron from representing this rule?**
- A) The positive cases lie on opposite corners of the input square.
- B) The inputs are binary rather than real valued, so the rule cannot use a separator.
- C) The output has only two possible labels.
- D) The rule uses two input variables.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  In the unit square $[0, 1]^2$, the positive cases $(0, 1)$ and $(1, 0)$ lie on one diagonal, and the negative cases $(0, 0)$ and $(1, 1)$ lie on the opposite diagonal. Any single straight line dividing the square will leave at least one pair on the wrong side. Thus, the positive and negative points cannot be separated by a single hyperplane.

---

#### Question 14
**Which modified lab rule would be linearly separable?**
- A) Flag exactly one returned item.
- B) Flag when the two item statuses are different.
- C) Flag only when both items have been returned.
- D) Flag when neither or both items have been returned.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  "Flag only when both items have been returned" represents the **AND** rule: $(1, 1) \to 1$ and all other inputs $\to 0$. The AND rule is linearly separable (e.g., using $x_1 + x_2 - 1.5 = 0$). Options A and B are XOR, and Option D is XNOR, both of which are non-linearly separable.

---

#### Question 15
**What does linear separability mean in this setting?**
- A) Every input must have label 1 before a straight boundary can be checked.
- B) A line can separate the 1-labeled and 0-labeled inputs.
- C) The perceptron must avoid using weights.
- D) The input values must be stored in a sequence.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Linear separability in 2D Euclidean space means there exists a straight line $w_1 x_1 + w_2 x_2 + b = 0$ such that all positive points lie strictly on one side of the line and all negative points lie strictly on the other side.

---

### Case Study 4: Parcel Routing Decision Review (Questions 16 to 20)

#### Context & Data
A courier service trains a perceptron without bias ($b = 0$).
- Features: $x_1$ (compactness), $x_2$ (priority label).
- Labels: $y \in \{-1, +1\}$ ($+1$ for express, $-1$ for standard).
- Perceptron update rule:
  $$\mathbf{w} \leftarrow \mathbf{w} + y\mathbf{x} \quad \text{whenever } y(\mathbf{w}^\top \mathbf{x}) \le 0$$
- Training dataset:
  - $P_1$: $\mathbf{x} = (2, 1)$, $y = +1$
  - $P_2$: $\mathbf{x} = (1, 2)$, $y = +1$
  - $P_3$: $\mathbf{x} = (-1, -1)$, $y = -1$
- Initial weights: $\mathbf{w}^{(0)} = (0, 0)$.

---

#### Question 16
**What is the weight vector after processing $P_1$ once?**
- A) $\mathbf{w} = (2, 1)$
- B) $\mathbf{w} = (1, 2)$
- C) $\mathbf{w} = (-1, -1)$
- D) $\mathbf{w} = (0, 0)$

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  - Initially $\mathbf{w}^{(0)} = (0, 0)$.
  - Evaluate on $P_1 = (2, 1)$ with $y_1 = +1$:
    $$\mathbf{w}^{(0)\top} \mathbf{x}_1 = 0(2) + 0(1) = 0$$
    $$y_1 (\mathbf{w}^{(0)\top} \mathbf{x}_1) = 1(0) = 0 \le 0 \implies \text{Update triggered!}$$
  - Perform update:
    $$\mathbf{w}^{(1)} = \mathbf{w}^{(0)} + y_1 \mathbf{x}_1 = (0, 0) + 1(2, 1) = (2, 1)$$

---

#### Question 17
**After the update from $P_1$, what is $\mathbf{w}^\top \mathbf{x}$ for $P_2$?**
- A) 2
- B) 3
- C) 4
- D) 5

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Current weight vector: $\mathbf{w} = (2, 1)$.
  Feature vector for $P_2$: $\mathbf{x}_2 = (1, 2)$.
  Dot product:
  $$\mathbf{w}^\top \mathbf{x}_2 = w_1 x_{2,1} + w_2 x_{2,2} = 2(1) + 1(2) = 2 + 2 = 4$$

---

#### Question 18
**With $\mathbf{w} = (2, 1)$, is $P_3$ correctly classified?**
- A) No, because the raw score $\mathbf{w}^\top \mathbf{x}$ is greater than zero.
- B) No, because $y(\mathbf{w}^\top \mathbf{x}) < 0$.
- C) Yes, because $\mathbf{w}^\top \mathbf{x} = 0$.
- D) Yes, because $y(\mathbf{w}^\top \mathbf{x}) > 0$.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Current weight vector: $\mathbf{w} = (2, 1)$.
  Sample $P_3$: $\mathbf{x}_3 = (-1, -1)$, $y_3 = -1$.
  Score:
  $$\mathbf{w}^\top \mathbf{x}_3 = 2(-1) + 1(-1) = -2 - 1 = -3$$
  Check classification condition:
  $$y_3 (\mathbf{w}^\top \mathbf{x}_3) = (-1) \cdot (-3) = +3 > 0$$
  Since $y(\mathbf{w}^\top \mathbf{x}) > 0$, the classification is correct.

---

#### Question 19
**Which condition is needed for the perceptron convergence guarantee?**
- A) Every feature must be binary.
- B) The data must be linearly separable.
- C) The learning rate must be zero throughout all training updates.
- D) All labels must be positive.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Novikoff's Perceptron Convergence Theorem guarantees that the Perceptron Learning Algorithm will terminate and find a separating hyperplane in a finite number of steps if and only if the training dataset is **linearly separable**. If the data is not linearly separable, the algorithm can cycle indefinitely.

---

#### Question 20
**Which vector separates the three listed parcels?**
- A) $\mathbf{w} = (-1, 1)$
- B) $\mathbf{w} = (1, -1)$
- C) $\mathbf{w} = (-1, -1)$
- D) $\mathbf{w} = (1, 1)$

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  A separating vector $\mathbf{w}$ must satisfy $y_i (\mathbf{w}^\top \mathbf{x}_i) > 0$ for all three parcels:
  Let's test option D: $\mathbf{w} = (1, 1)$:
  - For $P_1 = (2, 1), y = +1$: $\mathbf{w}^\top \mathbf{x} = 1(2) + 1(1) = 3 > 0 \implies y(\mathbf{w}^\top \mathbf{x}) = 3 > 0$ (Correct)
  - For $P_2 = (1, 2), y = +1$: $\mathbf{w}^\top \mathbf{x} = 1(1) + 1(2) = 3 > 0 \implies y(\mathbf{w}^\top \mathbf{x}) = 3 > 0$ (Correct)
  - For $P_3 = (-1, -1), y = -1$: $\mathbf{w}^\top \mathbf{x} = 1(-1) + 1(-1) = -2 < 0 \implies y(\mathbf{w}^\top \mathbf{x}) = (-1)(-2) = 2 > 0$ (Correct)
  All samples are strictly correctly separated by $\mathbf{w} = (1, 1)$.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Boundary Condition on Updates**:
   - In PLA, an update is triggered if $y_i(\mathbf{w}^\top \mathbf{x}_i) \le 0$. If the score is exactly $0$, **an update is performed** because the point lies directly on the boundary, which is not strictly separated.
2. **MP Neuron Weights**:
   - Remember that in a standard McCulloch-Pitts neuron, all weights are strictly uniform ($w_i = 1$) and inputs are binary ($0$ or $1$). There are no negative weights; negative influences are modeled strictly through **veto/inhibitory inputs**.
3. **Number of Boolean Functions Formula**:
   - Be careful with exponents: $2^{2^n}$, not $2^n$ and not $2^{n^2}$.
   - For $n=2$: $2^{2^2} = 2^4 = 16$.
   - For $n=3$: $2^{2^3} = 2^8 = 256$.
   - For $n=4$: $2^{2^4} = 2^{16} = 65,536$.
4. **Perceptron Margin and Bound**:
   - The convergence bound is $k \le \frac{R^2}{\gamma^2}$. If margin $\gamma$ is doubled, the maximum number of updates drops by a factor of $4$.
