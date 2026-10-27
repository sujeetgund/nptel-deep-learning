# Week 10: Convolutional Neural Networks (CNNs), Classic Architectures, and Visualization Techniques

---

## 1. Executive Summary & Exam Roadmap

Week 10 covers computer vision architectures and visual interpretability:
1. **The Convolution Operation**: Mathematical formulation (cross-correlation), receptive fields, kernel sliding, stride ($S$), and padding ($P$).
2. **Output Dimensions & Parameter Calculations**:
   - Spatial dimension formula: $O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$.
   - Convolutional layer parameter count: $(\text{kernel\_h} \times \text{kernel\_w} \times C_{\text{in}} + 1) \times C_{\text{out}}$.
3. **Pooling Layers**: Max pooling and Average pooling; parameter-free spatial downsampling; translation invariance.
4. **Global Average Pooling (GAP)**: Collapsing $H \times W \times C$ tensors into $1 \times 1 \times C$ vectors, eliminating multi-million parameter dense layers.
5. **Evolution of Classic Architectures**:
   - **LeNet-5 (1998)** & **AlexNet (2012)**.
   - **VGGNet (2014)**: The power of stacked $3 \times 3$ filters.
   - **GoogLeNet / Inception (2014)**: Multi-scale parallel convolutions and $1 \times 1$ bottleneck reductions.
   - **ResNet (2015)**: Residual learning $\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ and identity skip connections solving network degradation.
6. **Visualizing and Interpreting CNNs (Lecture 12)**:
   - Occlusion sensitivity analysis.
   - Guided Backpropagation and Class Activation Maps (CAM).
   - Deep Dream, Neural Style Transfer (Gram matrices), and Adversarial Perturbations (FGSM).

---

## 2. Comprehensive Lecture Notes & Mathematical Foundations

### 2.1 The Convolution Operation in Neural Networks

In deep learning, 2D convolution is implemented as cross-correlation between an input tensor $I$ and a learnable kernel/filter $K$:
$$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) K(m, n)$$
For multi-channel inputs (e.g. RGB with $C_{\text{in}} = 3$):
A single filter has dimensions $K_h \times K_w \times C_{\text{in}}$. It performs 3D spatial-channel dot products and sums over all input channels:
$$S(i, j) = b + \sum_{c=1}^{C_{\text{in}}} \sum_{m=1}^{K_h} \sum_{n=1}^{K_w} I(i + m, j + n, c) K(m, n, c)$$
If a layer has $C_{\text{out}}$ distinct filters, it outputs an activation tensor with $C_{\text{out}}$ channels/feature maps.

#### Spatial Dimension Formula
Given input spatial dimension $W$ (height or width), filter size $K$, padding $P$, and stride $S$:
$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
- **Valid Padding ($P = 0$)**: No padding; output shrinks: $O = \frac{W - K}{S} + 1$.
- **Same Padding**: Preserves input spatial resolution ($O = W$) when $S = 1$:
  $$W = \frac{W - K + 2P}{1} + 1 \implies 2P = K - 1 \implies P = \frac{K - 1}{2}$$
  *(Requires an odd filter size $K$, e.g. $K = 3 \implies P = 1$; $K = 5 \implies P = 2$).*

---

### 2.2 Pooling Layers & Global Average Pooling

#### Max Pooling & Average Pooling
- Operates independently on each feature map channel.
- A $2 \times 2$ window with stride $2$ extracts the maximum (or average) in non-overlapping $2 \times 2$ blocks, exactly **halving the spatial height and width**:
  $$O = \frac{W - 2}{2} + 1 = \frac{W}{2}$$
- **Key Characteristics**:
  1. **Zero learnable parameters**: Pooling has no weights or biases.
  2. **Translation Invariance**: If an object shifts by a few pixels, the maximum value within the pooling window remains identical.

#### Global Average Pooling (GAP) (Lin et al., 2013)
Instead of flattening an $H \times W \times C$ feature tensor into a huge 1D vector of length $H \cdot W \cdot C$ and multiplying by a massive weight matrix, GAP takes the spatial average over each $H \times W$ feature map:
$$y_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W A(i, j, c)$$
- Maps an $H \times W \times C$ tensor directly to a $1 \times 1 \times C$ vector.
- Eliminates millions of dense parameters, drastically reducing overfitting.

---

### 2.3 Evolution of Seminal CNN Architectures

```
LeNet-5 (1998) ---> AlexNet (2012) ---> VGGNet (2014) ---> GoogLeNet (2014) ---> ResNet (2015)
(5x5 filters)       (ReLU, Dropout)     (Stacked 3x3)      (Inception, 1x1)     (Skip Connections)
```

#### 1. VGGNet (Simonyan & Zisserman, 2014)
**Core Innovation**: Replace large filters ($5 \times 5, 7 \times 7$) with homogeneous stacks of small **$3 \times 3$ filters** with stride 1.
- **Receptive Field Equivalence**:
  - Two stacked $3 \times 3$ layers cover a receptive field of $(3 - 1) + (3 - 1) + 1 = 5 \times 5$.
  - Three stacked $3 \times 3$ layers cover a receptive field of $7 \times 7$.
- **Parameter Savings**:
  - One $5 \times 5$ filter across $C$ channels has $5^2 \cdot C^2 = 25 C^2$ parameters.
  - Two $3 \times 3$ filters across $C$ channels have $2 \times (3^2 \cdot C^2) = 18 C^2$ parameters (**$28\%$ fewer parameters**!).
  - Furthermore, incorporating two nonlinearities (e.g. ReLUs) instead of one makes the decision function significantly more discriminative.

#### 2. GoogLeNet / Inception (Szegedy et al., 2014)
**Core Innovation**: **Inception Modules** with multi-scale parallel filter paths.
- Instead of picking whether to use $1 \times 1$, $3 \times 3$, or $5 \times 5$, an Inception module runs all of them in parallel and concatenates their outputs depth-wise.
- **$1 \times 1$ Convolutions (Network-in-Network)**:
  Used as computational bottlenecks before $3 \times 3$ and $5 \times 5$ filters to reduce channel depth ($C_{\text{in}} \to C_{\text{reduced}}$), drastically slashing multiply-accumulate operations.

#### 3. ResNet (He et al., 2015)
**The Degradation Problem**: As plain networks get deeper (e.g. 56 layers vs 20 layers), training accuracy *saturates and then degrades rapidly*, not due to overfitting, but due to severe optimization difficulty.

**Residual Learning Formulation**:
Instead of hoping a stack of layers fits an underlying mapping $\mathcal{H}(\mathbf{x})$, let the layers fit a residual mapping $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$.
The original mapping is recast as:
$$\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$
via a feedforward **identity shortcut (skip connection)**.

**Why Residual Connections Cure Gradient Vanishing**:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \cdot \frac{\partial \mathcal{H}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right) = \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \frac{\partial \mathcal{L}}{\partial \mathcal{H}}$$
- Even if the layer gradients $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$ vanish to zero, the additive $+ \mathbf{I}$ term ensures that the gradient $\frac{\partial \mathcal{L}}{\partial \mathcal{H}}$ flows directly and unimpeded backwards through the skip connection across hundreds of layers!

---

### 2.4 Visualizing and Understanding CNNs (Lecture 12)

1. **Occlusion Sensitivity Analysis (Zeiler & Fergus, 2013)**:
   - Systematically place a gray masking square over different spatial patches $(i, j)$ of the input image and record the model's output probability for the correct class.
   - If masking a specific region causes the class confidence to plummet, that region contains the critical discriminative visual evidence.
2. **Gradient-Based Saliency & Guided Backpropagation**:
   - Compute the gradient of the target class score w.r.t input pixels: $\nabla_{\mathbf{x}} S_c(\mathbf{x})$. Pixels with large gradient magnitudes have the highest influence on the classification.
   - **Guided Backpropagation**: Zeroes out negative gradients during backpropagation through ReLUs, suppressing noise and producing sharp, human-interpretable feature visualizations.
3. **Adversarial Examples (Goodfellow et al., 2014)**:
   - Deep CNNs learn linear decision boundaries in high-dimensional pixel space that can be fooled by adding human-imperceptible perturbations aligned with the sign of the loss gradient:
     $$\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon \, \text{sign}(\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, y))$$
     *(Fast Gradient Sign Method - FGSM).*

---

## 3. Core Formulas & Cheat Sheet

| Concept | Mathematical Equation | Key Notes |
|:---|:---|:---|
| **Output Spatial Size** | $O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$ | $W$: input, $K$: kernel, $P$: padding, $S$: stride |
| **"Same" Padding** | $P = \frac{K - 1}{2}$ | Valid when $S = 1$ and $K$ is odd |
| **Conv Layer Weights** | $K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}$ | Ignores biases |
| **Conv Layer Biases** | $C_{\text{out}}$ | Exactly 1 bias per output feature map |
| **Pooling Output Size** | $O = \frac{W - K_{\text{pool}}}{S_{\text{pool}}} + 1$ | Has 0 learnable parameters |
| **Global Avg Pooling** | $y_c = \frac{1}{HW}\sum_{i,j} A_{ijc}$ | Collapses $H \times W \times C \to 1 \times 1 \times C$ |
| **Residual Block** | $\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ | Identity skip connection guarantees gradient highway |
| **FGSM Adversarial** | $\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon \, \text{sign}(\nabla_{\mathbf{x}} \mathcal{L})$ | Perturbation of magnitude $\epsilon$ fools CNN |

---

## 4. Assignment Solutions with Detailed Step-by-Step Explanations

### Case Study 1: Warehouse Package Image Inspection (Questions 1 to 5)

#### Context & Data
A logistics CNN inspects grayscale package images of size $96 \times 96$ ($C_{\text{in}} = 1$).
- First convolution: $3 \times 3$ filters, stride $S = 1$, padding $P = 1$.
- 24 filters in the layer.
- Followed by $2 \times 2$ max pooling with stride $S = 2$.

---

#### Question 1
**What is the output spatial size after the $3 \times 3$ convolution with stride 1 and padding 1?**
- A) $96 \times 96$
- B) $94 \times 94$
- C) $98 \times 98$
- D) $48 \times 48$

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Using the dimension formula:
  $$O = \frac{W - K + 2P}{S} + 1 = \frac{96 - 3 + 2(1)}{1} + 1 = \frac{95}{1} + 1 = 96$$
  The spatial size is completely preserved ($96 \times 96$).

---

#### Question 2
**If the convolution layer uses 24 filters, how many output feature maps are produced?**
- A) 3
- B) 1
- C) 24
- D) 96

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Each individual convolutional filter produces exactly one 2D output feature map. Therefore, 24 filters produce 24 output feature maps ($C_{\text{out}} = 24$).

---

#### Question 3
**What is the output size after applying $2 \times 2$ max pooling with stride 2 to a $96 \times 96$ feature map?**
- A) $96 \times 96$
- B) $24 \times 24$
- C) $94 \times 94$
- D) $48 \times 48$

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  $$O_{\text{pool}} = \frac{W - K_{\text{pool}}}{S_{\text{pool}}} + 1 = \frac{96 - 2}{2} + 1 = \frac{94}{2} + 1 = 47 + 1 = 48$$
  Both dimensions are halved to $48 \times 48$.

---

#### Question 4
**Why is pooling useful for this inspection task?**
- A) It removes the need for convolution under the case assumptions.
- B) It reduces spatial size and gives some robustness to small shifts.
- C) It increases every image dimension in the stated case setting.
- D) It converts grayscale images into RGB images in this specific scenario.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Pooling downsamples feature maps (reducing computation in deeper layers) and provides local translation invariance, ensuring that minor shifts in package placement do not alter feature activations.

---

#### Question 5
**What do early convolution filters usually learn in such image tasks?**
- A) Low-level patterns such as edges and corners
- B) The final class probability only
- C) The batch size for the described model
- D) The optimizer momentum in this specific scenario

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  Hierarchical feature learning in CNNs dictates that early layers learn primitive, low-level spatial features (Gabor-like edge detectors, color blobs, corners), while deeper layers compose them into complex semantic parts.

---

### Case Study 2: Crop Image Disease Detection (Questions 6 to 10)

#### Context & Data
A CNN classifies crop leaves.
Techniques used:
- Occlusion analysis (masking rectangular patches).
- Gradient-based pixel saliency.

---

#### Question 6
**If masking one leaf patch sharply reduces disease probability, what does that suggest?**
- A) The patch is irrelevant to the prediction and all class evidence.
- B) The patch contains information important for the prediction.
- C) The model has no learned filters for the data shown here.
- D) The image has no disease signal for the data shown here.

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  A steep drop in target class confidence upon occluding a region indicates that the model's decision boundary is heavily reliant on visual features located within that specific region.

---

#### Question 7
**What is the name of the technique that masks image regions and observes prediction changes?**
- A) Batch normalization
- B) Weight decay
- C) Dropout
- D) Occlusion analysis

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Systematically covering parts of an input image with a neutral gray or black mask to track changes in output classification probability is called **Occlusion Analysis** (Zeiler & Fergus).

---

#### Question 8
**Why are first-layer filters usually easier to interpret?**
- A) They operate directly on local pixel patterns.
- B) They contain no weights in this specific scenario.
- C) They always output class labels.
- D) They are identical across all filters.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  First-layer filters have small receptive fields and interact directly with raw input RGB pixels, making them readily visualizable as edge detectors, color contrasts, or oriented gradients.

---

#### Question 9
**A neuron that activates strongly for diseased leaves but weakly for healthy leaves has likely learned what?**
- A) The file name pattern in this specific scenario
- B) A disease-related visual feature
- C) The learning rate schedule
- D) A fixed image size only

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Class-selective activation across samples indicates that the filter is tuned to visual hallmarks characteristic of the disease class (e.g. necrotic spots, yellow lesions).

---

#### Question 10
**In gradient-based visualization, what does a large gradient magnitude for a pixel indicate?**
- A) The pixel has no influence for the data shown here.
- B) The pixel is outside the image.
- C) The activation is sensitive to that pixel.
- D) The filter has zero parameters.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  By definition of partial derivatives, $\frac{\partial S_c}{\partial I_{ij}}$ represents the sensitivity of score $S_c$ to pixel $I_{ij}$. A high absolute value means small modifications to that pixel produce large changes in model confidence.

---

### Case Study 3: Road Surface Image Classification (Questions 11 to 15)

#### Context & Data
Architectures compared: VGG ($3 \times 3$ stacks), Inception (multi-scale parallel filters), and ResNet (residual skip connections).

---

#### Question 11
**Why does VGG prefer stacks of small $3 \times 3$ filters?**
- A) To remove nonlinearities in the stated case setting.
- B) To avoid feature maps throughout the network.
- C) To force all images to be grayscale before every convolutional layer.
- D) To build deeper representations with controlled parameter growth.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Stacking two $3 \times 3$ filters matches the receptive field of a $5 \times 5$ filter while using $28\%$ fewer parameters and inserting two ReLU nonlinearities, enabling deeper, more expressive representations.

---

#### Question 12
**What is the main purpose of Inception-style parallel filters?**
- A) To delete pooling layers.
- B) To use only one spatial scale.
- C) To capture features at multiple spatial scales.
- D) To remove convolution in this specific scenario.

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  The Inception block applies $1 \times 1, 3 \times 3$, and $5 \times 5$ convolutions in parallel, allowing the network to simultaneously extract fine details and large contextual features at the same layer.

---

#### Question 13
**Why do ResNet skip connections help very deep networks train?**
- A) They improve information and gradient flow.
- B) They remove the input image.
- C) They replace all filters with one scalar.
- D) They make the output size independent of stride.

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The identity skip connection creates an additive gradient highway ($\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \frac{\partial \mathcal{L}}{\partial \mathcal{H}}$) that allows error signals to backpropagate directly without vanishing.

---

#### Question 14
**A module computes $\mathcal{H}(x) = \mathcal{F}(x) + x$. What architecture idea does this represent?**
- A) Bag-of-words
- B) Residual learning
- C) One-hot encoding
- D) Average pooling only

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Formulating the output as the sum of a learned residual function $\mathcal{F}(x)$ and the identity shortcut $x$ is the defining equation of **Residual Learning (ResNet)**.

---

#### Question 15
**Which architecture is most directly suited to mixing $1 \times 1, 3 \times 3$, and $5 \times 5$ convolutions in parallel?**
- A) Inception in this specific scenario
- B) A single-layer perceptron
- C) Plain linear regression
- D) A one-hot encoder for the described model

**Correct Answer: A**
- **Detailed Step-by-Step Explanation**:
  The **Inception architecture** (GoogLeNet) is explicitly designed around parallel multi-branch modules mixing $1 \times 1, 3 \times 3$, and $5 \times 5$ kernels.

---

### Case Study 4: Satellite Land-Cover Image Classification (Questions 16 to 20)

#### Context & Data
A satellite classification network processes $64 \times 64$ RGB patches ($C_{\text{in}} = 3$).
- Convolution: 32 filters, $5 \times 5$, stride $S = 1$, padding $P = 0$.
- Later stage: Global Average Pooling applied to an $8 \times 8 \times 128$ tensor.

---

#### Question 16
**What is the output spatial size after the $5 \times 5$, stride 1, no-padding convolution on a $64 \times 64$ image?**
- A) $64 \times 64$
- B) $32 \times 32$
- C) $60 \times 60$
- D) $68 \times 68$

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  $$O = \frac{W - K + 2P}{S} + 1 = \frac{64 - 5 + 0}{1} + 1 = 59 + 1 = 60$$
  The spatial size is $60 \times 60$.

---

#### Question 17
**How many output channels are produced by 32 filters?**
- A) 5
- B) 32
- C) 64
- D) 128

**Correct Answer: B**
- **Detailed Step-by-Step Explanation**:
  Each filter generates exactly one output channel. Thus, 32 filters generate 32 output channels ($C_{\text{out}} = 32$).

---

#### Question 18
**How many weight parameters are in one $5 \times 5$ RGB filter, ignoring bias?**
- A) 25
- B) 50
- C) 125
- D) 75

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  An RGB image has 3 color channels ($C_{\text{in}} = 3$). One filter spans all 3 channels:
  $$\text{Parameters} = K_h \times K_w \times C_{\text{in}} = 5 \times 5 \times 3 = 75$$

---

#### Question 19
**What is the output dimension after global average pooling of an $8 \times 8 \times 128$ tensor?**
- A) 64
- B) 1024
- C) 128
- D) 8192

**Correct Answer: C**
- **Detailed Step-by-Step Explanation**:
  Global average pooling computes the spatial average across the $8 \times 8 = 64$ positions for each channel independently, reducing each 2D slice to 1 scalar. With 128 channels, the output is a 128-dimensional vector.

---

#### Question 20
**Why can global average pooling reduce parameters compared with flattening followed by a large fully connected layer?**
- A) It increases the spatial dimensions.
- B) It removes all learned filters in this specific scenario.
- C) It increases the number of dense connections.
- D) It produces one summary value per feature map.

**Correct Answer: D**
- **Detailed Step-by-Step Explanation**:
  Flattening an $8 \times 8 \times 128$ tensor yields an 8,192-dimensional vector. Connecting that to a dense layer of 1024 units requires $8192 \times 1024 \approx 8.4 \text{ million}$ weights. In contrast, GAP has **zero** weights and condenses each channel to a single summary statistic.

---

## 5. NPTEL Exam Traps & High-Yield Pitfalls

1. **Number of Parameters in Conv Layer**:
   - For 1 filter: $K_h \times K_w \times C_{\text{in}}$ weights $+ 1$ bias.
   - For all filters in the layer: $(K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$.
   - **Common Trap**: Forgetting to multiply by the input channels $C_{\text{in}}$!
2. **Pooling Parameters**:
   - Pooling layers (Max pooling, Average pooling, Global Average Pooling) have **EXACTLY ZERO learnable parameters**.
3. **ResNet Residual Formula**:
   - $\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$. The gradient is $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathcal{H}}\left(\frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}\right)$. The identity matrix $\mathbf{I}$ prevents gradient vanishing.
