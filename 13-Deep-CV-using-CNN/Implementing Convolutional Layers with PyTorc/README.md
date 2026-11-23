# CNN Fundamentals: Kernel Size, Padding, and Stride

## 1. Introduction to CNN

A Convolutional Neural Network (CNN) is a deep learning architecture mainly used for image processing and computer vision tasks.

CNNs automatically learn important features from images, such as:
- Edges
- Shapes
- Textures
- Patterns
- Objects

The convolution operation uses a filter (kernel) to extract these features from an input image.

---

## 2. Kernel Size (Filter Size)

A kernel is a small matrix that slides over an input image to extract features.

Common kernel sizes:
- 3 × 3
- 5 × 5
- 7 × 7

### Example

A 3 × 3 kernel:

```text
[1  0  1]
[0  1  0]
[1  0  1]
```

The kernel moves across the image and performs element-wise multiplication followed by summation to generate feature values.

**Why do we use kernels?**

- Detect edges and patterns.
- Extract useful image features.
- Learn spatial information.

---

## 3. Padding

Padding means adding extra pixels around the borders of an input image before applying convolution.

There are two common types:

### A. Valid Padding

- No padding is added.
- Output dimensions become smaller.
- Some border information may not be fully captured.

### B. Same Padding

- Padding is added to maintain the output's spatial dimensions when stride = 1.
- Helps preserve border information.
- Commonly used in CNN architectures.

### Example

Input image: 5 × 5

Kernel: 3 × 3

With valid padding:

Output = 3 × 3

With same padding and stride = 1:

Output = 5 × 5

---

## 4. Stride

Stride defines how many pixels the kernel moves at each step.

### Stride = 1

The kernel moves one pixel at a time.

- More detailed feature extraction.
- Larger output feature map.

### Stride = 2

The kernel moves two pixels at a time.

- Smaller output feature map.
- Reduces spatial dimensions.
- Can reduce computation.

---

## 5. Output Size Formula

The output dimension of a convolution operation is calculated using:

```text
Output Size = floor((N + 2P - K) / S) + 1
```

Where:

- N = Input size
- P = Padding
- K = Kernel size
- S = Stride

### Example

Suppose:

```text
Input Size = 28 × 28
Kernel Size = 3 × 3
Padding = 0
Stride = 1
```

Calculation:

```text
Output = ((28 + 2(0) - 3) / 1) + 1

Output = 26 × 26
```

Therefore, the output feature map is 26 × 26.

---

## 6. Practical Example Using TensorFlow / Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation='relu',
        input_shape=(28, 28, 1)
    )
)

model.summary()
```

### Understanding Parameters

- `filters=32`: Creates 32 different feature maps.
- `kernel_size=(3,3)`: Uses a 3 × 3 filter.
- `strides=(1,1)`: Moves one pixel at a time.
- `padding='valid'`: Adds no padding.
- `activation='relu'`: Applies ReLU activation.
- `input_shape=(28,28,1)`: Input image has height 28, width 28, and one channel.

Output shape:

```text
(None, 26, 26, 32)
```

---

## 7. Comparison Table

| Parameter | Meaning | Effect |
|---|---|---|
| Kernel Size | Filter dimensions | Controls feature extraction area |
| Padding | Extra border pixels | Controls output dimensions |
| Stride | Kernel movement | Controls output dimensions |

---

## 8. Key Takeaways

1. Kernel extracts features from images.
2. Padding helps control output dimensions and preserve border information.
3. Stride controls how far the kernel moves.
4. Smaller strides generally produce larger feature maps.
5. Same padding with stride 1 maintains spatial dimensions.
6. These three concepts are fundamental to understanding CNN architecture.

**Next Topics:** Convolution Operation, Feature Maps, Activation Functions (ReLU), Pooling Layers, and CNN Architectures.