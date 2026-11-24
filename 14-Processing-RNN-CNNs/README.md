
# Recurrent Neural Network (RNN)

## 1. What is RNN?

A **Recurrent Neural Network (RNN)** is a type of Artificial Neural Network designed to process sequential data.

Unlike traditional neural networks, RNN can use information from previous time steps to understand the current input.

**Example:** When reading the sentence:

"I am learning Deep Learning."

The meaning of "Learning" depends partly on the previous words. RNN processes words sequentially and maintains information through its hidden state.

## 2. Why Do We Need RNN?

Traditional Neural Networks treat inputs independently and do not naturally maintain information about previous inputs.

RNN is useful when the order of data matters.

Examples:

- Text and language processing
- Speech recognition
- Time series forecasting
- Sentiment analysis
- Language translation

## 3. RNN Architecture

RNN consists of three main components:

1. **Input Layer (X):** Receives current input.
2. **Hidden State (H):** Maintains information from previous time steps.
3. **Output Layer (Y):** Produces the prediction.

### Important Components

- `X_t`: Current input at time step t.
- `H_t`: Current hidden state.
- `H_(t-1)`: Previous hidden state.
- `Y_t`: Current output.

The hidden state is passed from one time step to the next.

## 4. How Does RNN Work?

Suppose we have the sentence:

"I love Machine Learning."

RNN processes it step by step:

| Time Step | Input    | Processing                                      |
| --------- | -------- | ----------------------------------------------- |
| 1         | I        | Processes the first word                        |
| 2         | love     | Uses previous hidden state                      |
| 3         | Machine  | Uses information from earlier words             |
| 4         | Learning | Processes the final word using previous context |

Each time step uses the same weights.

## 5. Mathematical Formula

The hidden state is calculated using:

`h_t = tanh(W_xh * x_t + W_hh * h_(t-1) + b_h)`

Output calculation:

`y_t = W_hy * h_t + b_y`

Where:

- `x_t`: Current input
- `h_t`: Current hidden state
- `h_(t-1)`: Previous hidden state
- `W_xh`: Input-to-hidden weights
- `W_hh`: Hidden-to-hidden weights
- `W_hy`: Hidden-to-output weights
- `b`: Bias
- `tanh`: Activation function

## 6. Applications of RNN

1. **Speech Recognition:** Converts spoken language into text.
2. **Language Translation:** Translates one language into another.
3. **Sentiment Analysis:** Identifies positive or negative reviews.
4. **Time Series Forecasting:** Predicts future values using historical data.
5. **Text Generation:** Predicts the next word in a sequence.

## 7. Disadvantages of RNN

### 1. Vanishing Gradient Problem

During backpropagation through time, gradients can become extremely small, making it difficult to learn long-term dependencies.

### 2. Exploding Gradient Problem

Gradients can become extremely large, resulting in unstable training.

### 3. Sequential Processing

RNN processes sequence elements step by step, making parallel training difficult.

### 4. Short-Term Memory

Basic RNN struggles to remember information from very long sequences.

## 8. Advanced Types of RNN

### LSTM (Long Short-Term Memory)

- Designed to handle long-term dependencies.
- Uses gates to control information.
- Helps reduce the vanishing gradient problem.

### GRU (Gated Recurrent Unit)

- A simplified version of LSTM.
- Uses fewer gates.
- Often requires fewer parameters than LSTM.

## 9. CNN vs RNN

| CNN                      | RNN                                                           |
| ------------------------ | ------------------------------------------------------------- |
| Processes spatial data   | Processes sequential data                                     |
| C                        | Commonly used for text and timommonly used for imagese series |
| Uses convolution filters | Uses recurrent hidden states                                  |
| Learns spatial features  | Learns sequential relationships                               |

## 10. Important Key Points

- RNN stands for Recurrent Neural Network.
- RNN is designed for sequential data.
- Hidden state carries information from previous time steps.
- RNN uses shared weights across time steps.
- RNN training uses Backpropagation Through Time (BPTT).
- LSTM and GRU are advanced recurrent architectures.
- RNN suffers from vanishing and exploding gradients.

## Conclusion

RNN is an important Deep Learning architecture for sequential data. It maintains information through hidden states and uses previous information to process current inputs. However, basic RNN struggles with long-term dependencies, which led to the development of LSTM and GRU.

---

**Learning Topics:** Deep Learning | Recurrent Neural Networks | Sequential Data | Neural Network Architectures
