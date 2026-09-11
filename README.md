# Persian Restaurant Review Sentiment Classifier

A lightweight, self-contained sentiment analysis model built from scratch in Python using **NumPy** (without external machine learning frameworks like scikit-learn, PyTorch, or TensorFlow). 

The project classifies Persian restaurant customer reviews into **Positive** (`1`) and **Negative** (`0`) sentiments using rule-based Persian lexicon feature extraction and a custom-implemented **Logistic Regression** trained via **Stochastic Gradient Descent (SGD)**.

---

## 📌 Features

- **Zero Heavy Dependencies**: Implemented entirely with vanilla Python and NumPy.
- **Persian Domain-Specific Feature Engineering**:
  - Positive keyword frequency count (`عالی`, `خوشمزه`, `تازه`, etc.)
  - Negative keyword frequency count (`بد`, `سرد`, `شور`, etc.)
  - Intensifier/Amplifier presence flag (`حتما`, `پیشنهاد`, etc.)
  - Service criticism keyword count (`معطلی`, `تاخیر`, etc.)
  - Text length (total word count).
- **Z-Score Normalization**: Standardizes feature vectors with zero mean and unit variance.
- **From-Scratch Logistic Regression**:
  - Numerically stable Sigmoid function with overflow guards (`np.clip`).
  - Binary Cross-Entropy Loss computation.
  - Stochastic Gradient Descent (SGD) with shuffling at each epoch.
- **Evaluation & Diagnostics**:
  - Outputs training loss progression.
  - Generates sample-by-sample inference with positive-class probabilities.
  - Computes Accuracy, Precision, and Recall on the test dataset.

---

## 🛠 Prerequisites & Installation

Ensure you have Python 3.7+ installed. The only required package is `numpy`.
```bash
pip install numpy
