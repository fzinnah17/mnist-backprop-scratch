# MNIST Backpropagation from Scratch
**Designed and developed by:** Farnaz Zinnah  

---

## 🔗 Project Overview
This project implements a fully connected neural network (MLP) to classify handwritten digits from the MNIST dataset. The implementation follows a **manually coded forward and backward pass** (backpropagation) without using external deep learning frameworks like TensorFlow or PyTorch.  

---

## 📝 **About**
This project is an implementation of backpropagation for educational purposes.  

The model is trained on the **MNIST dataset** with:
- **Input Layer:** 784 neurons (28x28 pixel images flattened)
- **Two Hidden Layers:** 32 neurons each, with **sigmoid** activation
- **Output Layer:** 10 neurons with **softmax** activation

It uses **stochastic gradient descent (SGD)** to update weights after each sample, computing gradients manually for all layers.

---

## 🎯 **Description & Purpose**
✔ Implements a **Multi-Layer Perceptron (MLP)**  
✔ Uses **explicit loops** for matrix multiplications  
✔ **No external ML libraries** for training (only NumPy)  
✔ Performs training using **sample-by-sample** updates  
✔ Tracks accuracy improvement across **3 epochs**  

Expected accuracy: **50%-70%** after training.

---

## 🚀 **Tech Stack**
- **Python**
- **NumPy**
- **Matplotlib (for visualization)**
- **TensorFlow (only for loading MNIST dataset)**

---

## 🔧 **Features**
✅ **Manual Forward Pass:** Uses **sigmoid activation** for hidden layers, softmax for the output layer  
✅ **Manual Backpropagation:** Computes weight & bias gradients using explicit **Jacobian matrices**  
✅ **Cross-Entropy Loss:** Computes loss between predicted and true labels  
✅ **Weight Initialization:** Random weights from a normal distribution (`1/sqrt(n)`)  
✅ **Stochastic Gradient Descent (SGD):** Updates weights after **each sample**  
✅ **Evaluation:** Prints accuracy and loss after each epoch  
✅ **No Mini-batching:** Processes **one sample at a time**  

---

## 📊 **Results & Performance**
- **Initial Accuracy (before training):** ~10% (random guessing)  
- **After 3 epochs:** ~50%-70% accuracy  

Test loss is initially **higher** than train loss due to **overfitting** on training data and lack of regularization.

---

## 🛠 **Installation Instructions**
### 1️⃣ Clone the Repository:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/mnist-backprop-scratch.git
cd mnist-backprop-scratch
