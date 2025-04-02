# Foundations of Applied Machine Learning

These notes summarize the weekly sessions from the *Foundations of Applied Machine Learning* course at UC Riverside. Each session builds upon foundational ideas, with explanations, use cases, and key theoretical insights.

---

## Session 1: Introduction to Machine Learning

We began by discussing what machine learning is—automating pattern recognition through data-driven algorithms. We also covered basic programming concepts in Python as a foundation.

**Covered Topics:**
- What is ML? Supervised vs. Unsupervised vs. Reinforcement Learning
- How computers “learn” from data
- Common ML applications (e.g., image recognition, finance, healthcare)
- Basic Python: variables, loops, conditionals, built-in functions

---

## Session 2: Object-Oriented Programming and Probabilistic Thinking

This session introduced Object-Oriented Programming (OOP), followed by a conceptual entry into probabilistic reasoning and Bayesian thinking.

**OOP Concepts:**
- Class, object, method, attribute
- Encapsulation and reusability
- Importance in structuring large ML projects

**Probabilistic Programming:**
- Probability distributions and Bayes' Rule
- Modeling uncertainty and learning from data
- Motivation behind using prior beliefs in probabilistic models

---

## Session 3: Least Squares and Model Fitting

We covered curve fitting using least squares, a classical technique in statistics and ML. The idea is to choose parameters that minimize the total squared error between predictions and actual data.

**Theory:**
- Least squares loss: \( \sum (y_i - f(x_i))^2 \)
- Used for linear and nonlinear regression models
- Can be interpreted probabilistically as maximum likelihood under Gaussian noise

**Strengths:** Simple, interpretable, analytical solution available  
**Limitations:** Sensitive to outliers, assumes linearity unless extended

---

## Session 4: Monte Carlo Simulation & Bayesian Inference

We simulated a coin-toss experiment and explored how uncertainty is handled using **Bayesian inference**.

**Key Ideas:**
- Monte Carlo: random sampling to approximate solutions
- Binomial distribution for discrete outcomes
- Bayes’ Theorem: updating beliefs given evidence
- Priors (uniform) and posteriors over probability of fairness

**Use Cases:** A/B testing, diagnostics, reliability modeling

---

## Session 5: Clustering with K-Means

This session marked the transition to **unsupervised learning**. We explored K-means clustering, where data is grouped by similarity without labels.

**K-Means Process:**
1. Randomly initialize K cluster centers
2. Assign points to nearest cluster
3. Update cluster centers as means of assigned points

**Use Cases:** Customer segmentation, topic modeling, image compression  
**Challenges:** Choice of K, sensitivity to initialization, assumes spherical clusters

---

## Session 6: Classification and K-Nearest Neighbors (KNN)

We introduced **supervised learning**, particularly classification, where the goal is to map feature vectors to discrete labels.

**KNN Algorithm:**
- Memory-based: no explicit training
- Assigns label based on majority of k nearest labeled examples
- Distance metric critical (e.g., Euclidean)

**Use Cases:** Pattern recognition, recommender systems  
**Pros:** Intuitive, non-parametric  
**Cons:** Sensitive to noise, slow with large datasets

---

## Session 6 & 7 (Combined): Practical KNN and Feature Spaces

We continued developing intuition for classification with hands-on examples. Visualizing decision boundaries helped show how KNN adapts to different distributions in feature space.

**Topics:**
- Influence of K and decision boundaries
- Curse of dimensionality
- Role of normalization and preprocessing

---

## Session 8: Regression and Gradient Descent

We studied linear regression and how to **optimize weights** using **gradient descent**, a numerical method to minimize loss functions.

**Concepts:**
- Regression setup with matrix notation: \( y = Xw \)
- Loss function: squared error
- Gradient descent update: \( w := w - \eta 
abla L(w) \)

**Strengths:** Scales to large data  
**Drawbacks:** Sensitive to learning rate, local minima in non-convex loss

---

## Session 9: Mini-Batch Gradient Descent

We extended gradient descent to handle larger datasets using **mini-batch updates**.

**Ideas:**
- Full-batch: stable but slow
- Stochastic (SGD): fast but noisy
- Mini-batch: trade-off for efficiency and convergence stability

**Context:** Backbone of modern neural networks and deep learning frameworks

---

## Session 10: Neural Networks

We closed the course by introducing **feedforward neural networks**—a foundational architecture in deep learning.

**Core Ideas:**
- Layers of neurons apply linear transformation + non-linear activation
- Forward propagation passes input through layers to output
- Backpropagation updates weights using chain rule and loss gradients

**Activation Functions:** Sigmoid, ReLU  
**Applications:** Image/speech recognition, NLP, game AI  
**Challenges:** Overfitting, vanishing gradients, need for large data

---

This concludes the course outline. These sessions covered theoretical insights, practical techniques, and implementation skills in applied machine learning.