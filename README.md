# Equation Discovery from Noisy Dynamical Systems using Hybrid Machine Learning

This project explores how interpretable governing equations can be recovered from noisy dynamical system data using a hybrid scientific machine learning pipeline.

The system studied is the **Van der Pol oscillator**, a nonlinear dynamical system commonly used to model self-sustained oscillations. The main challenge addressed in this project is that real-world trajectory measurements are often noisy, and direct numerical differentiation of noisy signals can produce highly unstable derivative estimates.

To handle this, the project combines:

- Neural network-based trajectory smoothing
- Numerical derivative estimation
- Sparse regression-based equation discovery
- Robustness analysis under increasing noise levels

The goal is not only to fit the observed trajectory, but to recover a compact and interpretable representation of the underlying dynamics.

---

## Project Motivation

Many physical, biological, and engineering systems are governed by differential equations. However, in many practical situations, the exact governing equations are unknown and only noisy observations of the system are available.

A major difficulty in equation discovery is derivative estimation. If the observed states are noisy, then numerical derivatives such as `dx/dt` and `dy/dt` become highly unstable. Since sparse equation discovery depends on these derivatives, noise can severely affect the recovered equations.

This project asks:

> Can we improve equation discovery from noisy dynamical data by using a neural network as a smoothing module before sparse regression?

Instead of using a neural network as a black-box final predictor, this project uses it as a preprocessing step to produce smoother trajectories. Sparse regression is then used to recover interpretable governing equations.

---

## System Studied: Van der Pol Oscillator

The Van der Pol oscillator is defined by:

```text
dx/dt = y
dy/dt = μ(1 - x²)y - x