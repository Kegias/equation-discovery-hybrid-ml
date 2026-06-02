# Equation Discovery from Noisy Dynamical Systems using Hybrid Machine Learning

This project investigates how interpretable governing equations can be recovered from noisy dynamical-system data using a hybrid scientific machine learning pipeline.

The system studied is the **Van der Pol oscillator**, a nonlinear dynamical system commonly used to model self-sustained oscillations. The central challenge addressed in this project is that real-world trajectory measurements are often noisy, and direct numerical differentiation of noisy signals can produce unstable derivative estimates.

To address this, the project combines:

- Neural network-based trajectory smoothing
- Numerical derivative estimation
- Sparse regression-based equation discovery
- Robustness analysis under increasing noise levels

The goal is not only to fit observed trajectories, but to recover a compact and interpretable representation of the underlying dynamics.

---

## Project Motivation

Many physical, biological, and engineering systems are governed by differential equations. However, in practical settings, the exact governing equations may be unknown, and only noisy observations of the system may be available.

A major difficulty in equation discovery is derivative estimation. If observed states are noisy, then numerical derivatives such as `dx/dt` and `dy/dt` can become highly unstable. Since sparse equation discovery depends on these derivatives, noise can severely affect the recovered equations.

This project asks:

> Can equation discovery from noisy dynamical data be improved by using a neural network as a smoothing module before sparse regression?

Instead of using a neural network as a black-box final predictor, this project uses it as a preprocessing step to produce smoother trajectories. Sparse regression is then used to recover interpretable governing equations.

---

## System Studied: Van der Pol Oscillator


---

## Methodology

The project follows a four-stage pipeline.

### 1. Simulate noisy dynamical-system data

A clean Van der Pol trajectory is generated using numerical integration. Controlled observational noise is then added to simulate realistic measurement conditions.

### 2. Learn a smoothed trajectory representation

A neural network is trained to approximate the noisy state trajectories. The purpose of the neural network is not to replace equation discovery, but to reduce the effect of noise before derivative estimation.

### 3. Estimate derivatives

Derivatives are estimated from the smoothed trajectory rather than directly from noisy measurements. This helps reduce the noise amplification that usually occurs during numerical differentiation.

### 4. Discover sparse governing equations

A polynomial candidate library is constructed using terms such as:

```text
1, x, y, x², xy, y², x²y

The Van der Pol oscillator is defined as:

```text
dx/dt = y
dy/dt = μ(1 - x²)y - x
