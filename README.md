# Cricket Intelligence Projects

## Live Demo

[Launch the IPL Intelligence Production app on Hugging Face Spaces](https://huggingface.co/spaces/anmol-unitmole/ipl-intelligence-production)

Production-oriented machine learning systems for cricket match prediction, probabilistic forecasting, tournament simulation, model evaluation and deployment.

This repository is structured as a cricket machine-learning portfolio containing separate end-to-end projects under one common repository.

## Projects

### 01 - IPL Intelligence Production

A production-oriented IPL match intelligence and championship simulation platform built from a four-version modeling investigation.

Final production champion: V2

Strict out-of-time evaluation: IPL 2025-2026

- Accuracy: 54.23%
- Balanced Accuracy: 55.00%
- F1 Score: 52.55%
- ROC-AUC: 0.5713
- Log Loss: 0.68585
- Brier Score: 0.24636
- ECE: 0.08786

Four modeling architectures were evaluated under controlled temporal testing. V2 outperformed the V1 architecture replay, V3 feature-selection/stacking challenger, and V4 player-first challenger and was therefore frozen for production.

[Explore the IPL project](./01-ipl-intelligence-production)

### 02 - ICC World Cup Intelligence Engine

The second project will extend the same production-oriented approach to ICC ODI World Cup match forecasting and tournament simulation. It will be added after its modeling and validation pipeline is completed.

## Engineering Principles

- Point-in-time feature generation
- Chronological and out-of-time evaluation
- Probability calibration
- Explicit model acceptance and rejection
- Reproducible pipelines
- Monte Carlo tournament simulation
- Explainability and historical backtesting
- Automated testing
- Gradio application deployment
- Hugging Face Spaces deployment

## Portfolio Structure

01 - IPL Intelligence Production
02 - ICC World Cup Intelligence Engine - upcoming

The objective of this repository is not to maximize an attractive offline accuracy number. The emphasis is on leakage-safe sports forecasting, rigorous experimentation, honest model comparison, production engineering and reproducibility.
