WiCaP: Reliable WiFi-Based Human Activity Recognition under Environment Shift via Weighted Conformal Prediction
Overview

WiFi-based Human Activity Recognition (HAR) enables contactless sensing using Channel State Information (CSI). While deep learning models achieve high accuracy in controlled environments, they often fail in real-world scenarios due to environmental shifts such as changes in room layout, hardware, or users.

This project proposes WiCaP, a reliability-aware framework designed to detect and reduce overconfident incorrect predictions in such conditions.

Problem Statement

Traditional HAR models assume that training and deployment data follow the same distribution. However, in practice:

Environments change (different rooms, furniture, devices)
Signal propagation shifts
Model confidence remains high even when predictions are wrong

This leads to silent failures, which are critical in applications like healthcare and smart monitoring.

Experimental Setup
Dataset
Experiments are performed on the UT-HAR dataset
The dataset consists of WiFi CSI signals collected for multiple human activities
Activities include common indoor actions such as walking, sitting, etc.
What We Did

We evaluated the model under environment shift conditions:

Train the model on one environment
Test it on a different environment (distribution shift)
Compare reliability of predictions using different methods
Methods Compared
ERM (Empirical Risk Minimization)
Standard deep learning baseline
APS (Adaptive Prediction Sets)
Conformal prediction for uncertainty estimation
RAPS (Regularized APS)
Improved version with better control over prediction size
Selective Prediction
Rejects predictions when confidence is low
Results
Key Observations
ERM achieves high accuracy (~98%) in controlled settings
→ but fails under environment shift
Under shift:
Model still gives high-confidence wrong predictions
This leads to silent failures
Conformal methods (APS, RAPS):
Provide uncertainty-aware predictions
Maintain coverage guarantees
Reduce incorrect confident outputs
Selective Prediction:
Further improves reliability by rejecting uncertain predictions
Reduces risk in critical scenarios
Silent Failure Rate (SFR)

We introduce a new metric:

SFR = fraction of high-confidence incorrect predictions

Insight:
ERM → High SFR ❌ (dangerous)
WiCaP → Low SFR ✅ (reliable)
What We Learned (Conclusion)
Accuracy alone is not sufficient for real-world deployment
Models must be reliability-aware, not just accurate
Conformal prediction is effective for:
Handling uncertainty
Providing trustable outputs
Selective prediction helps avoid risky decisions

👉 Final takeaway:
WiCaP significantly improves reliability of WiFi HAR systems under real-world conditions.

Repository Structure
wifi-har-reliability-conformal/
├── paper.pdf
├── notebook/
├── models/
├── results/
├── scripts/
└── requirements.txt
How to Run
pip install -r requirements.txt
python train.py
python test.py
Applications
Smart homes
Healthcare monitoring
Elderly assistance systems
Reliable wireless sensing
